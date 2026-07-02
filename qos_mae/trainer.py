from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import trange

from .model import PatchMeta, QoSMAE, patch_reputation, patchify, unpatchify
from .utils import ensure_dir, save_json


@dataclass
class LogStandardScaler:
    mean_: float = 0.0
    std_: float = 1.0
    eps: float = 1e-8

    def fit(self, values: np.ndarray) -> LogStandardScaler:
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("Cannot fit scaler with no finite values.")
        logs = np.log1p(np.clip(values, a_min=0.0, a_max=None))
        self.mean_ = float(logs.mean())
        self.std_ = float(logs.std())
        if self.std_ < self.eps:
            self.std_ = 1.0
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        logs = np.log1p(np.clip(arr, a_min=0.0, a_max=None))
        return ((logs - self.mean_) / (self.std_ + self.eps)).astype(np.float32)

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        logs = arr * (self.std_ + self.eps) + self.mean_
        return np.expm1(logs).clip(min=0.0).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MAEResult:
    pred_matrix_raw: np.ndarray
    train_history: list[dict[str, float]]
    final_metrics: dict[str, float]
    best_metrics: dict[str, float]
    scaler: LogStandardScaler


def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    t = y_true[mask].astype(np.float64)
    p = y_pred[mask].astype(np.float64)
    finite = np.isfinite(t) & np.isfinite(p)
    if finite.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    err = p[finite] - t[finite]
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))), "n": int(finite.sum())}


def _build_training_tensors(
    completed_raw: np.ndarray,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    qos_reputation: np.ndarray,
    patch_size: int,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, PatchMeta, LogStandardScaler]:
    scaler = LogStandardScaler().fit(completed_raw[train_mask])
    normalized = scaler.transform(completed_raw)
    x_patches, meta = patchify(normalized, patch_size, fill_value=0.0)
    train_patches, _ = patchify(train_mask.astype(np.float32), patch_size, fill_value=0.0)
    valid_patches, _ = patchify(valid_mask.astype(np.float32), patch_size, fill_value=0.0)
    rep_patch, _ = patch_reputation(qos_reputation, valid_mask, patch_size)
    weights = np.where(train_patches > 0.5, 1.0, float(gamma)).astype(np.float32)
    weights *= valid_patches.astype(np.float32)
    return (
        torch.tensor(x_patches, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
        torch.tensor(valid_patches, dtype=torch.float32),
        torch.tensor(rep_patch, dtype=torch.float32),
        meta,
        scaler,
    )


def predict_full_matrix(
    model: QoSMAE,
    patches: torch.Tensor,
    patch_rep: torch.Tensor,
    meta: PatchMeta,
    scaler: LogStandardScaler,
    device: torch.device,
    samples: int = 5,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(max(1, samples)):
            pred_patches, _ = model(patches.to(device), patch_rep.to(device))
            preds.append(pred_patches.detach().cpu().numpy())
    pred_norm_patches = np.mean(np.stack(preds, axis=0), axis=0)
    pred_norm = unpatchify(pred_norm_patches, meta)
    return scaler.inverse_transform(pred_norm)


def train_qos_mae(
    completed_raw: np.ndarray,
    full_raw: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    valid_mask: np.ndarray,
    qos_reputation: np.ndarray,
    cfg: dict[str, Any],
    output_dir: str | Path,
    device: torch.device,
) -> MAEResult:
    output_dir = ensure_dir(output_dir)
    patch_size = int(cfg.get("patch_size", 32))
    gamma = float(cfg.get("gamma", 0.3))
    patches, loss_weights, valid_patches, patch_rep, meta, scaler = _build_training_tensors(completed_raw, train_mask, valid_mask, qos_reputation, patch_size, gamma)
    patches = patches.to(device)
    loss_weights = loss_weights.to(device)
    patch_rep = patch_rep.to(device)
    model = QoSMAE(
        num_patches=meta.num_patches,
        patch_dim=meta.patch_dim,
        encoder_dim=int(cfg.get("encoder_dim", 256)),
        decoder_dim=int(cfg.get("decoder_dim", 128)),
        encoder_layers=int(cfg.get("encoder_layers", 6)),
        decoder_layers=int(cfg.get("decoder_layers", 3)),
        encoder_heads=int(cfg.get("encoder_heads", 8)),
        decoder_heads=int(cfg.get("decoder_heads", 4)),
        mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
        dropout=float(cfg.get("dropout", 0.0)),
        base_mask_rate=float(cfg.get("base_mask_rate", 0.75)),
        beta=float(cfg.get("beta", 0.5)),
    ).to(device)
    epochs = int(cfg.get("epochs", 300))
    lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 0.05))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    eval_every = int(cfg.get("eval_every", 10))
    inference_samples = int(cfg.get("inference_samples", 5))
    verbose = bool(cfg.get("verbose", True))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    history: list[dict[str, float]] = []
    best_metrics = {"mae": float("inf"), "rmse": float("inf"), "epoch": -1}
    best_state = None
    iterator = trange(epochs, desc="QoS-MAE", disable=not verbose)
    for epoch in iterator:
        model.train()
        pred_patches, patch_mask = model(patches, patch_rep)
        cell_mask = patch_mask.to(device).float().unsqueeze(1)
        weights = loss_weights * cell_mask
        sq = (pred_patches - patches) ** 2
        loss = (sq * weights).sum() / (weights.sum() + 1e-8)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        scheduler.step()
        row = {"epoch": epoch + 1, "loss": float(loss.detach().cpu())}
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            pred_raw = predict_full_matrix(model, patches, patch_rep, meta, scaler, device, samples=inference_samples)
            metrics = mae_rmse(full_raw, pred_raw, test_mask)
            row.update({"test_mae": metrics["mae"], "test_rmse": metrics["rmse"]})
            iterator.set_postfix(loss=f"{row['loss']:.5f}", mae=f"{metrics['mae']:.5f}", rmse=f"{metrics['rmse']:.5f}")
            if metrics["mae"] < best_metrics["mae"]:
                best_metrics = {"mae": metrics["mae"], "rmse": metrics["rmse"], "epoch": epoch + 1}
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            iterator.set_postfix(loss=f"{row['loss']:.5f}")
        history.append(row)
    if best_state is not None:
        torch.save(best_state, output_dir / "qos_mae_best.pt")
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / "qos_mae_final.pt")
    pred_raw = predict_full_matrix(model, patches, patch_rep, meta, scaler, device, samples=inference_samples)
    final_metrics = mae_rmse(full_raw, pred_raw, test_mask)
    save_json({"history": history, "best_metrics": best_metrics, "final_metrics": final_metrics, "scaler": scaler.to_dict()}, output_dir / "mae_training.json")
    np.save(output_dir / "pred_matrix_rearranged.npy", pred_raw.astype(np.float32))
    return MAEResult(pred_raw.astype(np.float32), history, final_metrics, best_metrics, scaler)
