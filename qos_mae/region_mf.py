from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import trange


def encode_regions(regions: list[str]) -> np.ndarray:
    mapping: dict[str, int] = {}
    ids = []
    for rg in regions:
        key = str(rg)
        if key not in mapping:
            mapping[key] = len(mapping)
        ids.append(mapping[key])
    return np.asarray(ids, dtype=np.int64)


class RegionAwareMF(nn.Module):
    def __init__(self, num_users: int, num_services: int, latent_dim: int) -> None:
        super().__init__()
        self.user_factors = nn.Parameter(torch.empty(num_users, latent_dim))
        self.service_factors = nn.Parameter(torch.empty(num_services, latent_dim))
        nn.init.normal_(self.user_factors, std=0.05)
        nn.init.normal_(self.service_factors, std=0.05)

    def forward(self) -> torch.Tensor:
        return self.user_factors @ self.service_factors.t()


def _region_mean_excluding_self(emb: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    n_groups = int(group_ids.max().item()) + 1
    dim = emb.shape[1]
    sums = emb.new_zeros((n_groups, dim))
    sums.index_add_(0, group_ids, emb)
    counts = torch.bincount(group_ids, minlength=n_groups).to(emb.dtype).unsqueeze(1)
    counts_i = counts[group_ids]
    sums_i = sums[group_ids]
    denom = counts_i - 1.0
    mean_excl = (sums_i - emb) / torch.clamp(denom, min=1.0)
    singleton = counts_i.squeeze(1) <= 1.0
    if singleton.any():
        mean_excl = mean_excl.clone()
        mean_excl[singleton] = emb[singleton]
    return mean_excl


@dataclass
class MFResult:
    completed_matrix: np.ndarray
    predicted_matrix: np.ndarray
    train_loss: list[float]


def fit_region_aware_mf(
    matrix: np.ndarray,
    train_mask: np.ndarray,
    user_regions: list[str],
    service_regions: list[str],
    cfg: dict[str, Any],
    device: torch.device,
) -> MFResult:
    matrix = np.asarray(matrix, dtype=np.float32)
    train_mask = train_mask.astype(bool)
    m, n = matrix.shape
    latent_dim = int(cfg.get("latent_dim", 64))
    model = RegionAwareMF(m, n, latent_dim).to(device)
    q = torch.tensor(np.nan_to_num(matrix, nan=0.0), dtype=torch.float32, device=device)
    mask = torch.tensor(train_mask, dtype=torch.bool, device=device)
    user_group = torch.tensor(encode_regions(user_regions), dtype=torch.long, device=device)
    service_group = torch.tensor(encode_regions(service_regions), dtype=torch.long, device=device)
    lr = float(cfg.get("lr", 0.01))
    epochs = int(cfg.get("epochs", 300))
    lambda_reg = float(cfg.get("lambda_reg", 1.0))
    eta_user = float(cfg.get("eta_user", 0.001))
    eta_service = float(cfg.get("eta_service", 0.001))
    patience = int(cfg.get("early_stop_patience", 60))
    clip_min = cfg.get("clip_min", 0.0)
    verbose = bool(cfg.get("verbose", True))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    best_loss = float("inf")
    best_state = None
    bad_epochs = 0
    iterator = trange(epochs, desc="Region-aware MF", disable=not verbose)
    for _ in iterator:
        model.train()
        pred = model()
        rec = 0.5 * torch.mean((pred[mask] - q[mask]) ** 2)
        l2 = 0.5 * lambda_reg * (model.user_factors.pow(2).mean() + model.service_factors.pow(2).mean())
        u_mean = _region_mean_excluding_self(model.user_factors, user_group)
        s_mean = _region_mean_excluding_self(model.service_factors, service_group)
        region_loss = 0.5 * eta_user * (model.user_factors - u_mean).pow(2).mean() + 0.5 * eta_service * (model.service_factors - s_mean).pow(2).mean()
        loss = rec + l2 + region_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        value = float(loss.detach().cpu())
        history.append(value)
        iterator.set_postfix(loss=f"{value:.6f}", rec=f"{float(rec.detach().cpu()):.6f}")
        if value + 1e-8 < best_loss:
            best_loss = value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_np = model().detach().cpu().numpy().astype(np.float32)
    if clip_min is not None:
        pred_np = np.maximum(pred_np, float(clip_min))
    completed = pred_np.copy()
    completed[train_mask] = matrix[train_mask]
    return MFResult(completed, pred_np, history)
