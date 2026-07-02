from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from qos_mae.data import (
    compute_qos_reputation,
    inverse_rearrange,
    load_wsdream,
    make_train_test_split,
    rearrange_by_region,
    stable_region_order,
)
from qos_mae.region_mf import fit_region_aware_mf
from qos_mae.trainer import train_qos_mae
from qos_mae.utils import deep_update, ensure_dir, get_device, load_yaml, print_config, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QoS-MAE on WS-DREAM.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--qos", type=str, default="rt", choices=["rt", "tp", "response_time", "throughput"])
    parser.add_argument("--md", type=float, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mf_epochs", type=int, default=None)
    parser.add_argument("--mae_epochs", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--encoder_dim", type=int, default=None)
    parser.add_argument("--decoder_dim", type=int, default=None)
    parser.add_argument("--encoder_layers", type=int, default=None)
    parser.add_argument("--decoder_layers", type=int, default=None)
    parser.add_argument("--encoder_heads", type=int, default=None)
    parser.add_argument("--decoder_heads", type=int, default=None)
    parser.add_argument("--inference_samples", type=int, default=None)
    parser.add_argument("--mf_latent_dim", type=int, default=None)
    parser.add_argument("--cache_mf", action="store_true")
    return parser.parse_args()


def make_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {"qos_metric": args.qos}
    if args.md is not None:
        overrides["matrix_density"] = args.md
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.device is not None:
        overrides["device"] = args.device
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.mf_epochs is not None:
        overrides.setdefault("mf", {})["epochs"] = args.mf_epochs
    if args.mf_latent_dim is not None:
        overrides.setdefault("mf", {})["latent_dim"] = args.mf_latent_dim
    if args.mae_epochs is not None:
        overrides.setdefault("mae", {})["epochs"] = args.mae_epochs
    mae_overrides = overrides.setdefault("mae", {})
    if args.patch_size is not None:
        mae_overrides["patch_size"] = args.patch_size
    if args.encoder_dim is not None:
        mae_overrides["encoder_dim"] = args.encoder_dim
    if args.decoder_dim is not None:
        mae_overrides["decoder_dim"] = args.decoder_dim
    if args.encoder_layers is not None:
        mae_overrides["encoder_layers"] = args.encoder_layers
    if args.decoder_layers is not None:
        mae_overrides["decoder_layers"] = args.decoder_layers
    if args.encoder_heads is not None:
        mae_overrides["encoder_heads"] = args.encoder_heads
    if args.decoder_heads is not None:
        mae_overrides["decoder_heads"] = args.decoder_heads
    if args.inference_samples is not None:
        mae_overrides["inference_samples"] = args.inference_samples
    if not mae_overrides:
        overrides.pop("mae", None)
    return overrides


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config) if args.config else Path("configs") / ("tp.yaml" if args.qos in {"tp", "throughput"} else "rt.yaml")
    cfg = load_yaml(cfg_path)
    cfg = deep_update(cfg, make_overrides(args))
    set_seed(int(cfg.get("seed", 2026)))
    device = get_device(cfg.get("device", "auto"))
    print_config(cfg)
    data = load_wsdream(cfg["data_dir"], cfg["qos_metric"])
    full_matrix = data.matrix.astype(np.float32)
    print(f"[DATA] matrix={data.matrix_path}, shape={full_matrix.shape}, userlist={data.userlist_path}, wslist={data.wslist_path}")
    split = make_train_test_split(
        full_matrix,
        matrix_density=float(cfg["matrix_density"]),
        seed=int(cfg.get("seed", 2026)),
        valid_min=float(cfg.get("split", {}).get("valid_min", 0.0)),
    )
    print(f"[SPLIT] MD={float(cfg['matrix_density']):.2%}, train={split.train_mask.sum()}, test={split.test_mask.sum()}, valid={split.valid_mask.sum()}")
    qos_tag = "rt" if cfg["qos_metric"] in {"rt", "response_time"} else "tp"
    md_tag = f"md_{int(round(float(cfg['matrix_density']) * 100)):02d}"
    run_dir = ensure_dir(Path(cfg.get("output_dir", "outputs")) / qos_tag / md_tag / f"seed_{int(cfg.get('seed', 2026))}")
    save_json(cfg, run_dir / "config.json")
    mf_completed_path = run_dir / "mf_completed.npy"
    mf_pred_path = run_dir / "mf_pred.npy"
    if args.cache_mf and mf_completed_path.exists() and mf_pred_path.exists():
        print(f"[MF] Loading cached completion from {mf_completed_path}")
        completed = np.load(mf_completed_path)
    else:
        mf_result = fit_region_aware_mf(full_matrix, split.train_mask, data.user_regions, data.service_regions, cfg.get("mf", {}), device=device)
        completed = mf_result.completed_matrix
        np.save(mf_completed_path, completed.astype(np.float32))
        np.save(mf_pred_path, mf_result.predicted_matrix.astype(np.float32))
        save_json({"loss": mf_result.train_loss}, run_dir / "mf_training.json")
    rep = compute_qos_reputation(full_matrix, split.train_mask, alpha=float(cfg.get("reputation", {}).get("alpha", 0.6)))
    np.save(run_dir / "qos_reputation.npy", rep["qos_rep"])
    save_json({k: v for k, v in rep.items() if isinstance(v, float)}, run_dir / "reputation_stats.json")
    print(f"[REP] mu={rep['mu']:.6f}, sigma={rep['sigma']:.6f}, global_rep={rep['global_rep']:.6f}")
    user_order = stable_region_order(data.user_regions)
    service_order = stable_region_order(data.service_regions)
    np.save(run_dir / "user_order.npy", user_order)
    np.save(run_dir / "service_order.npy", service_order)
    completed_r = rearrange_by_region(completed, user_order, service_order)
    full_r = rearrange_by_region(full_matrix, user_order, service_order)
    train_mask_r = rearrange_by_region(split.train_mask, user_order, service_order)
    test_mask_r = rearrange_by_region(split.test_mask, user_order, service_order)
    valid_mask_r = rearrange_by_region(split.valid_mask, user_order, service_order)
    rep_r = rearrange_by_region(rep["qos_rep"], user_order, service_order)
    mae_result = train_qos_mae(
        completed_raw=completed_r,
        full_raw=full_r,
        train_mask=train_mask_r,
        test_mask=test_mask_r,
        valid_mask=valid_mask_r,
        qos_reputation=rep_r,
        cfg=cfg.get("mae", {}),
        output_dir=run_dir,
        device=device,
    )
    pred_orig = inverse_rearrange(mae_result.pred_matrix_raw, user_order, service_order)
    np.save(run_dir / "pred_matrix_original_order.npy", pred_orig.astype(np.float32))
    final = {
        "qos_metric": qos_tag,
        "matrix_density": float(cfg["matrix_density"]),
        "seed": int(cfg.get("seed", 2026)),
        "train_count": int(split.train_mask.sum()),
        "test_count": int(split.test_mask.sum()),
        "best_metrics_rearranged": mae_result.best_metrics,
        "final_metrics_rearranged": mae_result.final_metrics,
    }
    save_json(final, run_dir / "metrics.json")
    print("\n========== Final ==========")
    print(final)
    print(f"Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
