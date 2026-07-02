from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class QoSData:
    matrix: np.ndarray
    user_regions: list[str]
    service_regions: list[str]
    matrix_path: Path
    userlist_path: Path | None
    wslist_path: Path | None


@dataclass
class SplitData:
    train_mask: np.ndarray
    test_mask: np.ndarray
    valid_mask: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray


def _find_file(data_dir: str | Path, candidates: Iterable[str]) -> Path:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    lower_to_path = {p.name.lower(): p for p in data_dir.iterdir() if p.is_file()}
    for cand in candidates:
        p = lower_to_path.get(cand.lower())
        if p is not None:
            return p
    existing = ", ".join(sorted(p.name for p in data_dir.iterdir()))
    raise FileNotFoundError(f"Cannot find any of {list(candidates)} in {data_dir}. Existing: {existing}")


def _maybe_find_file(data_dir: str | Path, candidates: Iterable[str]) -> Path | None:
    try:
        return _find_file(data_dir, candidates)
    except FileNotFoundError:
        return None


def read_qos_matrix(path: str | Path) -> np.ndarray:
    path = Path(path)
    try:
        arr = np.loadtxt(path, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("matrix is not 2-D")
        return arr
    except Exception:
        df = pd.read_csv(path, sep=r"[\s,]+", header=None, engine="python", comment="#")
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        arr = df.to_numpy(dtype=np.float32)
        if arr.ndim != 2 or arr.size == 0:
            raise ValueError(f"Failed to parse a non-empty 2-D matrix from {path}")
        return arr


def _is_header_line(line: str) -> bool:
    parts = line.strip().split()
    first = parts[0].lower() if parts else ""
    return first in {"user", "userid", "user_id", "service", "serviceid", "service_id", "id"}


def _looks_like_as(token: str) -> bool:
    token_u = token.upper()
    return token_u.startswith("AS") and any(ch.isdigit() for ch in token_u)


def _parse_region_line(line: str, kind: str) -> str | None:
    raw = line.strip()
    if not raw or raw.startswith("#") or _is_header_line(raw):
        return None
    tab_cols = [c.strip() for c in raw.split("\t") if c.strip()]
    if len(tab_cols) >= 4:
        if kind == "user" and len(tab_cols) >= 3:
            return tab_cols[2]
        if kind == "service" and len(tab_cols) >= 4:
            return tab_cols[3]
    toks = raw.split()
    start = 2 if kind == "user" else 3
    if len(toks) <= start:
        return None
    as_idx = None
    for idx in range(start, len(toks)):
        if _looks_like_as(toks[idx]):
            as_idx = idx
            break
    if as_idx is not None and as_idx > start:
        return " ".join(toks[start:as_idx])
    return toks[start]


def read_regions(path: str | Path | None, expected_len: int, kind: str) -> list[str]:
    if path is None:
        return ["UNKNOWN"] * expected_len
    path = Path(path)
    regions: list[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rg = _parse_region_line(line, kind)
            if rg is not None:
                regions.append(rg)
    if len(regions) < expected_len:
        print(f"[WARN] Parsed only {len(regions)} {kind} regions from {path}; expected {expected_len}. Filling UNKNOWN.")
        regions.extend(["UNKNOWN"] * (expected_len - len(regions)))
    elif len(regions) > expected_len:
        print(f"[WARN] Parsed {len(regions)} {kind} regions from {path}; expected {expected_len}. Truncating extras.")
        regions = regions[:expected_len]
    return regions


def _resolve_wsdream_dir(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    direct_names = {p.name.lower() for p in data_dir.iterdir() if p.is_file()}
    if any(name in direct_names for name in {"rtmatrix", "rtmatrix.txt", "tpmatrix", "tpmatrix.txt"}):
        return data_dir
    nested = data_dir / "wsdream"
    if nested.exists() and nested.is_dir():
        nested_names = {p.name.lower() for p in nested.iterdir() if p.is_file()}
        if any(name in nested_names for name in {"rtmatrix", "rtmatrix.txt", "tpmatrix", "tpmatrix.txt"}):
            return nested
    return data_dir


def load_wsdream(data_dir: str | Path, qos_metric: str) -> QoSData:
    data_dir = _resolve_wsdream_dir(data_dir)
    metric = qos_metric.lower()
    if metric in {"rt", "response", "response_time", "responsetime"}:
        matrix_path = _find_file(data_dir, ["rtMatrix", "rtMatrix.txt", "rtmatrix", "rtmatrix.txt"])
    elif metric in {"tp", "throughput"}:
        matrix_path = _find_file(data_dir, ["tpMatrix", "tpMatrix.txt", "tpmatrix", "tpmatrix.txt"])
    else:
        raise ValueError("qos_metric must be one of: rt, response_time, tp, throughput")
    matrix = read_qos_matrix(matrix_path)
    m, n = matrix.shape
    userlist_path = _maybe_find_file(data_dir, ["userlist", "userlist.txt", "userList", "userList.txt"])
    wslist_path = _maybe_find_file(data_dir, ["wslist", "wslist.txt", "wsList", "wsList.txt"])
    user_regions = read_regions(userlist_path, m, "user")
    service_regions = read_regions(wslist_path, n, "service")
    return QoSData(matrix, user_regions, service_regions, matrix_path, userlist_path, wslist_path)


def make_train_test_split(matrix: np.ndarray, matrix_density: float, seed: int, valid_min: float = 0.0) -> SplitData:
    if not (0.0 < matrix_density < 1.0):
        raise ValueError("matrix_density must be in (0, 1), e.g. 0.05 for MD=5%")
    valid_mask = np.isfinite(matrix) & (matrix >= valid_min)
    flat_valid = np.flatnonzero(valid_mask.ravel())
    if flat_valid.size == 0:
        raise ValueError("No valid QoS entries found. Check valid_min and the matrix file.")
    rng = np.random.default_rng(seed)
    n_train = max(1, int(round(matrix_density * flat_valid.size)))
    train_flat = rng.choice(flat_valid, size=n_train, replace=False)
    train_mask = np.zeros(matrix.size, dtype=bool)
    train_mask[train_flat] = True
    train_mask = train_mask.reshape(matrix.shape)
    test_mask = valid_mask & (~train_mask)
    return SplitData(
        train_mask=train_mask,
        test_mask=test_mask,
        valid_mask=valid_mask,
        train_indices=np.column_stack(np.where(train_mask)),
        test_indices=np.column_stack(np.where(test_mask)),
    )


def compute_qos_reputation(matrix: np.ndarray, observed_mask: np.ndarray, alpha: float = 0.6) -> dict[str, np.ndarray | float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    observed_mask = observed_mask.astype(bool) & np.isfinite(matrix)
    obs = matrix[observed_mask]
    if obs.size == 0:
        raise ValueError("Cannot compute reputation with zero observed entries.")
    mu = float(obs.mean())
    sigma = float(obs.std())
    if sigma < 1e-12:
        sigma = 1.0
    rho_obs = np.zeros_like(matrix, dtype=np.float32)
    rho_obs[observed_mask] = np.exp(-((matrix[observed_mask] - mu) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    global_rep = float(rho_obs[observed_mask].mean())
    user_count = observed_mask.sum(axis=1)
    service_count = observed_mask.sum(axis=0)
    user_rep = np.full(matrix.shape[0], global_rep, dtype=np.float32)
    service_rep = np.full(matrix.shape[1], global_rep, dtype=np.float32)
    user_nonzero = user_count > 0
    service_nonzero = service_count > 0
    user_rep[user_nonzero] = (rho_obs.sum(axis=1)[user_nonzero] / user_count[user_nonzero]).astype(np.float32)
    service_rep[service_nonzero] = (rho_obs.sum(axis=0)[service_nonzero] / service_count[service_nonzero]).astype(np.float32)
    qos_rep = alpha * user_rep[:, None] + (1.0 - alpha) * service_rep[None, :]
    qos_rep = np.clip(qos_rep, 0.0, 1.0).astype(np.float32)
    return {
        "mu": mu,
        "sigma": sigma,
        "rho_obs": rho_obs,
        "user_rep": user_rep,
        "service_rep": service_rep,
        "qos_rep": qos_rep,
        "global_rep": global_rep,
    }


def stable_region_order(regions: list[str]) -> np.ndarray:
    return np.array(sorted(range(len(regions)), key=lambda i: (str(regions[i]), i)), dtype=np.int64)


def rearrange_by_region(arr: np.ndarray, user_order: np.ndarray, service_order: np.ndarray) -> np.ndarray:
    return arr[np.ix_(user_order, service_order)]


def inverse_rearrange(arr_rearranged: np.ndarray, user_order: np.ndarray, service_order: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr_rearranged)
    out[np.ix_(user_order, service_order)] = arr_rearranged
    return out
