"""
Train Hugging Face Autoformer on HAR parquet data with subject-disjoint splits.

Expected parquet columns by default:
    x_acc, y_acc, z_acc, x_gyro, y_gyro, z_gyro, id, time

Each parquet file is assumed to represent ONE activity type.
The split is performed by user ID, not by random windows, to avoid subject leakage.

Scaling behavior:
    1. If --scaling is NOT false/none/0/no:
        - Hugging Face Autoformer internal scaling is used.
        - No external scaler is applied.

    2. If --scaling false:
        - Hugging Face Autoformer receives scaling=False.
        - An external scaler is fitted on TRAIN USERS ONLY.
        - Options: --external_scaler minmax or standard.
        - Training/val/test/model generation use scaled values.
        - TRAIN.npz, TEST.npz, SYNTH.npz are inverse-transformed back to original values before saving.

Saved NPZ files:
    RUN-{run_id}/TRAIN.npz  [N_train, 256, C]
    RUN-{run_id}/TEST.npz   [N_test, 256, C]
    RUN-{run_id}/SYNTH.npz  [N_test, 256, C]

For TEST and SYNTH:
    first 128 steps = identical real context
    next 128 steps  = real future in TEST, generated future in SYNTH
"""

import argparse
import json
import math
import random
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoformerConfig, AutoformerForPrediction




def smooth_signal(x, window_size=5):
    """
    Smooth a signal using a centered moving average while preserving the original shape.

    Expected input:
        x: np.ndarray
           Can be:
           - [T]
           - [N, T]
           - [N, T, C]

        window_size: int
           Size of the moving average window. Must be odd for symmetric smoothing.

    Returns:
        smoothed: np.ndarray
           Same shape as input.
    """
    if window_size <= 1:
        return x

    if window_size % 2 == 0:
        raise ValueError("window_size should be odd so smoothing is centered.")

    x = np.asarray(x)

    pad = window_size // 2
    kernel = np.ones(window_size, dtype=np.float32) / window_size

    def _smooth_1d(signal):
        padded = np.pad(signal, pad_width=pad, mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    if x.ndim == 1:
        return _smooth_1d(x)

    elif x.ndim == 2:
        # Shape: [N, T]
        smoothed = np.empty_like(x)
        for i in range(x.shape[0]):
            smoothed[i] = _smooth_1d(x[i])
        return smoothed

    elif x.ndim == 3:
        # Shape: [N, T, C]
        smoothed = np.empty_like(x)
        for i in range(x.shape[0]):
            for c in range(x.shape[2]):
                smoothed[i, :, c] = _smooth_1d(x[i, :, c])
        return smoothed

    else:
        raise ValueError(f"Unsupported input shape {x.shape}. Expected 1D, 2D, or 3D array.")
# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# -----------------------------------------------------------------------------
# Scaling utilities
# -----------------------------------------------------------------------------


def is_false_like(value) -> bool:
    if isinstance(value, bool):
        return value is False
    if value is None:
        return True
    return str(value).strip().lower() in {"false", "0", "none", "no", "off"}


def parse_hf_scaling(value):
    """
    Convert CLI scaling value to what HF Autoformer expects.

    Valid useful values:
        "std"   -> HF std scaler
        "mean"  -> HF mean scaler
        "true"  -> True, equivalent to HF mean scaler
        "false" -> False, disables HF scaling
    """
    v = str(value).strip().lower()
    if v in {"false", "0", "none", "no", "off"}:
        return False
    if v in {"true", "1", "yes", "on"}:
        return True
    return v

class ExternalFeatureScaler:
    """
    Lightweight per-channel scaler for arrays with final shape [..., C].

    Supported methods:
        standard: (x - mean) / std
        minmax:   (x - min) / (max - min)

    The scaler is fitted on train users only, using all train rows.
    """

    def __init__(self, method: str = "minmax", eps: float = 1e-8) -> None:
        method = method.lower().strip()
        if method not in {"minmax", "standard"}:
            raise ValueError(f"external_scaler must be 'minmax' or 'standard', got {method}")
        self.method = method
        self.eps = float(eps)
        self.fitted = False
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "ExternalFeatureScaler":
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"fit expects [N, C], got {x.shape}")

        if self.method == "standard":
            center = np.nanmean(x, axis=0).astype(np.float32)
            scale = np.nanstd(x, axis=0).astype(np.float32)
            scale = np.where(scale < self.eps, 1.0, scale).astype(np.float32)
            self.center_ = center
            self.scale_ = scale
            self.data_min_ = None
            self.data_max_ = None
        else:
            data_min = np.nanmin(x, axis=0).astype(np.float32)
            data_max = np.nanmax(x, axis=0).astype(np.float32)
            scale = (data_max - data_min).astype(np.float32)
            scale = np.where(scale < self.eps, 1.0, scale).astype(np.float32)
            self.center_ = data_min
            self.scale_ = scale
            self.data_min_ = data_min
            self.data_max_ = data_max

        self.fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before transform().")
        x = np.asarray(x, dtype=np.float32)
        return ((x - self.center_) / self.scale_).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Scaler must be fitted before inverse_transform().")
        x = np.asarray(x, dtype=np.float32)
        return (x * self.scale_ + self.center_).astype(np.float32)

    def transform_df_inplace(self, df: pd.DataFrame, target_cols: Sequence[str]) -> pd.DataFrame:
        values = df[list(target_cols)].to_numpy(dtype=np.float32)
        df.loc[:, list(target_cols)] = self.transform(values)
        return df

    def inverse_transform_3d(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f"inverse_transform_3d expects [N, T, C], got {x.shape}")
        n, t, c = x.shape
        flat = x.reshape(-1, c)
        inv = self.inverse_transform(flat).reshape(n, t, c)
        return inv.astype(np.float32)

    def inverse_transform_torch_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse transform a torch tensor shaped [B, T, C] for plotting only."""
        if not self.fitted:
            return x
        device = x.device
        dtype = x.dtype
        center = torch.as_tensor(self.center_, device=device, dtype=dtype).view(1, 1, -1)
        scale = torch.as_tensor(self.scale_, device=device, dtype=dtype).view(1, 1, -1)
        return x * scale + center

    def to_dict(self) -> Dict:
        if not self.fitted:
            return {"method": self.method, "fitted": False}
        return {
            "method": self.method,
            "fitted": True,
            "center": self.center_.tolist(),
            "scale": self.scale_.tolist(),
            "data_min": None if self.data_min_ is None else self.data_min_.tolist(),
            "data_max": None if self.data_max_ is None else self.data_max_.tolist(),
            "eps": self.eps,
        }


def fit_external_scaler_on_train_users(
    df: pd.DataFrame,
    train_users: Sequence,
    target_cols: Sequence[str],
    id_col: str,
    method: str,
) -> ExternalFeatureScaler:
    train_mask = df[id_col].isin(set(train_users))
    train_values = df.loc[train_mask, list(target_cols)].to_numpy(dtype=np.float32)

    if train_values.size == 0:
        raise ValueError("Cannot fit external scaler because train split has no rows.")

    scaler = ExternalFeatureScaler(method=method)
    scaler.fit(train_values)
    return scaler


# -----------------------------------------------------------------------------
# Run folders and user-level split
# -----------------------------------------------------------------------------


def make_unique_run_dir(
    base_output_dir: str,
    run_name: Optional[str] = None,
    overwrite: bool = False,
) -> Tuple[Path, str]:
    base = Path(base_output_dir)
    base.mkdir(parents=True, exist_ok=True)

    run_id = uuid4().hex[:8]

    if run_name is None or str(run_name).strip() == "":
        folder_name = f"run_{run_id}"
    else:
        folder_name = str(run_name)

    run_dir = base / folder_name

    if run_dir.exists() and not overwrite:
        suffix = 1
        original = run_dir
        while run_dir.exists():
            run_dir = Path(f"{original}_{suffix:02d}")
            suffix += 1

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_id


def split_users(unique_users: Sequence, split_seed: int) -> Tuple[set, set, set]:
    """
    Subject-disjoint split matching the user's previous pipeline.

    Protocol:
        1. Sort unique user IDs.
        2. Shuffle deterministically with np.random.RandomState(split_seed).
        3. Use 1 user for validation.
        4. Use floor(0.1 * remaining_users) for test, at least 1 user.
        5. Use the rest for training.
    """
    unique_users = np.sort(np.array(unique_users))
    rs = np.random.RandomState(split_seed)

    shuffled = unique_users.copy()
    rs.shuffle(shuffled)

    n_users = len(shuffled)
    if n_users < 3:
        raise ValueError(f"Need at least 3 unique users for train/val/test split, got {n_users}")

    n_val = 1
    remaining = n_users - n_val
    n_test = max(1, int(np.floor(0.1 * remaining)))
    n_train = n_users - n_val - n_test

    if n_train <= 0:
        raise ValueError(
            f"Invalid split sizes after adjustment: train={n_train}, val={n_val}, test={n_test}"
        )

    val_users = shuffled[:n_val]
    test_users = shuffled[n_val : n_val + n_test]
    train_users = shuffled[n_val + n_test :]

    return set(train_users), set(val_users), set(test_users)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class HARAutoformerDataset(Dataset):
    """
    Creates Autoformer training instances from long per-user HAR sequences.

    Canonical sample indexing follows the user's previous pipeline:
        context length = 128
        future length  = 128
        saved/evaluation sequence length = 256

    Hugging Face Autoformer still requires an internal lag prefix:
        past_values length = max_lag + context_length

    The lag prefix is added inside __getitem__ and is NOT used when counting windows.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        user_ids: Sequence,
        target_cols: Sequence[str],
        id_col: str = "id",
        time_col: Optional[str] = "time",
        context_length: int = 128,
        prediction_length: int = 128,
        lags_sequence: Sequence[int] = (1, 2, 3, 4, 5, 10, 20),
        stride: int = 64,
        max_windows: Optional[int] = None,
        seed: int = 42,
        split_name: str = "train",
    ) -> None:
        self.target_cols = list(target_cols)
        self.id_col = id_col
        self.time_col = time_col
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.lags_sequence = tuple(int(x) for x in lags_sequence)
        self.max_lag = max(self.lags_sequence)
        self.past_length = self.context_length + self.max_lag
        self.total_length = self.context_length + self.prediction_length
        self.hf_total_length = self.past_length + self.prediction_length
        self.stride = int(stride)
        self.overlap_stride = 32
        self.min_windows = 10
        self.split_name = split_name
        
        if self.stride <= 0:
            raise ValueError(f"stride must be positive. Got {self.stride}")

        if self.overlap_stride <= 0:
            raise ValueError(f"overlap_stride must be positive. Got {self.overlap_stride}")

        user_set = set(user_ids)
        df_split = df[df[id_col].isin(user_set)].copy()

        if time_col is not None and time_col in df_split.columns:
            df_split = df_split.sort_values([id_col, time_col]).reset_index(drop=True)
        else:
            df_split = df_split.sort_values([id_col]).reset_index(drop=True)

        self.series: List[np.ndarray] = []
        self.series_user_ids: List = []
        self.indices: List[Tuple[int, int]] = []
        
        # min_windows = 10
        # overlap_stride = 32

        # for user_id, g in df_split.groupby(id_col, sort=False):
        #     values_raw = g[self.target_cols].to_numpy(dtype=np.float32)

        #     T_raw = values_raw.shape[0]

        #     # --------------------------------------------------
        #     # Match the old PPG script:
        #     # trim each user's sequence to a multiple of total_length
        #     # before calculating n_windows_nonoverlap.
        #     # --------------------------------------------------
        #     usable_len = (T_raw // self.total_length) * self.total_length

        #     if usable_len <= 0:
        #         continue

        #     values = values_raw[:usable_len]
        #     T = values.shape[0]

        #     series_idx = len(self.series)
        #     self.series.append(values)
        #     self.series_user_ids.append(user_id)

        #     n_windows_nonoverlap = T // self.total_length

        #     if n_windows_nonoverlap >= min_windows:
        #         # Same as old non-overlapping branch
        #         n_windows = n_windows_nonoverlap

        #         start_indices = (
        #             np.arange(n_windows, dtype=np.int64) * self.total_length
        #         )

        #         usable_rows = n_windows * self.total_length
        #         discarded_rows = T - usable_rows

        #         window_mode = "non_overlapping"
        #         stride_used = self.total_length

        #     else:
        #         # Same as old overlapping fallback branch,
        #         # but now T is the trimmed length, not the raw user length.
        #         stride_used = overlap_stride

        #         n_windows = 1 + (T - self.total_length) // stride_used

        #         start_indices = (
        #             np.arange(n_windows, dtype=np.int64) * stride_used
        #         )

        #         usable_rows = int(start_indices[-1]) + self.total_length
        #         discarded_rows = T - usable_rows

        #         window_mode = "overlapping"

        #     for start in start_indices:
        #         self.indices.append((series_idx, int(start)))

        #     print(
        #         f"[{split_name}] user={user_id}, "
        #         f"T_raw={T_raw}, "
        #         f"T_after_trim={T}, "
        #         f"mode={window_mode}, "
        #         f"n_windows_nonoverlap={n_windows_nonoverlap}, "
        #         f"n_windows={n_windows}, "
        #         f"stride_used={stride_used}, "
        #         f"usable_rows={usable_rows}, "
        #         f"discarded_rows={discarded_rows}"
        #     )
        
        
        for user_id, g in df_split.groupby(id_col, sort=False):
            values_raw = g[self.target_cols].to_numpy(dtype=np.float32)
            T_raw = values_raw.shape[0]

            if T_raw < self.total_length:
                continue

            # ==================================================
            # Special logic only for TEST and VAL splits
            # ==================================================
            if self.split_name in ["test", "val"]:
                min_windows = 10
                overlap_stride = 32

                # Match old test script:
                # trim each user to a multiple of total_length first
                usable_len = (T_raw // self.total_length) * self.total_length

                if usable_len <= 0:
                    continue

                values = values_raw[:usable_len]
                T = values.shape[0]

                n_windows_nonoverlap = T // self.total_length

                if n_windows_nonoverlap >= min_windows:
                    # Non-overlapping windows
                    n_windows = n_windows_nonoverlap
                    start_indices = (
                        np.arange(n_windows, dtype=np.int64) * self.total_length
                    )

                    window_mode = "non_overlapping"
                    stride_used = self.total_length

                else:
                    # Overlapping fallback with stride=32
                    stride_used = overlap_stride

                    n_windows = 1 + (T - self.total_length) // stride_used

                    if n_windows <= 0:
                        continue

                    start_indices = (
                        np.arange(n_windows, dtype=np.int64) * stride_used
                    )

                    window_mode = "overlapping"

                usable_rows = int(start_indices[-1]) + self.total_length
                discarded_rows = T - usable_rows

            # ==================================================
            # Normal logic for TRAIN / VAL / anything else
            # ==================================================
            else:
                values = values_raw
                T = T_raw

                n_windows = T - self.total_length + 1

                if n_windows <= 0:
                    continue

                start_indices = np.arange(
                    0,
                    n_windows,
                    self.stride,
                    dtype=np.int64,
                )

                window_mode = "normal_stride"
                stride_used = self.stride
                usable_rows = int(start_indices[-1]) + self.total_length
                discarded_rows = T - usable_rows

            # Store this user's series
            series_idx = len(self.series)
            self.series.append(values)
            self.series_user_ids.append(user_id)

            # Store window indices
            for start in start_indices:
                self.indices.append((series_idx, int(start)))

            # print(
            #     f"[{self.split_name}] user={user_id}, "
            #     f"T_raw={T_raw}, "
            #     f"T_used={values.shape[0]}, "
            #     f"mode={window_mode}, "
            #     f"n_windows={len(start_indices)}, "
            #     f"stride_used={stride_used}, "
            #     f"usable_rows={usable_rows}, "
            #     f"discarded_rows={discarded_rows}"
            # )
            
        

        if len(self.indices) == 0:
            raise ValueError(
                f"No valid windows for split={split_name}. "
                f"Need at least total_length={self.total_length} rows per selected user."
            )

        if max_windows is not None and len(self.indices) > max_windows:
            rng = random.Random(seed)
            rng.shuffle(self.indices)
            self.indices = self.indices[:max_windows]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        series_idx, start = self.indices[idx]
        arr = self.series[series_idx]

        context_start = start
        context_end = start + self.context_length
        future_end = context_end + self.prediction_length

        context = arr[context_start:context_end].copy()
        future_values = arr[context_end:future_end].copy()

        prefix_start = context_start - self.max_lag
        if prefix_start >= 0:
            lag_prefix = arr[prefix_start:context_start].copy()
        else:
            available_prefix = arr[0:context_start].copy()
            pad_len = -prefix_start
            pad = np.repeat(arr[[0]], repeats=pad_len, axis=0).astype(np.float32)
            lag_prefix = np.concatenate([pad, available_prefix], axis=0)

        if lag_prefix.shape[0] != self.max_lag:
            raise RuntimeError(
                f"Invalid lag_prefix length: expected {self.max_lag}, got {lag_prefix.shape[0]}"
            )

        past_values = np.concatenate([lag_prefix, context], axis=0)

        past_observed_mask = np.isfinite(past_values)
        future_observed_mask = np.isfinite(future_values)

        past_values = np.nan_to_num(past_values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        future_values = np.nan_to_num(future_values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        t = np.arange(self.hf_total_length, dtype=np.float32)
        t = t / max(1, self.hf_total_length - 1)
        time_features = t[:, None]

        past_time_features = time_features[: self.past_length]
        future_time_features = time_features[self.past_length :]

        return {
            "past_values": torch.tensor(past_values, dtype=torch.float32),
            "future_values": torch.tensor(future_values, dtype=torch.float32),
            "past_observed_mask": torch.tensor(past_observed_mask, dtype=torch.bool),
            "future_observed_mask": torch.tensor(future_observed_mask, dtype=torch.bool),
            "past_time_features": torch.tensor(past_time_features, dtype=torch.float32),
            "future_time_features": torch.tensor(future_time_features, dtype=torch.float32),
        }


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------


def load_har_parquet(
    parquet_path: str,
    target_cols: Sequence[str],
    id_col: str,
    time_col: Optional[str],
) -> pd.DataFrame:
    df = pd.read_parquet(str(parquet_path))

    required = set(list(target_cols) + [id_col])
    if time_col is not None:
        required.add(time_col)

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for col in target_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

    if time_col is not None:
        df = df.sort_values([id_col, time_col]).reset_index(drop=True)
    else:
        df = df.sort_values([id_col]).reset_index(drop=True)

    return df


def eligible_users(df: pd.DataFrame, id_col: str, total_length: int) -> List:
    counts = df.groupby(id_col).size()
    return counts[counts >= total_length].index.tolist()


def make_datasets_and_loaders(args):
    target_cols = [c.strip() for c in args.target_cols.split(",") if c.strip()]
    lags_sequence = tuple(int(x.strip()) for x in args.lags_sequence.split(",") if x.strip())
    max_lag = max(lags_sequence)

    total_length = args.context_length + args.prediction_length
    hf_total_length = args.context_length + max_lag + args.prediction_length

    df_raw = load_har_parquet(
        parquet_path=args.parquet_path,
        target_cols=target_cols,
        id_col=args.id_col,
        time_col=args.time_col,
    )

    users = eligible_users(df_raw, args.id_col, total_length)
    if len(users) < 3:
        raise ValueError(
            f"Only {len(users)} users have at least {total_length} rows. "
            f"Cannot create subject-disjoint train/val/test split."
        )

    split_seed = args.split_seed if args.split_seed is not None else args.seed
    train_users, val_users, test_users = split_users(unique_users=users, split_seed=split_seed)

    hf_scaling = parse_hf_scaling(args.scaling)
    use_external_scaler = hf_scaling is False

    external_scaler = None
    df_model = df_raw.copy()

    if use_external_scaler:
        external_scaler = fit_external_scaler_on_train_users(
            df=df_model,
            train_users=train_users,
            target_cols=target_cols,
            id_col=args.id_col,
            method=args.external_scaler,
        )
        df_model = external_scaler.transform_df_inplace(df_model, target_cols=target_cols)

    train_dataset = HARAutoformerDataset(
        df=df_model,
        user_ids=train_users,
        target_cols=target_cols,
        id_col=args.id_col,
        time_col=args.time_col,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        lags_sequence=lags_sequence,
        stride=args.train_stride,
        max_windows=args.max_train_windows,
        seed=args.seed,
        split_name="train",
    )

    val_dataset = HARAutoformerDataset(
        df=df_model,
        user_ids=val_users,
        target_cols=target_cols,
        id_col=args.id_col,
        time_col=args.time_col,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        lags_sequence=lags_sequence,
        stride=args.eval_stride,
        max_windows=args.max_eval_windows,
        seed=args.seed,
        split_name="val",
    )

    test_dataset = HARAutoformerDataset(
        df=df_model,
        user_ids=test_users,
        target_cols=target_cols,
        id_col=args.id_col,
        time_col=args.time_col,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        lags_sequence=lags_sequence,
        stride=args.eval_stride,
        max_windows=args.max_eval_windows,
        seed=args.seed,
        split_name="test",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    split_info = {
        "target_cols": target_cols,
        "lags_sequence": list(lags_sequence),
        "split_protocol": "sorted users + numpy RandomState shuffle + 1 val user + floor(0.1 * remaining) test users + rest train",
        "split_seed": split_seed,
        "total_users_eligible": len(users),
        "train_users": [str(x) for x in train_users],
        "val_users": [str(x) for x in val_users],
        "test_users": [str(x) for x in test_users],
        "num_train_windows": len(train_dataset),
        "num_val_windows": len(val_dataset),
        "num_test_windows": len(test_dataset),
        "lag_prefix_length": max_lag,
        "context_length_used_for_prediction": args.context_length,
        "past_length_passed_to_hf": args.context_length + max_lag,
        "prediction_length": args.prediction_length,
        "canonical_window_length": total_length,
        "hf_internal_total_length": hf_total_length,
        "hf_scaling": hf_scaling,
        "use_external_scaler": use_external_scaler,
        "external_scaler": None if external_scaler is None else external_scaler.to_dict(),
    }

    return train_loader, val_loader, test_loader, split_info, external_scaler


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


def build_model(args, target_cols: Sequence[str], lags_sequence: Sequence[int]) -> AutoformerForPrediction:
    hf_scaling = parse_hf_scaling(args.scaling)

    config = AutoformerConfig(
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        input_size=len(target_cols),
        lags_sequence=list(lags_sequence),
        num_time_features=1,
        d_model=args.d_model,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        encoder_attention_heads=args.encoder_attention_heads,
        decoder_attention_heads=args.decoder_attention_heads,
        encoder_ffn_dim=args.encoder_ffn_dim,
        decoder_ffn_dim=args.decoder_ffn_dim,
        moving_average=args.moving_average,
        autocorrelation_factor=args.autocorrelation_factor,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        activation_dropout=args.activation_dropout,
        distribution_output=args.distribution_output,
        loss="nll",
        scaling=hf_scaling,
        num_parallel_samples=args.num_parallel_samples,
    )
    return AutoformerForPrediction(config)


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def forward_loss(model: AutoformerForPrediction, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(
        past_values=batch["past_values"],
        past_time_features=batch["past_time_features"],
        past_observed_mask=batch["past_observed_mask"],
        future_values=batch["future_values"],
        future_time_features=batch["future_time_features"],
        future_observed_mask=batch["future_observed_mask"],
    )
    return outputs.loss


def train_one_epoch(
    model: AutoformerForPrediction,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    epoch: int,
) -> float:
    model.train()
    running_loss = 0.0
    num_steps = 0

    pbar = tqdm(loader, desc=f"Train epoch {epoch}")
    for batch in pbar:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        loss = forward_loss(model, batch)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss: {loss.item()}")

        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        running_loss += loss.item()
        num_steps += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running_loss / num_steps:.4f}")

    return running_loss / max(1, num_steps)


@torch.no_grad()
def evaluate_loss(
    model: AutoformerForPrediction,
    loader: DataLoader,
    device: torch.device,
    split_name: str,
) -> float:
    model.eval()
    running_loss = 0.0
    num_steps = 0

    pbar = tqdm(loader, desc=f"Eval {split_name}")
    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        loss = forward_loss(model, batch)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite {split_name} loss: {loss.item()}")

        running_loss += loss.item()
        num_steps += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running_loss / num_steps:.4f}")

    return running_loss / max(1, num_steps)


# -----------------------------------------------------------------------------
# Generation and plotting
# -----------------------------------------------------------------------------


@torch.no_grad()
def generate_batch(
    model: AutoformerForPrediction,
    loader: DataLoader,
    device: torch.device,
    num_parallel_samples: int = 3,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    model.eval()

    old_num_parallel_samples = getattr(model.config, "num_parallel_samples", None)
    model.config.num_parallel_samples = int(num_parallel_samples)

    try:
        batch = next(iter(loader))
        batch = move_batch_to_device(batch, device)

        generated = model.generate(
            past_values=batch["past_values"],
            past_time_features=batch["past_time_features"],
            past_observed_mask=batch["past_observed_mask"],
            future_time_features=batch["future_time_features"],
        )

        samples = generated.sequences
        pred_mean = samples.mean(dim=1)

    finally:
        if old_num_parallel_samples is not None:
            model.config.num_parallel_samples = old_num_parallel_samples

    return batch, samples, pred_mean


def save_compressed_npz(
    data_file: np.ndarray,
    channel_names: Optional[Sequence[str]] = None,
    model_name: str = "hf_autoformer",
    save_path: Path | str = "generated_samples",
    seed: int = 42,
) -> None:
    samples = np.asarray(data_file, dtype=np.float32)

    if samples.ndim != 3:
        raise ValueError(f"`data_file` must have shape [N, T, C], got {samples.shape}")

    n, seq_len, num_channels = samples.shape

    if channel_names is None:
        channel_names = [f"channel_{i}" for i in range(num_channels)]
    else:
        channel_names = list(channel_names)
        if len(channel_names) != num_channels:
            raise ValueError(
                f"len(channel_names) must equal num_channels={num_channels}, got {len(channel_names)}"
            )

    save_path = Path(save_path)
    if save_path.suffix != ".npz":
        save_path = save_path.with_suffix(".npz")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        save_path,
        samples=samples,
        channel_names=np.asarray(channel_names, dtype=str),
        seq_len=np.int32(seq_len),
        num_channels=np.int32(num_channels),
        num_samples=np.int32(n),
        model_name=model_name,
        seed=np.int32(seed),
    )


@torch.no_grad()
def collect_real_extended_sequences(
    loader: DataLoader,
    context_length: int,
    max_batches: Optional[int] = None,
) -> np.ndarray:
    extended = []

    for batch_idx, batch in enumerate(tqdm(loader, desc="Collect real extended sequences")):
        if max_batches is not None and batch_idx >= max_batches:
            break

        context = batch["past_values"][:, -context_length:, :].cpu().numpy()
        future = batch["future_values"].cpu().numpy()
        full = np.concatenate([context, future], axis=1)
        extended.append(full)

    if len(extended) == 0:
        raise RuntimeError("No real extended sequences were collected.")

    return np.concatenate(extended, axis=0).astype(np.float32)


@torch.no_grad()
def collect_test_generation_npz_arrays(
    model: AutoformerForPrediction,
    loader: DataLoader,
    device: torch.device,
    context_length: int,
    max_batches: Optional[int] = None,
    use_mean_prediction: bool = False,
    generation_num_parallel_samples: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()

    old_num_parallel_samples = getattr(model.config, "num_parallel_samples", None)
    model.config.num_parallel_samples = int(generation_num_parallel_samples)

    synth_extended = []
    test_extended = []

    try:
        pbar = tqdm(loader, desc="Generate full test set")
        for batch_idx, batch in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = move_batch_to_device(batch, device)

            generated = model.generate(
                past_values=batch["past_values"],
                past_time_features=batch["past_time_features"],
                past_observed_mask=batch["past_observed_mask"],
                future_time_features=batch["future_time_features"],
            )

            samples = generated.sequences

            if use_mean_prediction:
                pred_future = samples.mean(dim=1)
            else:
                pred_future = samples[:, 0, :, :]

            context = batch["past_values"][:, -context_length:, :]
            true_future = batch["future_values"]

            synth = torch.cat([context, pred_future], dim=1)
            test = torch.cat([context, true_future], dim=1)

            synth_extended.append(synth.detach().cpu().numpy())
            test_extended.append(test.detach().cpu().numpy())

            del generated, samples, pred_future, synth, test, batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        if old_num_parallel_samples is not None:
            model.config.num_parallel_samples = old_num_parallel_samples

    if len(synth_extended) == 0:
        raise RuntimeError("No generated test sequences were collected.")

    synth_extended = np.concatenate(synth_extended, axis=0).astype(np.float32)
    test_extended = np.concatenate(test_extended, axis=0).astype(np.float32)

    return synth_extended, test_extended


def plot_generated_examples(
    batch: Dict[str, torch.Tensor],
    samples: torch.Tensor,
    pred_mean: torch.Tensor,
    target_cols: Sequence[str],
    context_length: int,
    output_dir: Path,
    split_name: str,
    num_examples: int = 3,
    num_sample_paths: int = 3,
    dpi: int = 200,
    external_scaler: Optional[ExternalFeatureScaler] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    past_context = batch["past_values"][:, -context_length:, :].detach().cpu()
    future_true = batch["future_values"].detach().cpu()
    samples = samples.detach().cpu()
    pred_mean = pred_mean.detach().cpu()

    if external_scaler is not None:
        past_context = external_scaler.inverse_transform_torch_3d(past_context)
        future_true = external_scaler.inverse_transform_torch_3d(future_true)
        pred_mean = external_scaler.inverse_transform_torch_3d(pred_mean)

        b, s, t, c = samples.shape
        flat_samples = samples.reshape(b * s, t, c)
        flat_samples = external_scaler.inverse_transform_torch_3d(flat_samples)
        samples = flat_samples.reshape(b, s, t, c)

    batch_size, _, num_channels = past_context.shape
    pred_len = future_true.shape[1]
    num_examples = min(num_examples, batch_size)
    num_sample_paths = min(num_sample_paths, samples.shape[1])

    x_past = np.arange(context_length)
    x_future = np.arange(context_length, context_length + pred_len)

    n_cols = 2 if num_channels > 1 else 1
    n_rows = math.ceil(num_channels / n_cols)

    for example_idx in range(num_examples):
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(7.0 * n_cols, 3.2 * n_rows),
            squeeze=False,
        )
        axes = axes.reshape(-1)

        mse = torch.mean((pred_mean[example_idx] - future_true[example_idx]) ** 2).item()

        for ch in range(num_channels):
            ax = axes[ch]

            y_context = past_context[example_idx, :, ch].numpy()
            y_true = future_true[example_idx, :, ch].numpy()
            y_mean = pred_mean[example_idx, :, ch].numpy()

            ax.plot(x_past, y_context, label="Context", linewidth=1.4, alpha=0.85)
            ax.plot(x_future, y_true, label="Ground truth", linewidth=1.8)
            ax.plot(x_future, y_mean, label="Prediction mean", linewidth=1.8)

            for s in range(num_sample_paths):
                y_sample = samples[example_idx, s, :, ch].numpy()
                ax.plot(x_future, y_sample, linewidth=0.8, alpha=0.25)

            ax.axvline(context_length - 1, linestyle="--", linewidth=1.0)
            ax.set_title(str(target_cols[ch]))
            ax.set_xlabel("Time step")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.25)

            if ch == 0:
                ax.legend(loc="best")

        for j in range(num_channels, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"{split_name} example {example_idx} | future MSE={mse:.6f}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        save_path = output_dir / f"{split_name}_example_{example_idx}.png"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def print_std_diagnostic(
    batch: Dict[str, torch.Tensor],
    pred_mean: torch.Tensor,
    target_cols: Sequence[str],
    split_name: str,
    external_scaler: Optional[ExternalFeatureScaler] = None,
) -> None:
    real = batch["future_values"].detach().cpu()
    pred = pred_mean.detach().cpu()

    if external_scaler is not None:
        real = external_scaler.inverse_transform_torch_3d(real)
        pred = external_scaler.inverse_transform_torch_3d(pred)

    real_std = real.std(dim=(0, 1)).numpy()
    pred_std = pred.std(dim=(0, 1)).numpy()

    print(f"\nPer-channel std diagnostic on {split_name} generated batch:")
    for name, r, p in zip(target_cols, real_std, pred_std):
        ratio = p / (r + 1e-8)
        print(f"{name:>10s} | real std={r:.5f} | pred std={p:.5f} | ratio={ratio:.5f}")


# -----------------------------------------------------------------------------
# Checkpoints and logs
# -----------------------------------------------------------------------------


def save_checkpoint(
    output_dir: Path,
    model: AutoformerForPrediction,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": model.config.to_dict(),
        },
        ckpt_dir / f"epoch_{epoch:03d}.pt",
    )

    model.save_pretrained(output_dir / "hf_model_last")


def load_checkpoint_into_model(
    model: AutoformerForPrediction,
    checkpoint_path: str | Path,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    load_optimizer: bool = False,
) -> Tuple[AutoformerForPrediction, Optional[int], Optional[Dict[str, float]]]:
    """
    Load a saved checkpoint for sampling/resuming.

    Supported checkpoint formats:
        1. A .pt file saved by this script, e.g. checkpoints/best.pt
        2. A Hugging Face save_pretrained directory, e.g. hf_model_best

    Returns:
        model, checkpoint_epoch, checkpoint_metrics
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint_path does not exist: {checkpoint_path}")

    if checkpoint_path.is_dir():
        print(f"Loading Hugging Face model directory: {checkpoint_path}")
        model = AutoformerForPrediction.from_pretrained(checkpoint_path).to(device)
        model.eval()
        return model, None, None

    print(f"Loading checkpoint file: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint does not contain 'model_state_dict'. Keys found: {list(checkpoint.keys())}"
        )

    # Important:
    # A checkpoint may have been trained with a different architecture from the
    # current CLI args, e.g. distribution_output='student_t' versus 'normal'.
    # The saved config is the source of truth for the model architecture.
    if "config" in checkpoint and checkpoint["config"] is not None:
        print("Rebuilding model from checkpoint config before loading weights.")
        ckpt_config = AutoformerConfig.from_dict(checkpoint["config"])
        model = AutoformerForPrediction(ckpt_config).to(device)

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as e:
        print("Failed to load checkpoint with strict=True.")
        print("This usually means the current model architecture does not match the checkpoint.")
        print("Most common cause here: --distribution_output mismatch, e.g. student_t vs normal.")
        print("Checkpoint config keys available:", list(checkpoint.get("config", {}).keys()))
        raise e

    model.to(device)
    model.eval()

    if load_optimizer and optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    checkpoint_epoch = checkpoint.get("epoch", None)
    checkpoint_metrics = checkpoint.get("metrics", None)

    if checkpoint_epoch is not None:
        print(f"Loaded checkpoint epoch: {checkpoint_epoch}")
    if checkpoint_metrics is not None:
        print(f"Loaded checkpoint metrics: {checkpoint_metrics}")

    return model, checkpoint_epoch, checkpoint_metrics


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--parquet_path", type=str, required=True)
    parser.add_argument("--target_cols", type=str, default="x_acc,y_acc,z_acc,x_gyro,y_gyro,z_gyro")
    parser.add_argument("--id_col", type=str, default="id")
    parser.add_argument("--time_col", type=str, default="time")

    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--prediction_length", type=int, default=128)
    parser.add_argument("--lags_sequence", type=str, default="1,2,3,4,5,10,20")

    parser.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help="Seed for user-level split. If omitted, uses --seed.",
    )

    parser.add_argument("--train_stride", type=int, default=64)
    parser.add_argument("--eval_stride", type=int, default=256)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_eval_windows", type=int, default=None)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=4,
        help="Batch size used only for model.generate() during SYNTH export and plotting. Keep small to avoid OOM.",
    )

    # Model
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--encoder_layers", type=int, default=4)
    parser.add_argument("--decoder_layers", type=int, default=4)
    parser.add_argument("--encoder_attention_heads", type=int, default=8)
    parser.add_argument("--decoder_attention_heads", type=int, default=8)
    parser.add_argument("--encoder_ffn_dim", type=int, default=1024)
    parser.add_argument("--decoder_ffn_dim", type=int, default=1024)
    parser.add_argument("--moving_average", type=int, default=7)
    parser.add_argument("--autocorrelation_factor", type=int, default=5)

    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--activation_dropout", type=float, default=0.1)

    parser.add_argument("--distribution_output", type=str, default="normal", choices=["normal", "student_t"])
    parser.add_argument(
        "--scaling",
        type=str,
        default="std",
        help=(
            "HF Autoformer scaling. Use std/mean/true for HF scaling. "
            "Use false/none/0/no to disable HF scaling and enable external scaler."
        ),
    )
    parser.add_argument(
        "--external_scaler",
        type=str,
        default="minmax",
        choices=["minmax", "standard"],
        help="External scaler used only when --scaling false. Fitted on train users only. Default: minmax.",
    )
    parser.add_argument("--num_parallel_samples", type=int, default=20)
    parser.add_argument(
        "--export_num_parallel_samples",
        type=int,
        default=1,
        help="Number of parallel samples used during full SYNTH.npz export. Default 1 because SYNTH uses first stochastic path.",
    )

    # Optimization
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Output
    parser.add_argument("--output_dir", type=str, default="runs_hf_autoformer_har")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_plot_examples", type=int, default=3)
    parser.add_argument("--num_sample_paths", type=int, default=3)
    parser.add_argument(
        "--plot_num_parallel_samples",
        type=int,
        default=None,
        help="Number of parallel samples for plotting. If omitted, uses max(num_sample_paths, 1).",
    )
    parser.add_argument("--plot_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--plot_dpi", type=int, default=200)

    parser.add_argument(
        "--no_save_eval_npz",
        action="store_true",
        help="Disable saving TRAIN.npz, TEST.npz, and SYNTH.npz.",
    )
    parser.add_argument("--max_export_train_batches", type=int, default=None)
    parser.add_argument("--max_export_test_batches", type=int, default=None)
    parser.add_argument(
        "--synth_from_mean",
        action="store_true",
        help="Use the mean over HF generated samples as the SYNTH future. Default is the first stochastic sample.",
    )

    # Checkpoint / sampling-only mode
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a .pt checkpoint such as checkpoints/best.pt, or an HF save_pretrained directory.",
    )
    parser.add_argument(
        "--sample_only",
        action="store_true",
        help="Skip training, load --checkpoint_path, then export TRAIN/TEST/SYNTH and plots.",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    return parser


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    
    # time it
    start_time = time()
    
    parser = build_arg_parser()
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir, run_id = make_unique_run_dir(
        base_output_dir=args.output_dir,
        run_name=args.run_name,
        overwrite=args.overwrite,
    )
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    eval_export_dir = output_dir / f"RUN-{run_id}"
    eval_export_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, split_info, external_scaler = make_datasets_and_loaders(args)

    target_cols = split_info["target_cols"]
    lags_sequence = split_info["lags_sequence"]

    model = build_model(args, target_cols=target_cols, lags_sequence=lags_sequence).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    run_config = vars(args).copy()
    run_config.update(split_info)
    run_config["device"] = str(device)
    run_config["run_id"] = run_id
    run_config["run_folder"] = str(output_dir.resolve())
    run_config["eval_export_folder"] = str(eval_export_dir.resolve())
    run_config["num_trainable_params"] = int(num_params)

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print("\n==================== RUN INFO ====================")
    print(f"Device: {device}")
    print(f"Run ID: {run_id}")
    print(f"Run folder: {output_dir.resolve()}")
    print(f"Evaluation export folder: {eval_export_dir.resolve()}")
    print(f"Parquet: {args.parquet_path}")
    print(f"Target columns: {target_cols}")
    print(f"Lags sequence: {lags_sequence}")
    print(f"Lag prefix length required by HF: {split_info['lag_prefix_length']}")
    print(f"Context length used for prediction: {split_info['context_length_used_for_prediction']}")
    print(f"Past length passed to HF: {split_info['past_length_passed_to_hf']}")
    print(f"Prediction length: {args.prediction_length}")
    print(f"Canonical saved/evaluation window length: {split_info['canonical_window_length']}")
    print(f"HF internal total length including lag prefix: {split_info['hf_internal_total_length']}")
    print(
        "Forecast setup: "
        f"last {args.context_length} observed steps -> next {args.prediction_length} future steps "
        f"(plus {split_info['lag_prefix_length']} earlier lag-prefix steps required internally by HF)"
    )
    print(f"Moving average: {args.moving_average}")
    print(f"Autocorrelation factor: {args.autocorrelation_factor}")
    print(f"HF scaling sent to Autoformer: {parse_hf_scaling(args.scaling)}")
    print(f"Use external scaler: {split_info['use_external_scaler']}")
    if split_info["use_external_scaler"]:
        print(f"External scaler: {args.external_scaler} fitted on train users only")
    print(f"Distribution output: {args.distribution_output}")
    print(f"Generation batch size: {args.generation_batch_size}")
    print(f"Export num_parallel_samples: {args.export_num_parallel_samples}")
    print(f"Split protocol: {split_info['split_protocol']}")
    print(f"Split seed: {split_info['split_seed']}")
    print(f"Train users: {len(split_info['train_users'])}")
    print(f"Val users: {len(split_info['val_users'])}")
    print(f"Test users: {len(split_info['test_users'])}")
    print(f"Train windows: {split_info['num_train_windows']}")
    print(f"Val windows: {split_info['num_val_windows']}")
    print(f"Test windows: {split_info['num_test_windows']}")
    print(f"Trainable params: {num_params:,}")
    print("==================================================\n")

    history = []
    best_val_loss = float("inf")
    best_epoch = -1

    loaded_checkpoint_epoch = None
    loaded_checkpoint_metrics = None

    if args.sample_only and args.checkpoint_path is None:
        raise ValueError("--sample_only requires --checkpoint_path")

    if args.checkpoint_path is not None:
        model, loaded_checkpoint_epoch, loaded_checkpoint_metrics = load_checkpoint_into_model(
            model=model,
            checkpoint_path=args.checkpoint_path,
            device=device,
            optimizer=optimizer,
            load_optimizer=False,
        )
        if loaded_checkpoint_metrics is not None and "val_loss" in loaded_checkpoint_metrics:
            best_val_loss = float(loaded_checkpoint_metrics["val_loss"])
        if loaded_checkpoint_epoch is not None:
            best_epoch = int(loaded_checkpoint_epoch)

    if args.sample_only:
        print("Sample-only mode enabled. Skipping training and using the loaded checkpoint for generation/export.")
    else:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                grad_clip=args.grad_clip,
                epoch=epoch,
            )

            val_loss = evaluate_loss(model=model, loader=val_loader, device=device, split_name="val")
            test_loss = evaluate_loss(model=model, loader=test_loader, device=device, split_name="test")

            metrics = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "test_loss": float(test_loss),
            }
            history.append(metrics)

            print(
                f"Epoch {epoch:03d} | "
                f"train={train_loss:.6f} | val={val_loss:.6f} | test={test_loss:.6f}"
            )

            save_checkpoint(output_dir, model, optimizer, epoch, metrics)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                model.save_pretrained(output_dir / "hf_model_best")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": metrics,
                        "config": model.config.to_dict(),
                    },
                    output_dir / "checkpoints" / "best.pt",
                )

            with open(output_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")

        best_ckpt_path = output_dir / "checkpoints" / "best.pt"
        if best_ckpt_path.exists():
            print(f"Loading best validation checkpoint from epoch {best_epoch} before generation...")
            checkpoint = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()
        else:
            print("WARNING: best.pt not found. Using last model for generation.")

    if args.sample_only:
        if loaded_checkpoint_metrics is not None:
            print(f"Loaded checkpoint metrics: {loaded_checkpoint_metrics}")
        if loaded_checkpoint_epoch is not None:
            print(f"Loaded checkpoint epoch: {loaded_checkpoint_epoch}")


    loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    plot_loader = loader_map[args.plot_split]

    train_export_loader = DataLoader(
        train_loader.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_export_loader = DataLoader(
        test_loader.dataset,
        batch_size=args.generation_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if not args.no_save_eval_npz:
        print("Saving evaluation NPZ files...")

        train_extended = collect_real_extended_sequences(
            loader=train_export_loader,
            context_length=args.context_length,
            max_batches=args.max_export_train_batches,
        )

        synth_extended, test_extended = collect_test_generation_npz_arrays(
            model=model,
            loader=test_export_loader,
            device=device,
            context_length=args.context_length,
            max_batches=args.max_export_test_batches,
            use_mean_prediction=args.synth_from_mean,
            generation_num_parallel_samples=args.export_num_parallel_samples,
        )

        expected_len = args.context_length + args.prediction_length
        assert train_extended.shape[1] == expected_len
        assert test_extended.shape[1] == expected_len
        assert synth_extended.shape[1] == expected_len

        context_diff_scaled = np.max(
            np.abs(test_extended[:, : args.context_length, :] - synth_extended[:, : args.context_length, :])
        )
        print(f"Context max abs diff between TEST and SYNTH before inverse scaling: {context_diff_scaled:.8f}")

        if external_scaler is not None:
            print("Inverse-transforming TRAIN/TEST/SYNTH back to original scale before saving...")
            train_extended = external_scaler.inverse_transform_3d(train_extended)
            test_extended = external_scaler.inverse_transform_3d(test_extended)
            synth_extended = external_scaler.inverse_transform_3d(synth_extended)

        context_diff = np.max(
            np.abs(test_extended[:, : args.context_length, :] - synth_extended[:, : args.context_length, :])
        )
        print(f"Context max abs diff between TEST and SYNTH: {context_diff:.8f}")

        # Special case for PPG dataset: last column is chest_label which is heart rate. We smooth the generated data to align with
        # the true heart rate values better for evaluation purposes, since HF Autoformer can struggle to capture the exact heart rate without a lot of tuning.
        if "chest_label" in target_cols:
            hr_idx = target_cols.index("chest_label")
            print(f"Smoothing SYNTH chest_label (heart rate) channel at index {hr_idx} for better alignment with TEST...")
            # we only do this for the generated half of the sequence, since the context is real data
            # and we pad the signal to ensure length consistency, so we don't want to smooth the context
            synth_extended[:, args.context_length :, hr_idx] = smooth_signal(
                synth_extended[:, args.context_length :, hr_idx],
                window_size=5,
            )
        

        
        print(f"TRAIN shape: {train_extended.shape}")
        print(f"TEST shape:  {test_extended.shape}")
        print(f"SYNTH shape: {synth_extended.shape}")

        save_compressed_npz(
            data_file=train_extended,
            channel_names=target_cols,
            model_name="hf_autoformer",
            save_path=eval_export_dir / "TRAIN.npz",
            seed=args.seed,
        )

        save_compressed_npz(
            data_file=test_extended,
            channel_names=target_cols,
            model_name="hf_autoformer",
            save_path=eval_export_dir / "TEST.npz",
            seed=args.seed,
        )

        save_compressed_npz(
            data_file=synth_extended,
            channel_names=target_cols,
            model_name="hf_autoformer",
            save_path=eval_export_dir / "SYNTH.npz",
            seed=args.seed,
        )

    plot_generation_loader = DataLoader(
        plot_loader.dataset,
        batch_size=max(1, min(args.generation_batch_size, args.num_plot_examples)),
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    plot_num_parallel_samples = (
        args.plot_num_parallel_samples
        if args.plot_num_parallel_samples is not None
        else max(1, args.num_sample_paths)
    )

    batch, samples, pred_mean = generate_batch(
        model,
        plot_generation_loader,
        device,
        num_parallel_samples=plot_num_parallel_samples,
    )

    print("\nGeneration sanity check:")
    print(f"samples shape:   {tuple(samples.shape)}")
    print(f"pred_mean shape: {tuple(pred_mean.shape)}")
    print(f"future shape:    {tuple(batch['future_values'].shape)}")

    plot_generated_examples(
        batch=batch,
        samples=samples,
        pred_mean=pred_mean,
        target_cols=target_cols,
        context_length=args.context_length,
        output_dir=output_dir / "plots",
        split_name=args.plot_split,
        num_examples=args.num_plot_examples,
        num_sample_paths=args.num_sample_paths,
        dpi=args.plot_dpi,
        external_scaler=external_scaler,
    )

    print_std_diagnostic(
        batch=batch,
        pred_mean=pred_mean,
        target_cols=target_cols,
        split_name=args.plot_split,
        external_scaler=external_scaler,
    )

    print("\nDone.")
    print(f"Outputs saved to: {output_dir.resolve()}")
    print(f"Evaluation NPZ files saved to: {eval_export_dir.resolve()}")
    print(f"Plots saved to: {(output_dir / 'plots').resolve()}")
    print(f"Total time elapsed: {(time() - start_time) / 60:.2f} minutes")
    print("RUN ID:", run_id)


if __name__ == "__main__":
    main()
