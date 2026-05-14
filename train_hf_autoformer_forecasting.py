"""
Train Hugging Face Autoformer on generic forecasting datasets with chronological splits.

Designed for non-HAR forecasting datasets such as:
    - ETT / ETTh / ETTm
    - Air Quality
    - Weather
    - Sine wave / synthetic forecasting datasets

Key differences from the HAR script:
    - No user/subject split.
    - The dataset is treated as one continuous time series.
    - Train/val/test splits are chronological, based on user-provided fractions.
    - External scalers, if enabled, are fitted on the TRAIN time segment only.

Saved NPZ files:
    RUN-{run_id}/TRAIN.npz  [N_train, context_length + prediction_length, C]
    RUN-{run_id}/TEST.npz   [N_test,  context_length + prediction_length, C]
    RUN-{run_id}/SYNTH.npz  [N_test,  context_length + prediction_length, C]

For TEST and SYNTH:
    first context_length steps = identical real context
    next prediction_length steps = real future in TEST, generated future in SYNTH
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


def parse_hf_scaling(value):
    """
    Convert CLI scaling value to what HF Autoformer expects.

    Useful values:
        std     -> HF std scaler
        mean    -> HF mean scaler
        true    -> True, equivalent to HF mean scaler
        false   -> False, disables HF scaling and enables external scaler
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

    The scaler is fitted only on the chronological training segment.
    """

    def __init__(self, method: str = "standard", eps: float = 1e-8) -> None:
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

    def inverse_transform_3d(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f"inverse_transform_3d expects [N, T, C], got {x.shape}")
        n, t, c = x.shape
        flat = x.reshape(-1, c)
        inv = self.inverse_transform(flat).reshape(n, t, c)
        return inv.astype(np.float32)

    def inverse_transform_torch_3d(self, x: torch.Tensor) -> torch.Tensor:
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


# -----------------------------------------------------------------------------
# Run folders
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


# -----------------------------------------------------------------------------
# Data loading and chronological splitting
# -----------------------------------------------------------------------------


def load_forecasting_dataframe(
    data_path: str,
    target_cols_arg: str,
    time_col: Optional[str],
    exclude_cols_arg: str = "",
) -> Tuple[pd.DataFrame, List[str]]:
    data_path = str(data_path)
    suffix = Path(data_path).suffix.lower()

    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(data_path)
    elif suffix in {".csv", ".txt"}:
        df = pd.read_csv(data_path)
    else:
        raise ValueError(
            f"Unsupported file extension {suffix}. Use .csv, .txt, .parquet, or .pq."
        )

    if time_col is not None and time_col.lower() in {"none", "null", ""}:
        time_col = None

    if time_col is not None:
        if time_col not in df.columns:
            raise ValueError(f"time_col={time_col} not found in columns: {list(df.columns)}")
        df = df.sort_values(time_col).reset_index(drop=True)

    exclude_cols = [c.strip() for c in exclude_cols_arg.split(",") if c.strip()]
    if time_col is not None:
        exclude_cols.append(time_col)
    exclude_set = set(exclude_cols)

    if target_cols_arg.strip().lower() == "all":
        target_cols = [c for c in df.columns if c not in exclude_set]
    else:
        target_cols = [c.strip() for c in target_cols_arg.split(",") if c.strip()]

    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing target columns: {missing}")

    for col in target_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

    if len(target_cols) == 0:
        raise ValueError("No target columns selected.")

    return df, target_cols


def chronological_split_indices(
    n_rows: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    total_length: int,
) -> Dict[str, Tuple[int, int]]:
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios <= 0):
        raise ValueError(f"Split ratios must be positive. Got {ratios.tolist()}")
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError(
            f"Split ratios must sum to 1. Got train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

    n_train = int(np.floor(n_rows * train_ratio))
    n_val = int(np.floor(n_rows * val_ratio))
    n_test = n_rows - n_train - n_val

    if min(n_train, n_val, n_test) < total_length:
        raise ValueError(
            "One or more splits are too short to produce a single window. "
            f"Got n_train={n_train}, n_val={n_val}, n_test={n_test}, "
            f"but each split needs at least context_length + prediction_length = {total_length}."
        )

    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_rows),
    }


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class ForecastingAutoformerDataset(Dataset):
    """
    Windowed dataset for generic forecasting time series.

    Canonical sample:
        [context, future]
        context length = context_length
        future length  = prediction_length

    Hugging Face Autoformer internal input:
        past_values = [lag_prefix, context]
        future_values = future

    The lag prefix is required by HF Autoformer but is not included in saved TRAIN/TEST/SYNTH arrays.
    """

    def __init__(
        self,
        values: np.ndarray,
        context_length: int,
        prediction_length: int,
        lags_sequence: Sequence[int],
        stride: int,
        split_name: str,
        max_windows: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"values must have shape [T, C], got {values.shape}")

        self.values = values
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.lags_sequence = tuple(int(x) for x in lags_sequence)
        self.max_lag = max(self.lags_sequence)
        self.past_length = self.context_length + self.max_lag
        self.total_length = self.context_length + self.prediction_length
        self.hf_total_length = self.past_length + self.prediction_length
        self.stride = int(stride)
        self.split_name = split_name

        if self.stride <= 0:
            raise ValueError(f"stride must be positive. Got {self.stride}")

        n_windows = len(self.values) - self.total_length + 1
        if n_windows <= 0:
            raise ValueError(
                f"No windows for split={split_name}. "
                f"Need at least {self.total_length} rows, got {len(self.values)}."
            )

        self.indices = list(range(0, n_windows, self.stride))

        if max_windows is not None and len(self.indices) > max_windows:
            rng = random.Random(seed)
            rng.shuffle(self.indices)
            self.indices = self.indices[:max_windows]
            self.indices = sorted(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = self.indices[idx]
        arr = self.values

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

        # Local normalized position feature. We do not pass raw calendar features here.
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
# Datasets and loaders
# -----------------------------------------------------------------------------


def make_datasets_and_loaders(args):
    lags_sequence = tuple(int(x.strip()) for x in args.lags_sequence.split(",") if x.strip())
    max_lag = max(lags_sequence)
    total_length = args.context_length + args.prediction_length
    hf_total_length = args.context_length + max_lag + args.prediction_length

    df_raw, target_cols = load_forecasting_dataframe(
        data_path=args.data_path,
        target_cols_arg=args.target_cols,
        time_col=args.time_col,
        exclude_cols_arg=args.exclude_cols,
    )

    split_bounds = chronological_split_indices(
        n_rows=len(df_raw),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        total_length=total_length,
    )

    train_start, train_end = split_bounds["train"]
    val_start, val_end = split_bounds["val"]
    test_start, test_end = split_bounds["test"]

    values_raw = df_raw[target_cols].to_numpy(dtype=np.float32)

    hf_scaling = parse_hf_scaling(args.scaling)
    use_external_scaler = hf_scaling is False
    external_scaler = None

    if use_external_scaler:
        external_scaler = ExternalFeatureScaler(method=args.external_scaler)
        external_scaler.fit(values_raw[train_start:train_end])
        values_model = external_scaler.transform(values_raw)
    else:
        values_model = values_raw.copy()

    train_values = values_model[train_start:train_end]
    val_values = values_model[val_start:val_end]
    test_values = values_model[test_start:test_end]

    train_dataset = ForecastingAutoformerDataset(
        values=train_values,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        lags_sequence=lags_sequence,
        stride=args.train_stride,
        max_windows=args.max_train_windows,
        seed=args.seed,
        split_name="train",
    )

    val_dataset = ForecastingAutoformerDataset(
        values=val_values,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        lags_sequence=lags_sequence,
        stride=args.eval_stride,
        max_windows=args.max_eval_windows,
        seed=args.seed,
        split_name="val",
    )

    test_dataset = ForecastingAutoformerDataset(
        values=test_values,
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
        "split_protocol": "chronological contiguous split by fractions",
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "n_rows_total": len(df_raw),
        "train_range": [train_start, train_end],
        "val_range": [val_start, val_end],
        "test_range": [test_start, test_end],
        "n_rows_train": len(train_values),
        "n_rows_val": len(val_values),
        "n_rows_test": len(test_values),
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
# Generation / export / plotting
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
# Checkpoints
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

    if "config" in checkpoint and checkpoint["config"] is not None:
        print("Rebuilding model from checkpoint config before loading weights.")
        ckpt_config = AutoformerConfig.from_dict(checkpoint["config"])
        model = AutoformerForPrediction(ckpt_config).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
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
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--target_cols",
        type=str,
        default="all",
        help="Comma-separated target columns, or 'all' to use all numeric columns except excluded/time cols.",
    )
    parser.add_argument("--time_col", type=str, default="date")
    parser.add_argument(
        "--exclude_cols",
        type=str,
        default="",
        help="Comma-separated non-target columns to exclude when --target_cols all.",
    )

    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--prediction_length", type=int, default=128)
    parser.add_argument("--lags_sequence", type=str, default="1,2,3,4,5,10,20")

    parser.add_argument("--train_ratio", type=float, default=0.80)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)

    parser.add_argument("--train_stride", type=int, default=32)
    parser.add_argument("--eval_stride", type=int, default=256)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_eval_windows", type=int, default=None)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--generation_batch_size", type=int, default=4)

    # Model
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--encoder_layers", type=int, default=2)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--encoder_attention_heads", type=int, default=4)
    parser.add_argument("--decoder_attention_heads", type=int, default=4)
    parser.add_argument("--encoder_ffn_dim", type=int, default=256)
    parser.add_argument("--decoder_ffn_dim", type=int, default=256)
    parser.add_argument("--moving_average", type=int, default=25)
    parser.add_argument("--autocorrelation_factor", type=int, default=5)

    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--activation_dropout", type=float, default=0.1)

    parser.add_argument("--distribution_output", type=str, default="normal", choices=["normal", "student_t"])
    parser.add_argument(
        "--scaling",
        type=str,
        default="std",
        help="HF Autoformer scaling. Use false/none/0/no to disable HF scaling and use external scaler.",
    )
    parser.add_argument(
        "--external_scaler",
        type=str,
        default="standard",
        choices=["minmax", "standard"],
        help="External scaler used only when --scaling false. Fitted on train time segment only.",
    )
    parser.add_argument("--num_parallel_samples", type=int, default=20)
    parser.add_argument("--export_num_parallel_samples", type=int, default=1)

    # Optimization
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Output
    parser.add_argument("--output_dir", type=str, default="runs_hf_autoformer_forecasting")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_plot_examples", type=int, default=3)
    parser.add_argument("--num_sample_paths", type=int, default=3)
    parser.add_argument("--plot_num_parallel_samples", type=int, default=None)
    parser.add_argument("--plot_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--plot_dpi", type=int, default=200)

    parser.add_argument("--no_save_eval_npz", action="store_true")
    parser.add_argument("--max_export_train_batches", type=int, default=None)
    parser.add_argument("--max_export_test_batches", type=int, default=None)
    parser.add_argument("--synth_from_mean", action="store_true")

    # Checkpoint / sample-only
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--sample_only", action="store_true")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    return parser


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
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
    print(f"Data path: {args.data_path}")
    print(f"Target columns: {target_cols}")
    print(f"Time column: {args.time_col}")
    print(f"Split protocol: {split_info['split_protocol']}")
    print(f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"Rows: total={split_info['n_rows_total']}, train={split_info['n_rows_train']}, val={split_info['n_rows_val']}, test={split_info['n_rows_test']}")
    print(f"Train row range: {split_info['train_range']}")
    print(f"Val row range: {split_info['val_range']}")
    print(f"Test row range: {split_info['test_range']}")
    print(f"Train windows: {split_info['num_train_windows']}")
    print(f"Val windows: {split_info['num_val_windows']}")
    print(f"Test windows: {split_info['num_test_windows']}")
    print(f"Lags sequence: {lags_sequence}")
    print(f"Lag prefix length required by HF: {split_info['lag_prefix_length']}")
    print(f"Past length passed to HF: {split_info['past_length_passed_to_hf']}")
    print(f"Canonical saved/evaluation window length: {split_info['canonical_window_length']}")
    print(f"HF internal total length including lag prefix: {split_info['hf_internal_total_length']}")
    print(f"Moving average: {args.moving_average}")
    print(f"Autocorrelation factor: {args.autocorrelation_factor}")
    print(f"HF scaling sent to Autoformer: {parse_hf_scaling(args.scaling)}")
    print(f"Use external scaler: {split_info['use_external_scaler']}")
    if split_info["use_external_scaler"]:
        print(f"External scaler: {args.external_scaler} fitted on train time segment only")
    print(f"Distribution output: {args.distribution_output}")
    print(f"Generation batch size: {args.generation_batch_size}")
    print(f"Export num_parallel_samples: {args.export_num_parallel_samples}")
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

        print(f"TRAIN shape: {train_extended.shape}")
        print(f"TEST shape:  {test_extended.shape}")
        print(f"SYNTH shape: {synth_extended.shape}")

        save_compressed_npz(
            data_file=train_extended,
            channel_names=target_cols,
            model_name="hf_autoformer_forecasting",
            save_path=eval_export_dir / "TRAIN.npz",
            seed=args.seed,
        )

        save_compressed_npz(
            data_file=test_extended,
            channel_names=target_cols,
            model_name="hf_autoformer_forecasting",
            save_path=eval_export_dir / "TEST.npz",
            seed=args.seed,
        )

        save_compressed_npz(
            data_file=synth_extended,
            channel_names=target_cols,
            model_name="hf_autoformer_forecasting",
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
