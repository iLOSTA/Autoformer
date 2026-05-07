"""
Train Hugging Face Autoformer on HAR parquet data with subject-disjoint splits.

Expected parquet columns by default:
    x_acc, y_acc, z_acc, x_gyro, y_gyro, z_gyro, id, time

Each parquet file is assumed to represent ONE activity type.
The split is performed by user ID, not by random windows, to avoid subject leakage.
"""

import argparse
import json
import math
import os
import random
from uuid import uuid4
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
# User-level split
# -----------------------------------------------------------------------------


def make_unique_run_dir(
    base_output_dir: str,
    run_name: Optional[str] = None,
    overwrite: bool = False,
) -> Tuple[Path, str]:
    """
    Create a unique folder for each run so experiments do not overwrite each other.

    If run_name is provided, the final folder is:
        base_output_dir / run_name

    Otherwise, the final folder is:
        base_output_dir / run_{run_id}

    Returns:
        run_dir: folder for the whole run
        run_id: generated ID used for export folder naming
    """
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
# User-level split
# -----------------------------------------------------------------------------


def split_users(
    unique_users: Sequence,
    split_seed: int,
) -> Tuple[set, set, set]:
    """
    Subject-disjoint split matching the user's previous pipeline.

    Protocol:
        1. Sort unique user IDs.
        2. Shuffle deterministically with np.random.RandomState(split_seed).
        3. Use 1 user for validation.
        4. Use floor(0.1 * remaining_users) for test, at least 1 user.
        5. Use the rest for training.

    Returns:
        train_users, val_users, test_users as sets.
    """
    unique_users = np.sort(np.array(unique_users))
    rs = np.random.RandomState(split_seed)

    shuffled = unique_users.copy()
    rs.shuffle(shuffled)

    n_users = len(shuffled)
    if n_users < 3:
        raise ValueError(
            f"Need at least 3 unique users for train/val/test split, got {n_users}"
        )

    n_val = 1
    remaining = n_users - n_val
    n_test = max(1, int(np.floor(0.1 * remaining)))
    n_train = n_users - n_val - n_test

    if n_train <= 0:
        raise ValueError(
            f"Invalid split sizes after adjustment: "
            f"train={n_train}, val={n_val}, test={n_test}"
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
        # HF internal length passed as past_values.
        self.past_length = self.context_length + self.max_lag

        # Canonical window length used for indexing and saving.
        # This must match the previous pipeline: [context, future] = [128, 128].
        self.total_length = self.context_length + self.prediction_length

        # Total length seen by HF if you include lag-prefix + context + future.
        self.hf_total_length = self.past_length + self.prediction_length
        self.stride = int(stride)
        self.split_name = split_name

        if self.stride <= 0:
            raise ValueError(f"stride must be positive. Got {self.stride}")

        user_set = set(user_ids)
        df_split = df[df[id_col].isin(user_set)].copy()

        if time_col is not None and time_col in df_split.columns:
            df_split = df_split.sort_values([id_col, time_col]).reset_index(drop=True)
        else:
            df_split = df_split.sort_values([id_col]).reset_index(drop=True)

        self.series: List[np.ndarray] = []
        self.series_user_ids: List = []
        self.indices: List[Tuple[int, int]] = []

        for user_id, g in df_split.groupby(id_col, sort=False):
            values = g[self.target_cols].to_numpy(dtype=np.float32)

            if len(values) < self.total_length:
                continue

            series_idx = len(self.series)
            self.series.append(values)
            self.series_user_ids.append(user_id)

            # Match the previous pipeline:
            #   n_windows = T - (context_length + prediction_length) + 1
            #   for start_idx in range(0, n_windows, stride)
            # The HF lag prefix is NOT included in n_windows.
            n_windows = len(values) - self.total_length + 1
            for start in range(0, n_windows, self.stride):
                self.indices.append((series_idx, start))

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

        # Canonical 256-step window, exactly matching the old pipeline:
        #   context = arr[start : start + context_length]
        #   future  = arr[start + context_length : start + context_length + prediction_length]
        context_start = start
        context_end = start + self.context_length
        future_end = context_end + self.prediction_length

        context = arr[context_start:context_end].copy()
        future_values = arr[context_end:future_end].copy()

        # HF Autoformer requires max_lag additional values before the context.
        # These extra values are used internally for lag features but are NOT part of
        # the saved 256-step TRAIN/TEST/SYNTH arrays.
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

        # Simple non-calendar age/position feature.
        # Shape: [time, num_time_features]
        # Use the HF internal length here: [lag_prefix, context, future].
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
    parquet_path = str(parquet_path)
    df = pd.read_parquet(parquet_path)

    required = set(target_cols + [id_col])
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


def eligible_users(
    df: pd.DataFrame,
    id_col: str,
    total_length: int,
) -> List:
    counts = df.groupby(id_col).size()
    users = counts[counts >= total_length].index.tolist()
    return users


def make_datasets_and_loaders(args):
    target_cols = [c.strip() for c in args.target_cols.split(",") if c.strip()]
    lags_sequence = tuple(int(x.strip()) for x in args.lags_sequence.split(",") if x.strip())
    max_lag = max(lags_sequence)

    # Canonical/evaluation window length, matching the old pipeline.
    # The HF lag prefix is added separately inside the dataset and should not
    # affect the number of windows.
    total_length = args.context_length + args.prediction_length

    # Internal HF length if including lag prefix + context + future.
    hf_total_length = args.context_length + max_lag + args.prediction_length

    df = load_har_parquet(
        parquet_path=args.parquet_path,
        target_cols=target_cols,
        id_col=args.id_col,
        time_col=args.time_col,
    )

    users = eligible_users(df, args.id_col, total_length)
    if len(users) < 3:
        raise ValueError(
            f"Only {len(users)} users have at least {total_length} rows. "
            f"Cannot create subject-disjoint train/val/test split."
        )

    split_seed = args.split_seed if args.split_seed is not None else args.seed

    train_users, val_users, test_users = split_users(
        unique_users=users,
        split_seed=split_seed,
    )

    train_dataset = HARAutoformerDataset(
        df=df,
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
        df=df,
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
        df=df,
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
    }

    return train_loader, val_loader, test_loader, split_info


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


def build_model(args, target_cols: Sequence[str], lags_sequence: Sequence[int]) -> AutoformerForPrediction:
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
        scaling=args.scaling,
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
    """
    Save one dataset already formatted as [N, T, C] into a compressed .npz file.

    This intentionally matches the user's existing downstream pipeline schema.

    Saved keys:
        samples
        channel_names
        seq_len
        num_channels
        num_samples
        model_name
        seed
    """
    samples = np.asarray(data_file, dtype=np.float32)

    if samples.ndim != 3:
        raise ValueError(f"`data_file` must have shape [N, T, C], got {samples.shape}")

    N, seq_len, num_channels = samples.shape

    if channel_names is None:
        channel_names = [f"channel_{i}" for i in range(num_channels)]
    else:
        channel_names = list(channel_names)
        if len(channel_names) != num_channels:
            raise ValueError(
                f"len(channel_names) must equal num_channels={num_channels}, "
                f"got {len(channel_names)}"
            )

    save_path = Path(save_path)
    if save_path.suffix != ".npz":
        save_path = save_path.with_suffix(".npz")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        save_path,
        samples=samples,
        # Do not use dtype=object here.
        # Object arrays are pickled and can break across NumPy versions, e.g.
        # ModuleNotFoundError: No module named 'numpy._core'.
        channel_names=np.asarray(channel_names, dtype=str),
        seq_len=np.int32(seq_len),
        num_channels=np.int32(num_channels),
        num_samples=np.int32(N),
        model_name=model_name,
        seed=np.int32(seed),
    )


@torch.no_grad()
def collect_real_extended_sequences(
    loader: DataLoader,
    context_length: int,
    max_batches: Optional[int] = None,
) -> np.ndarray:
    """
    Collect real sequences as:
        [context, true_future]

    Output shape:
        [N, context_length + prediction_length, C]

    This deliberately removes the HF lag prefix from past_values.
    """
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
    """
    Generate SYNTH and TEST arrays over the test loader.

    Returns:
        synth_extended: [N, context_length + prediction_length, C]
            [context, generated_future]
        test_extended: [N, context_length + prediction_length, C]
            [context, true_future]

    By default, generated_future is the first stochastic trajectory.

    Important memory note:
        HF generate() expands memory roughly with num_parallel_samples.
        For SYNTH.npz using the first stochastic path, use generation_num_parallel_samples=1.
    """
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

            samples = generated.sequences  # [B, S, pred_len, C]

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

            # Free GPU tensors before the next generation step.
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


def save_generated_npz(
    output_path: Path,
    batch: Dict[str, torch.Tensor],
    samples: torch.Tensor,
    pred_mean: torch.Tensor,
    target_cols: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        past_values=batch["past_values"].detach().cpu().numpy(),
        future_values=batch["future_values"].detach().cpu().numpy(),
        samples=samples.detach().cpu().numpy(),
        pred_mean=pred_mean.detach().cpu().numpy(),
        target_cols=np.asarray(list(target_cols), dtype=object),
    )


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
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    past_context = batch["past_values"][:, -context_length:, :].detach().cpu()
    future_true = batch["future_values"].detach().cpu()
    samples = samples.detach().cpu()
    pred_mean = pred_mean.detach().cpu()

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
) -> None:
    real = batch["future_values"].detach().cpu()
    pred = pred_mean.detach().cpu()

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

    parser.add_argument("--distribution_output", type=str, default="normal")
    parser.add_argument("--scaling", type=str, default="std")
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
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run folder name. If omitted, a timestamped folder is created.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing run folder. Default is False.",
    )
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

    # Evaluation/export NPZ files compatible with the previous pipeline.
    parser.add_argument(
        "--no_save_eval_npz",
        action="store_true",
        help="Disable saving TRAIN.npz, TEST.npz, SYNTH.npz, and SYNTH_SAMPLES.npz.",
    )
    parser.add_argument(
        "--max_export_train_batches",
        type=int,
        default=None,
        help="Optional cap for TRAIN.npz export. Default saves the full train split.",
    )
    parser.add_argument(
        "--max_export_test_batches",
        type=int,
        default=None,
        help="Optional cap for TEST/SYNTH export. Default saves the full test split.",
    )
    parser.add_argument(
        "--synth_from_mean",
        action="store_true",
        help="Use the mean over HF generated samples as the SYNTH future. Default is the first stochastic sample.",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    return parser


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Each execution gets its own run folder under args.output_dir.
    # This prevents accidental overwriting across experiments.
    output_dir, run_id = make_unique_run_dir(
        base_output_dir=args.output_dir,
        run_name=args.run_name,
        overwrite=args.overwrite,
    )
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    # Main evaluation export folder requested by the previous pipeline convention.
    # Example:
    #   runs_hf_autoformer_har/run_ab12cd34/RUN-ab12cd34/
    eval_export_dir = output_dir / f"RUN-{run_id}"
    eval_export_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, split_info = make_datasets_and_loaders(args)

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
    print(f"Scaling: {args.scaling}")
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

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            epoch=epoch,
        )

        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            device=device,
            split_name="val",
        )

        test_loss = evaluate_loss(
            model=model,
            loader=test_loader,
            device=device,
            split_name="test",
        )

        metrics = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "test_loss": float(test_loss),
        }
        history.append(metrics)

        print(
            f"\nEpoch {epoch:03d} | "
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

    print(f"\nBest validation loss: {best_val_loss:.6f} at epoch {best_epoch}")

    # Load the best validation checkpoint before generating SYNTH.npz
    best_ckpt_path = output_dir / "checkpoints" / "best.pt"

    if best_ckpt_path.exists():
        print(f"\nLoading best validation checkpoint from epoch {best_epoch} before generation...")
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
    else:
        print("\nWARNING: best.pt not found. Using last model for generation.")
        
    loader_map = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    plot_loader = loader_map[args.plot_split]

    # Export loaders should be deterministic and should not drop the last incomplete batch.
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

        # Sanity check: first 128 steps in TEST and SYNTH should be identical context.
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
    )

    print_std_diagnostic(
        batch=batch,
        pred_mean=pred_mean,
        target_cols=target_cols,
        split_name=args.plot_split,
    )

    print("\nDone.")
    print(f"Outputs saved to: {output_dir.resolve()}")
    print(f"Evaluation NPZ files saved to: {eval_export_dir.resolve()}")
    print(f"Plots saved to: {(output_dir / 'plots').resolve()}")
    print("RUN ID:", run_id)


if __name__ == "__main__":
    main()
