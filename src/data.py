from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import (
    IDENTITY_FILE, MAX_ROWS, RELATION_COLUMNS, TRAIN_RATIO, VAL_RATIO,
    TRANSACTION_FILE
)


def load_transactions() -> pd.DataFrame:
    if not TRANSACTION_FILE.exists():
        raise FileNotFoundError(
            f"Missing {TRANSACTION_FILE}. Download IEEE-CIS data from Kaggle "
            "and place train_transaction.csv in data/raw/."
        )

    tx = pd.read_csv(TRANSACTION_FILE)
    if MAX_ROWS is not None:
        tx = tx.head(MAX_ROWS).copy()

    if IDENTITY_FILE.exists():
        identity = pd.read_csv(IDENTITY_FILE)
        tx = tx.merge(identity, on="TransactionID", how="left")

    required = {"TransactionID", "TransactionDT", "isFraud"}
    missing = required.difference(tx.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    return tx.sort_values("TransactionDT").reset_index(drop=True)


def chronological_masks(frame: pd.DataFrame):
    n = len(frame)
    if n < 20:
        raise ValueError("Need at least 20 transactions to create time splits.")
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True
    return train_mask, val_mask, test_mask


def build_features(frame: pd.DataFrame, train_mask: torch.Tensor):
    """Fit imputers/scalers and categorical vocabularies on training rows only."""
    excluded = {
        "isFraud", "TransactionID", *RELATION_COLUMNS
    }
    feature_frame = frame.drop(
        columns=[c for c in excluded if c in frame.columns],
        errors="ignore",
    ).copy()

    # Keep numeric columns and a bounded set of categorical columns.
    numeric_cols = feature_frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [
        c for c in feature_frame.columns
        if c not in numeric_cols and feature_frame[c].nunique(dropna=True) <= 100
    ]

    train_idx = train_mask.numpy()
    parts = []

    for col in numeric_cols:
        values = pd.to_numeric(feature_frame[col], errors="coerce")
        median = values.iloc[train_idx].median()
        if pd.isna(median):
            median = 0.0
        values = values.fillna(median).replace([np.inf, -np.inf], median)
        mean = values.iloc[train_idx].mean()
        std = values.iloc[train_idx].std()
        if pd.isna(std) or std == 0:
            std = 1.0
        scaled = ((values - mean) / std).clip(-10, 10)
        parts.append(scaled.to_numpy(dtype=np.float32).reshape(-1, 1))

    for col in categorical_cols:
        values = feature_frame[col].astype("string").fillna("__MISSING__")
        train_values = set(values.iloc[train_idx].unique().tolist())
        # Stable train-only vocabulary; unseen validation/test values map to 0.
        vocab = {value: i + 1 for i, value in enumerate(sorted(train_values))}
        encoded = values.map(vocab).fillna(0).to_numpy(dtype=np.float32)
        parts.append(encoded.reshape(-1, 1))

    if not parts:
        raise ValueError("No usable feature columns were found.")
    x = np.concatenate(parts, axis=1)
    return torch.tensor(x, dtype=torch.float32)


def get_labels(frame: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(frame["isFraud"].astype("int64").to_numpy(), dtype=torch.long)
