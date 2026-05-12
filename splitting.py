from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    y = np.asarray(y)
    idx = np.arange(len(y))

    n_splits = 5
    outer = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    val_fraction_of_outer_train = val_size / (1.0 - (1.0 / n_splits))
    val_fraction_of_outer_train = float(np.clip(val_fraction_of_outer_train, 0.05, 0.4))

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []

    for fold_idx, (idx_train_val, idx_test) in enumerate(outer.split(idx, y)):
        y_train_val = y[idx_train_val]
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=val_fraction_of_outer_train,
            random_state=random_state + fold_idx,
            stratify=y_train_val,
        )
        splits.append((np.asarray(idx_train), np.asarray(idx_val), np.asarray(idx_test)))

    return splits