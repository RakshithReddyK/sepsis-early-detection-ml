"""Feature selection for the sepsis model.

This module exists because of a bug: the original `train_sepsis.py` trained
on `data/balanced_sepsis.csv` while leaving in:

  * `diagnosis_Heart Failure`, `diagnosis_Normal`, `diagnosis_Pneumonia`,
    `diagnosis_Sepsis` -- one-hot columns encoding the *same diagnosis label*
    the target is derived from. `diagnosis_Sepsis` is literally
    `is_sepsis` under a different name (see `data.py`:
    `is_sepsis = 1 if diagnosis == "Sepsis" else 0`).
  * `mortality` and `readmission_30d` -- outcomes that are only known
    *after* the prediction would need to be made, i.e. they cannot exist at
    inference time in a real early-warning system.

Both leaks independently explain the previously-reported ROC-AUC / PR-AUC
of 1.0.

`get_feature_columns` / `prepare_features` are the single source of truth
for what is and is not allowed into the feature matrix. `train.py` and
`tests/test_leakage.py` both import from here on purpose: if someone
reintroduces a leaking column in the training path, the regression test
(which calls this same function) will catch it.
"""
from __future__ import annotations

import pandas as pd

# Columns that must never reach the feature matrix for the sepsis model,
# because they leak the label or leak post-prediction-time information.
LEAKAGE_COLUMNS: frozenset[str] = frozenset(
    {
        "diagnosis_Heart Failure",
        "diagnosis_Normal",
        "diagnosis_Pneumonia",
        "diagnosis_Sepsis",
        "mortality",
        "readmission_30d",
    }
)

# Any raw (non-one-hot) identifier / label-adjacent columns to also exclude
# if present, e.g. when working from a less-processed dataframe.
NON_FEATURE_COLUMNS: frozenset[str] = frozenset({"patient_id", "diagnosis"})


def get_feature_columns(df: pd.DataFrame, target_col: str = "is_sepsis") -> list[str]:
    """Return the list of columns in `df` that are safe to use as features.

    Excludes the target column, every column in LEAKAGE_COLUMNS, every
    column in NON_FEATURE_COLUMNS, and (defensively) any column whose name
    starts with "diagnosis_" -- so this still works if the one-hot encoding
    step changes category names/order in the future.
    """
    drop = set(LEAKAGE_COLUMNS) | set(NON_FEATURE_COLUMNS) | {target_col}
    feature_cols = [
        c for c in df.columns if c not in drop and not c.startswith("diagnosis_")
    ]
    return feature_cols


def prepare_features(
    df: pd.DataFrame, target_col: str = "is_sepsis"
) -> tuple[pd.DataFrame, pd.Series]:
    """Split `df` into a leakage-free feature matrix X and target vector y."""
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' in dataframe")

    feature_cols = get_feature_columns(df, target_col=target_col)
    X = df[feature_cols].copy()
    y = df[target_col].astype(int)
    return X, y
