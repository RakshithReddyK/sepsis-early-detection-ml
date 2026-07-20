from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def leaky_sepsis_df() -> pd.DataFrame:
    """A tiny dataframe shaped like data/balanced_sepsis.csv, including the
    leakage columns that must be dropped before modeling.
    """
    rng = np.random.default_rng(0)
    n = 60
    diagnosis = rng.choice(
        ["Normal", "Pneumonia", "Heart Failure", "Sepsis"], size=n
    )
    is_sepsis = (diagnosis == "Sepsis").astype(int)

    df = pd.DataFrame(
        {
            "age": rng.normal(55, 15, n).clip(18, 90),
            "bmi": rng.normal(27, 5, n),
            "systolic_bp": rng.normal(120, 15, n),
            "diastolic_bp": rng.normal(78, 10, n),
            "glucose": rng.normal(105, 20, n),
            "cholesterol": rng.normal(190, 30, n),
            "creatinine": rng.normal(1.1, 0.3, n),
            "diabetes": rng.integers(0, 2, n).astype(float),
            "hypertension": rng.integers(0, 2, n).astype(float),
            "readmission_30d": rng.integers(0, 2, n).astype(float),
            "mortality": rng.integers(0, 2, n).astype(float),
            "sex_Female": 0.0,
            "sex_Male": 1.0,
            "sex_Other": 0.0,
            "diagnosis_Heart Failure": (diagnosis == "Heart Failure").astype(float),
            "diagnosis_Normal": (diagnosis == "Normal").astype(float),
            "diagnosis_Pneumonia": (diagnosis == "Pneumonia").astype(float),
            "diagnosis_Sepsis": (diagnosis == "Sepsis").astype(float),
            "is_sepsis": is_sepsis,
        }
    )
    return df


@pytest.fixture
def tiny_balanced_sepsis_csv(tmp_path: Path, leaky_sepsis_df: pd.DataFrame) -> Path:
    """Write leaky_sepsis_df to a temp CSV, mimicking data/balanced_sepsis.csv,
    for end-to-end pipeline tests.
    """
    path = tmp_path / "balanced_sepsis.csv"
    leaky_sepsis_df.to_csv(path, index=False)
    return path
