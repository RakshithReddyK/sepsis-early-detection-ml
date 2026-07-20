"""Data loading, and a reconstructed synthetic data generation pipeline.

Context (be honest about this, it matters for anyone reusing this repo):

The README this project shipped with described a source file
`data/synthetic_clinical_dataset.csv` and a SMOTE/SMOTENC balancing step
that supposedly produced the committed `data/balanced_*.csv` files. Neither
the source CSV nor any generation code was actually present in the repo,
and `imbalanced-learn` was not even a listed dependency -- so the balanced
CSVs could not have been reproduced from anything in the repository as it
shipped.

This module is a **documented reconstruction**, not a recovery of the
original generator. It was written after the fact, is fully seeded, and
produces data of a similar shape and similar (plausible, clinically
motivated) structure to the committed `balanced_*.csv` files -- but it is
NOT guaranteed to be bit-identical to them, because the original seed/logic
no longer exists anywhere.

By default this module writes to `data/reconstructed/` and will NOT touch
the original committed `data/balanced_*.csv` files, which remain the
reference artifacts that `reports/sepsis_metrics.json` and the README's
reported metrics are computed from. Pass `--overwrite-canonical` to the CLI
if you explicitly want to regenerate the canonical `data/` files instead
(not recommended -- it would make the "real metrics" reported in the README
un-reproducible from the committed data again).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from imblearn.over_sampling import SMOTE, SMOTENC
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "imbalanced-learn is required for data generation. "
        "Install it with `pip install imbalanced-learn` "
        "(it is listed in requirements.txt / pyproject.toml)."
    ) from exc

RANDOM_STATE = 42

DIAGNOSES = ["Normal", "Pneumonia", "Heart Failure", "Sepsis"]
DIAGNOSIS_PROBS = [0.62, 0.15, 0.13, 0.10]
SEXES = ["Female", "Male", "Other"]
SEX_PROBS = [0.55, 0.43, 0.02]


def generate_synthetic_clinical_dataset(
    n_patients: int = 10_000, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Generate a synthetic, PII-free clinical dataset.

    Each row is a patient with demographics, vitals/labs, comorbidities, a
    diagnosis, and two post-admission outcomes (mortality, 30-day
    readmission). Vitals/labs are drawn from diagnosis-conditioned
    distributions so the data has plausible (not perfect) clinical signal,
    e.g. septic patients tend towards hypotension, elevated lactate-adjacent
    creatinine, and stress hyperglycemia.

    This is synthetic data generated from parametric distributions -- it
    does not encode real patient records of any kind.
    """
    rng = np.random.default_rng(random_state)

    patient_id = np.arange(1, n_patients + 1)
    diagnosis = rng.choice(DIAGNOSES, size=n_patients, p=DIAGNOSIS_PROBS)
    sex = rng.choice(SEXES, size=n_patients, p=SEX_PROBS)

    age = np.clip(rng.normal(55, 18, n_patients), 18, 95).round(0)

    is_sepsis_mask = diagnosis == "Sepsis"
    is_hf_mask = diagnosis == "Heart Failure"
    is_pneumonia_mask = diagnosis == "Pneumonia"

    bmi = rng.normal(27, 5, n_patients)
    bmi += np.where(is_hf_mask, 3.0, 0.0)
    bmi = np.clip(bmi, 15, 55).round(1)

    # Vitals: sepsis trends hypotensive/tachycardic; heart failure trends
    # hyper/variable; pneumonia mild derangement; normal near-baseline.
    systolic_bp = rng.normal(120, 12, n_patients)
    systolic_bp += np.where(is_sepsis_mask, -22.0, 0.0)
    systolic_bp += np.where(is_hf_mask, 8.0, 0.0)
    systolic_bp += np.where(is_pneumonia_mask, -5.0, 0.0)
    systolic_bp = np.clip(systolic_bp, 70, 200).round(0)

    diastolic_bp = rng.normal(78, 9, n_patients)
    diastolic_bp += np.where(is_sepsis_mask, -12.0, 0.0)
    diastolic_bp += np.where(is_hf_mask, 4.0, 0.0)
    diastolic_bp = np.clip(diastolic_bp, 40, 130).round(0)

    glucose = rng.normal(100, 18, n_patients)
    glucose += np.where(is_sepsis_mask, 35.0, 0.0)  # stress hyperglycemia
    glucose = np.clip(glucose, 60, 400)

    cholesterol = rng.normal(190, 35, n_patients)
    cholesterol += np.where(is_hf_mask, 15.0, 0.0)
    cholesterol = np.clip(cholesterol, 100, 350)

    creatinine = rng.normal(1.0, 0.25, n_patients)
    creatinine += np.where(is_sepsis_mask, 0.7, 0.0)  # AKI signal
    creatinine = np.clip(creatinine, 0.4, 6.0)

    diabetes_prob = 0.10 + 0.15 * (age > 55) + 0.10 * (bmi > 30)
    diabetes = (rng.random(n_patients) < np.clip(diabetes_prob, 0, 0.9)).astype(float)

    hypertension_prob = 0.12 + 0.25 * (age > 55) + 0.10 * (bmi > 30)
    hypertension_prob += np.where(is_hf_mask, 0.2, 0.0)
    hypertension = (
        rng.random(n_patients) < np.clip(hypertension_prob, 0, 0.95)
    ).astype(float)

    # Outcomes are deliberately noisy functions of diagnosis + age, and are
    # only known post-admission -- never usable as sepsis-prediction
    # features (see features.py LEAKAGE_COLUMNS).
    base_mortality = {
        "Normal": 0.02,
        "Pneumonia": 0.08,
        "Heart Failure": 0.15,
        "Sepsis": 0.27,
    }
    mortality_prob = np.array([base_mortality[d] for d in diagnosis])
    mortality_prob += 0.10 * (age > 75)
    mortality = (rng.random(n_patients) < np.clip(mortality_prob, 0, 0.9)).astype(int)

    base_readmit = {
        "Normal": 0.04,
        "Pneumonia": 0.14,
        "Heart Failure": 0.22,
        "Sepsis": 0.20,
    }
    readmit_prob = np.array([base_readmit[d] for d in diagnosis])
    readmit_prob += 0.05 * (age > 75)
    readmission_30d = (rng.random(n_patients) < np.clip(readmit_prob, 0, 0.9)).astype(
        int
    )
    # Clinically: you cannot be readmitted if you died during the index stay.
    readmission_30d = np.where(mortality == 1, 0, readmission_30d)

    df = pd.DataFrame(
        {
            "patient_id": patient_id,
            "sex": sex,
            "diagnosis": diagnosis,
            "age": age,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "glucose": glucose.round(1),
            "cholesterol": cholesterol.round(1),
            "creatinine": creatinine.round(2),
            "diabetes": diabetes,
            "hypertension": hypertension,
            "mortality": mortality,
            "readmission_30d": readmission_30d,
        }
    )
    return df


def make_balanced_sepsis(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Reproduce the shape of `data/balanced_sepsis.csv`: numeric features +
    one-hot `sex_*` / `diagnosis_*` columns, balanced 50/50 on `is_sepsis`
    via SMOTE.

    NOTE: this intentionally reproduces the *original, leaky* column set
    (including `diagnosis_*` and the outcome columns) because that is what
    the shipped `data/balanced_sepsis.csv` actually contains. The fix for
    the leakage bug lives in `features.py` / `train.py`, which drop those
    columns before modeling -- not here. This function exists to explain
    how such a file could plausibly have been produced, and to give anyone
    extending this repo a reference implementation with the leak already
    called out.
    """
    encoded = pd.get_dummies(df, columns=["sex", "diagnosis"], prefix=["sex", "diagnosis"])
    encoded = encoded.drop(columns=["patient_id"])
    encoded["is_sepsis"] = (df["diagnosis"] == "Sepsis").astype(int)

    bool_cols = encoded.select_dtypes(include="bool").columns
    encoded[bool_cols] = encoded[bool_cols].astype(float)

    y = encoded["is_sepsis"]
    X = encoded.drop(columns=["is_sepsis"])

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)

    ordered_cols = [
        "age",
        "bmi",
        "systolic_bp",
        "diastolic_bp",
        "glucose",
        "cholesterol",
        "creatinine",
        "diabetes",
        "hypertension",
        "readmission_30d",
        "mortality",
        "sex_Female",
        "sex_Male",
        "sex_Other",
        "diagnosis_Heart Failure",
        "diagnosis_Normal",
        "diagnosis_Pneumonia",
        "diagnosis_Sepsis",
    ]
    ordered_cols = [c for c in ordered_cols if c in X_res.columns]
    out = X_res[ordered_cols].copy()
    out["is_sepsis"] = y_res.values
    return out


def _make_balanced_outcome(
    df: pd.DataFrame, target_col: str, drop_cols: list[str], random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Balance `target_col` (mortality or readmission_30d) 50/50 via SMOTENC,
    keeping the raw (non-one-hot) categorical/numeric shape of
    `balanced_mortality.csv` / `balanced_readmission.csv`.
    """
    work = df.drop(columns=drop_cols + ["patient_id"]).reset_index(drop=True)
    y = work[target_col]
    X = work.drop(columns=[target_col])

    categorical_cols = ["sex", "diagnosis"]
    categorical_idx = [X.columns.get_loc(c) for c in categorical_cols]

    smote_nc = SMOTENC(categorical_features=categorical_idx, random_state=random_state)
    X_res, y_res = smote_nc.fit_resample(X, y)

    out = X_res.copy()
    out.insert(0, "patient_id", np.arange(1, len(out) + 1))
    out[target_col] = y_res.values
    return out


def make_balanced_mortality(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    return _make_balanced_outcome(
        df, target_col="mortality", drop_cols=["readmission_30d"], random_state=random_state
    )


def make_balanced_readmission(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    return _make_balanced_outcome(
        df, target_col="readmission_30d", drop_cols=["mortality"], random_state=random_state
    )


def load_balanced_sepsis(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run `sepsis-ml data-prep` to generate a "
            "reconstructed dataset, or restore the committed data/balanced_sepsis.csv."
        )
    return pd.read_csv(path)


def _cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstructed synthetic data generator + SMOTE/SMOTENC balancing. "
            "Writes to data/reconstructed/ by default -- see module docstring."
        )
    )
    parser.add_argument("--n-patients", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/reconstructed"),
        help="Directory to write generated CSVs to (default: data/reconstructed).",
    )
    parser.add_argument(
        "--overwrite-canonical",
        action="store_true",
        help=(
            "Write into data/ instead, overwriting the committed reference "
            "balanced_*.csv files. Not recommended: reports/sepsis_metrics.json "
            "and the README metrics are computed from the committed files."
        ),
    )
    args = parser.parse_args(argv)

    out_dir = Path("data") if args.overwrite_canonical else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_patients} synthetic patients (seed={args.random_state})...")
    base = generate_synthetic_clinical_dataset(args.n_patients, args.random_state)
    base_path = out_dir / "synthetic_clinical_dataset.csv"
    base.to_csv(base_path, index=False)
    print(f"  wrote {base_path} ({len(base)} rows)")

    sepsis_df = make_balanced_sepsis(base, args.random_state)
    sepsis_path = out_dir / "balanced_sepsis.csv"
    sepsis_df.to_csv(sepsis_path, index=False)
    print(f"  wrote {sepsis_path} ({len(sepsis_df)} rows, is_sepsis balance="
          f"{sepsis_df['is_sepsis'].value_counts().to_dict()})")

    mortality_df = make_balanced_mortality(base, args.random_state)
    mortality_path = out_dir / "balanced_mortality.csv"
    mortality_df.to_csv(mortality_path, index=False)
    print(f"  wrote {mortality_path} ({len(mortality_df)} rows)")

    readmission_df = make_balanced_readmission(base, args.random_state)
    readmission_path = out_dir / "balanced_readmission.csv"
    readmission_df.to_csv(readmission_path, index=False)
    print(f"  wrote {readmission_path} ({len(readmission_df)} rows)")

    if args.overwrite_canonical:
        print(
            "WARNING: wrote into data/, overwriting the canonical committed "
            "files. reports/sepsis_metrics.json will no longer match the "
            "data these files describe until you retrain."
        )


if __name__ == "__main__":
    _cli()
