"""Regression test for the target-leakage bug found in the original
train_sepsis.py.

The original script dropped only `is_sepsis` from the feature matrix and
left in `diagnosis_Sepsis` (identical to the label), the other
`diagnosis_*` one-hot columns, and the post-admission outcomes `mortality`
/ `readmission_30d`. That produced a fabricated ROC-AUC / PR-AUC of 1.0.

This test asserts, directly against the real feature-selection code path
used by training (`sepsis_ml.features.prepare_features` /
`get_feature_columns`), that none of those columns can reach the model.
If someone reintroduces the leak in features.py or train.py, this test
must fail.
"""
from sepsis_ml.features import (
    LEAKAGE_COLUMNS,
    get_feature_columns,
    prepare_features,
)

EXPECTED_LEAKAGE_COLUMNS = {
    "diagnosis_Heart Failure",
    "diagnosis_Normal",
    "diagnosis_Pneumonia",
    "diagnosis_Sepsis",
    "mortality",
    "readmission_30d",
}


def test_leakage_columns_constant_is_complete():
    assert EXPECTED_LEAKAGE_COLUMNS.issubset(set(LEAKAGE_COLUMNS))


def test_get_feature_columns_excludes_leakage(leaky_sepsis_df):
    feature_cols = get_feature_columns(leaky_sepsis_df, target_col="is_sepsis")

    for leaky_col in EXPECTED_LEAKAGE_COLUMNS:
        assert leaky_col not in feature_cols, (
            f"Leakage column '{leaky_col}' must not appear in the feature "
            "matrix -- it leaks the label or a post-prediction-time outcome."
        )

    assert "is_sepsis" not in feature_cols


def test_get_feature_columns_excludes_any_diagnosis_prefixed_column(
    leaky_sepsis_df,
):
    # Defensive check: even a diagnosis one-hot column not explicitly listed
    # (e.g. if categories change) must still be excluded via the prefix rule.
    df = leaky_sepsis_df.copy()
    df["diagnosis_Unknown"] = 0.0

    feature_cols = get_feature_columns(df, target_col="is_sepsis")
    assert "diagnosis_Unknown" not in feature_cols


def test_prepare_features_returns_clean_matrix(leaky_sepsis_df):
    X, y = prepare_features(leaky_sepsis_df, target_col="is_sepsis")

    leaked = EXPECTED_LEAKAGE_COLUMNS.intersection(set(X.columns))
    assert not leaked, f"Leakage columns present in X: {leaked}"

    # Sanity: y should be exactly the sepsis label, and X should still have
    # legitimate clinical features left over.
    assert set(y.unique()).issubset({0, 1})
    assert "age" in X.columns
    assert "creatinine" in X.columns


def test_diagnosis_sepsis_is_literally_the_label(leaky_sepsis_df):
    # Documents *why* this bug was so severe: diagnosis_Sepsis is not just
    # correlated with is_sepsis, it is bit-for-bit identical to it.
    assert (
        leaky_sepsis_df["diagnosis_Sepsis"] == leaky_sepsis_df["is_sepsis"]
    ).all()
