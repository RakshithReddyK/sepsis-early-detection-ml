"""End-to-end smoke test: the training pipeline runs on a small fixture
and produces the expected artifacts, without erroring and without a
suspiciously perfect (leaked) score.
"""
from pathlib import Path

from sepsis_ml.train import run_training


def test_run_training_end_to_end(tmp_path: Path, tiny_balanced_sepsis_csv: Path):
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"

    metrics = run_training(
        data_path=tiny_balanced_sepsis_csv,
        models_dir=models_dir,
        reports_dir=reports_dir,
        n_splits=3,
        n_repeats=2,
        random_state=0,
    )

    # Artifacts exist
    assert (models_dir / "sepsis_gb_model.joblib").exists()
    assert (reports_dir / "sepsis_metrics.json").exists()
    assert (reports_dir / "figures" / "confusion_matrix.png").exists()
    assert (reports_dir / "figures" / "pr_curve.png").exists()

    # Metrics are well-formed
    for key in ("cv_roc_auc_mean", "cv_pr_auc_mean", "out_of_fold_roc_auc"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0

    # None of the leakage columns made it into the trained feature set.
    leaked = {
        "diagnosis_Heart Failure",
        "diagnosis_Normal",
        "diagnosis_Pneumonia",
        "diagnosis_Sepsis",
        "mortality",
        "readmission_30d",
    }
    assert leaked.isdisjoint(set(metrics["feature_columns"]))
