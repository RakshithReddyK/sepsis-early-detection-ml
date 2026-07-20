"""Training entry point for the sepsis early-detection model.

Replaces the old `train_sepsis.py` script. Key differences from the
original:

  * Feature selection goes through `features.prepare_features`, which
    drops the `diagnosis_*` / `mortality` / `readmission_30d` leakage
    columns (see features.py for why).
  * Evaluation uses repeated stratified k-fold cross-validation instead of
    a single 80/20 split, and also reports out-of-fold metrics from a
    plain stratified k-fold (used to produce a single confusion matrix /
    PR curve).
  * All paths are CLI arguments (argparse), not hardcoded module constants.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_balanced_sepsis
from .evaluate import compute_metrics, save_confusion_matrix_plot, save_pr_curve_plot
from .features import prepare_features


def build_pipeline(random_state: int = 42) -> Pipeline:
    """StandardScaler -> GradientBoostingClassifier, matching the original
    project's model choice."""
    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=random_state,
    )
    return Pipeline(steps=[("scaler", StandardScaler()), ("clf", clf)])


def run_training(
    data_path: Path,
    models_dir: Path,
    reports_dir: Path,
    target_col: str = "is_sepsis",
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
) -> dict:
    """Run the full train + cross-validate + evaluate + persist pipeline.

    Returns the metrics dict that is also written to
    `reports_dir/sepsis_metrics.json`.
    """
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path} ...")
    df = load_balanced_sepsis(Path(data_path))
    X, y = prepare_features(df, target_col=target_col)

    dropped = [c for c in df.columns if c not in list(X.columns) + [target_col]]
    print(f"Loaded {len(df)} rows, {df.shape[1]} raw columns.")
    print(f"Dropped as leakage / non-feature columns: {dropped}")
    print(f"Using {X.shape[1]} features: {list(X.columns)}")
    print("Class balance:", y.value_counts().to_dict())

    pipeline = build_pipeline(random_state=random_state)

    # --- Repeated stratified k-fold CV: robust headline metrics ---
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=rskf,
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
        n_jobs=-1,
    )
    roc_auc_scores = cv_results["test_roc_auc"]
    pr_auc_scores = cv_results["test_pr_auc"]

    print(
        f"CV ROC-AUC: {roc_auc_scores.mean():.4f} +/- {roc_auc_scores.std():.4f} "
        f"over {n_splits}x{n_repeats} folds"
    )
    print(
        f"CV PR-AUC : {pr_auc_scores.mean():.4f} +/- {pr_auc_scores.std():.4f} "
        f"over {n_splits}x{n_repeats} folds"
    )

    # --- Plain stratified k-fold: single set of out-of-fold predictions
    # for a confusion matrix / PR curve (repeated CV would score each row
    # multiple times, which doesn't make sense for a single plot). ---
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_proba = cross_val_predict(
        pipeline, X, y, cv=skf, method="predict_proba", n_jobs=-1
    )[:, 1]
    oof_metrics = compute_metrics(y, oof_proba)

    save_confusion_matrix_plot(
        y, (oof_proba >= 0.5).astype(int), figures_dir / "confusion_matrix.png"
    )
    save_pr_curve_plot(y, oof_proba, figures_dir / "pr_curve.png")

    # --- Fit the final model on all available data for persistence ---
    pipeline.fit(X, y)
    model_path = models_dir / "sepsis_gb_model.joblib"
    joblib.dump(pipeline, model_path)

    metrics = {
        "feature_columns": list(X.columns),
        "dropped_leakage_columns": dropped,
        "n_rows": int(len(df)),
        "class_balance": {str(k): int(v) for k, v in y.value_counts().items()},
        "cv_scheme": f"RepeatedStratifiedKFold(n_splits={n_splits}, n_repeats={n_repeats})",
        "cv_roc_auc_mean": float(roc_auc_scores.mean()),
        "cv_roc_auc_std": float(roc_auc_scores.std()),
        "cv_pr_auc_mean": float(pr_auc_scores.mean()),
        "cv_pr_auc_std": float(pr_auc_scores.std()),
        "cv_roc_auc_folds": roc_auc_scores.tolist(),
        "cv_pr_auc_folds": pr_auc_scores.tolist(),
        # Out-of-fold metrics from a single stratified 5-fold pass, used for
        # the confusion matrix / classification report / PR curve artifacts.
        "out_of_fold_roc_auc": oof_metrics["roc_auc"],
        "out_of_fold_pr_auc": oof_metrics["pr_auc"],
        "out_of_fold_confusion_matrix": oof_metrics["confusion_matrix"],
        "out_of_fold_classification_report": oof_metrics["classification_report"],
        "artifacts": {
            "model": str(model_path),
            "confusion_matrix_plot": str(figures_dir / "confusion_matrix.png"),
            "pr_curve_plot": str(figures_dir / "pr_curve.png"),
        },
    }

    metrics_path = reports_dir / "sepsis_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    return metrics


def _cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the sepsis early-detection model.")
    parser.add_argument(
        "--data-path", type=Path, default=Path("data/balanced_sepsis.csv"),
        help="Path to the balanced sepsis CSV (default: data/balanced_sepsis.csv).",
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--target-col", type=str, default="is_sepsis")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    run_training(
        data_path=args.data_path,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        target_col=args.target_col,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    _cli()
