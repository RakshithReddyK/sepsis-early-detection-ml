#!/usr/bin/env python3
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/balanced_sepsis.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def load_data(path: Path = DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    df = pd.read_csv(path)

    if "is_sepsis" not in df.columns:
        raise ValueError("Expected 'is_sepsis' column in balanced_sepsis.csv")

    X = df.drop(columns=["is_sepsis"])
    y = df["is_sepsis"].astype(int)
    return X, y


def build_pipeline(n_features: int) -> Pipeline:
    """
    Build a simple numeric-only pipeline:
    StandardScaler -> GradientBoostingClassifier
    """
    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )
    return pipe


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    print(f"📂 Loading data from {DATA_PATH} ...")
    X, y = load_data()

    print(f"✅ Loaded {len(X)} rows, {X.shape[1]} features.")
    print("   Class balance:", y.value_counts().to_dict())

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print(f"🧪 Train size: {len(X_train)}, Val size: {len(X_val)}")

    model = build_pipeline(X.shape[1])

    print("🚀 Training GradientBoostingClassifier...")
    model.fit(X_train, y_train)

    # Evaluate
    proba_val = model.predict_proba(X_val)[:, 1]
    preds_val = (proba_val >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_val, proba_val)
    pr_auc = average_precision_score(y_val, proba_val)
    cm = confusion_matrix(y_val, preds_val)
    report = classification_report(y_val, preds_val, output_dict=True)

    print(f"✅ Training complete.")
    print(f"📈 ROC-AUC: {roc_auc:.4f}")
    print(f"📈 PR-AUC : {pr_auc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # Save model + metrics
    model_path = MODELS_DIR / "sepsis_gb_model.joblib"
    metrics_path = REPORTS_DIR / "sepsis_metrics.json"

    joblib.dump(model, model_path)
    with metrics_path.open("w") as f:
        json.dump(
            {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "confusion_matrix": cm.tolist(),
                "classification_report": report,
            },
            f,
            indent=2,
        )

    print(f"💾 Model saved to: {model_path}")
    print(f"📝 Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()

