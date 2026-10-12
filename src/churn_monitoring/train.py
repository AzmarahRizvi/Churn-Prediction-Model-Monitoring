"""Train and evaluate an XGBoost churn classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from churn_monitoring.data import generate_customers
from churn_monitoring.schema import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET


def train(output_dir: Path, n_rows: int, seed: int) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = generate_customers(n_rows, seed=seed)
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessing),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.045,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.5,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=seed,
                    n_jobs=4,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_valid)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "rows": int(n_rows),
        "validation_rows": int(len(X_valid)),
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y_valid, probabilities)),
        "accuracy": float(accuracy_score(y_valid, predictions)),
        "precision": float(precision_score(y_valid, predictions, zero_division=0)),
        "recall": float(recall_score(y_valid, predictions, zero_division=0)),
        "f1": float(f1_score(y_valid, predictions, zero_division=0)),
    }

    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "threshold": 0.5}, model_dir / "model.joblib")
    data.to_csv(output_dir / "training_baseline.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--rows", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metrics = train(args.output_dir, args.rows, args.seed)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

