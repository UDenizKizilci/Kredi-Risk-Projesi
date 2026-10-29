"""Train the credit-risk model and persist it with joblib.

Usage (from project root):
    python models/train_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # makes `src` importable and picklable

import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing import (
    RAW_FEATURES,
    TARGET,
    build_preprocessor,
    feature_engineering_step,
    synthetic_dataset,
    validate_schema,
)

DATA_PATH = ROOT / "data" / "german_credit_data.csv"
MODEL_PATH = ROOT / "models" / "credit_model.pkl"
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        if validate_schema(df):
            print(f"Loaded {len(df)} rows from {DATA_PATH}")
            return df
        print("CSV does not match the unified schema -> generating synthetic data.")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = synthetic_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Synthetic dataset written to {DATA_PATH}")
    return df


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    """Imbalance is handled inside the model via scale_pos_weight."""
    return Pipeline(
        steps=[
            ("features", feature_engineering_step()),
            ("preprocess", build_preprocessor()),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="auc",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    df = load_data()
    X, y = df[RAW_FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"Default rate: {y.mean():.2%} | scale_pos_weight = {spw:.2f}")

    pipeline = build_pipeline(spw)
    cv_auc = cross_val_score(
        pipeline, X_train, y_train, scoring="roc_auc",
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
    )
    print(f"5-fold CV ROC-AUC: {cv_auc.mean():.3f} (+/- {cv_auc.std():.3f})")

    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, proba)
    print(f"Test ROC-AUC: {test_auc:.3f}")
    print(classification_report(y_test, (proba >= 0.5).astype(int)))

    joblib.dump({"pipeline": pipeline, "test_auc": float(test_auc)}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()