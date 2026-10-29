"""Feature schema, feature engineering and the scikit-learn preprocessing stage."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "age",
    "monthly_income",
    "loan_amount",
    "loan_term_months",
    "debt_to_income",
    "employment_years",
    "loan_to_annual_income",  # derived
]
CATEGORICAL_FEATURES = ["housing", "savings_level"]
RAW_FEATURES = [
    "age", "monthly_income", "loan_amount", "loan_term_months",
    "debt_to_income", "employment_years", "housing", "savings_level",
]
TARGET = "default"  # 1 = defaulted / risky

HOUSING_OPTIONS = ["own", "rent", "free"]
SAVINGS_OPTIONS = ["none", "low", "medium", "high"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ratio features. Pure function so it can live inside a Pipeline."""
    out = df.copy()
    annual_income = (out["monthly_income"].astype(float) * 12).clip(lower=1.0)
    out["loan_to_annual_income"] = out["loan_amount"].astype(float) / annual_income
    return out


def build_preprocessor() -> ColumnTransformer:
    """Scaling for numeric features, one-hot encoding for categorical ones."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def feature_engineering_step() -> FunctionTransformer:
    return FunctionTransformer(add_features, validate=False)


def validate_schema(df: pd.DataFrame) -> bool:
    """True if the dataframe already follows the unified schema."""
    return set(RAW_FEATURES + [TARGET]).issubset(df.columns)


def synthetic_dataset(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate an imbalanced (~15-20% default) demo dataset in the unified schema."""
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 70, n)
    income = np.clip(rng.lognormal(mean=8.6, sigma=0.45, size=n), 1500, 40000)
    term = rng.choice([12, 24, 36, 48, 60], n)
    loan = np.clip(income * rng.uniform(1, 20, n), 1000, 400000)
    dti = np.clip(rng.beta(2.2, 4.5, n) * 0.9, 0.0, 0.9)
    emp = np.clip(rng.gamma(2.0, 3.0, n), 0, 40)
    housing = rng.choice(HOUSING_OPTIONS, n, p=[0.55, 0.35, 0.10])
    savings = rng.choice(SAVINGS_OPTIONS, n, p=[0.25, 0.35, 0.25, 0.15])

    lti = loan / (income * 12)
    logit = (
        -3.4
        + 4.0 * dti
        + 0.9 * lti
        - 0.07 * np.minimum(emp, 15)
        - 0.012 * (age - 21).clip(0, 25)
        + 0.35 * (housing == "rent")
        + 0.2 * (housing == "free")
        - 0.35 * np.array([SAVINGS_OPTIONS.index(s) for s in savings])
        + 0.006 * term
        + rng.normal(0, 0.4, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    return pd.DataFrame(
        {
            "age": age,
            "monthly_income": income.round(0),
            "loan_amount": loan.round(0),
            "loan_term_months": term,
            "debt_to_income": dti.round(3),
            "employment_years": emp.round(1),
            "housing": housing,
            "savings_level": savings,
            TARGET: (rng.uniform(size=n) < prob).astype(int),
        }
    )