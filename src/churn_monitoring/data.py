"""Deterministic synthetic telco data for a self-contained demo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn_monitoring.schema import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET


def _blend_probabilities(
    rng: np.random.Generator,
    baseline: list[float],
    shifted: list[float],
    strength: float,
    size: int,
) -> np.ndarray:
    probabilities = (1.0 - strength) * np.asarray(baseline) + strength * np.asarray(shifted)
    return rng.choice(len(probabilities), size=size, p=probabilities)


def generate_customers(
    n_rows: int,
    *,
    seed: int = 42,
    drift_strength: float = 0.0,
    include_target: bool = True,
) -> pd.DataFrame:
    """Create plausible telco records; ``drift_strength`` moves production to a new mix."""
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if not 0.0 <= drift_strength <= 1.0:
        raise ValueError("drift_strength must be between 0 and 1")

    rng = np.random.default_rng(seed)
    tenure = np.clip(
        np.rint(
            (1.0 - drift_strength) * rng.gamma(shape=2.2, scale=15.0, size=n_rows)
            + drift_strength * rng.gamma(shape=1.4, scale=7.0, size=n_rows)
        ),
        0,
        72,
    ).astype(int)

    monthly_charges = np.clip(
        rng.normal(loc=70.0 + 18.0 * drift_strength, scale=22.0, size=n_rows), 20.0, 120.0
    )
    total_charges = np.maximum(
        0.0, tenure * monthly_charges + rng.normal(0.0, 90.0, size=n_rows)
    )

    contract_idx = _blend_probabilities(
        rng, [0.54, 0.25, 0.21], [0.88, 0.08, 0.04], drift_strength, n_rows
    )
    internet_idx = _blend_probabilities(
        rng, [0.22, 0.46, 0.32], [0.10, 0.72, 0.18], drift_strength, n_rows
    )
    payment_idx = _blend_probabilities(
        rng, [0.34, 0.24, 0.22, 0.20], [0.69, 0.08, 0.13, 0.10], drift_strength, n_rows
    )

    contracts = np.array(["Month-to-month", "One year", "Two year"])[contract_idx]
    internet = np.array(["No", "Fiber optic", "DSL"])[internet_idx]
    payment = np.array(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    )[payment_idx]
    tech_support = rng.choice(["Yes", "No"], size=n_rows, p=[0.30, 0.70])
    online_security = rng.choice(["Yes", "No"], size=n_rows, p=[0.34, 0.66])
    paperless = rng.choice(["Yes", "No"], size=n_rows, p=[0.62, 0.38])
    senior_citizen = rng.binomial(1, 0.17, size=n_rows)

    frame = pd.DataFrame(
        {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges.round(2),
            "TotalCharges": total_charges.round(2),
            "SeniorCitizen": senior_citizen,
            "Contract": contracts,
            "InternetService": internet,
            "PaymentMethod": payment,
            "TechSupport": tech_support,
            "OnlineSecurity": online_security,
            "PaperlessBilling": paperless,
        },
        columns=FEATURES,
    )

    if include_target:
        contract_effect = pd.Series(contracts).map(
            {"Month-to-month": 1.15, "One year": -0.25, "Two year": -1.05}
        ).to_numpy()
        log_odds = (
            -2.15
            + contract_effect
            - 0.028 * tenure
            + 0.016 * (monthly_charges - 70.0)
            + 0.38 * senior_citizen
            + 0.22 * (internet == "Fiber optic")
            + 0.35 * (payment == "Electronic check")
            - 0.62 * (tech_support == "Yes")
            - 0.48 * (online_security == "Yes")
            + 0.16 * (paperless == "Yes")
        )
        churn_probability = 1.0 / (1.0 + np.exp(-log_odds))
        frame[TARGET] = rng.binomial(1, churn_probability, size=n_rows)

    return frame


def validate_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and order an inference frame, rejecting missing or extra fields."""
    missing = sorted(set(FEATURES) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(FEATURES))
    if missing or extra:
        raise ValueError(f"Invalid feature columns; missing={missing}, extra={extra}")
    result = frame.loc[:, FEATURES].copy()
    for column in NUMERIC_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="raise")
    for column in CATEGORICAL_FEATURES:
        result[column] = result[column].astype(str)
    return result

