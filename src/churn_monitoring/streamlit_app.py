"""Streamlit interface for the churn prediction API."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st


API_BASE_URL = os.getenv("CHURN_API_BASE_URL", "http://127.0.0.1:8080").rstrip("/")

st.set_page_config(
    page_title="Churn Signal | Customer risk",
    page_icon="📊",
    layout="wide",
)

st.title("Customer churn prediction")
st.caption("Enter a customer profile to estimate its churn probability.")
st.info(
    "Demo model: this project uses synthetic telco data. Scores demonstrate the prediction "
    "pipeline and should not be treated as real customer assessments."
)


def get_api_status() -> tuple[bool, str]:
    """Check whether the prediction API is reachable and has loaded its model."""
    try:
        with urlopen(f"{API_BASE_URL}/health", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        is_ready = response.status == 200 and payload.get("status") == "ok"
        return is_ready, "Prediction API connected" if is_ready else "Prediction API is starting"
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return False, f"Prediction API unavailable at {API_BASE_URL}"


def request_prediction(instance: dict[str, Any]) -> dict[str, Any]:
    """Send one customer record to the FastAPI prediction endpoint."""
    request = Request(
        f"{API_BASE_URL}/predict",
        data=json.dumps({"instances": [instance]}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("detail", str(exc))
        except (json.JSONDecodeError, AttributeError):
            detail = str(exc)
        if isinstance(detail, list):
            detail = "; ".join(item.get("msg", str(item)) for item in detail)
        raise RuntimeError(str(detail)) from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Could not connect to {API_BASE_URL}. Start the FastAPI server and try again."
        ) from exc


status_placeholder = st.empty()
api_ready, status_text = get_api_status()
if api_ready:
    status_placeholder.success(status_text)
else:
    status_placeholder.warning(status_text)

st.subheader("Customer profile")
st.caption("All fields are required. The example values are prefilled for a quick prediction.")

with st.form("customer-profile"):
    left, right = st.columns(2)

    with left:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=2, step=1)
        monthly_charges = st.number_input(
            "Monthly charges ($)", min_value=0.0, value=95.50, step=1.0, format="%.2f"
        )
        total_charges = st.number_input(
            "Total charges ($)", min_value=0.0, value=191.00, step=10.0, format="%.2f"
        )
        senior_citizen_label = st.selectbox("Senior citizen", ["No", "Yes"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with right:
        internet_service = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        payment_method = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        )
        tech_support = st.selectbox("Tech support", ["No", "Yes"])
        online_security = st.selectbox("Online security", ["No", "Yes"])
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])

    submitted = st.form_submit_button("Predict churn risk", type="primary", use_container_width=True)

if submitted:
    instance = {
        "tenure": int(tenure),
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
        "SeniorCitizen": int(senior_citizen_label == "Yes"),
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        "TechSupport": tech_support,
        "OnlineSecurity": online_security,
        "PaperlessBilling": paperless_billing,
    }

    try:
        prediction_response = request_prediction(instance)
        prediction = prediction_response["predictions"][0]
        probability = float(prediction["churn_probability"])
        churn_prediction = int(prediction["churn_prediction"])

        st.subheader("Prediction result")
        metric_column, signal_column = st.columns([1, 2])
        with metric_column:
            st.metric("Churn probability", f"{probability:.1%}")
        with signal_column:
            st.progress(max(0.0, min(1.0, probability)), text="Model churn probability")

        if churn_prediction == 1:
            st.warning("Higher churn risk — the model predicts this customer may churn.")
        else:
            st.success("Lower churn risk — the model predicts this customer is less likely to churn.")
    except (RuntimeError, KeyError, IndexError, TypeError, ValueError) as exc:
        st.error(f"Prediction failed: {exc}")
