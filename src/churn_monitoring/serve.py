"""FastAPI server compatible with local Docker and Vertex AI custom prediction."""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from churn_monitoring.data import validate_feature_frame


def _load_bundle() -> dict[str, Any]:
    explicit_path = os.getenv("MODEL_PATH")
    if explicit_path:
        return joblib.load(explicit_path)

    storage_uri = os.getenv("AIP_STORAGE_URI")
    if storage_uri:
        parsed = urlparse(storage_uri)
        if parsed.scheme != "gs" or not parsed.netloc:
            raise RuntimeError("AIP_STORAGE_URI must be a gs:// URI")
        from google.cloud import storage

        bucket = storage.Client().bucket(parsed.netloc)
        prefix = parsed.path.lstrip("/").rstrip("/")
        with tempfile.TemporaryDirectory(prefix="churn-model-") as tmp_dir:
            local_dir = Path(tmp_dir)
            for blob in bucket.list_blobs(prefix=f"{prefix}/" if prefix else ""):
                relative = blob.name[len(prefix) :].lstrip("/") if prefix else blob.name
                if not relative:
                    continue
                destination = local_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(destination)
            model_path = local_dir / "model.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"No model.joblib found under {storage_uri}")
            return joblib.load(model_path)

    fallback_path = Path("artifacts/model/model.joblib")
    if fallback_path.exists():
        return joblib.load(fallback_path)
    raise RuntimeError("Set MODEL_PATH locally or provide model artifacts through AIP_STORAGE_URI")


@asynccontextmanager
async def lifespan(_: FastAPI):
    app.state.bundle = _load_bundle()
    yield


app = FastAPI(title="Churn Prediction API", version="0.1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[dict[str, Any]]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if getattr(app.state, "bundle", None) else "starting"}


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, list[dict[str, Any]]]:
    if not request.instances:
        raise HTTPException(status_code=422, detail="instances must contain at least one record")
    try:
        frame = validate_feature_frame(pd.DataFrame(request.instances))
        bundle = app.state.bundle
        probabilities = bundle["model"].predict_proba(frame)[:, 1]
        threshold = float(bundle.get("threshold", 0.5))
        results = [
            {
                "churn_probability": float(probability),
                "churn_prediction": int(probability >= threshold),
            }
            for probability in probabilities
        ]
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AttributeError as exc:
        raise HTTPException(status_code=503, detail="Model is not loaded") from exc
    return {"predictions": results}

