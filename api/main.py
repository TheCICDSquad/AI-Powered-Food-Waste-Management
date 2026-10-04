"""
REST API for waste level predictions.

Run locally:  uvicorn api.main:app --reload
Docs:         http://localhost:8000/docs
"""
import json
from datetime import date, datetime, timezone
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from foodwaste import config
from foodwaste.predict import load_metrics, load_model, predict


class InventoryRecord(BaseModel):
    """One business + food category, at the start of the day."""
    Date: date
    Business_Type: Literal["Restaurant", "Supermarket", "Cafeteria", "Cloud Kitchen"]
    Food_Category: Literal["Fruits", "Vegetables", "Dairy", "Bakery", "Meat", "Prepared Food"]
    Purchase_Quantity: int = Field(gt=0, description="Units ordered/stocked for the day")
    Shelf_Life_Days: Optional[float] = Field(None, ge=1)
    Storage_Temperature: Optional[float] = Field(None, description="Degrees Celsius")
    Daily_Demand: Optional[float] = Field(None, ge=0, description="Expected customer demand (units)")
    Promotion: Literal["Yes", "No"] = "No"
    Weather: Optional[Literal["Sunny", "Rainy", "Cloudy", "Hot"]] = None

    model_config = {"json_schema_extra": {"example": {
        "Date": "2025-07-14", "Business_Type": "Restaurant", "Food_Category": "Prepared Food",
        "Purchase_Quantity": 180, "Shelf_Life_Days": 1, "Storage_Temperature": 24.0,
        "Daily_Demand": 110, "Promotion": "No", "Weather": "Rainy"}}}


class Prediction(BaseModel):
    waste_level: Literal["Low", "Medium", "High"]
    probabilities: dict[str, float]
    model_version: str


app = FastAPI(title="Food Waste Level API", version="1.0.0",
              description="Predicts Low / Medium / High food waste at the start of the day.")


def _log(records, results):
    """Append every request + prediction, so drift can be checked on real traffic later."""
    config.LOGS_DIR.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    with open(config.PREDICTION_LOG, "a") as f:
        for rec, res in zip(records, results):
            f.write(json.dumps({"timestamp": now, "input": rec, "prediction": res["waste_level"]}) + "\n")


def _predict(items):
    records = [item.model_dump(mode="json") for item in items]
    try:
        results = predict(records)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    _log(records, results)
    version = load_metrics().get("model_version", "unknown")
    return [Prediction(**r, model_version=version) for r in results]


@app.get("/health")
def health():
    try:
        load_model()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    m = load_metrics()
    return {k: m.get(k) for k in ["model_version", "best_model", "macro_f1", "accuracy", "rows", "data_md5",
                                  "sklearn_version", "candidates"]}


@app.post("/predict", response_model=Prediction)
def predict_one(item: InventoryRecord):
    return _predict([item])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(items: list[InventoryRecord]):
    if not items:
        raise HTTPException(status_code=422, detail="Send at least one record")
    return _predict(items)
