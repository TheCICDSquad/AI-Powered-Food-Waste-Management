"""Load the saved model and make predictions from raw records."""
import json
from functools import lru_cache

import joblib
import pandas as pd

from foodwaste import config
from foodwaste.data import validate


@lru_cache(maxsize=1)
def load_model():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"No model at {config.MODEL_PATH}. Run: python -m foodwaste.train")
    return joblib.load(config.MODEL_PATH)


@lru_cache(maxsize=1)
def load_metrics():
    return json.loads(config.METRICS_PATH.read_text()) if config.METRICS_PATH.exists() else {}


def predict(records):
    """records: list of dicts with the raw feature columns. Returns one result dict per record."""
    df = validate(pd.DataFrame(records), require_target=False)
    model = load_model()
    labels = model.predict(df[config.RAW_FEATURES])
    probs = model.predict_proba(df[config.RAW_FEATURES])
    classes = list(model.classes_)
    return [
        {"waste_level": label,
         "probabilities": {c: round(float(p[classes.index(c)]), 4) for c in config.CLASSES}}
        for label, p in zip(labels, probs)
    ]
