import pandas as pd

from foodwaste import config


def test_model_passes_quality_gate(trained):
    _, metrics, passed = trained
    assert passed, f"macro F1 {metrics['macro_f1']} below {config.MIN_MACRO_F1}"


def test_model_beats_always_predicting_low(trained, raw_data):
    _, metrics, _ = trained
    majority_share = raw_data[config.TARGET].value_counts(normalize=True).max()
    assert metrics["accuracy"] > majority_share


def test_predictions_are_valid_classes(trained, raw_data):
    model, _, _ = trained
    preds = model.predict(raw_data.head(50)[config.RAW_FEATURES])
    assert set(preds) <= set(config.CLASSES)


def test_model_handles_missing_values(trained, sample_record):
    model, _, _ = trained
    record = {**sample_record, "Shelf_Life_Days": None, "Storage_Temperature": None,
              "Daily_Demand": None, "Weather": None}
    assert model.predict(pd.DataFrame([record]))[0] in config.CLASSES


def test_overstocking_raises_waste_risk(trained, sample_record):
    """Behaviour check: ordering far more than demand should not lower the High-waste probability."""
    model, _, _ = trained
    normal = {**sample_record, "Purchase_Quantity": 110}
    over = {**sample_record, "Purchase_Quantity": 400}
    high = list(model.classes_).index("High")
    p_normal, p_over = model.predict_proba(pd.DataFrame([normal, over]))[:, high]
    assert p_over >= p_normal
