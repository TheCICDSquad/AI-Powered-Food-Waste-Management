import pytest
from fastapi.testclient import TestClient

from foodwaste import config


@pytest.fixture
def client(trained, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(config, "PREDICTION_LOG", tmp_path / "predictions.jsonl")
    from api.main import app
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict(client, sample_record):
    r = client.post("/predict", json=sample_record)
    assert r.status_code == 200
    body = r.json()
    assert body["waste_level"] in config.CLASSES
    assert abs(sum(body["probabilities"].values()) - 1) < 0.01


def test_predict_logs_request(client, sample_record):
    client.post("/predict", json=sample_record)
    assert config.PREDICTION_LOG.read_text().count("\n") == 1


def test_batch(client, sample_record):
    r = client.post("/predict/batch", json=[sample_record, {**sample_record, "Promotion": "Yes"}])
    assert r.status_code == 200 and len(r.json()) == 2


def test_optional_fields_can_be_missing(client, sample_record):
    record = {k: v for k, v in sample_record.items() if k not in ("Weather", "Daily_Demand")}
    assert client.post("/predict", json=record).status_code == 200


@pytest.mark.parametrize("field,value", [("Food_Category", "Seafood"), ("Purchase_Quantity", -5),
                                         ("Date", "not-a-date")])
def test_invalid_input_rejected(client, sample_record, field, value):
    assert client.post("/predict", json={**sample_record, field: value}).status_code == 422
