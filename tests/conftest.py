import pytest

from foodwaste import config
from foodwaste.data import load_data


@pytest.fixture(scope="session")
def raw_data():
    return load_data()


@pytest.fixture(scope="session")
def trained():
    """Train once per test session. Returns (pipeline, metrics, passed_quality_gate)."""
    from foodwaste.train import train
    return train(save=not config.MODEL_PATH.exists())


@pytest.fixture
def sample_record():
    return {"Date": "2025-07-14", "Business_Type": "Restaurant", "Food_Category": "Prepared Food",
            "Purchase_Quantity": 180, "Shelf_Life_Days": 1.0, "Storage_Temperature": 24.0,
            "Daily_Demand": 110.0, "Promotion": "No", "Weather": "Rainy"}
