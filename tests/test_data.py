import pytest

from foodwaste import config
from foodwaste.data import DataValidationError, add_features, validate


def test_real_data_passes_validation(raw_data):
    validate(raw_data)


def test_no_leakage_columns_in_features():
    assert not set(config.LEAKAGE_COLUMNS) & set(config.RAW_FEATURES)
    assert config.TARGET not in config.RAW_FEATURES


def test_missing_column_is_rejected(raw_data):
    with pytest.raises(DataValidationError, match="Missing columns"):
        validate(raw_data.drop(columns=["Food_Category"]))


def test_unknown_category_is_rejected(raw_data):
    bad = raw_data.head(5).copy()
    bad.loc[0, "Food_Category"] = "Seafood"
    with pytest.raises(DataValidationError, match="Seafood"):
        validate(bad)


def test_non_positive_purchase_is_rejected(raw_data):
    bad = raw_data.head(5).copy()
    bad.loc[0, "Purchase_Quantity"] = 0
    with pytest.raises(DataValidationError, match="Purchase_Quantity"):
        validate(bad)


def test_feature_engineering(raw_data):
    out = add_features(raw_data.head(3)[config.RAW_FEATURES])
    assert {"Month", "Day_Of_Week", "Is_Weekend", "Order_To_Demand_Ratio"} <= set(out.columns)
    assert "Date" not in out.columns
    assert out.loc[0, "Month"] == 1   # first row is 2025-01-01
