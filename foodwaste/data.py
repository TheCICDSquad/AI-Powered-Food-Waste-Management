"""Load, validate and engineer features for the food waste data."""
import numpy as np
import pandas as pd

from foodwaste import config


class DataValidationError(ValueError):
    pass


def load_data(path=config.DATA_PATH):
    return pd.read_csv(path)


def validate(df, require_target=True):
    """Check the data before it reaches the model. Raises DataValidationError listing every problem."""
    errors = []
    required = config.RAW_FEATURES + ([config.TARGET] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")

    if df.empty:
        errors.append("No rows")

    for col, allowed in config.ALLOWED_VALUES.items():
        if col not in df.columns:
            continue
        unknown = set(df[col].dropna().unique()) - allowed
        if unknown:
            errors.append(f"{col}: unknown values {sorted(unknown)}")

    if (df["Purchase_Quantity"] <= 0).any():
        errors.append("Purchase_Quantity must be > 0")
    if (df["Shelf_Life_Days"].dropna() < 1).any():
        errors.append("Shelf_Life_Days must be >= 1")
    if (df["Daily_Demand"].dropna() < 0).any():
        errors.append("Daily_Demand must be >= 0")
    if pd.to_datetime(df["Date"], errors="coerce").isna().any():
        errors.append("Date has invalid values")
    if require_target and df[config.TARGET].isna().any():
        errors.append(f"{config.TARGET} has missing values")

    if errors:
        raise DataValidationError("; ".join(errors))
    return df


def add_features(df):
    """Feature engineering. Runs inside the model pipeline, so the API can send raw columns."""
    df = df.copy()
    date = pd.to_datetime(df["Date"])
    df["Month"] = date.dt.month
    df["Day_Of_Week"] = date.dt.dayofweek
    df["Is_Weekend"] = (df["Day_Of_Week"] >= 5).astype(int)
    # Ordering much more than expected demand is the main driver of waste
    df["Order_To_Demand_Ratio"] = df["Purchase_Quantity"] / df["Daily_Demand"].replace(0, np.nan)
    return df.drop(columns=["Date"])


ENGINEERED_NUMERIC = ["Month", "Day_Of_Week", "Is_Weekend", "Order_To_Demand_Ratio"]


def split_xy(df):
    return df[config.RAW_FEATURES], df[config.TARGET]
