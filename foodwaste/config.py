"""Single place for paths, column lists and quality thresholds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "food_waste_dataset.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
REFERENCE_PATH = MODELS_DIR / "reference_data.csv"   # training data snapshot, used for drift checks
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"
PREDICTION_LOG = LOGS_DIR / "predictions.jsonl"

TARGET = "Waste_Level"
CLASSES = ["Low", "Medium", "High"]

# Inputs known at the START of the day, before any food is sold.
CATEGORICAL_FEATURES = ["Business_Type", "Food_Category", "Promotion", "Weather"]
NUMERIC_FEATURES = ["Purchase_Quantity", "Shelf_Life_Days", "Storage_Temperature", "Daily_Demand"]
RAW_FEATURES = ["Date"] + CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Only known AFTER the day ends (or computed from the waste itself).
# Using them would leak the answer into the model.
LEAKAGE_COLUMNS = ["Units_Sold", "Remaining_Stock", "Waste_Quantity", "Waste_Reason", "Donation_Made"]

ALLOWED_VALUES = {
    "Business_Type": {"Restaurant", "Supermarket", "Cafeteria", "Cloud Kitchen"},
    "Food_Category": {"Fruits", "Vegetables", "Dairy", "Bakery", "Meat", "Prepared Food"},
    "Promotion": {"Yes", "No"},
    "Weather": {"Sunny", "Rainy", "Cloudy", "Hot"},
    TARGET: set(CLASSES),
}

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Quality gate: training (and CI) fails if the best model scores below this.
MIN_MACRO_F1 = 0.80

# Drift rules: a feature has drifted when the change is statistically
# significant AND big enough to matter.
DRIFT_P_VALUE = 0.05
DRIFT_MIN_KS = 0.10          # numeric features: Kolmogorov-Smirnov statistic
DRIFT_MIN_TVD = 0.10         # categorical features: total variation distance
