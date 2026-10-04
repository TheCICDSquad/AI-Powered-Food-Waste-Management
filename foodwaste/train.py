"""
Train the waste level model, compare candidates and save the best one.

Usage:
  python -m foodwaste.train
Fails (exit code 1) without saving anything if the best model is below the quality gate.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone

import joblib
import matplotlib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from foodwaste import config
from foodwaste.data import ENGINEERED_NUMERIC, add_features, load_data, split_xy, validate

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def candidate_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                                random_state=config.RANDOM_STATE, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(class_weight="balanced",
                                                                 random_state=config.RANDOM_STATE),
    }


def build_pipeline(model):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocess = ColumnTransformer([
        ("num", numeric, config.NUMERIC_FEATURES + ENGINEERED_NUMERIC),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("preprocess", preprocess),
        ("model", model),
    ])


def file_hash(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def train(data_path=config.DATA_PATH, save=True):
    df = validate(load_data(data_path))
    x, y = split_xy(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y)
    print(f"Loaded {len(df)} rows -> train {len(x_train)}, test {len(x_test)}")

    scores, fitted = {}, {}
    for name, model in candidate_models().items():
        pipe = build_pipeline(model).fit(x_train, y_train)
        pred = pipe.predict(x_test)
        scores[name] = {"macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
                        "accuracy": round(accuracy_score(y_test, pred), 4)}
        fitted[name] = pipe
        print(f"  {name:24s} macro F1 {scores[name]['macro_f1']:.4f}   accuracy {scores[name]['accuracy']:.4f}")

    best_name = max(scores, key=lambda n: scores[n]["macro_f1"])
    best = fitted[best_name]
    y_pred = best.predict(x_test)
    best_f1 = scores[best_name]["macro_f1"]
    print(f"\nBest model: {best_name} (macro F1 {best_f1:.4f})\n")
    print(classification_report(y_test, y_pred, labels=config.CLASSES))

    metrics = {
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        "best_model": best_name,
        "macro_f1": best_f1,
        "accuracy": scores[best_name]["accuracy"],
        "quality_gate_min_macro_f1": config.MIN_MACRO_F1,
        "candidates": scores,
        "per_class": classification_report(y_test, y_pred, labels=config.CLASSES, output_dict=True),
        "rows": len(df),
        "data_md5": file_hash(data_path),
        "features": config.RAW_FEATURES,
        "sklearn_version": sklearn.__version__,
    }

    if best_f1 < config.MIN_MACRO_F1:
        print(f"QUALITY GATE FAILED: macro F1 {best_f1:.4f} < {config.MIN_MACRO_F1}. Model NOT saved.")
        return best, metrics, False

    if save:
        config.MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(best, config.MODEL_PATH)
        config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
        # Snapshot of what the model was trained on, for drift monitoring
        x_train.assign(**{config.TARGET: y_train}).to_csv(config.REFERENCE_PATH, index=False)

        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred, labels=config.CLASSES),
                               display_labels=config.CLASSES).plot(cmap="Blues")
        plt.title(f"Confusion matrix - {best_name}")
        plt.savefig(config.MODELS_DIR / "confusion_matrix.png", bbox_inches="tight")
        plt.close()
        print(f"Quality gate passed (>= {config.MIN_MACRO_F1}). Saved model version {metrics['model_version']} "
              f"to {config.MODEL_PATH.relative_to(config.ROOT)}")
    return best, metrics, True


if __name__ == "__main__":
    _, _, passed = train()
    sys.exit(0 if passed else 1)
