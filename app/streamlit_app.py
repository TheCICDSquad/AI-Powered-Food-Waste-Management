"""
Streamlit UI. Calls the FastAPI service; falls back to the local model if the API is not running.

Run:  streamlit run app/streamlit_app.py
"""
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from foodwaste import config  # noqa: E402
from foodwaste.drift import detect_drift, simulate  # noqa: E402
from foodwaste.predict import load_metrics, predict  # noqa: E402

API_URL = os.getenv("API_URL", "http://localhost:8000")
COLORS = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}
ADVICE = {
    "Low": "Order looks right for expected demand.",
    "Medium": "Some waste likely - consider a smaller order or a promotion.",
    "High": "High waste risk - cut the order, run a promotion or plan a donation.",
}

st.set_page_config(page_title="Food Waste Predictor", page_icon="🥗", layout="wide")


def get_prediction(record):
    try:
        r = requests.post(f"{API_URL}/predict", json=record, timeout=5)
        r.raise_for_status()
        return r.json(), "FastAPI service"
    except requests.RequestException:
        result = predict([record])[0]
        return {**result, "model_version": load_metrics().get("model_version", "unknown")}, "local model (API offline)"


st.title("🥗 Food Waste Level Predictor")
st.caption("Predicts at the start of the day whether today's stock will end in Low, Medium or High waste.")
predict_tab, model_tab, monitor_tab = st.tabs(["Predict", "Model", "Monitoring"])

with predict_tab:
    with st.form("record"):
        c1, c2, c3 = st.columns(3)
        day = c1.date_input("Date", date(2025, 7, 14))
        business = c1.selectbox("Business type", sorted(config.ALLOWED_VALUES["Business_Type"]))
        category = c1.selectbox("Food category", sorted(config.ALLOWED_VALUES["Food_Category"]), index=4)
        purchase = c2.number_input("Purchase quantity (units)", 1, 2000, 180)
        demand = c2.number_input("Expected demand (units)", 0, 2000, 110)
        shelf = c2.number_input("Shelf life (days)", 1, 30, 1)
        temp = c3.number_input("Storage temperature (°C)", -10.0, 40.0, 24.0, step=0.5)
        weather = c3.selectbox("Weather", sorted(config.ALLOWED_VALUES["Weather"]), index=1)
        promo = c3.radio("Promotion today?", ["No", "Yes"], horizontal=True)
        submitted = st.form_submit_button("Predict waste level", type="primary")

    if submitted:
        record = {"Date": day.isoformat(), "Business_Type": business, "Food_Category": category,
                  "Purchase_Quantity": int(purchase), "Shelf_Life_Days": float(shelf),
                  "Storage_Temperature": float(temp), "Daily_Demand": float(demand),
                  "Promotion": promo, "Weather": weather}
        result, source = get_prediction(record)
        level = result["waste_level"]
        st.subheader(f"{COLORS[level]} Predicted waste level: **{level}**")
        st.write(ADVICE[level])
        st.bar_chart(pd.Series(result["probabilities"]).reindex(config.CLASSES), horizontal=True)
        st.caption(f"Served by: {source} · model version {result['model_version']}")

with model_tab:
    m = load_metrics()
    if not m:
        st.warning("No trained model yet. Run: python -m foodwaste.train")
    else:
        st.write(f"**Best model:** `{m['best_model']}` · version `{m['model_version']}` · "
                 f"trained on {m['rows']:,} rows · quality gate: macro F1 ≥ {m['quality_gate_min_macro_f1']}")
        cols = st.columns(len(m["candidates"]))
        for col, (name, s) in zip(cols, m["candidates"].items()):
            col.metric(name.replace("_", " ").title(), f"F1 {s['macro_f1']:.3f}", f"acc {s['accuracy']:.1%}",
                       delta_color="off")
        cm = config.MODELS_DIR / "confusion_matrix.png"
        if cm.exists():
            st.image(str(cm), width=450)

with monitor_tab:
    st.write("Compare a batch of new data with the training data to check for **data drift**.")
    scenario = st.radio("Batch to check", ["normal", "heatwave"], horizontal=True,
                        captions=["Fresh sample, same conditions", "70% hot days, storage ~6 °C warmer"])
    if st.button("Run drift check"):
        reference = pd.read_csv(config.REFERENCE_PATH)
        result = detect_drift(reference, simulate(scenario))
        n = int(result["drifted"].sum())
        (st.error if n else st.success)(f"Drift detected in {n} feature(s)" if n else "No drift detected")
        st.dataframe(result.style.apply(
            lambda r: ["background-color: #f8d7da" if r["drifted"] else ""] * len(r), axis=1),
            hide_index=True, width="stretch")
