"""
Data drift check: compare new data against the data the model was trained on.

Usage:
  python -m foodwaste.drift --simulate normal      # a fresh sample of normal data  -> expect no drift
  python -m foodwaste.drift --simulate heatwave    # hot weather + warmer storage   -> expect drift
  python -m foodwaste.drift --from-logs            # requests the API has received
  python -m foodwaste.drift --current new.csv      # any CSV with the raw feature columns
Add --fail-on-drift to exit with code 1 when drift is found (for schedulers / CI).
"""
import argparse
import base64
import io
import json
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

from foodwaste import config
from foodwaste.data import load_data

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

MIN_ROWS = 100


def numeric_drift(ref, cur):
    ref, cur = ref.dropna(), cur.dropna()
    ks, p = stats.ks_2samp(ref, cur)
    drifted = p < config.DRIFT_P_VALUE and ks >= config.DRIFT_MIN_KS
    return {"test": "KS", "statistic": round(float(ks), 4), "p_value": float(p), "drifted": bool(drifted),
            "reference": f"mean {ref.mean():.2f}", "current": f"mean {cur.mean():.2f}"}


def categorical_drift(ref, cur):
    ref, cur = ref.fillna("missing"), cur.fillna("missing")
    cats = sorted(set(ref) | set(cur))
    ref_counts = ref.value_counts().reindex(cats, fill_value=0)
    cur_counts = cur.value_counts().reindex(cats, fill_value=0)
    _, p, _, _ = stats.chi2_contingency(np.array([ref_counts, cur_counts]))
    tvd = 0.5 * (ref_counts / ref_counts.sum() - cur_counts / cur_counts.sum()).abs().sum()
    drifted = p < config.DRIFT_P_VALUE and tvd >= config.DRIFT_MIN_TVD
    top = lambda s: f"top {s.value_counts(normalize=True).idxmax()} ({s.value_counts(normalize=True).max():.0%})"
    return {"test": "Chi-square", "statistic": round(float(tvd), 4), "p_value": float(p), "drifted": bool(drifted),
            "reference": top(ref), "current": top(cur)}


def detect_drift(reference, current):
    """Returns one row per feature with the test used, statistic, p-value and whether it drifted."""
    rows = []
    for col in config.NUMERIC_FEATURES:
        rows.append({"feature": col, **numeric_drift(reference[col], current[col])})
    for col in config.CATEGORICAL_FEATURES:
        rows.append({"feature": col, **categorical_drift(reference[col], current[col])})
    return pd.DataFrame(rows)


def simulate(kind, n=1000, seed=0):
    """Make a batch of 'new' data. heatwave = most days hot, storage running warmer."""
    rng = np.random.default_rng(seed)
    batch = load_data().sample(n, random_state=seed).reset_index(drop=True)
    if kind == "heatwave":
        hot = rng.random(n) < 0.7
        batch.loc[hot, "Weather"] = "Hot"
        batch["Storage_Temperature"] = batch["Storage_Temperature"] + rng.normal(6, 1.5, n)
    return batch


def load_logged_requests():
    if not config.PREDICTION_LOG.exists():
        raise FileNotFoundError(f"No prediction log at {config.PREDICTION_LOG}. Send some requests to the API first.")
    with open(config.PREDICTION_LOG) as f:
        return pd.DataFrame([json.loads(line)["input"] for line in f])


def _plot(reference, current, col):
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    if col in config.NUMERIC_FEATURES:
        bins = np.histogram_bin_edges(pd.concat([reference[col], current[col]]).dropna(), bins=30)
        ax.hist(reference[col].dropna(), bins=bins, alpha=0.6, density=True, label="training")
        ax.hist(current[col].dropna(), bins=bins, alpha=0.6, density=True, label="current")
    else:
        share = pd.DataFrame({"training": reference[col].value_counts(normalize=True),
                              "current": current[col].value_counts(normalize=True)}).fillna(0)
        share.plot.bar(ax=ax, rot=0, alpha=0.8)
    ax.set_title(col, fontsize=10)
    ax.legend(fontsize=8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=90)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def write_report(result, reference, current, name):
    config.REPORTS_DIR.mkdir(exist_ok=True)
    n_drift = int(result["drifted"].sum())
    status = f"DRIFT DETECTED in {n_drift} feature(s)" if n_drift else "No drift detected"
    table = result.assign(p_value=result["p_value"].map(lambda p: f"{p:.2e}")).to_html(index=False)
    charts = "".join(f'<img src="data:image/png;base64,{_plot(reference, current, c)}">'
                     for c in result.loc[result["drifted"], "feature"]) or "<p>Nothing to show.</p>"
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Drift report - {name}</title>
<style>body{{font-family:sans-serif;margin:24px;max-width:1100px}} table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 8px}} .s{{font-size:20px;color:{'#b00020' if n_drift else '#1b7f3b'}}}</style>
</head><body><h1>Data drift report: {name}</h1>
<p class="s">{status}</p><p>Training rows: {len(reference)} &nbsp; Current rows: {len(current)}</p>
{table}<h2>Drifted features</h2>{charts}</body></html>"""
    html_path = config.REPORTS_DIR / f"drift_{name}.html"
    html_path.write_text(html, encoding="utf-8")
    (config.REPORTS_DIR / f"drift_{name}.json").write_text(
        json.dumps({"status": status, "features": result.to_dict(orient="records")}, indent=2))
    return html_path


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--simulate", choices=["normal", "heatwave"])
    source.add_argument("--from-logs", action="store_true")
    source.add_argument("--current", help="CSV file with new data")
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()

    reference = pd.read_csv(config.REFERENCE_PATH)
    if args.simulate:
        current, name = simulate(args.simulate), args.simulate
    elif args.from_logs:
        current, name = load_logged_requests(), "api_requests"
    else:
        current, name = pd.read_csv(args.current), "custom"

    if len(current) < MIN_ROWS:
        print(f"WARNING: only {len(current)} rows - drift tests are unreliable below {MIN_ROWS}. "
              f"Collect more data before acting on this.\n")
    result = detect_drift(reference, current)
    print(result[["feature", "test", "statistic", "p_value", "drifted", "reference", "current"]].to_string(index=False))
    path = write_report(result, reference, current, name)
    drifted = result["drifted"].any()
    print(f"\n{'DRIFT DETECTED' if drifted else 'No drift detected'} - report: {path.relative_to(config.ROOT)}")
    if drifted and args.fail_on_drift:
        sys.exit(1)


if __name__ == "__main__":
    main()
