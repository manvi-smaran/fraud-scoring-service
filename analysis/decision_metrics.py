"""Offline batch scoring script — generates decision diagnostics and 3 plots.

Run from any directory:
    python analysis/decision_metrics.py
"""
import sys
import logging
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Matplotlib must use non-interactive backend BEFORE any pyplot import.
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)

import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Project root on sys.path so that `app.*` imports resolve correctly.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rules.engine import apply_rules, RuleResult  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CSV_PATH = ROOT / "data" / "creditcard.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
DIAGRAMS_DIR = ROOT / "diagrams"

# Sampling constants
SAMPLE_FRAUD = 492       # take all fraud rows
SAMPLE_LEGIT = 4508      # complement to reach ~5 000 total
RANDOM_STATE = 42

DECISIONS = ("APPROVE", "CHALLENGE", "DECLINE")


# ---------------------------------------------------------------------------
# Step 1: Load + stratified sample
# ---------------------------------------------------------------------------
def load_sample() -> pd.DataFrame:
    """Load creditcard.csv and return a stratified 5 000-row sample."""
    log.info("Loading dataset from %s", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    log.info("Full dataset: %d rows (fraud=%d, legit=%d)",
             len(df), (df["Class"] == 1).sum(), (df["Class"] == 0).sum())

    fraud = df[df["Class"] == 1].sample(
        n=min(SAMPLE_FRAUD, (df["Class"] == 1).sum()), random_state=RANDOM_STATE
    )
    legit = df[df["Class"] == 0].sample(n=SAMPLE_LEGIT, random_state=RANDOM_STATE)

    sample = pd.concat([fraud, legit]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    log.info(
        "Sampled %d rows (fraud=%d, legit=%d)",
        len(sample), (sample["Class"] == 1).sum(), (sample["Class"] == 0).sum(),
    )
    return sample


# ---------------------------------------------------------------------------
# Step 2: Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (feature_matrix, raw_amount_series, raw_hour_series)."""
    raw_amount: pd.Series = df["Amount"].copy()
    raw_hour: pd.Series = ((df["Time"] / 3600) % 24).copy()

    df = df.copy()
    df["log_amount"] = np.log1p(df["Amount"])
    df["Hour"] = (df["Time"] / 3600) % 24
    df = df.drop(columns=["Time", "Amount"])

    return df, raw_amount, raw_hour


# ---------------------------------------------------------------------------
# Step 3: Load model artifact
# ---------------------------------------------------------------------------
def load_model(path: Path) -> dict:
    log.info("Loading model from %s", path)
    artifact: dict = joblib.load(path)
    log.info(
        "Model loaded — version=%s  threshold=%.4f  features=%d",
        artifact.get("version", "unknown"),
        artifact.get("threshold", float("nan")),
        len(artifact.get("feature_names", [])),
    )
    return artifact


# ---------------------------------------------------------------------------
# Step 4: Inference
# ---------------------------------------------------------------------------
def predict_proba(artifact: dict, feature_df: pd.DataFrame) -> np.ndarray:
    """Return 1-D array of fraud probabilities."""
    model = artifact["model"]
    feature_names: list[str] = artifact["feature_names"]

    # Align columns to the exact feature order the model was trained on.
    X = feature_df[feature_names]
    log.info("Running predict_proba on %d rows with %d features", len(X), len(feature_names))
    proba: np.ndarray = model.predict_proba(X)[:, 1]
    return proba


# ---------------------------------------------------------------------------
# Step 5: Apply rules engine
# ---------------------------------------------------------------------------
def apply_rules_batch(
    proba: np.ndarray,
    raw_amount: pd.Series,
    raw_hour: pd.Series,
) -> list[RuleResult]:
    log.info("Applying rules engine to %d rows", len(proba))
    results: list[RuleResult] = [
        apply_rules(float(prob), float(amount), float(hour))
        for prob, amount, hour in zip(proba, raw_amount, raw_hour)
    ]
    return results


# ---------------------------------------------------------------------------
# Step 6: Build results DataFrame
# ---------------------------------------------------------------------------
def build_results_df(
    sample: pd.DataFrame,
    proba: np.ndarray,
    rule_results: list[RuleResult],
    raw_hour: pd.Series,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Class": sample["Class"].values,
            "fraud_probability": proba,
            "decision": [r.decision for r in rule_results],
            "triggered_rules": [",".join(r.triggered_rules) for r in rule_results],
            "Hour": raw_hour.values,
        }
    )


# ---------------------------------------------------------------------------
# Step 7: Summary statistics
# ---------------------------------------------------------------------------
def print_summary(results: pd.DataFrame) -> None:
    total = len(results)
    fraud_mask = results["Class"] == 1
    legit_mask = results["Class"] == 0
    fraud_count = fraud_mask.sum()
    legit_count = legit_mask.sum()

    log.info("=" * 60)
    log.info("BATCH SCORING SUMMARY")
    log.info("=" * 60)
    log.info("Total scored : %d", total)
    log.info("Fraud (Class=1): %d", fraud_count)
    log.info("Legit (Class=0): %d", legit_count)
    log.info("-" * 60)
    log.info("Decision distribution:")
    for decision in DECISIONS:
        n = (results["decision"] == decision).sum()
        pct = 100.0 * n / total
        log.info("  %-10s %5d  (%5.1f%%)", decision, n, pct)
    log.info("-" * 60)

    # Fraud outcome breakdown — all three decision buckets
    fraud_decline = (results.loc[fraud_mask, "decision"] == "DECLINE").sum()
    fraud_challenge = (results.loc[fraud_mask, "decision"] == "CHALLENGE").sum()
    fraud_approve = (results.loc[fraud_mask, "decision"] == "APPROVE").sum()

    fraud_decline_rate = 100.0 * fraud_decline / fraud_count if fraud_count else 0.0
    fraud_challenge_rate = 100.0 * fraud_challenge / fraud_count if fraud_count else 0.0
    fraud_approve_rate = 100.0 * fraud_approve / fraud_count if fraud_count else 0.0
    fraud_flagged_rate = fraud_decline_rate + fraud_challenge_rate  # key metric

    log.info("Fraud DECLINE rate   (fraud blocked)     : %5.1f%%", fraud_decline_rate)
    log.info("Fraud CHALLENGE rate (fraud flagged)      : %5.1f%%", fraud_challenge_rate)
    log.info("Fraud APPROVE rate   (fraud missed)       : %5.1f%%", fraud_approve_rate)
    log.info("Fraud flagged rate   (DECLINE+CHALLENGE)  : %5.1f%%", fraud_flagged_rate)
    log.info("-" * 60)

    # False positive rate: % of legit rows that received DECLINE
    legit_decline = (results.loc[legit_mask, "decision"] == "DECLINE").sum()
    fp_rate = 100.0 * legit_decline / legit_count if legit_count else 0.0
    log.info("False positive rate  (legit DECLINE / total legit) : %5.1f%%", fp_rate)

    # False challenge rate: % of legit rows that received CHALLENGE
    legit_challenge = (results.loc[legit_mask, "decision"] == "CHALLENGE").sum()
    fc_rate = 100.0 * legit_challenge / legit_count if legit_count else 0.0
    log.info("False challenge rate (legit CHALLENGE / total legit): %5.1f%%", fc_rate)
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Step 8a: Plot 1 — Decision distribution stacked bar
# ---------------------------------------------------------------------------
def plot_decision_distribution(results: pd.DataFrame, out_dir: Path) -> None:
    log.info("Generating plot: decision_distribution.png")

    fraud_counts = {
        d: ((results["decision"] == d) & (results["Class"] == 1)).sum()
        for d in DECISIONS
    }
    legit_counts = {
        d: ((results["decision"] == d) & (results["Class"] == 0)).sum()
        for d in DECISIONS
    }

    x = np.arange(len(DECISIONS))
    width = 0.5

    fig, ax = plt.subplots(figsize=(8, 5))

    legit_bars = ax.bar(x, [legit_counts[d] for d in DECISIONS], width, label="Legit", color="#4878CF")
    fraud_bars = ax.bar(
        x,
        [fraud_counts[d] for d in DECISIONS],
        width,
        bottom=[legit_counts[d] for d in DECISIONS],
        label="Fraud",
        color="#D65F5F",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(DECISIONS, fontsize=12)
    ax.set_ylabel("Transaction Count", fontsize=11)
    ax.set_title("Decision Distribution by True Class", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path = out_dir / "decision_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Step 8b: Plot 2 — DECLINE rate by hour
# ---------------------------------------------------------------------------
def plot_approval_by_hour(results: pd.DataFrame, out_dir: Path) -> None:
    log.info("Generating plot: approval_by_hour.png")

    hours = list(range(24))

    def decline_rate_by_hour(mask: pd.Series) -> list[float]:
        rates: list[float] = []
        subset = results[mask].copy()
        for h in hours:
            hour_rows = subset[subset["Hour"].astype(int) == h]
            if len(hour_rows) == 0:
                rates.append(float("nan"))
            else:
                rate = 100.0 * (hour_rows["decision"] == "DECLINE").sum() / len(hour_rows)
                rates.append(rate)
        return rates

    fraud_rates = decline_rate_by_hour(results["Class"] == 1)
    legit_rates = decline_rate_by_hour(results["Class"] == 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hours, fraud_rates, marker="o", color="#D65F5F", linewidth=2, label="Fraud")
    ax.plot(hours, legit_rates, marker="s", color="#4878CF", linewidth=2, label="Legit")
    ax.set_xlabel("Hour of Day (0–23)", fontsize=11)
    ax.set_ylabel("DECLINE Rate (%)", fontsize=11)
    ax.set_title("DECLINE Rate by Hour of Day", fontsize=13, fontweight="bold")
    ax.set_xticks(hours)
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path = out_dir / "approval_by_hour.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Step 8c: Plot 3 — Fraud transaction outcomes horizontal bar
# ---------------------------------------------------------------------------
def plot_fraud_caught(results: pd.DataFrame, out_dir: Path) -> None:
    log.info("Generating plot: fraud_caught.png")

    fraud_rows = results[results["Class"] == 1]
    total_fraud = len(fraud_rows)

    caught = (fraud_rows["decision"] == "DECLINE").sum()
    flagged = (fraud_rows["decision"] == "CHALLENGE").sum()
    missed = (fraud_rows["decision"] == "APPROVE").sum()

    labels = ["Fraud DECLINE\n(Caught)", "Fraud CHALLENGE\n(Flagged)", "Fraud APPROVE\n(Missed)"]
    values = [caught, flagged, missed]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.5)

    for bar, val in zip(bars, values):
        pct = 100.0 * val / total_fraud if total_fraud else 0.0
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val}  ({pct:.1f}%)",
            va="center",
            fontsize=10,
        )

    ax.set_xlabel("Transaction Count", fontsize=11)
    ax.set_title(
        "Fraud Transaction Outcomes\n(DECLINE=blocked, CHALLENGE=flagged via step-up auth, APPROVE=missed)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlim(0, max(values) * 1.25)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.invert_yaxis()

    fig.tight_layout()
    out_path = out_dir / "fraud_caught.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("decision_metrics.py started at %s", datetime.now().isoformat(timespec="seconds"))

    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load + sample
    sample = load_sample()

    # 2. Feature engineering — preserve raw Amount and Hour for rules engine
    feature_df, raw_amount, raw_hour = engineer_features(sample)

    # 3. Load model
    artifact = load_model(MODEL_PATH)

    # 4. Inference
    proba = predict_proba(artifact, feature_df)

    # 5. Rules engine
    rule_results = apply_rules_batch(proba, raw_amount, raw_hour)

    # 6. Results DataFrame
    results = build_results_df(sample, proba, rule_results, raw_hour)

    # 7. Summary statistics
    print_summary(results)

    # 8. Plots
    plot_decision_distribution(results, DIAGRAMS_DIR)
    plot_approval_by_hour(results, DIAGRAMS_DIR)
    plot_fraud_caught(results, DIAGRAMS_DIR)

    log.info("All 3 diagrams saved.")


if __name__ == "__main__":
    main()
