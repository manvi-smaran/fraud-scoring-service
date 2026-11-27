"""
Train XGBClassifier for credit card fraud detection.

Usage:  python notebooks/train.py

Outputs
-------
models/model.joblib          — serialised model dict
diagrams/feature_importance.png
diagrams/roc_curve.png
diagrams/confusion_matrix.png
"""

# ── stdlib ─────────────────────────────────────────────────────────────────────
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── third-party ────────────────────────────────────────────────────────────────
import joblib
import matplotlib
import xgboost
matplotlib.use("Agg")           # headless — must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA     = ROOT / "data" / "creditcard.csv"
MODELS   = ROOT / "models"
DIAGRAMS = ROOT / "diagrams"

MODELS.mkdir(parents=True, exist_ok=True)
DIAGRAMS.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS / "model.joblib"

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

RANDOM_STATE = 42


# ── 1. load data ───────────────────────────────────────────────────────────────
def load_data(path: Path) -> pd.DataFrame:
    log.info("Loading data from %s …", path)
    df = pd.read_csv(path)
    log.info("Loaded  %s rows × %s columns", f"{len(df):,}", df.shape[1])
    fraud_rate = df["Class"].mean() * 100
    log.info("Fraud rate: %.4f %%", fraud_rate)
    return df


# ── 2. feature engineering ─────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    log.info("Engineering features …")
    df = df.copy()
    df["log_amount"] = np.log1p(df["Amount"])
    df["Hour"] = (df["Time"] / 3600) % 24
    df = df.drop(columns=["Time", "Amount"])

    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols]
    y = df["Class"]

    log.info("Feature matrix: %s columns", len(feature_cols))
    return X, y, feature_cols


# ── 3. train/test split ────────────────────────────────────────────────────────
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    log.info("Stratified 80/20 train-test split (random_state=%d) …", RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    log.info(
        "Train: %s rows  |  Test: %s rows",
        f"{len(X_train):,}",
        f"{len(X_test):,}",
    )
    return X_train, X_test, y_train, y_test


# ── 4. build early-stopping eval set from training data ───────────────────────
def make_eval_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Reserve 20 % of the training set for early stopping."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )
    return X_tr, X_val, y_tr, y_val


# ── 5. train model ─────────────────────────────────────────────────────────────
def train_model(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
) -> XGBClassifier:
    log.info("Training XGBClassifier (scale_pos_weight=%.2f) …", scale_pos_weight)
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=20,       # xgboost 2.x: set in constructor
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    log.info("Training complete. Best iteration: %d", xgb.best_iteration)
    return xgb


# ── 6. threshold tuning ────────────────────────────────────────────────────────
def tune_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    beta: float = 2.0,
) -> float:
    """Return the threshold in [0.05, 0.95] that maximises F-beta on the validation set."""
    log.info("Tuning decision threshold (beta=%.1f) …", beta)
    thresholds = np.arange(0.05, 1.00, 0.05)
    best_threshold = 0.5
    best_f2 = -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f2 = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_threshold = float(t)

    log.info("Best threshold: %.2f  (F%.0f = %.4f)", best_threshold, beta, best_f2)
    return best_threshold


# ── 7. evaluate ────────────────────────────────────────────────────────────────
def evaluate(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    """Print metrics and return (roc_auc, pr_auc)."""
    roc_auc = roc_auc_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    y_pred = (y_prob >= threshold).astype(int)

    log.info("=" * 60)
    log.info("ROC-AUC : %.4f", roc_auc)
    log.info("PR-AUC  : %.4f", pr_auc)
    log.info("Threshold: %.2f", threshold)
    log.info("Confusion matrix:\n%s", confusion_matrix(y_true, y_pred))
    log.info(
        "Classification report:\n%s",
        classification_report(y_true, y_pred, target_names=["Legit", "Fraud"]),
    )
    log.info("=" * 60)
    return roc_auc, pr_auc


# ── 8. plots ───────────────────────────────────────────────────────────────────
def _save_feature_importance(
    xgb: XGBClassifier,
    feature_cols: list[str],
    out_path: Path,
    top_n: int = 20,
) -> None:
    importances = xgb.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    names = [feature_cols[i] for i in indices]
    values = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(names, values, color="steelblue")
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(f"Top {top_n} XGBoost Feature Importances")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out_path)


def _save_roc_curve(
    y_true: pd.Series,
    y_prob: np.ndarray,
    roc_auc: float,
    out_path: Path,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Credit Card Fraud Detection")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out_path)


def _save_confusion_matrix(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    out_path: Path,
) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Legit", "Fraud"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (threshold = {threshold:.2f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out_path)


def save_plots(
    xgb: XGBClassifier,
    feature_cols: list[str],
    y_test: pd.Series,
    y_prob: np.ndarray,
    roc_auc: float,
    threshold: float,
) -> None:
    log.info("Saving diagnostic plots …")
    _save_feature_importance(xgb, feature_cols, DIAGRAMS / "feature_importance.png")
    _save_roc_curve(y_test, y_prob, roc_auc, DIAGRAMS / "roc_curve.png")
    _save_confusion_matrix(y_test, y_prob, threshold, DIAGRAMS / "confusion_matrix.png")


# ── 9. persist ─────────────────────────────────────────────────────────────────
def save_model(
    xgb: XGBClassifier,
    threshold: float,
    feature_cols: list[str],
    n_train_rows: int,
) -> None:
    payload = {
        "model": xgb,
        "threshold": threshold,
        "feature_names": feature_cols,
        "version": "xgb_v1",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "xgb_version": xgboost.__version__,
        "n_train_rows": n_train_rows,
    }
    joblib.dump(payload, MODEL_PATH)
    log.info("Model saved to %s", MODEL_PATH)


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    log.info("=== Credit Card Fraud Detection — Training Pipeline ===")

    # 1. load
    df = load_data(DATA)

    # 2. feature engineering
    X, y, feature_cols = engineer_features(df)

    # 3. train/test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 4. eval split for early stopping / threshold tuning (carved from training data only)
    X_tr, X_val, y_tr, y_val = make_eval_split(X_train, y_train)

    # 5. scale_pos_weight computed from X_tr/y_tr only (Fix 2)
    n_legit = int((y_tr == 0).sum())
    n_fraud = int((y_tr == 1).sum())
    scale_pos_weight = n_legit / n_fraud
    log.info("n_legit=%s  n_fraud=%s  scale_pos_weight=%.2f",
             f"{n_legit:,}", f"{n_fraud:,}", scale_pos_weight)

    # 6. train
    xgb = train_model(X_tr, y_tr, X_val, y_val, scale_pos_weight)

    # 7. threshold tuning on validation set — test set not touched here (Fix 1)
    y_prob_val = xgb.predict_proba(X_val)[:, 1]
    best_threshold = tune_threshold(y_val, y_prob_val, beta=2.0)

    # 8. final evaluation on held-out test set (touched only once)
    log.info("Generating test-set predictions …")
    y_prob = xgb.predict_proba(X_test)[:, 1]
    roc_auc, pr_auc = evaluate(y_test, y_prob, best_threshold)

    # 9. plots
    save_plots(xgb, feature_cols, y_test, y_prob, roc_auc, best_threshold)

    # 10. save model
    save_model(xgb, best_threshold, feature_cols, n_train_rows=int(len(X_tr)))

    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
