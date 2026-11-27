# Real-Time Credit Card Fraud Scoring Service
### Amex-style CFR Demo · XGBoost · FastAPI · SQLite Audit Log

`Python 3.12` | `FastAPI 0.115` | `XGBoost 2.1`

---

## Overview

Credit card fraud costs tens of billions of dollars globally each year. Card networks and issuers like American Express must detect fraudulent transactions in real time — within milliseconds of each authorization. This project is a simplified end-to-end implementation of such a system: a trained ML model scores each transaction, a business rules engine maps the score to an action (APPROVE / CHALLENGE / DECLINE), and every decision is persisted to an audit log for governance and analysis.

---

## Architecture

```
creditcard.csv
      │
      ▼
┌─────────────────┐
│  Offline Train  │  notebooks/train.py
│  XGBoost model  │  → models/model.joblib
└────────┬────────┘
         │ loaded at startup
         ▼
┌─────────────────────────────────┐
│         FastAPI Service         │
│  POST /score_transaction        │
│  ┌──────────────┐               │
│  │ Model Scorer │ → prob        │
│  └──────┬───────┘               │
│         ▼                       │
│  ┌──────────────┐               │
│  │ Rules Engine │ → decision    │
│  └──────┬───────┘               │
│         ▼                       │
│  ┌──────────────┐               │
│  │  Audit Log   │ → decisions.db│
│  └──────────────┘               │
└─────────────────────────────────┘
```

---

## Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9710 |
| PR-AUC | 0.8412 |
| F2-optimised threshold (offline) | 0.95 |
| Fraud recall | 78.6% |
| Fraud precision | 88.5% |
| False positives (per 56K test txns) | 10 |

> Threshold tuned on a held-out validation set; test set touched exactly once.
> The recall/precision figures above are measured at this F2 threshold.

> **Known limitation:** the F2-optimised threshold is computed during training and stored in the
> model artifact (`threshold` key), but the live API does **not** use it. API decisions come from the
> fixed probability cut-offs in `app/rules/engine.py` (see Business Rules below). Wiring the tuned
> threshold into the rules engine is a planned next step.

---

## Business Rules

| Rule | Condition | Decision |
|------|-----------|----------|
| HIGH_SCORE_HIGH_AMOUNT | prob >= 0.9 AND amount > $2,000 | DECLINE |
| HIGH_SCORE | prob >= 0.7 | CHALLENGE |
| EXCESSIVE_AMOUNT | amount > $5,000 AND prob > 0.5 | DECLINE |
| SUSPICIOUS_HOUR | prob > 0.4 AND hour in 0–5am | CHALLENGE |
| Default | — | APPROVE |

Rules only escalate decisions, never de-escalate. Cut-offs are hard-coded constants in
`app/rules/engine.py`; they are independent of the F2 threshold stored in the model artifact.

**Note on `hour`:** the dataset's `Time` field is *seconds elapsed since the first transaction in the
dataset*, not a wall-clock timestamp. `Hour` is derived as `(Time / 3600) % 24`, so the
SUSPICIOUS_HOUR rule operates on this relative hour, not the cardholder's local time.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place dataset
# Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# and place it in data/creditcard.csv

# 3. Train the model (required — models/model.joblib is git-ignored and not shipped)
python notebooks/train.py

# 4. Run the tests (API tests are skipped if models/model.joblib is absent)
python -m pytest -q

# 5. Start the API
uvicorn app.main:app --reload
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

> This is a local demo: there is **no authentication and no rate limiting** on the API.

---

## API Reference

**GET /health**

```json
{"status": "ok", "model_version": "xgb_v1", "uptime_seconds": 12.3}
```

**POST /score_transaction**

Request body — all of `V1`–`V28`, `Amount`, and `Time` are required; unknown fields, NaN/Infinity,
and malformed `transaction_id`s (must match `^[A-Za-z0-9_-]+$`, 1–64 chars) are rejected with 422.
`Amount` must be within `[0, 10,000,000]`; `Time` must be `>= 0`.

```json
{
  "transaction_id": "TXN001",
  "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782, "V5": -0.3383,
  "V6": 0.4624, "V7": 0.2396, "V8": 0.0987, "V9": 0.3638, "V10": 0.0908,
  "V11": -0.5516, "V12": -0.6178, "V13": -0.9914, "V14": -0.3112, "V15": 1.4682,
  "V16": -0.4704, "V17": 0.2080, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
  "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669, "V25": 0.1285,
  "V26": -0.1891, "V27": 0.1336, "V28": -0.0211,
  "Amount": 149.62,
  "Time": 43200.0
}
```

Response (actual output from the trained `xgb_v1` model; score < 0.4 and daytime hour, so no rule fires):

```json
{
  "transaction_id": "TXN001",
  "score": 0.0034,
  "fraud_probability": 0.0034,
  "decision": "APPROVE",
  "reasons": [],
  "triggered_rules": [],
  "model_version": "xgb_v1",
  "scored_at": "2026-11-27T12:45:21.157383+00:00"
}
```

---

## Project Structure

```
fraud-scoring-service/
├── data/                    # creditcard.csv (git-ignored)
├── notebooks/
│   ├── eda.py               # EDA + 4 diagnostic plots
│   └── train.py             # XGBoost training pipeline
├── app/
│   ├── main.py              # FastAPI app + lifespan
│   ├── schemas.py           # Pydantic v2 request/response
│   ├── model_loader.py      # Artifact loading + inference
│   ├── rules/
│   │   └── engine.py        # Business rules engine
│   └── audit/
│       └── logger.py        # SQLite decision audit log
├── models/                  # model.joblib (git-ignored; produced by notebooks/train.py)
├── analysis/
│   └── decision_metrics.py  # Offline batch scoring + plots
├── diagrams/                # All saved PNGs
├── tests/
│   ├── conftest.py          # Shared fixtures / payload builder
│   ├── test_rules_engine.py # Rule decision boundaries
│   ├── test_schemas.py      # Request validation
│   └── test_api.py          # TestClient API tests (audit log stubbed)
├── requirements.txt
└── README.md
```

---

## Data Governance — Audit Log

Every scored transaction is persisted to `decisions.db` (SQLite, git-ignored) with:

- `transaction_id`, `timestamp` (UTC), `fraud_probability`, `decision`
- `triggered_rules_json` — which rules fired
- `features_json` — full feature vector for reproducibility
- `model_version` — artifact version for model lineage

Indexed on `transaction_id`, `decision`, and `timestamp` for efficient audit queries.

---

## Why This Is Relevant for Credit & Fraud Risk at Amex

- **Real-time authorization pipeline:** mirrors the shape of real-time authorization decisioning — a single synchronous request runs model inference, rule evaluation, and audit logging (latency has not been formally benchmarked)
- **ML + rules combination:** production CFR systems combine statistical models with policy rules; this project implements both layers with clean separation of concerns (`app/model_loader.py` → `app/rules/engine.py`)
- **Risk vs. customer experience tradeoffs:** escalation-only rules and an offline F2-optimised (recall-weighted) threshold reflect the real tension between catching fraud and minimising false declines. Currently the API uses fixed rule cut-offs; feeding the tuned threshold into the rules engine is a next step
- **Data governance:** every decision is logged with full feature vectors and model version, enabling audit trails, model monitoring, and regulatory compliance — a core requirement in financial services

---

## Dataset

Credit Card Fraud Detection dataset — ULB Machine Learning Group  
284,807 transactions · 492 fraud · 0.17% fraud rate · 28 PCA features  
Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
