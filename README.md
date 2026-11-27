# Real-Time Credit Card Fraud Scoring Service
### Amex-style CFR Demo · XGBoost · FastAPI · SQLite Audit Log

`Python 3.12` | `FastAPI 0.115` | `XGBoost 2.1` (model) | `scikit-learn 1.5` (train/val/test split + metrics) | `PyYAML` (hot-reloaded rules config)

**Model:** an XGBoost gradient-boosted classifier (`xgboost.XGBClassifier`). scikit-learn is used only
for the stratified data split and evaluation metrics (ROC-AUC, PR-AUC, F2) — not as the model.

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

> **Using the tuned threshold:** the F2-optimised threshold is stored in the model artifact
> (`threshold` key). Setting `challenge_prob: model_threshold` in `config/rules.yaml` makes the
> HIGH_SCORE (CHALLENGE) rule use it live — no restart needed. Because the CHALLENGE cut-off must not
> exceed the DECLINE cut-off, you must also raise `high_score_amount_prob` above the tuned value
> (e.g. to `0.97`); see [Tunable thresholds](#tunable-thresholds--no-restart-no-redeploy). The shipped default stays at `0.7`
> because the current tuned value (0.95) sits at the upper edge of the threshold search range, which
> suggests the range should be widened and the threshold re-tuned before relying on it.

---

## Business Rules

| Rule | Condition (default values) | Config keys | Decision |
|------|-----------|-------------|----------|
| HIGH_SCORE_HIGH_AMOUNT | prob >= 0.9 AND amount > $2,000 | `high_score_amount_prob`, `high_score_amount_threshold` | DECLINE |
| HIGH_SCORE | prob >= 0.7 | `challenge_prob` (number or `model_threshold`) | CHALLENGE |
| EXCESSIVE_AMOUNT | amount > $5,000 AND prob > 0.5 | `excessive_amount`, `excessive_amount_prob` | DECLINE |
| SUSPICIOUS_HOUR | prob > 0.4 AND hour in 0–5 | `suspicious_hour_prob`, `suspicious_hours` | CHALLENGE |
| Default | — | — | APPROVE |

Rules only escalate decisions, never de-escalate. The rules engine (`app/rules/engine.py`) is a
pure function with no hard-coded thresholds: every cut-off comes from `config/rules.yaml`.

### Tunable thresholds — no restart, no redeploy

- **File:** `config/rules.yaml` (override the path with the `RULES_CONFIG_PATH` env var).
- **Hot reload:** on every request the service reads the (tiny) file and compares a SHA-256 content
  hash with the active version, so any change — even one that keeps the same mtime and size — is
  re-validated and applied to the next request after the new file is in place. No restart or
  redeploy is needed.
- **Complete files only:** every threshold key must be present. An empty or partially written file
  is rejected (it does **not** fall back to defaults). Make edits **atomic** — write a temp file in
  the same directory, then rename it over `rules.yaml` (`mv` / `os.replace`) — so the service never
  sees a half-written file. (Saving in an editor also works; a partial save is simply rejected and
  picked up once the complete file lands.)
- **Validation** (`app/rules/config.py`, frozen pydantic model): probabilities in `[0, 1]`, amounts
  `>= 0`, hours are unique integers `0–23`, unknown keys rejected, plus cross-field checks
  (`challenge_prob <= high_score_amount_prob`, `high_score_amount_threshold <= excessive_amount`).
- **Fail-safe:** if an edit is invalid or the file becomes unreadable, the error is logged once and
  the **last good config stays active**. Only an invalid config at startup prevents the service
  from starting.
- **`challenge_prob: model_threshold`** uses the F2-tuned threshold from the loaded model artifact
  instead of a fixed number. It is resolved and validated when the config is loaded, including the
  `challenge_prob <= high_score_amount_prob` check. The shipped artifact threshold is ~0.95, so with
  the default `high_score_amount_prob: 0.9` this is **rejected** — raise `high_score_amount_prob`
  too (note this also raises the HIGH_SCORE_HIGH_AMOUNT DECLINE cut-off):

  ```yaml
  high_score_amount_prob: 0.97
  challenge_prob: model_threshold
  ```
- **Observability:** `GET /health` shows the active thresholds, a content hash
  (`rules_config_version`), when they were loaded, and the effective CHALLENGE cut-off, so you can
  confirm a reload took effect. Every audit-log row also records the `rules_config_version` and
  `effective_challenge_prob` that produced the decision.

```bash
# e.g. tighten the CHALLENGE cut-off while the API is running (portable, atomic replace)
python - <<'PY'
import os, pathlib, re
path = pathlib.Path("config/rules.yaml")
tmp = path.with_suffix(".yaml.tmp")
tmp.write_text(re.sub(r"(?m)^challenge_prob: .*$", "challenge_prob: 0.6", path.read_text()))
os.replace(tmp, path)  # atomic rename: the service never sees a partial file
PY
curl -s localhost:8000/health   # rules_config.challenge_prob == 0.6, new rules_config_version
```

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
{
  "status": "ok",
  "model_version": "xgb_v1",
  "uptime_seconds": 12.3,
  "rules_config": {
    "high_score_amount_prob": 0.9, "high_score_amount_threshold": 2000.0,
    "challenge_prob": 0.7, "excessive_amount": 5000.0, "excessive_amount_prob": 0.5,
    "suspicious_hour_prob": 0.4, "suspicious_hours": [0, 1, 2, 3, 4, 5]
  },
  "rules_config_version": "3f1c2a9b7d10",
  "rules_config_loaded_at": "2026-11-27T12:00:00.000000+00:00",
  "effective_challenge_prob": 0.7
}
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
│   │   ├── config.py        # Validated, hot-reloaded threshold config
│   │   └── engine.py        # Business rules engine (pure function)
│   └── audit/
│       └── logger.py        # SQLite decision audit log
├── config/
│   └── rules.yaml           # Rule thresholds (hot-reloaded, no restart)
├── models/                  # model.joblib (git-ignored; produced by notebooks/train.py)
├── analysis/
│   └── decision_metrics.py  # Offline batch scoring + plots
├── diagrams/                # All saved PNGs
├── tests/
│   ├── conftest.py          # Shared fixtures / payload builder
│   ├── test_rules_engine.py # Rule decision boundaries
│   ├── test_rules_config.py # Config validation + hot reload
│   ├── test_audit_logger.py # Audit columns + schema migration (temp SQLite)
│   ├── test_api_config_reload.py # Decision changes after live config edit
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
- `rules_config_version` — content hash of the `config/rules.yaml` used for this decision
- `effective_challenge_prob` — the CHALLENGE cut-off actually applied (resolves `model_threshold`)

Both rules-config columns come from the same config snapshot that made the decision. They are
nullable; an existing `decisions.db` is migrated in place at startup (`ALTER TABLE ... ADD COLUMN`,
idempotent). Override the DB location with the `FRAUD_AUDIT_DB_URL` env var (SQLAlchemy URL;
default `sqlite:///<project>/decisions.db`).

Indexed on `transaction_id`, `decision`, and `timestamp` for efficient audit queries.

---

## Why This Is Relevant for Credit & Fraud Risk at Amex

- **Real-time authorization pipeline:** mirrors the shape of real-time authorization decisioning — a single synchronous request runs model inference, rule evaluation, and audit logging (latency has not been formally benchmarked)
- **ML + rules combination:** production CFR systems combine statistical models with policy rules; this project implements both layers with clean separation of concerns (`app/model_loader.py` → `app/rules/engine.py`)
- **Risk vs. customer experience tradeoffs:** escalation-only rules and an offline F2-optimised (recall-weighted) threshold reflect the real tension between catching fraud and minimising false declines. All rule cut-offs are tunable at runtime via `config/rules.yaml`, and the tuned threshold can be switched on with `challenge_prob: model_threshold`
- **Data governance:** every decision is logged with full feature vectors, model version, and the rules-config version that produced it, enabling audit trails, model monitoring, and regulatory compliance — a core requirement in financial services

---

## Dataset

Credit Card Fraud Detection dataset — ULB Machine Learning Group  
284,807 transactions · 492 fraud · 0.17% fraud rate · 28 PCA features  
Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
