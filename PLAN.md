# Fraud Scoring Service — Implementation Plan

**Target:** American Express CFR Product Development Internship portfolio project  
**Stack:** Python 3.11 · FastAPI · XGBoost · SQLAlchemy/SQLite · Pydantic v2

---

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Repo structure + data + EDA | ✅ DONE |
| 2 | Model training | ⬜ TODO |
| 3 | FastAPI scoring API | ⬜ TODO |
| 4 | Business rules engine + audit log | ⬜ TODO |
| 5 | Decision analysis + README | ⬜ TODO |

---

## Phase 2 — Model Training

### Files
- `notebooks/train.py`
- `models/xgb_fraud.pkl`
- `diagrams/feature_importance.png`
- `diagrams/roc_curve.png`
- `diagrams/confusion_matrix.png`

### Steps
1. Load `data/creditcard.csv`
2. Feature engineering: `log_amount = log1p(Amount)`, `Hour = (Time/3600)%24`
3. Drop raw `Time`, drop `Amount` (replaced by log_amount)
4. Stratified 80/20 train/test split (random_state=42)
5. Set `scale_pos_weight = n_legit / n_fraud` to handle 0.17% imbalance
6. Train XGBoostClassifier (n_estimators=500, early_stopping_rounds=20)
7. Tune decision threshold on val set for F2 score (recall-weighted)
8. Evaluate: ROC-AUC, PR-AUC, confusion matrix, classification report
9. Save model + threshold + feature names → `models/model.joblib` via joblib
10. Save feature importance plot → `diagrams/feature_importance.png`

### Verification
- [ ] `python notebooks/train.py` runs without errors
- [ ] `models/model.joblib` exists
- [ ] ROC-AUC ≥ 0.95 on test set
- [ ] `diagrams/feature_importance.png` exists

### Risks
- **Imbalance:** mitigated by `scale_pos_weight` + threshold tuning
- **Overfitting:** mitigated by early stopping on eval set

---

## Phase 3 — FastAPI Scoring API

### Files
- `app/__init__.py`
- `app/main.py`
- `app/schemas.py`
- `app/model_loader.py`

### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health + model version + uptime |
| POST | `/score_transaction` | Score a transaction, return decision |

### Request schema (`POST /score_transaction`)
```json
{
  "transaction_id": "TXN001",
  "V1": -1.35, "V2": -0.07, ..., "V28": -0.02,
  "Amount": 149.62,
  "Time": 0.0
}
```

### Response schema
```json
{
  "transaction_id": "TXN001",
  "score": 0.82,
  "fraud_probability": 0.82,
  "risk_score_percentile": 97,
  "decision": "DECLINE",
  "reasons": ["high_score_high_amount"],
  "triggered_rules": ["HIGH_FRAUD_PROB", "HIGH_AMOUNT"],
  "model_version": "xgb_v1",
  "scored_at": "2026-06-18T03:49:00Z"
}
```

### Verification
- [ ] `uvicorn app.main:app --reload` starts without errors
- [ ] `GET /health` returns 200
- [ ] `POST /score_transaction` returns correct decision
- [ ] Invalid input returns 422

---

## Phase 4 — Business Rules Engine + Audit Log

### Files
- `app/rules/engine.py`
- `app/rules/__init__.py`
- `app/audit/logger.py`
- `app/audit/__init__.py`
- `decisions.db` (SQLite, git-ignored)

### Rules Engine Logic
```
# Primary threshold rules (aligned with plan_context.md)
p_fraud >= 0.9 AND Amount > 2000  → DECLINE,    reason: "high_score_high_amount"
p_fraud >= 0.7                    → CHALLENGE,  reason: "high_score_requires_otp"
else                              → APPROVE

# Additional override rules (force DECLINE):
  - Amount > $5,000 AND prob > 0.50   → reason: "excessive_amount"
  - velocity_flag (>5 txns/hr) AND prob > 0.40 → reason: "velocity_breach"
  - foreign_transaction AND prob > 0.60 → reason: "foreign_high_risk"
```

### Audit Log Schema (`decision_log` table)
| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | auto-increment |
| transaction_id | TEXT | from request |
| timestamp | DATETIME | UTC |
| features_json | TEXT | full input serialized |
| fraud_probability | REAL | model output |
| decision | TEXT | APPROVE/CHALLENGE/DECLINE |
| triggered_rules_json | TEXT | list of rule names |
| model_version | TEXT | artifact version string |

### Verification
- [ ] POST fraud-like txn → DECLINE appears in DB
- [ ] POST legit txn → APPROVE appears in DB
- [ ] `decisions.db` file created automatically
- [ ] All fields populated correctly

---

## Phase 5 — Decision Analysis + README

### Files
- `analysis/decision_metrics.py`
- `README.md`
- `diagrams/architecture.png` (simple ASCII → PNG via matplotlib)

### Analysis script outputs
- Decision distribution pie chart → `diagrams/decision_distribution.png`
- Approval rate by hour → `diagrams/approval_by_hour.png`
- Fraud caught (DECLINE) vs missed → `diagrams/fraud_caught.png`

### README sections (per plan_context.md)
1. **Title:** "Real-Time Credit Card Fraud Scoring Service (Amex-style CFR Demo)"
2. **Overview:** 3–4 sentences on fraud, Amex-style real-time detection, what this project does
3. **Architecture:** offline training + online scoring + rules diagram
4. **Tech Stack**
5. **How to Run** — training command + uvicorn + example curl
6. **Metrics** — ROC-AUC, precision/recall, decision metrics at chosen threshold
7. **Why relevant for Credit & Fraud Risk at Amex** ← CRITICAL for portfolio
   - Real-time authorization pipeline simulation
   - ML + rules combination (mirrors production CFR systems)
   - Risk vs customer experience trade-offs
8. Dataset credit + link

---

## Agent & Tool Assignments

| Phase | Primary agent | Parallel checks |
|-------|---------------|-----------------|
| 2 | inline (train.py) | `python-reviewer` after |
| 3 | inline (FastAPI) | `security-reviewer` + `code-reviewer` after |
| 4 | inline (rules + audit) | `database-reviewer` after |
| 5 | inline (analysis + README) | `code-reviewer` after |

## Execution Order (dependencies)
```
Phase 2 → Phase 3 → Phase 4 → Phase 5
          (model needed)  (API needed)
```

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| ML model | XGBoost | Industry standard for tabular fraud; fast, interpretable via SHAP |
| Imbalance strategy | scale_pos_weight + threshold tuning | Better precision/recall control than SMOTE for this dataset size |
| API framework | FastAPI | Async, auto-docs, Pydantic v2 native |
| Audit store | SQLite + SQLAlchemy | Zero-infra, portable, queryable — right for a portfolio project |
| Decision threshold | F2-optimised (recall-weighted) | Fraud recall > precision in financial risk contexts |
