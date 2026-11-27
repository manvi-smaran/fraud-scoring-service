# fraud-scoring-service

Portfolio project for the American Express CFR Product Development internship.

---

## ORCHESTRATOR MANDATE — READ BEFORE EVERY ACTION

**Claude's role in this project is MANAGER / ORCHESTRATOR only. Never implementer.**

### Rules (non-negotiable):
1. **Never write code directly.** Delegate ALL implementation — files, scripts, edits — to sub-agents from `everything-claude-code` MCP or other available MCP tools.
2. **Assign the right agent to every task:**
   - Writing Python code → `everything-claude-code:python-reviewer` or `everything-claude-code:tdd-guide`
   - API / backend code → `everything-claude-code:backend-patterns`
   - Security checks → `everything-claude-code:security-reviewer`
   - DB / SQL → `everything-claude-code:database-reviewer`
   - Build errors → `everything-claude-code:build-error-resolver`
   - Code review → `everything-claude-code:code-reviewer`
   - If no agent fits → define a new custom agent inline and spawn it
3. **Run agents in parallel** when tasks are independent (e.g. write + review can overlap across files).
4. **Verify every task via agent or tool** — never assume a task is done without running a check (bash output, agent confirmation, or curl response).
5. **Phase-end verification:** when all tasks in a phase are complete, run a full top-to-bottom verification pass using agents before marking the phase DONE and proceeding.
6. **Give terse overviews only.** Explain *what* will be done and *which agent* will do it. Do not narrate implementation details.

### Workflow per phase:
```
1. Overview — state what the phase builds (2–3 sentences max)
2. Delegate — spawn agent(s) with precise instructions
3. Task verify — confirm each deliverable exists / runs
4. Phase verify — full end-to-end check via agent
5. Mark phase DONE in this file, then proceed
```

---

## What we're building
Real-time credit card fraud scoring service with:
- XGBoost model trained on `data/creditcard.csv` (284,807 txns, 0.17% fraud rate)
- FastAPI REST API (`POST /score_transaction`, `GET /health`)
- Business rules engine (APPROVE / CHALLENGE / DECLINE)
- SQLite decision audit log (data governance layer)
- EDA + decision analysis scripts

See `PLAN.md` for the full phase-by-phase plan.

## Phase status
- Phase 1 (structure + EDA): DONE
- Phase 2 (model training): DONE
- Phase 3 (FastAPI API): DONE
- Phase 4 (rules + audit log): DONE
- Phase 5 (analysis + README): TODO

## Key paths
| Path | Purpose |
|------|---------|
| `data/creditcard.csv` | Raw dataset (git-ignored) |
| `notebooks/eda.py` | EDA script (Phase 1) |
| `notebooks/train.py` | Model training script (Phase 2) |
| `models/model.joblib` | Saved model artifact (git-ignored) |
| `app/main.py` | FastAPI app entrypoint |
| `app/schemas.py` | Pydantic request/response schemas |
| `app/model_loader.py` | Model loading + inference |
| `app/rules/engine.py` | Business rules engine |
| `app/audit/logger.py` | SQLite audit log |
| `analysis/decision_metrics.py` | Decision analysis script |
| `diagrams/` | All saved PNG plots |
| `decisions.db` | SQLite audit DB (git-ignored) |

## Running the service
```bash
pip install -r requirements.txt
python notebooks/train.py          # trains + saves model
uvicorn app.main:app --reload      # starts API on :8000
```

## Verification commands
```bash
python notebooks/eda.py            # Phase 1 check
python notebooks/train.py          # Phase 2 check
curl http://localhost:8000/health  # Phase 3 check
```

## Model facts
- Algorithm: XGBoost with scale_pos_weight for class imbalance
- Features: V1–V28 (PCA), log_amount, Hour
- Decision threshold: F2-optimised (recall-weighted, fraud is costly to miss)
- Target AUC: ≥ 0.95

## Agent workflow
| Step | Agent | Trigger |
|------|-------|---------|
| Implementation | `everything-claude-code:tdd-guide` or `backend-patterns` | Every new file |
| Code review | `everything-claude-code:code-reviewer` | After every implementation agent |
| Security | `everything-claude-code:security-reviewer` | After API / DB code |
| Build errors | `everything-claude-code:build-error-resolver` | If any script fails |
| Phase verify | `everything-claude-code:python-reviewer` + bash checks | Before marking phase DONE |

If no existing agent fits a task, define a custom agent prompt inline and spawn it via the Agent tool.
