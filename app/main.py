from contextlib import asynccontextmanager
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from app.schemas import TransactionRequest, ScoreResponse
from app.model_loader import load_model, predict
from app.rules.engine import apply_rules
from app.audit.logger import log_decision

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.artifact = load_model(MODEL_PATH)
        app.state.start_time = time.time()
        logger.info("Model loaded: %s", app.state.artifact.version)
    except Exception as exc:
        logger.critical("Failed to load model from %s: %s", MODEL_PATH, exc)
        raise
    yield


app = FastAPI(title="Fraud Scoring Service", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict:
    artifact = request.app.state.artifact
    return {
        "status": "ok",
        "model_version": artifact.version,
        "uptime_seconds": round(time.time() - request.app.state.start_time, 1),
    }


@app.post("/score_transaction", response_model=ScoreResponse)
def score_transaction(txn: TransactionRequest, request: Request) -> ScoreResponse:
    t0 = time.time()
    artifact = request.app.state.artifact

    features = txn.to_feature_dict()
    prob, _ = predict(artifact, features)

    hour = (txn.Time / 3600) % 24
    rule_result = apply_rules(prob=prob, amount=txn.Amount, hour=hour)

    # Audit log (non-blocking — log failure must not fail the request)
    try:
        log_decision(
            transaction_id=txn.transaction_id,
            features=features,
            fraud_probability=prob,
            decision=rule_result.decision,
            triggered_rules=list(rule_result.triggered_rules),
            model_version=artifact.version,
        )
    except Exception as audit_exc:
        logger.warning("Audit log failed for %s: %s", txn.transaction_id, audit_exc)

    elapsed_ms = round((time.time() - t0) * 1000, 2)
    logger.info(
        "txn=%s decision=%s score=%.4f elapsed_ms=%s",
        txn.transaction_id,
        rule_result.decision,
        prob,
        elapsed_ms,
    )

    return ScoreResponse(
        transaction_id=txn.transaction_id,
        score=round(prob, 4),
        fraud_probability=round(prob, 4),
        decision=rule_result.decision,
        reasons=list(rule_result.reasons),
        triggered_rules=list(rule_result.triggered_rules),
        model_version=artifact.version,
        scored_at=datetime.now(timezone.utc).isoformat(),
    )
