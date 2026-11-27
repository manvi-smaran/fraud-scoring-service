from contextlib import asynccontextmanager
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from app.schemas import TransactionRequest, ScoreResponse
from app.model_loader import load_model, predict
from app.rules.config import RulesConfigProvider, config_path_from_env
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
    rules_path = config_path_from_env()
    try:
        # Hot-reloading provider: thresholds can change at runtime, no restart needed.
        # The artifact threshold lets it resolve + validate `challenge_prob: model_threshold`
        # at load/reload time, so an invalid combination is rejected, never served.
        app.state.rules_provider = RulesConfigProvider(
            rules_path, model_threshold=app.state.artifact.threshold
        )
    except Exception as exc:
        logger.critical("Invalid rules config at %s: %s", rules_path, exc)
        raise
    yield


app = FastAPI(title="Fraud Scoring Service", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict:
    artifact = request.app.state.artifact
    snapshot = request.app.state.rules_provider.snapshot()
    return {
        "status": "ok",
        "model_version": artifact.version,
        "uptime_seconds": round(time.time() - request.app.state.start_time, 1),
        "rules_config": snapshot.config.model_dump(mode="json"),
        "rules_config_version": snapshot.version,
        "rules_config_loaded_at": snapshot.loaded_at.isoformat(),
        "effective_challenge_prob": snapshot.effective_challenge_prob,
    }


@app.post("/score_transaction", response_model=ScoreResponse)
def score_transaction(txn: TransactionRequest, request: Request) -> ScoreResponse:
    t0 = time.time()
    artifact = request.app.state.artifact

    # One snapshot per request: the decision and its audit row use the same config.
    snapshot = request.app.state.rules_provider.snapshot()  # picks up config edits live

    features = txn.to_feature_dict()
    prob, _ = predict(artifact, features)

    hour = (txn.Time / 3600) % 24
    rule_result = apply_rules(
        prob=prob,
        amount=txn.Amount,
        hour=hour,
        config=snapshot.config,
        challenge_prob=snapshot.effective_challenge_prob,
    )

    # Audit log (non-blocking — log failure must not fail the request)
    try:
        log_decision(
            transaction_id=txn.transaction_id,
            features=features,
            fraud_probability=prob,
            decision=rule_result.decision,
            triggered_rules=list(rule_result.triggered_rules),
            model_version=artifact.version,
            rules_config_version=snapshot.version,
            effective_challenge_prob=snapshot.effective_challenge_prob,
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
