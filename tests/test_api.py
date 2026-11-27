"""API tests via TestClient. Never touches the project's decisions.db.

The real audit module is replaced in sys.modules by an in-memory stub
*before* app.main is imported, so no SQLite engine is created for the
project database and no rows are written.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from tests.conftest import build_payload

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"

if not MODEL_PATH.exists():
    pytest.skip(
        "models/model.joblib missing - run `python notebooks/train.py` first",
        allow_module_level=True,
    )

from fastapi.testclient import TestClient  # noqa: E402

_AUDIT_MODULE = "app.audit.logger"
_MAIN_MODULE = "app.main"


def _make_audit_stub(calls: list[dict]) -> types.ModuleType:
    stub = types.ModuleType(_AUDIT_MODULE)
    stub.log_decision = lambda **kwargs: calls.append(kwargs)
    return stub


@pytest.fixture(scope="module")
def audit_calls():
    calls: list[dict] = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, _AUDIT_MODULE, _make_audit_stub(calls))
        mp.delitem(sys.modules, _MAIN_MODULE, raising=False)
        yield calls
    sys.modules.pop(_MAIN_MODULE, None)


@pytest.fixture(scope="module")
def client(audit_calls):
    import app.main as main_module

    with TestClient(main_module.app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_version"], str)
    assert body["uptime_seconds"] >= 0


def test_score_transaction_valid(client: TestClient, audit_calls: list) -> None:
    before = len(audit_calls)
    response = client.post("/score_transaction", json=build_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "TXN_001"
    assert 0.0 <= body["score"] <= 1.0
    assert body["score"] == body["fraud_probability"]
    assert body["decision"] in {"APPROVE", "CHALLENGE", "DECLINE"}
    assert len(body["reasons"]) == len(body["triggered_rules"])
    assert len(audit_calls) == before + 1
    assert audit_calls[-1]["decision"] == body["decision"]


def test_score_transaction_rejects_extra_field(client: TestClient) -> None:
    response = client.post("/score_transaction", json=build_payload(Class=1))
    assert response.status_code == 422


def test_score_transaction_rejects_missing_feature(client: TestClient) -> None:
    payload = {k: v for k, v in build_payload().items() if k != "V1"}
    assert client.post("/score_transaction", json=payload).status_code == 422


def test_score_transaction_rejects_bad_id(client: TestClient) -> None:
    response = client.post(
        "/score_transaction", json=build_payload(transaction_id="bad id;")
    )
    assert response.status_code == 422


def test_audit_failure_does_not_fail_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.main as main_module

    def _boom(**_kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(main_module, "log_decision", _boom)
    response = client.post("/score_transaction", json=build_payload())
    assert response.status_code == 200


def test_audit_row_carries_rules_config_provenance(
    client: TestClient, audit_calls: list
) -> None:
    health = client.get("/health").json()
    response = client.post("/score_transaction", json=build_payload())
    assert response.status_code == 200
    call = audit_calls[-1]
    assert call["rules_config_version"] == health["rules_config_version"]
    assert call["effective_challenge_prob"] == health["effective_challenge_prob"]
