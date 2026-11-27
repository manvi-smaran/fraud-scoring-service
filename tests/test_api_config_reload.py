"""API tests proving rule thresholds change at runtime via config — no restart.

RULES_CONFIG_PATH points at a tmp file; the audit module is stubbed in
sys.modules (same pattern as tests/test_api.py) so decisions.db is never touched.
"""
from __future__ import annotations

import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from tests.conftest import build_payload, rules_yaml

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
DAYTIME = 12 * 3600.0  # Time -> hour 12, so SUSPICIOUS_HOUR never fires


def _write_bumped(path: Path, text: str) -> None:
    """Rewrite the file and push its mtime forward so the change is detected."""
    previous_ns = path.stat().st_mtime_ns if path.exists() else 0
    path.write_text(text)
    new_ns = max(path.stat().st_mtime_ns, previous_ns + 1_000_000_000)
    os.utime(path, ns=(new_ns, new_ns))


@pytest.fixture
def rules_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "rules.yaml"
    _write_bumped(path, rules_yaml(challenge_prob=0.7))
    monkeypatch.setenv("RULES_CONFIG_PATH", str(path))
    return path


@contextmanager
def _app_client(calls: list[dict]) -> Iterator[TestClient]:
    """Start the app with the audit module stubbed (decisions.db never touched)."""
    stub = types.ModuleType(_AUDIT_MODULE)
    stub.log_decision = lambda **kwargs: calls.append(kwargs)
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, _AUDIT_MODULE, stub)
            mp.delitem(sys.modules, _MAIN_MODULE, raising=False)
            import app.main as main_module

            with TestClient(main_module.app) as test_client:
                yield test_client
    finally:
        sys.modules.pop(_MAIN_MODULE, None)


@pytest.fixture
def audit_calls() -> list[dict]:
    return []


@pytest.fixture
def client(rules_path: Path, audit_calls: list[dict]):
    with _app_client(audit_calls) as test_client:
        yield test_client


def _score(client: TestClient) -> dict:
    response = client.post("/score_transaction", json=build_payload(Time=DAYTIME))
    assert response.status_code == 200
    return response.json()


def test_health_exposes_active_rules_config(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["rules_config"]["challenge_prob"] == 0.7
    assert body["effective_challenge_prob"] == 0.7
    assert isinstance(body["rules_config_version"], str)
    assert body["rules_config_loaded_at"]


def test_decision_changes_after_config_edit_without_restart(
    client: TestClient, rules_path: Path
) -> None:
    before = _score(client)
    assert before["score"] < 0.7
    assert before["decision"] == "APPROVE"
    version_before = client.get("/health").json()["rules_config_version"]

    _write_bumped(rules_path, rules_yaml(challenge_prob=0.0))

    after = _score(client)
    assert after["decision"] == "CHALLENGE"
    assert after["triggered_rules"] == ["HIGH_SCORE"]
    health = client.get("/health").json()
    assert health["rules_config"]["challenge_prob"] == 0.0
    assert health["rules_config_version"] != version_before


def test_invalid_edit_keeps_last_good_config(client: TestClient, rules_path: Path) -> None:
    _write_bumped(rules_path, rules_yaml(challenge_prob=0.0))
    assert _score(client)["decision"] == "CHALLENGE"

    _write_bumped(rules_path, rules_yaml(challenge_prob=1.5))

    assert _score(client)["decision"] == "CHALLENGE"
    assert client.get("/health").json()["rules_config"]["challenge_prob"] == 0.0


def test_model_threshold_option_uses_artifact_threshold(
    client: TestClient, rules_path: Path
) -> None:
    import app.main as main_module

    artifact_threshold = main_module.app.state.artifact.threshold
    _write_bumped(
        rules_path, rules_yaml(challenge_prob="model_threshold", high_score_amount_prob=0.97)
    )

    health = client.get("/health").json()
    assert health["rules_config"]["challenge_prob"] == "model_threshold"
    assert health["effective_challenge_prob"] == artifact_threshold

    body = _score(client)
    expected = "CHALLENGE" if body["fraud_probability"] >= artifact_threshold else "APPROVE"
    assert body["decision"] == expected


def test_model_threshold_with_default_decline_cutoff_is_rejected_on_reload(
    client: TestClient, rules_path: Path
) -> None:
    # artifact threshold 0.95 > default high_score_amount_prob 0.9 -> invalid
    _write_bumped(rules_path, rules_yaml(challenge_prob="model_threshold"))

    health = client.get("/health").json()
    assert health["rules_config"]["challenge_prob"] == 0.7
    assert health["effective_challenge_prob"] == 0.7
    assert client.post("/score_transaction", json=build_payload(Time=DAYTIME)).status_code == 200


def test_model_threshold_with_default_decline_cutoff_is_fatal_at_startup(
    rules_path: Path,
) -> None:
    _write_bumped(rules_path, rules_yaml(challenge_prob="model_threshold"))
    with pytest.raises(ValueError, match="high_score_amount_prob"):
        with _app_client([]):
            pass


def test_partial_file_is_fatal_at_startup(rules_path: Path) -> None:
    _write_bumped(rules_path, "challenge_prob: 0.7\n")
    with pytest.raises(ValueError, match="missing required keys"):
        with _app_client([]):
            pass


def test_audit_uses_same_snapshot_as_decision(
    client: TestClient, rules_path: Path, audit_calls: list[dict]
) -> None:
    _write_bumped(rules_path, rules_yaml(challenge_prob=0.0))
    body = _score(client)
    health = client.get("/health").json()
    assert body["decision"] == "CHALLENGE"
    assert audit_calls[-1]["rules_config_version"] == health["rules_config_version"]
    assert audit_calls[-1]["effective_challenge_prob"] == 0.0
