"""Tests for app.audit.logger against TEMP SQLite files only.

FRAUD_AUDIT_DB_URL is pointed at tmp_path *before* the module is imported, so
the project's decisions.db is never opened. The module is removed from
sys.modules afterwards (restored by monkeypatch).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable

import pytest
from sqlalchemy import create_engine, inspect, text

import app.audit as audit_package

ROOT = Path(__file__).resolve().parent.parent
_MODULE = "app.audit.logger"
NEW_COLUMNS = {"rules_config_version", "effective_challenge_prob"}

OLD_SCHEMA_DDL = """
CREATE TABLE decision_log (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    transaction_id VARCHAR(64) NOT NULL,
    timestamp DATETIME NOT NULL,
    features_json TEXT,
    fraud_probability FLOAT NOT NULL,
    decision VARCHAR(16) NOT NULL,
    triggered_rules_json TEXT,
    model_version VARCHAR(32) NOT NULL
)
"""
OLD_ROW_INSERT = """
INSERT INTO decision_log (transaction_id, timestamp, fraud_probability, decision, model_version)
VALUES ('OLD_TXN', '2026-01-01 00:00:00', 0.1, 'APPROVE', 'v0')
"""


def _sqlite_url(path: Path) -> str:
    return f"sqlite:///{path}"


def _create_old_schema_db(path: Path) -> None:
    engine = create_engine(_sqlite_url(path))
    with engine.begin() as conn:
        conn.execute(text(OLD_SCHEMA_DDL))
        conn.execute(text(OLD_ROW_INSERT))
    engine.dispose()


def _columns(path: Path) -> set[str]:
    engine = create_engine(_sqlite_url(path))
    try:
        return {col["name"] for col in inspect(engine).get_columns("decision_log")}
    finally:
        engine.dispose()


@pytest.fixture
def import_audit(monkeypatch: pytest.MonkeyPatch) -> Callable[[Path], ModuleType]:
    """Return a loader that imports a fresh audit module bound to a temp DB file."""
    loaded: list[ModuleType] = []

    def _load(db_path: Path) -> ModuleType:
        monkeypatch.setenv("FRAUD_AUDIT_DB_URL", _sqlite_url(db_path))
        monkeypatch.delitem(sys.modules, _MODULE, raising=False)
        monkeypatch.delattr(audit_package, "logger", raising=False)
        module = importlib.import_module(_MODULE)
        loaded.append(module)
        # Hard guard: never proceed (and never write) unless bound to the temp DB.
        if str(module.engine.url) != _sqlite_url(db_path):
            pytest.fail(f"audit module not bound to temp DB: {module.engine.url}")
        return module

    yield _load
    for module in loaded:
        module.engine.dispose()


def _log(module: ModuleType, **extra: object) -> None:
    module.log_decision(
        transaction_id="TXN_1",
        features={"V1": 0.0},
        fraud_probability=0.42,
        decision="APPROVE",
        triggered_rules=[],
        model_version="v1",
        **extra,
    )


def _only_row(module: ModuleType):
    session = module.SessionLocal()
    try:
        rows = session.query(module.DecisionLog).all()
        assert len(rows) == 1
        return rows[0]
    finally:
        session.close()


def test_db_url_uses_env_override(import_audit, tmp_path: Path) -> None:
    db = tmp_path / "audit.db"
    module = import_audit(db)
    assert module.audit_db_url() == _sqlite_url(db)
    assert str(module.engine.url) == _sqlite_url(db)


def test_db_url_default_is_project_decisions_db(
    import_audit, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = import_audit(tmp_path / "audit.db")
    monkeypatch.delenv("FRAUD_AUDIT_DB_URL")
    assert module.audit_db_url() == _sqlite_url(ROOT / "decisions.db")


def test_log_decision_records_rules_config_provenance(import_audit, tmp_path: Path) -> None:
    module = import_audit(tmp_path / "audit.db")
    _log(module, rules_config_version="3f1c2a9b7d10", effective_challenge_prob=0.7)
    row = _only_row(module)
    assert row.rules_config_version == "3f1c2a9b7d10"
    assert row.effective_challenge_prob == 0.7


def test_log_decision_provenance_columns_are_nullable(import_audit, tmp_path: Path) -> None:
    module = import_audit(tmp_path / "audit.db")
    _log(module)
    row = _only_row(module)
    assert row.rules_config_version is None
    assert row.effective_challenge_prob is None


def test_migrate_schema_adds_missing_columns_and_keeps_rows(
    import_audit, tmp_path: Path
) -> None:
    module = import_audit(tmp_path / "fresh.db")
    old_db = tmp_path / "old.db"
    _create_old_schema_db(old_db)
    assert not NEW_COLUMNS & _columns(old_db)

    engine = create_engine(_sqlite_url(old_db))
    try:
        module.migrate_schema(engine)
        module.migrate_schema(engine)  # idempotent: second run is a no-op
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT transaction_id, rules_config_version FROM decision_log")
            ).one()
    finally:
        engine.dispose()

    assert NEW_COLUMNS <= _columns(old_db)
    assert tuple(row) == ("OLD_TXN", None)


def test_import_migrates_existing_old_schema_db(import_audit, tmp_path: Path) -> None:
    old_db = tmp_path / "decisions.db"
    _create_old_schema_db(old_db)

    module = import_audit(old_db)  # startup migration runs at import
    assert NEW_COLUMNS <= _columns(old_db)

    _log(module, rules_config_version="abc123", effective_challenge_prob=0.6)
    session = module.SessionLocal()
    try:
        versions = {r.rules_config_version for r in session.query(module.DecisionLog).all()}
    finally:
        session.close()
    assert versions == {None, "abc123"}
