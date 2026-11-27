"""SQLAlchemy-based audit logger for fraud scoring decisions."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import DateTime, Engine, Float, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DB_URL_ENV = "FRAUD_AUDIT_DB_URL"
DEFAULT_DB_URL = f"sqlite:///{ROOT / 'decisions.db'}"
_TABLE = "decision_log"
# Columns added after the original schema. Fixed identifiers/DDL only -- no user input.
_ADDED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("rules_config_version", "VARCHAR(32)"),
    ("effective_challenge_prob", "FLOAT"),
)


def audit_db_url() -> str:
    """Audit DB URL from FRAUD_AUDIT_DB_URL, falling back to <project>/decisions.db."""
    return os.environ.get(DB_URL_ENV) or DEFAULT_DB_URL


class Base(DeclarativeBase):
    pass


class DecisionLog(Base):
    __tablename__ = "decision_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # SQLite stores as naive string; UTC enforced via default
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    features_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    fraud_probability: Mapped[float] = mapped_column(Float, nullable=False)
    decision: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    triggered_rules_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    # Rules-config provenance: which thresholds produced this decision.
    rules_config_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    effective_challenge_prob: Mapped[float | None] = mapped_column(Float, nullable=True)


def migrate_schema(target: Engine) -> None:
    """Idempotently add columns missing from an existing decision_log table.

    create_all() never alters existing tables, so DBs created before a column was
    added get it here via ALTER TABLE ... ADD COLUMN (nullable, so old rows -> NULL).
    """
    existing = {col["name"] for col in inspect(target).get_columns(_TABLE)}
    missing = [(name, ddl) for name, ddl in _ADDED_COLUMNS if name not in existing]
    if not missing:
        return
    with target.begin() as conn:
        for name, ddl in missing:
            conn.execute(text(f"ALTER TABLE {_TABLE} ADD COLUMN {name} {ddl}"))
            logger.info("Audit DB migrated: added column %s.%s", _TABLE, name)


def init_db(target: Engine) -> None:
    """Create the table if absent, then bring an existing table up to date."""
    Base.metadata.create_all(target)
    migrate_schema(target)


engine = create_engine(audit_db_url(), connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Startup schema setup on first import -- idempotent, safe to call repeatedly.
init_db(engine)


def log_decision(
    transaction_id: str,
    features: dict,
    fraud_probability: float,
    decision: str,
    triggered_rules: list[str],
    model_version: str,
    rules_config_version: Optional[str] = None,
    effective_challenge_prob: Optional[float] = None,
) -> None:
    """Persist a single fraud-scoring decision to the audit database.

    Opens a new session, writes one :class:`DecisionLog` row, commits, then
    closes the session.  Thread-safe: each call owns its own session.
    """
    session = SessionLocal()
    try:
        row = DecisionLog(
            transaction_id=transaction_id,
            timestamp=datetime.now(timezone.utc),
            features_json=json.dumps(features),
            fraud_probability=fraud_probability,
            decision=decision,
            triggered_rules_json=json.dumps(triggered_rules),
            model_version=model_version,
            rules_config_version=rules_config_version,
            effective_challenge_prob=effective_challenge_prob,
        )
        session.add(row)
        session.commit()
        logger.debug(
            "Logged decision: transaction_id=%s decision=%s prob=%.4f",
            transaction_id,
            decision,
            fraud_probability,
        )
    except Exception:
        session.rollback()
        logger.exception(
            "Failed to log decision for transaction_id=%s", transaction_id
        )
        raise
    finally:
        session.close()
