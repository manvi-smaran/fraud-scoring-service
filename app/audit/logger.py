"""SQLAlchemy-based audit logger for fraud scoring decisions."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
_DB_URL = f"sqlite:///{ROOT / 'decisions.db'}"

engine = create_engine(_DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


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


# Create table on first import — idempotent, safe to call repeatedly.
Base.metadata.create_all(engine)


def log_decision(
    transaction_id: str,
    features: dict,
    fraud_probability: float,
    decision: str,
    triggered_rules: list[str],
    model_version: str,
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
