"""Shared pytest fixtures and path setup."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

V_FEATURES = tuple(f"V{i}" for i in range(1, 29))


def build_payload(**overrides: object) -> dict:
    """Return a fresh valid request payload, with optional field overrides."""
    base = {"transaction_id": "TXN_001", "Amount": 149.62, "Time": 0.0}
    features = {name: 0.0 for name in V_FEATURES}
    return {**base, **features, **overrides}


@pytest.fixture
def valid_payload() -> dict:
    return build_payload()


DEFAULT_RULES: dict = {
    "high_score_amount_prob": 0.9,
    "high_score_amount_threshold": 2000.0,
    "challenge_prob": 0.7,
    "excessive_amount": 5000.0,
    "excessive_amount_prob": 0.5,
    "suspicious_hour_prob": 0.4,
    "suspicious_hours": [0, 1, 2, 3, 4, 5],
}


def rules_yaml(**overrides: object) -> str:
    """Return a COMPLETE rules.yaml (every key present), with optional overrides."""
    merged = {**DEFAULT_RULES, **overrides}
    return "".join(f"{key}: {value}\n" for key, value in merged.items())
