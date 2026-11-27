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
