"""Validation tests for app.schemas.TransactionRequest / ScoreResponse."""
from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from app.schemas import ScoreResponse, TransactionRequest
from tests.conftest import build_payload


def test_accepts_valid_payload(valid_payload: dict) -> None:
    txn = TransactionRequest(**valid_payload)
    assert txn.transaction_id == "TXN_001"


def test_accepts_zero_amount() -> None:
    txn = TransactionRequest(**build_payload(Amount=0))
    assert txn.Amount == 0
    assert txn.to_feature_dict()["log_amount"] == 0.0


def test_accepts_max_amount() -> None:
    assert TransactionRequest(**build_payload(Amount=1e7)).Amount == 1e7


@pytest.mark.parametrize("amount", [-0.01, 1e7 + 1])
def test_rejects_out_of_range_amount(amount: float) -> None:
    with pytest.raises(ValidationError):
        TransactionRequest(**build_payload(Amount=amount))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field", ["V1", "V28", "Amount", "Time"])
def test_rejects_nan_and_inf(field: str, bad: float) -> None:
    with pytest.raises(ValidationError):
        TransactionRequest(**build_payload(**{field: bad}))


@pytest.mark.parametrize(
    "txn_id",
    ["", "x" * 65, "TXN 001", "TXN;DROP", "../etc", "TXN\n", "交易"],
)
def test_rejects_bad_transaction_id(txn_id: str) -> None:
    with pytest.raises(ValidationError):
        TransactionRequest(**build_payload(transaction_id=txn_id))


@pytest.mark.parametrize("txn_id", ["a", "x" * 64, "TXN-001_abc"])
def test_accepts_good_transaction_id(txn_id: str) -> None:
    assert TransactionRequest(**build_payload(transaction_id=txn_id)).transaction_id == txn_id


def test_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TransactionRequest(**build_payload(Class=1))


def test_rejects_missing_feature(valid_payload: dict) -> None:
    payload = {k: v for k, v in valid_payload.items() if k != "V14"}
    with pytest.raises(ValidationError):
        TransactionRequest(**payload)


def test_feature_dict_has_30_features(valid_payload: dict) -> None:
    features = TransactionRequest(**valid_payload).to_feature_dict()
    assert len(features) == 30
    assert {"log_amount", "Hour"} <= features.keys()


def test_response_schema_unchanged() -> None:
    assert set(ScoreResponse.model_fields) == {
        "transaction_id", "score", "fraud_probability", "decision",
        "reasons", "triggered_rules", "model_version", "scored_at",
    }
