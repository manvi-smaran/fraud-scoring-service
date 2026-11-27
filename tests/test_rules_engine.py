"""Decision-boundary tests for app.rules.engine.apply_rules."""
from __future__ import annotations

import pytest

from app.rules import engine
from app.rules.engine import RuleResult, apply_rules

DAYTIME_HOUR = 12.0
NIGHT_HOUR = 3.0


def test_escalation_is_module_level_constant() -> None:
    assert engine._ESCALATION == {"APPROVE": 0, "CHALLENGE": 1, "DECLINE": 2}


def test_returns_immutable_rule_result() -> None:
    result = apply_rules(prob=0.1, amount=10.0, hour=DAYTIME_HOUR)
    assert isinstance(result, RuleResult)
    assert isinstance(result.reasons, tuple)
    with pytest.raises(AttributeError):
        result.decision = "DECLINE"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("prob", "amount", "hour", "decision", "rules"),
    [
        # Default approve
        (0.0, 0.0, DAYTIME_HOUR, "APPROVE", ()),
        (0.6999, 100.0, DAYTIME_HOUR, "APPROVE", ()),
        # HIGH_SCORE boundary (>= 0.7)
        (0.7, 100.0, DAYTIME_HOUR, "CHALLENGE", ("HIGH_SCORE",)),
        # HIGH_SCORE_HIGH_AMOUNT boundaries (prob >= 0.9 AND amount > 2000)
        (0.9, 2000.01, DAYTIME_HOUR, "DECLINE", ("HIGH_SCORE_HIGH_AMOUNT",)),
        (0.9, 2000.0, DAYTIME_HOUR, "CHALLENGE", ("HIGH_SCORE",)),
        (0.8999, 2500.0, DAYTIME_HOUR, "CHALLENGE", ("HIGH_SCORE",)),
        # EXCESSIVE_AMOUNT boundaries (amount > 5000 AND prob > 0.5)
        (0.51, 5000.01, DAYTIME_HOUR, "DECLINE", ("EXCESSIVE_AMOUNT",)),
        (0.5, 5000.01, DAYTIME_HOUR, "APPROVE", ()),
        (0.51, 5000.0, DAYTIME_HOUR, "APPROVE", ()),
        # SUSPICIOUS_HOUR boundaries (prob > 0.4 AND 0 <= hour < 6)
        (0.41, 100.0, NIGHT_HOUR, "CHALLENGE", ("SUSPICIOUS_HOUR",)),
        (0.4, 100.0, NIGHT_HOUR, "APPROVE", ()),
        (0.41, 100.0, 0.0, "CHALLENGE", ("SUSPICIOUS_HOUR",)),
        (0.41, 100.0, 5.99, "CHALLENGE", ("SUSPICIOUS_HOUR",)),
        (0.41, 100.0, 6.0, "APPROVE", ()),
        (0.41, 100.0, 23.99, "APPROVE", ()),
    ],
)
def test_decision_boundaries(prob, amount, hour, decision, rules) -> None:
    result = apply_rules(prob=prob, amount=amount, hour=hour)
    assert result.decision == decision
    assert result.triggered_rules == rules


def test_suspicious_hour_never_deescalates_decline() -> None:
    result = apply_rules(prob=0.95, amount=6000.0, hour=NIGHT_HOUR)
    assert result.decision == "DECLINE"
    assert result.triggered_rules == (
        "HIGH_SCORE_HIGH_AMOUNT",
        "EXCESSIVE_AMOUNT",
        "SUSPICIOUS_HOUR",
    )


def test_reasons_align_with_triggered_rules() -> None:
    result = apply_rules(prob=0.75, amount=100.0, hour=NIGHT_HOUR)
    assert result.decision == "CHALLENGE"
    assert result.reasons == ("high_score_requires_otp", "suspicious_hour")
