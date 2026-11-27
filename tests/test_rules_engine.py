"""Decision-boundary tests for app.rules.engine.apply_rules."""
from __future__ import annotations

import pytest

from app.rules import engine
from app.rules.config import RulesConfig
from app.rules.engine import RuleResult
from app.rules.engine import apply_rules as _apply_rules

DAYTIME_HOUR = 12.0
NIGHT_HOUR = 3.0
DEFAULT_CONFIG = RulesConfig()


def apply_rules(
    prob: float,
    amount: float,
    hour: float,
    config: RulesConfig = DEFAULT_CONFIG,
    challenge_prob: float | None = None,
) -> RuleResult:
    """Call the engine with the default config unless overridden."""
    resolved = config.resolve_challenge_prob(0.95) if challenge_prob is None else challenge_prob
    return _apply_rules(
        prob=prob, amount=amount, hour=hour, config=config, challenge_prob=resolved
    )


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


def test_engine_has_no_hard_coded_threshold_constants() -> None:
    for name in (
        "_HIGH_SCORE_AMOUNT_PROB",
        "_HIGH_SCORE_AMOUNT_THRESH",
        "_CHALLENGE_PROB",
        "_EXCESSIVE_AMOUNT",
        "_EXCESSIVE_AMOUNT_PROB",
        "_SUSPICIOUS_HOUR_PROB",
        "_SUSPICIOUS_HOURS",
    ):
        assert not hasattr(engine, name), name


def test_config_and_challenge_prob_are_required() -> None:
    with pytest.raises(TypeError):
        _apply_rules(prob=0.1, amount=10.0, hour=DAYTIME_HOUR)  # type: ignore[call-arg]


def test_custom_config_changes_every_cut_off() -> None:
    cfg = RulesConfig(
        high_score_amount_prob=0.8,
        high_score_amount_threshold=1000.0,
        challenge_prob=0.3,
        excessive_amount=3000.0,
        excessive_amount_prob=0.2,
        suspicious_hour_prob=0.1,
        suspicious_hours=[22, 23],
    )
    assert apply_rules(0.3, 10.0, DAYTIME_HOUR, cfg).decision == "CHALLENGE"
    assert apply_rules(0.8, 1000.01, DAYTIME_HOUR, cfg).decision == "DECLINE"
    assert apply_rules(0.21, 3000.01, DAYTIME_HOUR, cfg).triggered_rules == (
        "EXCESSIVE_AMOUNT",
    )
    assert apply_rules(0.11, 10.0, 22.5, cfg).triggered_rules == ("SUSPICIOUS_HOUR",)
    assert apply_rules(0.11, 10.0, NIGHT_HOUR, cfg).decision == "APPROVE"


def test_resolved_challenge_prob_overrides_numeric_value() -> None:
    # e.g. "model_threshold" resolved to the artifact's tuned 0.95
    assert apply_rules(0.9, 10.0, DAYTIME_HOUR, challenge_prob=0.95).decision == "APPROVE"
    assert apply_rules(0.95, 10.0, DAYTIME_HOUR, challenge_prob=0.95).decision == "CHALLENGE"


def test_model_threshold_config_resolution_feeds_engine() -> None:
    cfg = RulesConfig(challenge_prob="model_threshold", high_score_amount_prob=0.97)
    result = apply_rules(0.8, 10.0, DAYTIME_HOUR, cfg, cfg.resolve_challenge_prob(0.95))
    assert result.decision == "APPROVE"


def test_empty_suspicious_hours_disables_rule() -> None:
    cfg = RulesConfig(suspicious_hours=[])
    assert apply_rules(0.5, 10.0, NIGHT_HOUR, cfg).decision == "APPROVE"


def test_never_deescalates_with_custom_config() -> None:
    cfg = RulesConfig(challenge_prob=0.1, suspicious_hour_prob=0.0)
    result = apply_rules(0.6, 5000.01, NIGHT_HOUR, cfg)
    assert result.decision == "DECLINE"
    assert result.triggered_rules == ("HIGH_SCORE", "EXCESSIVE_AMOUNT", "SUSPICIOUS_HOUR")
