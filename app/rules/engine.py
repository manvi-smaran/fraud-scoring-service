"""Business rules engine — maps fraud probability + transaction context to a decision."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import logging

logger = logging.getLogger(__name__)

Decision = Literal["APPROVE", "CHALLENGE", "DECLINE"]

# Probability / amount thresholds for the rules engine
_HIGH_SCORE_AMOUNT_PROB   = 0.9
_HIGH_SCORE_AMOUNT_THRESH = 2000.0
_CHALLENGE_PROB           = 0.7
_EXCESSIVE_AMOUNT         = 5000.0
_EXCESSIVE_AMOUNT_PROB    = 0.5
_SUSPICIOUS_HOUR_PROB     = 0.4
_SUSPICIOUS_HOURS         = frozenset(range(0, 6))   # midnight–5am

# Severity ordering used so override rules only escalate, never de-escalate
_ESCALATION: dict[Decision, int] = {"APPROVE": 0, "CHALLENGE": 1, "DECLINE": 2}


@dataclass(frozen=True)
class RuleResult:
    decision: Decision
    reasons: tuple[str, ...]
    triggered_rules: tuple[str, ...]


def apply_rules(prob: float, amount: float, hour: float) -> RuleResult:
    """
    Apply business rules in priority order.
    Returns a RuleResult with the final decision and all triggered rule names.

    Priority:
      1. High score + high amount → DECLINE
      2. High score → CHALLENGE
      3. Default → APPROVE
    Then override rules (can only escalate, never de-escalate):
      4. Excessive amount → DECLINE
      5. Suspicious hour → at least CHALLENGE
    """
    decision: Decision = "APPROVE"
    reasons: list[str] = []
    triggered_rules: list[str] = []

    # Primary rules
    if prob >= _HIGH_SCORE_AMOUNT_PROB and amount > _HIGH_SCORE_AMOUNT_THRESH:
        decision = "DECLINE"
        reasons.append("high_score_high_amount")
        triggered_rules.append("HIGH_SCORE_HIGH_AMOUNT")
    elif prob >= _CHALLENGE_PROB:
        decision = "CHALLENGE"
        reasons.append("high_score_requires_otp")
        triggered_rules.append("HIGH_SCORE")

    # Override rules — only escalate, never de-escalate
    if amount > _EXCESSIVE_AMOUNT and prob > _EXCESSIVE_AMOUNT_PROB:
        decision = "DECLINE"  # always escalates; no de-escalation possible
        reasons.append("excessive_amount")
        triggered_rules.append("EXCESSIVE_AMOUNT")

    if prob > _SUSPICIOUS_HOUR_PROB and int(hour) in _SUSPICIOUS_HOURS:
        if _ESCALATION["CHALLENGE"] > _ESCALATION[decision]:
            decision = "CHALLENGE"
        reasons.append("suspicious_hour")
        triggered_rules.append("SUSPICIOUS_HOUR")

    logger.debug("prob=%.4f amount=%.2f hour=%.1f → %s rules=%s",
                 prob, amount, hour, decision, triggered_rules)
    return RuleResult(
        decision=decision,
        reasons=tuple(reasons),
        triggered_rules=tuple(triggered_rules),
    )
