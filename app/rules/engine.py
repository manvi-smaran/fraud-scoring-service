"""Business rules engine — maps fraud probability + transaction context to a decision."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import logging

from app.rules.config import RulesConfig

logger = logging.getLogger(__name__)

Decision = Literal["APPROVE", "CHALLENGE", "DECLINE"]

# Thresholds are NOT defined here: they come from RulesConfig (config/rules.yaml),
# which is hot-reloaded at runtime by app.rules.config.RulesConfigProvider.

# Severity ordering used so override rules only escalate, never de-escalate
_ESCALATION: dict[Decision, int] = {"APPROVE": 0, "CHALLENGE": 1, "DECLINE": 2}


@dataclass(frozen=True)
class RuleResult:
    decision: Decision
    reasons: tuple[str, ...]
    triggered_rules: tuple[str, ...]


def apply_rules(
    prob: float,
    amount: float,
    hour: float,
    *,
    config: RulesConfig,
    challenge_prob: float,
) -> RuleResult:
    """
    Apply business rules in priority order. Pure function: no module state.

    `config` supplies every threshold; `challenge_prob` is the already-resolved
    numeric CHALLENGE cut-off (config.resolve_challenge_prob(model_threshold)),
    so the "model_threshold" option is handled by the caller.

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
    if prob >= config.high_score_amount_prob and amount > config.high_score_amount_threshold:
        decision = "DECLINE"
        reasons.append("high_score_high_amount")
        triggered_rules.append("HIGH_SCORE_HIGH_AMOUNT")
    elif prob >= challenge_prob:
        decision = "CHALLENGE"
        reasons.append("high_score_requires_otp")
        triggered_rules.append("HIGH_SCORE")

    # Override rules — only escalate, never de-escalate
    if amount > config.excessive_amount and prob > config.excessive_amount_prob:
        decision = "DECLINE"  # always escalates; no de-escalation possible
        reasons.append("excessive_amount")
        triggered_rules.append("EXCESSIVE_AMOUNT")

    if prob > config.suspicious_hour_prob and int(hour) in config.suspicious_hours:
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
