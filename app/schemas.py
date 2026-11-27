import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TRANSACTION_ID_PATTERN = r"^[A-Za-z0-9_\-]+$"
MAX_AMOUNT = 1e7


class TransactionRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    transaction_id: str = Field(min_length=1, max_length=64, pattern=TRANSACTION_ID_PATTERN)
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0, le=MAX_AMOUNT)
    Time: float = Field(ge=0)

    def to_feature_dict(self) -> dict[str, float]:
        return {
            "log_amount": math.log1p(self.Amount),
            "Hour": (self.Time / 3600) % 24,
            "V1": self.V1,
            "V2": self.V2,
            "V3": self.V3,
            "V4": self.V4,
            "V5": self.V5,
            "V6": self.V6,
            "V7": self.V7,
            "V8": self.V8,
            "V9": self.V9,
            "V10": self.V10,
            "V11": self.V11,
            "V12": self.V12,
            "V13": self.V13,
            "V14": self.V14,
            "V15": self.V15,
            "V16": self.V16,
            "V17": self.V17,
            "V18": self.V18,
            "V19": self.V19,
            "V20": self.V20,
            "V21": self.V21,
            "V22": self.V22,
            "V23": self.V23,
            "V24": self.V24,
            "V25": self.V25,
            "V26": self.V26,
            "V27": self.V27,
            "V28": self.V28,
        }


class ScoreResponse(BaseModel):
    transaction_id: str
    score: float = Field(description="Fraud probability score (0–1), primary output field")
    fraud_probability: float = Field(description="Fraud probability score (alias of score for schema compatibility)")
    decision: Literal["APPROVE", "CHALLENGE", "DECLINE"]
    reasons: list[str]
    triggered_rules: list[str]
    model_version: str
    scored_at: str
