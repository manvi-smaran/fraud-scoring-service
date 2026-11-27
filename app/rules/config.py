"""Rules-engine configuration: validated YAML + thread-safe hot reload.

Thresholds live in config/rules.yaml (override via RULES_CONFIG_PATH). The
provider re-reads the (tiny) file on every check and compares a content hash,
so any saved edit is picked up without a restart or redeploy -- even one that
keeps the same mtime and size. A config file must list EVERY threshold: an
empty or partially written file is rejected rather than silently falling back
to defaults. An invalid edit is logged and ignored: the last good config stays
active. Only an invalid *initial* load is fatal.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config" / "rules.yaml"
CONFIG_PATH_ENV = "RULES_CONFIG_PATH"
MODEL_THRESHOLD = "model_threshold"

# Strict types reject YAML booleans (true/false) masquerading as numbers.
Probability = Annotated[Union[StrictFloat, StrictInt], Field(ge=0.0, le=1.0)]
Amount = Annotated[Union[StrictFloat, StrictInt], Field(ge=0.0)]
Hour = Annotated[StrictInt, Field(ge=0, le=23)]


class RulesConfig(BaseModel):
    """Immutable, validated rule thresholds. Defaults equal the original constants."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    high_score_amount_prob: Probability = 0.9
    high_score_amount_threshold: Amount = 2000.0
    challenge_prob: Union[Probability, Literal["model_threshold"]] = 0.7
    excessive_amount: Amount = 5000.0
    excessive_amount_prob: Probability = 0.5
    suspicious_hour_prob: Probability = 0.4
    suspicious_hours: tuple[Hour, ...] = (0, 1, 2, 3, 4, 5)

    @field_validator("suspicious_hours")
    @classmethod
    def _unique_sorted_hours(cls, hours: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(hours)) != len(hours):
            raise ValueError("suspicious_hours must not contain duplicates")
        return tuple(sorted(hours))

    @model_validator(mode="after")
    def _cross_field_checks(self) -> "RulesConfig":
        numeric_challenge = self.challenge_prob != MODEL_THRESHOLD
        if numeric_challenge and self.challenge_prob > self.high_score_amount_prob:
            raise ValueError("challenge_prob must be <= high_score_amount_prob")
        if self.high_score_amount_threshold > self.excessive_amount:
            raise ValueError("high_score_amount_threshold must be <= excessive_amount")
        return self

    def resolve_challenge_prob(self, model_threshold: Optional[float]) -> float:
        """Return the numeric CHALLENGE cut-off, substituting the model's tuned threshold.

        Raises ValueError if "model_threshold" is selected but the artifact value is
        missing, outside [0, 1], or above high_score_amount_prob.
        """
        if self.challenge_prob != MODEL_THRESHOLD:
            return float(self.challenge_prob)
        if model_threshold is None:
            raise ValueError("challenge_prob is 'model_threshold' but no model threshold is available")
        value = float(model_threshold)
        if math.isnan(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"model threshold {model_threshold!r} is not in [0, 1]")
        if value > self.high_score_amount_prob:
            raise ValueError(
                f"model threshold {value!r} exceeds high_score_amount_prob "
                f"{self.high_score_amount_prob!r}; challenge_prob must be <= high_score_amount_prob"
            )
        return value


@dataclass(frozen=True)
class ConfigSnapshot:
    """A loaded config plus provenance, for /health and logging."""

    config: RulesConfig
    effective_challenge_prob: float  # resolved + validated at load: never raises per request
    path: Path
    version: str
    loaded_at: datetime


def config_path_from_env() -> Path:
    """Path from RULES_CONFIG_PATH, falling back to config/rules.yaml."""
    raw = os.environ.get(CONFIG_PATH_ENV)
    return Path(raw) if raw else DEFAULT_CONFIG_PATH


REQUIRED_KEYS = frozenset(RulesConfig.model_fields)


def _parse(raw: bytes, path: Path) -> RulesConfig:
    """Parse a COMPLETE rules file. Empty / partial files are invalid (no defaults)."""
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        raise ValueError(f"{path} is empty; every rule threshold must be set")
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping of threshold names to values")
    missing = sorted(REQUIRED_KEYS - data.keys())
    if missing:
        raise ValueError(f"{path} is missing required keys: {', '.join(missing)}")
    return RulesConfig(**data)  # pydantic ValidationError is a ValueError


def _read_config(path: Path) -> Optional[bytes]:
    """Return the file's bytes, or None if it is missing or unreadable."""
    try:
        return path.read_bytes()
    except OSError:
        return None


def _digest(raw: Optional[bytes]) -> Optional[str]:
    return None if raw is None else hashlib.sha256(raw).hexdigest()


def _build_snapshot(raw: bytes, path: Path, model_threshold: Optional[float]) -> ConfigSnapshot:
    config = _parse(raw, path)
    return ConfigSnapshot(
        config=config,
        effective_challenge_prob=config.resolve_challenge_prob(model_threshold),
        path=path,
        version=hashlib.sha256(raw).hexdigest()[:12],
        loaded_at=datetime.now(timezone.utc),
    )


def load_rules_config(path: Path) -> RulesConfig:
    """Read and validate a complete rules YAML file. Raises OSError / ValueError."""
    path = Path(path)
    return _parse(path.read_bytes(), path)


class RulesConfigProvider:
    """Thread-safe holder of the active config snapshot with content-hash hot reload.

    `model_threshold` is the loaded artifact's tuned threshold; it is used to resolve
    and validate `challenge_prob: model_threshold` at load/reload time.
    """

    def __init__(self, path: Path, model_threshold: Optional[float] = None) -> None:
        self._path = Path(path)
        self._model_threshold = model_threshold
        self._lock = threading.Lock()
        raw = self._path.read_bytes()  # first load: errors propagate (fatal at startup)
        self._snapshot = _build_snapshot(raw, self._path, model_threshold)
        self._seen = _digest(raw)
        logger.info("Rules config loaded from %s (version %s)", self._path, self._snapshot.version)

    def get(self) -> RulesConfig:
        """Return the active config, reloading first if the file changed."""
        return self.snapshot().config

    def snapshot(self) -> ConfigSnapshot:
        """Return the active snapshot, reloading first if the file content changed."""
        if _digest(_read_config(self._path)) == self._seen:
            return self._snapshot
        with self._lock:
            # Re-read under the lock: a stale pre-lock read must never roll _seen back.
            raw = _read_config(self._path)
            digest = _digest(raw)
            if digest != self._seen:
                self._reload(raw, digest)
            return self._snapshot

    def _reload(self, raw: Optional[bytes], digest: Optional[str]) -> None:
        """Swap in a snapshot built from `raw`, or keep the last good one. Lock held."""
        self._seen = digest
        try:
            if raw is None:
                raise OSError(f"{self._path} is missing or unreadable")
            new_snapshot = _build_snapshot(raw, self._path, self._model_threshold)
        except (OSError, ValueError) as exc:
            logger.error(
                "Rules config reload failed; keeping version %s. Error: %s",
                self._snapshot.version,
                exc,
            )
            return
        self._snapshot = new_snapshot
        logger.info("Rules config reloaded from %s (version %s)", self._path, new_snapshot.version)
