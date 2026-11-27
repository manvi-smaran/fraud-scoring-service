"""Tests for app.rules.config: validation, loading, and hot reload."""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.rules.config import (
    DEFAULT_CONFIG_PATH,
    MODEL_THRESHOLD,
    RulesConfig,
    RulesConfigProvider,
    config_path_from_env,
    load_rules_config,
)
from tests.conftest import rules_yaml

ROOT = Path(__file__).resolve().parent.parent

VALID_YAML = """\
high_score_amount_prob: 0.9
high_score_amount_threshold: 2000.0
challenge_prob: 0.7
excessive_amount: 5000.0
excessive_amount_prob: 0.5
suspicious_hour_prob: 0.4
suspicious_hours: [0, 1, 2, 3, 4, 5]
"""


def _write(path: Path, text: str, bump_ns: int = 0) -> None:
    """Write text and force a distinct mtime so the change is always detected."""
    path.write_text(text)
    if bump_ns:
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + bump_ns))


def _bumped(path: Path, text: str, step: int) -> None:
    _write(path, text, bump_ns=step * 1_000_000_000)


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "rules.yaml"
    _write(path, VALID_YAML)
    return path


# ---------------------------------------------------------------- defaults


def test_defaults_match_previous_hard_coded_values() -> None:
    cfg = RulesConfig()
    assert cfg.high_score_amount_prob == 0.9
    assert cfg.high_score_amount_threshold == 2000.0
    assert cfg.challenge_prob == 0.7
    assert cfg.excessive_amount == 5000.0
    assert cfg.excessive_amount_prob == 0.5
    assert cfg.suspicious_hour_prob == 0.4
    assert cfg.suspicious_hours == (0, 1, 2, 3, 4, 5)


def test_shipped_yaml_equals_defaults() -> None:
    shipped = ROOT / "config" / "rules.yaml"
    assert DEFAULT_CONFIG_PATH == shipped
    assert load_rules_config(shipped) == RulesConfig()


def test_config_is_frozen() -> None:
    cfg = RulesConfig()
    with pytest.raises(ValidationError):
        cfg.challenge_prob = 0.1  # type: ignore[misc]


# -------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "overrides",
    [
        {"high_score_amount_prob": 1.01},
        {"high_score_amount_prob": -0.1},
        {"challenge_prob": 1.5},
        {"challenge_prob": "half"},
        {"challenge_prob": True},
        {"excessive_amount_prob": -0.01},
        {"suspicious_hour_prob": 2},
        {"high_score_amount_threshold": -1},
        {"excessive_amount": -5},
        {"suspicious_hours": [24]},
        {"suspicious_hours": [-1]},
        {"suspicious_hours": [1, 1]},
        {"unknown_key": 1},
        # cross-field: CHALLENGE cut-off must not exceed DECLINE cut-off
        {"challenge_prob": 0.95, "high_score_amount_prob": 0.9},
        # cross-field: high-amount threshold must not exceed excessive amount
        {"high_score_amount_threshold": 6000.0, "excessive_amount": 5000.0},
    ],
)
def test_rejects_invalid_values(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        RulesConfig(**overrides)


def test_accepts_boundary_values() -> None:
    cfg = RulesConfig(
        high_score_amount_prob=1.0,
        challenge_prob=0.0,
        excessive_amount_prob=0.0,
        suspicious_hour_prob=1.0,
        high_score_amount_threshold=0.0,
        suspicious_hours=[23, 0],
    )
    assert cfg.suspicious_hours == (0, 23)


def test_accepts_empty_suspicious_hours() -> None:
    assert RulesConfig(suspicious_hours=[]).suspicious_hours == ()


def test_accepts_model_threshold_sentinel() -> None:
    assert RulesConfig(challenge_prob="model_threshold").challenge_prob == MODEL_THRESHOLD


# --------------------------------------------------- challenge resolution


def test_resolve_numeric_challenge_prob_ignores_model() -> None:
    assert RulesConfig(challenge_prob=0.6).resolve_challenge_prob(0.95) == 0.6


def test_resolve_model_threshold_uses_artifact_value() -> None:
    cfg = RulesConfig(challenge_prob="model_threshold", high_score_amount_prob=0.97)
    assert cfg.resolve_challenge_prob(0.95) == 0.95


def test_resolve_model_threshold_rejects_value_above_decline_cutoff() -> None:
    cfg = RulesConfig(challenge_prob="model_threshold")  # high_score_amount_prob = 0.9
    with pytest.raises(ValueError, match="high_score_amount_prob"):
        cfg.resolve_challenge_prob(0.9500000000000001)


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
def test_resolve_model_threshold_rejects_out_of_range_model_value(bad: float) -> None:
    cfg = RulesConfig(challenge_prob="model_threshold")
    with pytest.raises(ValueError):
        cfg.resolve_challenge_prob(bad)


# ----------------------------------------------------------------- loading


def test_load_rules_config_reads_values(tmp_path: Path) -> None:
    path = tmp_path / "r.yaml"
    _write(path, VALID_YAML.replace("challenge_prob: 0.7", "challenge_prob: 0.65"))
    assert load_rules_config(path).challenge_prob == 0.65


def test_load_rules_config_rejects_partial_file(tmp_path: Path) -> None:
    path = tmp_path / "r.yaml"
    _write(path, "challenge_prob: 0.6\n")
    with pytest.raises(ValueError, match="missing required keys"):
        load_rules_config(path)


@pytest.mark.parametrize("text", ["", "   \n", "# only a comment\n"])
def test_load_rules_config_rejects_empty_file(tmp_path: Path, text: str) -> None:
    path = tmp_path / "r.yaml"
    _write(path, text)
    with pytest.raises(ValueError):
        load_rules_config(path)


def test_load_rules_config_accepts_complete_file(tmp_path: Path) -> None:
    path = tmp_path / "r.yaml"
    _write(path, rules_yaml(challenge_prob="model_threshold", high_score_amount_prob=0.97))
    assert load_rules_config(path).challenge_prob == MODEL_THRESHOLD


@pytest.mark.parametrize("text", ["- 1\n- 2\n", "just a string\n", "a: [unclosed\n"])
def test_load_rules_config_rejects_non_mapping_or_bad_yaml(tmp_path: Path, text: str) -> None:
    path = tmp_path / "r.yaml"
    _write(path, text)
    with pytest.raises(ValueError):
        load_rules_config(path)


def test_load_rules_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        load_rules_config(tmp_path / "nope.yaml")


def test_config_path_from_env_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RULES_CONFIG_PATH", raising=False)
    assert config_path_from_env() == DEFAULT_CONFIG_PATH


def test_config_path_from_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RULES_CONFIG_PATH", str(tmp_path / "x.yaml"))
    assert config_path_from_env() == tmp_path / "x.yaml"


# -------------------------------------------------------------- provider


def test_provider_first_load_invalid_raises(tmp_path: Path) -> None:
    path = tmp_path / "r.yaml"
    _write(path, "challenge_prob: 7\n")
    with pytest.raises(ValueError):
        RulesConfigProvider(path)


def test_provider_first_load_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        RulesConfigProvider(tmp_path / "missing.yaml")


def test_provider_hot_reloads_on_mtime_change(config_file: Path) -> None:
    provider = RulesConfigProvider(config_file)
    first = provider.get()
    assert first.challenge_prob == 0.7

    _bumped(config_file, VALID_YAML.replace("challenge_prob: 0.7", "challenge_prob: 0.55"), 1)
    second = provider.get()

    assert second.challenge_prob == 0.55
    assert first.challenge_prob == 0.7  # old object untouched (immutability)
    assert second is not first


def test_provider_returns_same_object_when_unchanged(config_file: Path) -> None:
    provider = RulesConfigProvider(config_file)
    assert provider.get() is provider.get()


def test_provider_snapshot_metadata_changes_on_reload(config_file: Path) -> None:
    provider = RulesConfigProvider(config_file)
    before = provider.snapshot()
    _bumped(config_file, VALID_YAML.replace("0.4", "0.45"), 1)
    after = provider.snapshot()
    assert after.config.suspicious_hour_prob == 0.45
    assert after.version != before.version
    assert after.loaded_at >= before.loaded_at
    assert after.path == config_file


def test_provider_keeps_last_good_on_invalid_edit(
    config_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    provider = RulesConfigProvider(config_file)
    good = provider.snapshot()

    with caplog.at_level(logging.ERROR, logger="app.rules.config"):
        _bumped(config_file, "challenge_prob: 1.5\n", 1)
        assert provider.get() is good.config
        assert provider.snapshot() == good

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1  # logged once per bad file version, not per request


def test_provider_keeps_last_good_on_unparseable_yaml(config_file: Path) -> None:
    provider = RulesConfigProvider(config_file)
    good = provider.get()
    _bumped(config_file, "challenge_prob: [oops\n", 1)
    assert provider.get() is good


def test_provider_keeps_last_good_when_file_deleted(config_file: Path) -> None:
    provider = RulesConfigProvider(config_file)
    good = provider.get()
    config_file.unlink()
    assert provider.get() is good


def test_provider_recovers_after_invalid_edit_is_fixed(config_file: Path) -> None:
    provider = RulesConfigProvider(config_file)
    _bumped(config_file, "challenge_prob: -3\n", 1)
    assert provider.get().challenge_prob == 0.7
    _bumped(config_file, rules_yaml(challenge_prob=0.33), 2)
    assert provider.get().challenge_prob == 0.33


def test_provider_is_thread_safe_under_concurrent_reload(config_file: Path) -> None:
    provider = RulesConfigProvider(config_file)
    seen: list[float] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def reader() -> None:
        try:
            for _ in range(200):
                value = provider.get().challenge_prob
                with lock:
                    seen.append(value)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(8)]
    for t in threads:
        t.start()
    _bumped(config_file, VALID_YAML.replace("challenge_prob: 0.7", "challenge_prob: 0.6"), 1)
    for t in threads:
        t.join()

    assert not errors
    assert set(seen) <= {0.7, 0.6}
    assert provider.get().challenge_prob == 0.6


# ------------------------------------------- partial / empty file on reload


@pytest.fixture
def tuned_file(tmp_path: Path) -> Path:
    """A config whose values differ from the defaults, so a silent reset is visible."""
    path = tmp_path / "tuned.yaml"
    _write(path, rules_yaml(challenge_prob=0.65, excessive_amount=4000.0))
    return path


@pytest.mark.parametrize("text", ["", "challenge_prob: 0.6\n", "high_score_amount_prob: 0.9\nhigh_"])
def test_provider_rejects_empty_or_partial_edit(tuned_file: Path, text: str) -> None:
    provider = RulesConfigProvider(tuned_file)
    good = provider.snapshot()
    _bumped(tuned_file, text, 1)
    assert provider.snapshot() is good
    assert provider.get().challenge_prob == 0.65
    assert provider.get().excessive_amount == 4000.0


def test_provider_first_load_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "r.yaml"
    _write(path, "")
    with pytest.raises(ValueError):
        RulesConfigProvider(path)


# --------------------------------------- effective challenge_prob in provider

ARTIFACT_THRESHOLD = 0.9500000000000001  # value shipped in models/model.joblib


def test_snapshot_effective_challenge_prob_numeric(config_file: Path) -> None:
    provider = RulesConfigProvider(config_file, model_threshold=ARTIFACT_THRESHOLD)
    assert provider.snapshot().effective_challenge_prob == 0.7


def test_snapshot_effective_challenge_prob_uses_model_threshold(tmp_path: Path) -> None:
    path = tmp_path / "r.yaml"
    _write(path, rules_yaml(challenge_prob="model_threshold", high_score_amount_prob=0.97))
    provider = RulesConfigProvider(path, model_threshold=ARTIFACT_THRESHOLD)
    assert provider.snapshot().effective_challenge_prob == ARTIFACT_THRESHOLD


def test_provider_startup_rejects_model_threshold_above_decline_cutoff(tmp_path: Path) -> None:
    path = tmp_path / "r.yaml"
    _write(path, rules_yaml(challenge_prob="model_threshold"))  # high_score_amount_prob 0.9
    with pytest.raises(ValueError, match="high_score_amount_prob"):
        RulesConfigProvider(path, model_threshold=ARTIFACT_THRESHOLD)


@pytest.mark.parametrize("bad", [None, -0.1, 1.1, float("nan")])
def test_provider_startup_rejects_unusable_model_threshold(tmp_path: Path, bad) -> None:
    path = tmp_path / "r.yaml"
    _write(path, rules_yaml(challenge_prob="model_threshold", high_score_amount_prob=1.0))
    with pytest.raises(ValueError):
        RulesConfigProvider(path, model_threshold=bad)


def test_provider_reload_rejects_model_threshold_above_decline_cutoff(config_file: Path) -> None:
    provider = RulesConfigProvider(config_file, model_threshold=ARTIFACT_THRESHOLD)
    good = provider.snapshot()
    _bumped(config_file, rules_yaml(challenge_prob="model_threshold"), 1)
    assert provider.snapshot() is good
    assert provider.snapshot().effective_challenge_prob == 0.7


# ------------------------------------------------ change detection / races


def test_provider_detects_edit_with_same_mtime_and_size(config_file: Path) -> None:
    provider = RulesConfigProvider(config_file)
    before = os.stat(config_file)
    edited = VALID_YAML.replace("challenge_prob: 0.7", "challenge_prob: 0.6")
    assert len(edited) == len(VALID_YAML)
    config_file.write_text(edited)
    os.utime(config_file, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert provider.get().challenge_prob == 0.6


def test_provider_stale_unlocked_read_does_not_regress(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.rules.config as config_module

    provider = RulesConfigProvider(config_file)
    _bumped(config_file, rules_yaml(challenge_prob=0.6), 1)
    current = provider.snapshot()
    assert current.config.challenge_prob == 0.6

    # Simulate a request that read the OLD file just before the edit landed and
    # only reaches the lock after another thread already loaded the new one.
    real_read = config_module._read_config
    stale = iter([VALID_YAML.encode()])
    monkeypatch.setattr(
        config_module, "_read_config", lambda path: next(stale, None) or real_read(path)
    )
    assert provider.snapshot() is current
    monkeypatch.undo()
    assert provider.snapshot() is current
