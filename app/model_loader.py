import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

REQUIRED_KEYS = {"model", "threshold", "feature_names", "version", "trained_at", "n_train_rows"}


@dataclass(frozen=True)
class ModelArtifact:
    model: object
    threshold: float
    feature_names: list[str]
    version: str
    trained_at: str
    n_train_rows: int


def load_model(path: Path) -> ModelArtifact:
    logger.info("Loading model from %s", path)
    artifact_dict: dict = joblib.load(path)

    missing = REQUIRED_KEYS - artifact_dict.keys()
    if missing:
        raise ValueError(f"Model artifact missing required keys: {missing}")

    artifact = ModelArtifact(
        model=artifact_dict["model"],
        threshold=float(artifact_dict["threshold"]),
        feature_names=list(artifact_dict["feature_names"]),
        version=str(artifact_dict["version"]),
        trained_at=str(artifact_dict["trained_at"]),
        n_train_rows=int(artifact_dict["n_train_rows"]),
    )
    logger.info(
        "Model loaded: version=%s threshold=%.4f features=%d n_train_rows=%d",
        artifact.version,
        artifact.threshold,
        len(artifact.feature_names),
        artifact.n_train_rows,
    )
    return artifact


def predict(artifact: ModelArtifact, features: dict) -> tuple[float, bool]:
    values = [features[name] for name in artifact.feature_names]
    array = np.array(values, dtype=np.float64).reshape(1, -1)
    prob: float = float(artifact.model.predict_proba(array)[0][1])
    return (round(prob, 6), prob >= artifact.threshold)
