"""Configuration loading and validation for CURV-TAIL.

A configuration is a single YAML file with the top-level sections

* ``experiment`` -- run identity (name, seed, output root),
* ``data``       -- data-cache / split location and expected sizes,
* ``model``      -- CURV-TAIL model hyper-parameters (see
  :class:`curv_tail.models.MultiViewLorentzTrafficCNN`),
* ``training``   -- schedule, batch size, mixed precision, etc.

The supervised loss is always cross-entropy (optionally joined by the masked
length-token reconstruction of ``model.rec_weight``), and the optimizer is
always AdamW; neither is configurable in this release.

Validation here is intentionally light: it checks structure and that the
mandatory data paths / sizes are present.  Reproducibility is instead
guaranteed by a content hash (:func:`config_hash`) that is stamped into every
checkpoint and run directory, so a checkpoint can only be resumed / tested
against an identical configuration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_config", "validate_config", "config_hash", "public_config"]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to a ``.yaml`` configuration.

    Returns:
        The parsed configuration dict, with an extra private key
        ``"_config_path"`` holding the resolved source path.

    Raises:
        TypeError: If the YAML root is not a mapping.
        ValueError: If required keys are missing or invalid (see
            :func:`validate_config`).
    """
    config_path = Path(path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise TypeError(f"configuration must be a mapping: {config_path}")
    cfg["_config_path"] = str(config_path)
    validate_config(cfg)
    return cfg


def public_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the configuration without private keys.

    Private keys (leading underscore), such as ``_config_path``, are load-time
    bookkeeping.  They are excluded from the content hash and from everything
    persisted (``resolved_config.yaml``, checkpoint payloads).

    Args:
        cfg: Parsed configuration mapping.

    Returns:
        A shallow copy of ``cfg`` with private keys removed.
    """
    return {key: value for key, value in cfg.items() if not str(key).startswith("_")}


def validate_config(cfg: dict[str, Any]) -> None:
    """Structural validation of a loaded configuration.

    Ensures the experiment / data / model / training sections exist and are
    mappings, that the expected sample count is positive and more than one class
    is declared, and that the mandatory data files are declared.

    Args:
        cfg: Parsed configuration mapping.

    Raises:
        ValueError: If any mandatory key is missing or invalid.
    """
    for section in ("experiment", "data", "model", "training"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"missing or invalid '{section}' section")
    experiment = cfg["experiment"]
    data = cfg["data"]
    model = cfg["model"]
    training = cfg["training"]
    if not experiment.get("name"):
        raise ValueError("experiment.name is required")
    if int(data.get("expected_num_samples", 0)) <= 0:
        raise ValueError("data.expected_num_samples must be positive")
    if int(data.get("expected_num_classes", 0)) <= 1:
        raise ValueError("data.expected_num_classes must be greater than one")
    for key in ("manifest", "split", "cache_dir"):
        if not data.get(key):
            raise ValueError(f"missing data.{key}")
    if not data.get("input_view"):
        raise ValueError("data.input_view is required (packet_bytes_multiview)")
    if model.get("family") != "lorentz_origin_multiview":
        raise ValueError("model.family is required and must be 'lorentz_origin_multiview'")
    if not training.get("epochs"):
        raise ValueError("training.epochs is required")


def config_hash(cfg: dict[str, Any]) -> str:
    """Deterministic content hash of a configuration.

    Private keys (leading underscore) such as ``_config_path`` are excluded so
    that the same logical configuration always hashes to the same value even
    when read from a different location.

    Args:
        cfg: Parsed configuration mapping.

    Returns:
        Hex SHA-256 digest of the canonicalized configuration.
    """
    clean = public_config(cfg)
    blob = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
