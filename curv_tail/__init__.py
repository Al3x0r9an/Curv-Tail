"""CURV-TAIL: curvature-adaptive origin-tangent Lorentz temporal CNN for
long-tailed encrypted traffic classification.

Public API
----------
``build_model(model_cfg, num_classes, max_packets)``
    Instantiate the multi-view CURV-TAIL model from a model-config mapping.
    This is the single entry point used by the training/evaluation CLI.

The package layout mirrors a standard research template:

- ``curv_tail.lorentz``   Hyperbolic (Lorentz) geometric primitives.
- ``curv_tail.models``    Network definition (discrete length tokenization,
                          byte stem, origin-tangent temporal blocks, Lorentz
                          prototype classifier, multi-view fusion).
- ``curv_tail.data``      Cache readers / datasets and the long-tail
                          head/body/tail grouping used for evaluation.
- ``curv_tail.metrics``   Standard + long-tail-aware metric computation.
- ``curv_tail.config``    YAML config loading and validation.
- ``curv_tail.train``     Training / evaluation CLI (``curv-train``).
- ``curv_tail.utils``     RNG seeding, JSONL logging, checkpoint I/O helpers.

Only the CURV-TAIL method itself is implemented here, plus its Euclidean twin
(``geometry: euclidean``, kept as a geometry ablation); the comparison methods
of the paper (pretrained / self-supervised traffic encoders and other baseline
families) are intentionally not included.
"""

from .models import build_model

__all__ = ["build_model"]

__version__ = "1.0.0"
