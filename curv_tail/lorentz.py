"""Hyperbolic geometry primitives on the (Lorentz) hyperboloid model.

CURV-TAIL represents flow encodings on the hyperboloid

    H^d_c = { x in R^{d+1} : <x,x>_L = -1/c, x_0 > 0 },

where the Lorentzian (Minkowski) inner product is

    <x, y>_L = -x_0 y_0 + sum_{i=1..d} x_i y_i,

and c > 0 is the *curvature*.  All operations here are real-valued and
numerically guarded so they can be called under ``torch.autocast`` or on the
raw float tensors produced by the model.

The three operations the network actually needs are:

* :func:`expmap_origin`    inject a tangent (Euclidean) vector into H^d_c at
                           the origin, the map that lifts neural features onto
                           the manifold;
* :func:`logmap_origin`    project a point of H^d_c back to the tangent space
                           at the origin (the inverse of ``expmap_origin``, up
                           to the tangent-norm cap that ``expmap`` applies);
* :func:`distance`         geodesic distance between two manifold points,
                           used by the Lorentz prototype classifier.

Padding handling is done with :func:`reset_masked_to_origin`, which places
masked (padding) tokens at the manifold origin so they carry no signal.

All functions broadcast over leading batch dimensions and are written to be
*differentiable* with respect to their float tensor arguments.
"""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = [
    "minkowski_dot",
    "origin_like",
    "project_to_tangent",
    "tangent_norm",
    "clip_tangent",
    "expmap",
    "logmap",
    "distance",
    "expmap_origin",
    "logmap_origin",
    "manifold_error",
    "reset_masked_to_origin",
    "Curvature",
]


def minkowski_dot(x: torch.Tensor, y: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    """Lorentzian inner product ``<x, y>_L`` along the last dimension.

    Args:
        x: Tensor of shape ``[..., d + 1]`` (time coordinate first).
        y: Tensor of shape ``[..., d + 1]``.
        keepdim: If ``True`` keep the reduced dimension instead of squeezing it.

    Returns:
        ``-x_0 y_0 + sum_i x_i y_i`` of shape ``[...]`` (or ``[..., 1]`` when
        ``keepdim=True``).
    """
    value = -x[..., :1] * y[..., :1] + (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)
    return value if keepdim else value.squeeze(-1)


def origin_like(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Manifold origin broadcast to the shape of ``x``.

    The origin of H^d_c is ``(1/sqrt(c), 0, ..., 0)``.

    Args:
        x: Reference tensor (only its shape, dtype and device are used).
        c: Positive scalar curvature.

    Returns:
        Tensor with the same shape/dtype as ``x`` whose spatial part is zero.
    """
    out = torch.zeros_like(x)
    out[..., 0] = torch.rsqrt(c)
    return out


def project_to_tangent(x: torch.Tensor, u: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Orthogonal projection of ``u`` onto the tangent space ``T_x H^d_c``.

    Args:
        x: Point on the hyperboloid ``[..., d + 1]``.
        u: Vector to project ``[..., d + 1]``.
        c: Curvature.

    Returns:
        ``u + c <x, u>_L x``, the component of ``u`` tangent at ``x``.
    """
    return u + c * minkowski_dot(x, u, keepdim=True) * x


def tangent_norm(v: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Norm of a tangent vector: ``sqrt(<v, v>_L)`` (clamped positive).

    Args:
        v: Tangent vector ``[..., d + 1]``.
        eps: Lower clamp on the squared norm for numerical safety.

    Returns:
        Positive scalar norm per leading element, shape ``[..., 1]``.
    """
    return torch.sqrt(torch.clamp(minkowski_dot(v, v, keepdim=True), min=eps))


def clip_tangent(v: torch.Tensor, max_norm: float, eps: float = 1e-9) -> torch.Tensor:
    """Scale a tangent vector down to ``max_norm`` when it exceeds it.

    Args:
        v: Tangent vector ``[..., d + 1]``.
        max_norm: Radius cap.
        eps: Safety epsilon for the norm.

    Returns:
        ``v`` if its norm is within ``max_norm``, otherwise ``v / ||v|| * max_norm``.
    """
    norm = tangent_norm(v, eps=eps)
    scale = torch.clamp(float(max_norm) / norm, max=1.0)
    return v * scale


def expmap(x: torch.Tensor, v: torch.Tensor, c: torch.Tensor, max_norm: float = 8.0) -> torch.Tensor:
    """Exponential map at a general point ``x``: ``exp_x(v)``.

    Args:
        x: Base point on the hyperboloid ``[..., d + 1]``.
        v: Tangent vector at ``x`` ``[..., d + 1]``.
        c: Curvature.
        max_norm: Tangent vector norm cap applied before mapping.

    Returns:
        Point on H^d_c reached from ``x``.  ``v`` is first projected onto the
        tangent space at ``x`` and clipped to norm ``max_norm``, so the geodesic
        length actually traversed is ``||clip(project(v), max_norm)||`` -- equal
        to ``||v||`` only when ``v`` is already tangent and within the cap.
    """
    x32, v32, c32 = x.float(), v.float(), c.float()
    v32 = project_to_tangent(x32, v32, c32)
    v32 = clip_tangent(v32, max_norm=max_norm)
    norm = tangent_norm(v32)
    theta = torch.sqrt(c32) * norm
    theta_safe = torch.clamp(theta, min=1e-7)
    sinhc = torch.where(theta > 1e-6, torch.sinh(theta) / theta_safe, 1.0 + theta.square() / 6.0)
    out = torch.cosh(theta) * x32 + sinhc * v32
    # Re-orthogonalize the time coordinate so the point stays exactly on H^d_c.
    spatial = out[..., 1:]
    time = torch.sqrt(torch.clamp(torch.reciprocal(c32) + spatial.square().sum(dim=-1, keepdim=True), min=1e-9))
    return torch.cat([time, spatial], dim=-1)


def logmap(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Logarithmic map: the tangent vector at ``x`` pointing to ``y``.

    Args:
        x: Base point on the hyperboloid ``[..., d + 1]``.
        y: Target point on the hyperboloid ``[..., d + 1]``.
        c: Curvature.

    Returns:
        Tangent vector at ``x`` (orthogonally projected onto ``T_x H^d_c``)
        whose exponential map reaches ``y`` when ``y`` lies on the manifold.

        The cosine ``-c <x, y>_L`` is floored at ``1`` and the argument of
        ``acosh`` at ``1 + 1e-7``, while the ratio itself falls back to ``1``
        whenever the cosine is at most ``1 + 1e-6``.  For coincident points
        the cosine floors to exactly ``1``, so the numerator of the final
        update (``y - alpha * x``) vanishes and the result is exactly zero;
        for points that coincide only up to float32 round-off the cosine can
        land just above ``1``, giving a vector of order ``1e-6`` instead.
    """
    x32, y32, c32 = x.float(), y.float(), c.float()
    alpha = torch.clamp(-c32 * minkowski_dot(x32, y32, keepdim=True), min=1.0)
    delta = torch.clamp(alpha.square() - 1.0, min=1e-12)
    acosh = torch.acosh(torch.clamp(alpha, min=1.0 + 1e-7))
    factor = acosh / torch.sqrt(delta)
    factor = torch.where(alpha <= 1.0 + 1e-6, torch.ones_like(factor), factor)
    return project_to_tangent(x32, factor * (y32 - alpha * x32), c32)


def distance(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Geodesic distance between two points of the hyperboloid.

    Args:
        x: Point ``[..., d + 1]``.
        y: Point ``[..., d + 1]``.
        c: Curvature.

    Returns:
        ``d_H(x, y) = acosh(-c <x, y>_L) / sqrt(c)``, shape ``[...]``.  The
        Lorentzian product is floored at ``1 + 1e-7`` before the ``acosh``, so
        the self-distance ``distance(x, x)`` is a small positive floor
        (``acosh(1 + 1e-7) / sqrt(c)``) rather than exactly zero.
    """
    alpha = torch.clamp(-c.float() * minkowski_dot(x.float(), y.float()), min=1.0 + 1e-7)
    return torch.acosh(alpha) / torch.sqrt(c.float())


def expmap_origin(spatial: torch.Tensor, c: torch.Tensor, max_norm: float = 8.0) -> torch.Tensor:
    """Lift a Euclidean (tangent) feature vector onto H^d_c at the origin.

    This is the entry map of CURV-TAIL: a neural feature ``v in R^d`` is first
    treated as the spatial part of a tangent vector at the origin
    ``(0, v) in T_o H`` and then mapped onto the manifold.

    Args:
        spatial: Features of shape ``[..., d]`` (no time coordinate).
        c: Curvature.
        max_norm: Tangent norm cap applied before the exponential map.

    Returns:
        Point on H^d_c of shape ``[..., d + 1]``.
    """
    tangent = torch.cat([torch.zeros_like(spatial[..., :1]), spatial.float()], dim=-1)
    return expmap(origin_like(tangent, c.float()), tangent, c.float(), max_norm=max_norm)


def logmap_origin(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Project a manifold point back to the tangent space at the origin.

    Args:
        x: Point on H^d_c ``[..., d + 1]``.
        c: Curvature.

    Returns:
        Spatial tangent coordinates of shape ``[..., d]``.
    """
    return logmap(origin_like(x.float(), c.float()), x.float(), c.float())[..., 1:]


def manifold_error(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Deviation of ``x`` from the hyperboloid H^d_c (should be ~0).

    Used as a numerical monitoring signal, not a loss.

    Args:
        x: Point tensor ``[..., d + 1]``.
        c: Curvature.

    Returns:
        ``| <x, x>_L + 1/c |`` per element.
    """
    return torch.abs(minkowski_dot(x.float(), x.float()) + torch.reciprocal(c.float()))


def reset_masked_to_origin(x: torch.Tensor, mask: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Put padded positions back to the manifold origin.

    ``expmap_origin(0) = origin`` is the additive identity on H^d_c, so masked
    tokens contribute nothing to downstream pooling / convolutions.

    Args:
        x: Manifold tensor ``[B, L, d + 1]``.
        mask: Boolean ``[B, L]``, ``True`` for valid tokens.
        c: Curvature.

    Returns:
        ``x`` with every masked position replaced by the origin.
    """
    return torch.where(mask.unsqueeze(-1), x, origin_like(x, c))


class Curvature(nn.Module):
    """Learnable curvature constrained to an open interval ``(minimum, maximum)``.

    The raw parameter lives in logit space and is passed through a sigmoid so
    the effective curvature ``c`` is always in ``(minimum, maximum)``.  CURV-TAIL
    initializes ``c = 1.0`` and lets it adapt during training (bounded in
    ``[0.05, 2.0]``).

    Args:
        value: Initial curvature (clamped inside the valid interval).
        learnable: Whether the curvature is updated by gradient descent.
        minimum: Lower bound of the interval.
        maximum: Upper bound of the interval.

    Attributes:
        raw: The sigmoid-parameterized ``nn.Parameter`` (logit space).
        minimum: Lower bound (float, not a parameter).
        maximum: Upper bound (float, not a parameter).
    """

    def __init__(self, value: float, learnable: bool, minimum: float = 0.05, maximum: float = 2.0) -> None:
        super().__init__()
        if not 0 < minimum < maximum:
            raise ValueError("curvature bounds must satisfy 0 < minimum < maximum")
        value = min(max(float(value), minimum + 1e-6), maximum - 1e-6)
        ratio = (value - minimum) / (maximum - minimum)
        raw = math.log(ratio / (1.0 - ratio))
        self.raw = nn.Parameter(torch.tensor(raw, dtype=torch.float32), requires_grad=learnable)
        self.minimum = float(minimum)
        self.maximum = float(maximum)

    def forward(self) -> torch.Tensor:
        """Return the effective curvature (a scalar tensor)."""
        return self.minimum + (self.maximum - self.minimum) * torch.sigmoid(self.raw)
