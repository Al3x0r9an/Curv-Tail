"""Network definition of CURV-TAIL.

CURV-TAIL is a **double-branch (multi-view) temporal convolutional network
whose per-view features live on the Lorentz hyperboloid**.  The two views are

* the *packet / length view*  -- per-token *exact discrete* packet-length
  tokens (the size is recovered as ``expm1(|log1p(size)|)``; its sign supplies
  only the direction bit) joined with direction and ``log1p(inter-arrival)``
  channels, and
* the *bidirectional byte view*  -- raw payload bytes of the forward and
  backward directions, patch-embedded into fixed-size tokens.

Both branches are encoded by :class:`OriginTangentTemporalBlock` stacks (a
gated dilated Conv1d applied in the tangent space at the origin, i.e. after
``logmap_origin`` and before ``expmap_origin``).  The two pooled branch
representations are concatenated and fused, then read out by a
:class:`LorentzPrototypeClassifier` that scores each flow by its geodesic
distance to a trainable class prototype on the hyperboloid.  The curvature
``c`` is *learnable* and bounds ``c`` to ``[0.05, 2.0]``.

Length tokenization is detailed in :func:`discretize_length`.  For packets no
larger than 1503 bytes the token equals the exact size (one token per byte);
larger (coalesced) packets fall into coarse buckets indexed by ``coarse_scale``
(R4 grid; ``coarse_scale = 1424`` for the released recipe).  A small auxiliary
head (:attr:`MultiViewLorentzTrafficCNN.rec_head`) reconstructs the length-token
sequence from the packet branch and is trained with a masked cross-entropy term
(``rec_weight``) that acts as a self-consistency regularizer.

Only this method is implemented: the paper's comparison methods (pretrained /
self-supervised traffic encoders) and the other baseline families are
intentionally omitted, with the sole exception of the Euclidean twin under
``geometry: euclidean``, which is kept as an in-repo geometry ablation.

The module-level entry point is :func:`build_model`.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .lorentz import (
    Curvature,
    distance,
    expmap_origin,
    logmap_origin,
    manifold_error,
    reset_masked_to_origin,
)

__all__ = [
    "discretize_length",
    "PacketStem",
    "BytePatchStem",
    "OriginTangentTemporalBlock",
    "EuclideanTemporalBlock",
    "LorentzPrototypeClassifier",
    "masked_mean",
    "MultiViewLorentzTrafficCNN",
    "build_model",
]

# Single-frame packet-size ceiling.  Under the released byte-exact setting
# (length_block = 1) sizes <= 1503 bytes keep an *exact* token (token == size)
# and larger coalesced packets use coarse buckets (see the R4 grid).  A coarser
# length_block quantizes sizes first, so the identity no longer holds there.
_FINE_SIZE_CEILING = 1503


def discretize_length(
    signed_log1p: torch.Tensor,
    length_block: float,
    length_vocab: int,
    coarse_scale: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover the integer packet size from a signed log1p signal and tokenize it.

    The packet-size channel stores ``signed_log1p(size)``: the *sign* is the
    direction and ``expm1(|·|)`` recovers the exact byte size.  CURV-TAIL uses
    this channel to build an *exact discrete length token* (with one byte of
    precision) rather than feeding the continuous log signal to the network.

    Args:
        signed_log1p: Float tensor ``[..., T]`` of signed log1p packet sizes.
        length_block: Size quantization step; tokens come from the quantized
            size ``round(size / length_block)``, so ``token == size`` only when
            ``length_block == 1`` (the released byte-exact setting).
        length_vocab: Number of length tokens.
        coarse_scale: R4 coarse grid scale.  When ``> 1`` every size above the
            single-frame ceiling is mapped to a coarse bucket
            ``1503 + round((size + 1) / coarse_scale) - 1``, never below
            ``1504`` (bounded by the vocab).  When ``<= 1`` the legacy path is
            taken instead: the quantized size is simply clamped into
            ``[0, length_vocab - 1]``, so sizes up to ``length_vocab - 1`` keep
            their own token and only larger ones saturate at the top -- not used
            by the released recipe.

    Returns:
        ``(token, direction)`` where ``token`` has dtype ``torch.long`` and
        ``direction`` is ``1`` for a positive (forward) size and ``0`` otherwise.
    """
    raw_len = torch.exp(torch.abs(signed_log1p.float())) - 1.0
    # Round (not floor): sizes are stored as float32 log1p and expm1 lands within
    # ~1e-3 of the integer; floor() would silently merge ~36% of sizes to size-1.
    rounded = (raw_len / float(length_block)).round().long()
    if float(coarse_scale) > 1:
        # k = round((size + 1) / coarse_scale) recovers the aggregate grid index;
        # the +1 absorbs float round-trips that store k*coarse_scale occasionally
        # as k*coarse_scale - 1.  Grid levels k = 2..32 map to tokens 1504..1534.
        k = ((rounded.float() + 1.0) / float(coarse_scale)).round().long()
        token = torch.where(
            rounded > _FINE_SIZE_CEILING, _FINE_SIZE_CEILING + torch.clamp(k - 1, min=1), rounded
        )
        token = token.clamp(0, int(length_vocab) - 1)
    else:
        token = rounded.clamp(0, int(length_vocab) - 1)
    direction = (signed_log1p > 0).long()
    return token, direction


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling over the token axis honoring a validity mask.

    Args:
        x: Tensor ``[B, T, D]``.
        mask: Boolean ``[B, T]``, ``True`` for valid tokens.

    Returns:
        Masked mean ``[B, D]`` (division by the number of valid tokens).
    """
    weights = mask.unsqueeze(-1).to(x.dtype)
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class PacketStem(nn.Module):
    """Discrete length-token stem for the packet / length view.

    Turns each packet into a ``dim``-dimensional embedding by concatenating

    * an embedding of the *exact* discrete length token,
    * (optionally) an embedding of the direction, and
    * a linear projection of ``log1p(inter-arrival time)``,

    projecting the concatenation up to ``dim``, adding a learned positional
    bias, and normalizing.  IAT stays a *continuous* channel on purpose: length
    is discrete (protocol-meaningful) while inter-arrival time is not.

    Args:
        input_dim: Number of continuous channels in the raw sequence (2: packet
            size + IAT).  Kept for interface compatibility; the discrete branch
            only consumes channel 0 (length) and channel 1 (IAT).
        dim: Model width.
        max_packets: Maximum sequence length (drives the positional bias size).
        dropout: Dropout probability applied after normalization.
        length_vocab: Size of the length-token vocabulary.
        length_block: Length quantization step (1 = byte-exact).
        length_embed_dim: Width of the length/direction embeddings.
        use_direction: Whether to add a direction embedding.
        use_position: Whether to add the learned positional bias.
        coarse_scale: R4 coarse grid scale (see :func:`discretize_length`).
    """

    def __init__(
        self,
        input_dim: int,
        dim: int,
        max_packets: int,
        dropout: float,
        length_vocab: int = 384,
        length_block: int = 4,
        length_embed_dim: int = 16,
        use_direction: bool = True,
        use_position: bool = True,
        coarse_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.coarse_scale = float(coarse_scale)
        self.length_block = float(length_block)
        self.length_vocab = int(length_vocab)
        self.length_embedding = nn.Embedding(int(length_vocab), int(length_embed_dim))
        parts = 1
        if use_direction:
            self.direction_embedding = nn.Embedding(2, int(length_embed_dim))
            parts += 1
        else:
            self.direction_embedding = None
        self.iat_proj = nn.Linear(1, int(length_embed_dim))
        parts += 1
        self.combined_proj = nn.Linear(parts * int(length_embed_dim), dim)
        if use_position:
            self.position = nn.Parameter(torch.zeros(1, max_packets, dim))
            nn.init.trunc_normal_(self.position, std=0.02)
        else:
            self.position = None
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Embed a packet sequence.

        Args:
            x: Float ``[B, T, 2]`` (``[signed_log1p_size, log1p_iat]``).
            mask: Boolean ``[B, T]`` validity mask.

        Returns:
            ``[B, T, dim]`` discrete-length embeddings, masked to zero at padding.
        """
        token, direction = discretize_length(x[..., 0], self.length_block, self.length_vocab, self.coarse_scale)
        parts = [self.length_embedding(token)]
        if self.direction_embedding is not None:
            parts.append(self.direction_embedding(direction))
        parts.append(self.iat_proj(x[..., 1:2].float()))
        h = self.combined_proj(torch.cat(parts, dim=-1))
        if self.position is not None:
            h = h + self.position[:, : x.shape[1]]
        h = self.dropout(F.gelu(self.norm(h)))
        return h * mask.unsqueeze(-1).to(h.dtype)


class OriginTangentTemporalBlock(nn.Module):
    """Origin-tangent dilated temporal convolution on the hyperboloid.

    The *origin-tangent* contract: at every block the manifold points are
    mapped to the tangent space at the origin with ``logmap_origin``, an ordinary
    dilated ``Conv1d`` (with LayerNorm + GELU + dropout) is applied *in tangent
    space*, and the result is mapped back with ``expmap_origin`` using a gated
    residual addition.  Because the convolution kernel is shared over positions,
    this stays a translation-equivariant temporal model while the state itself
    lives on the manifold.

    Args:
        dim: Model width (also the manifold spatial dimension).
        kernel_size: Temporal kernel width (odd).
        dropout: Dropout applied after the LayerNorm.
        max_norm: Tangent norm cap applied inside ``expmap_origin``.
        dilation: Dilation of the Conv1d (the recipe cycles 1, 2, 4).
    """

    def __init__(self, dim: int, kernel_size: int, dropout: float, max_norm: float, dilation: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            dim, dim, kernel_size, padding=(kernel_size // 2) * int(dilation), dilation=int(dilation)
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(-2.1972246))
        self.max_norm = float(max_norm)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Apply one origin-tangent temporal update.

        Args:
            x: Manifold points ``[B, T, dim + 1]`` on H^d_c.
            mask: Boolean ``[B, T]`` validity mask.
            c: Curvature scalar tensor.

        Returns:
            Updated manifold points ``[B, T, dim + 1]`` (padding reset to origin).
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            tangent = logmap_origin(x.float(), c)
            update = self.conv(tangent.transpose(1, 2)).transpose(1, 2)
            update = self.dropout(F.gelu(self.norm(update)))
            out = expmap_origin(tangent + torch.sigmoid(self.gate.float()) * update, c, self.max_norm)
            return reset_masked_to_origin(out, mask, c)


class EuclideanTemporalBlock(nn.Module):
    """Gated residual dilated Conv1d in Euclidean space.

    This is the *Euclidean twin* of :class:`OriginTangentTemporalBlock`: the
    same dilated-conv schedule (LayerNorm + GELU + dropout + gated residual) run
    directly on the stem features with no logmap / expmap round trip.  It is used
    by the ``geometry: euclidean`` ablation so CURV-TAIL can be compared against
    an otherwise identical model whose only difference is the geometry.

    Args:
        dim: Model width.
        kernel_size: Temporal kernel width.
        dilation: Dilation.
        dropout: Dropout probability.
    """

    def __init__(self, dim: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = dilation * (kernel_size // 2)
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=pad, dilation=dilation)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(-2.1972246))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply one residual dilated-conv update in Euclidean space."""
        update = self.conv(x.transpose(1, 2)).transpose(1, 2)
        update = self.dropout(F.gelu(self.norm(update)))
        out = x + torch.sigmoid(self.gate) * update
        return out * mask.unsqueeze(-1).to(out.dtype)


class BytePatchStem(nn.Module):
    """Bidirectional byte patch stem for the payload byte view.

    Raw bytes of the forward and backward directions are each patch-embedded
    with a ``Conv1d`` over byte embeddings (257 vocabulary: 0 = padding, bytes
    shifted by +1), the two directions are concatenated, a direction embedding
    is added per stream, then a learned positional bias and normalization are
    applied.

    Args:
        dim: Model width (output token width).
        byte_embed_dim: Width of the per-byte embedding.
        max_bytes: Maximum bytes kept per direction (must divide ``patch_size``).
        patch_size: Number of bytes merged into one token by the Conv1d stride.
        dropout: Dropout probability.

    Raises:
        ValueError: If ``max_bytes`` is not divisible by ``patch_size``.

    Attributes:
        patch_size: Bytes per patch.
    """

    def __init__(
        self, *, dim: int, byte_embed_dim: int, max_bytes: int, patch_size: int, dropout: float
    ) -> None:
        super().__init__()
        if max_bytes % patch_size:
            raise ValueError("max_bytes must be divisible by patch_size")
        self.patch_size = int(patch_size)
        self.embedding = nn.Embedding(257, byte_embed_dim, padding_idx=0)
        self.patch = nn.Conv1d(byte_embed_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.position = nn.Parameter(torch.zeros(1, 2 * (max_bytes // patch_size), dim))
        self.direction = nn.Embedding(2, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.position, std=0.02)

    def _stream(self, tokens: torch.Tensor, direction: int) -> tuple[torch.Tensor, torch.Tensor]:
        token_mask = tokens.ne(0)
        embedded = self.embedding(tokens).transpose(1, 2)
        patches = self.patch(embedded).transpose(1, 2)
        patch_mask = F.max_pool1d(
            token_mask.to(patches.dtype).unsqueeze(1), self.patch_size, self.patch_size
        ).squeeze(1).bool()
        direction_ids = torch.full(
            (tokens.shape[0], patches.shape[1]), direction, dtype=torch.long, device=tokens.device
        )
        patches = patches + self.direction(direction_ids)
        return patches, patch_mask

    def forward(self, fwd: torch.Tensor, bwd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed the two byte streams.

        Args:
            fwd: ``[B, max_bytes]`` int64 forward byte tokens (0 = padding).
            bwd: ``[B, max_bytes]`` int64 backward byte tokens (0 = padding).

        Returns:
            ``(h, mask)`` where ``h`` is ``[B, 2 * (max_bytes // patch_size), dim]``
            and ``mask`` marks valid patches.
        """
        fwd_h, fwd_mask = self._stream(fwd, 0)
        bwd_h, bwd_mask = self._stream(bwd, 1)
        h = torch.cat([fwd_h, bwd_h], dim=1)
        mask = torch.cat([fwd_mask, bwd_mask], dim=1)
        h = h + self.position[:, : h.shape[1]]
        h = self.dropout(F.gelu(self.norm(h)))
        return h * mask.unsqueeze(-1).to(h.dtype), mask


class LorentzPrototypeClassifier(nn.Module):
    """Geodesic prototype classifier on the hyperboloid.

    Class prototypes live on H^d_c.  A query (pooled flow) is expmapped onto the
    manifold and scored by the *squared geodesic distance* to every prototype,
    scaled by a learned temperature:

    .. code-block:: text

        logits = -d_H(query, proto_c)^2 / temperature + bias

    Args:
        dim: Manifold spatial dimension.
        num_classes: Number of prototypes (classes).
        temperature: Initial softmax temperature (learned, clamped to [0.05, 20]).
        max_norm: Tangent norm cap used when lifting the prototypes.

    Attributes:
        prototypes: Trainable ``nn.Parameter [num_classes, dim]`` (Euclidean
            coordinates lifted to the manifold in ``forward``).
        bias: Per-class bias ``nn.Parameter [num_classes]``.
        log_temperature: Learned log-temperature parameter.
    """

    def __init__(self, dim: int, num_classes: int, temperature: float, max_norm: float) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(num_classes, dim))
        nn.init.normal_(self.prototypes, std=0.02)
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(float(temperature))))
        self.max_norm = float(max_norm)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Compute logits for pooled flow features.

        Args:
            z: Manifold query points ``[B, dim + 1]`` (already expmapped).
            c: Curvature scalar tensor.

        Returns:
            Logits ``[B, num_classes]``.
        """
        with torch.autocast(device_type=z.device.type, enabled=False):
            prototypes = expmap_origin(self.prototypes.float(), c, self.max_norm)
            dist = distance(z.float().unsqueeze(1), prototypes.unsqueeze(0), c)
            temperature = torch.clamp(self.log_temperature.float().exp(), min=0.05, max=20.0)
            return -dist.square() / temperature + self.bias.float()


class MultiViewLorentzTrafficCNN(nn.Module):
    """Multi-view origin-tangent Lorentz traffic CNN (the CURV-TAIL method).

    Double-branch design (both branches always active):

    * **Packet / length branch** -- discrete length-token embeddings from the
      packet-size / IAT channel, lifted to the hyperboloid and refined by a
      stack of :class:`OriginTangentTemporalBlock`.  Its representation is
      pooled with learned attention (``packet_pooling: attention``, the released
      setting) or with a masked mean, and (optionally) a masked length-token
      reconstruction auxiliary head is attached.
    * **Byte branch** -- bidirectional payload bytes, patch embedded and
      refined by the same block stack, pooled with masked mean.

    The two pooled vectors are fused with ``Linear(2*dim -> dim) + LN + GELU +
    dropout``, and the fused feature is read out by the Lorentz prototype
    classifier (or by a plain linear head when ``geometry == "euclidean"``).

    Args:
        cfg: Model configuration mapping (see notes below for keys).
        num_classes: Number of classes (prototypes).
        max_packets: Maximum packet sequence length.

    Model-config keys understood (with defaults):
        family (must be ``lorentz_origin_multiview``), temporal (``conv``),
        view (``multiview``), geometry (``hyperbolic`` | ``euclidean``),
        dim, depth, kernel_size, dilation_cycle, dropout, input_dim,
        length_embed (True), length_vocab, coarse_scale, length_block,
        length_embed_dim, use_direction, use_position, rec_weight,
        byte_embed_dim, byte_patch_size, max_bytes_per_direction,
        packet_pooling (``attention`` | ``mean``), curvature, learnable_curvature,
        min_curvature, max_curvature, lift_scale, max_tangent_norm,
        prototype_temperature.

    For the released recipe the model/training blocks are identical across both
    datasets; only the ``data`` section of the config differs.

    Raises:
        ValueError: If ``family`` / ``temporal`` / ``view`` / ``length_embed`` /
            ``packet_pooling`` / ``geometry`` hold an unsupported value.
    """

    def __init__(self, cfg: dict[str, Any], num_classes: int, max_packets: int) -> None:
        super().__init__()
        dim = int(cfg["dim"])
        dropout = float(cfg["dropout"])
        self.max_norm = float(cfg.get("max_tangent_norm", 6.0))

        # -- method identity guards -------------------------------------------
        if str(cfg.get("family", "lorentz_origin_multiview")) != "lorentz_origin_multiview":
            raise ValueError("CURV-TAIL only supports family=lorentz_origin_multiview")
        if str(cfg.get("temporal", "conv")) != "conv":
            raise ValueError("CURV-TAIL only supports temporal=conv (origin-tangent blocks)")
        self.view = str(cfg.get("view", "multiview"))
        if self.view != "multiview":
            raise ValueError("CURV-TAIL releases only the double-branch multiview view")
        if not bool(cfg.get("length_embed", True)):
            raise ValueError("CURV-TAIL requires length_embed=True (exact discrete length tokens)")

        # -- readout pooling ---------------------------------------------------
        self.packet_pooling = str(cfg.get("packet_pooling", "attention"))
        if self.packet_pooling not in {"attention", "mean"}:
            raise ValueError(f"unsupported packet_pooling: {self.packet_pooling}")
        self.packet_query = (
            nn.Parameter(torch.randn(dim) * 0.02) if self.packet_pooling == "attention" else None
        )

        # -- discrete length tokenization --------------------------------------
        self.length_vocab = int(cfg.get("length_vocab", 384))
        self.length_block = float(cfg.get("length_block", 4))
        self.coarse_scale = float(cfg.get("coarse_scale", 0.0))

        # -- geometry: hyperbolic (method) or euclidean (twin ablation) --------
        self.geometry = str(cfg.get("geometry", "hyperbolic"))
        if self.geometry not in {"hyperbolic", "euclidean"}:
            raise ValueError(f"unsupported geometry: {self.geometry}")
        self.euclidean = self.geometry == "euclidean"

        # -- packet / length branch ---------------------------------------------
        self.packet_stem = PacketStem(
            int(cfg["input_dim"]),
            dim,
            max_packets,
            dropout,
            length_vocab=self.length_vocab,
            length_block=self.length_block,
            length_embed_dim=int(cfg.get("length_embed_dim", 16)),
            use_direction=bool(cfg.get("use_direction", True)),
            use_position=bool(cfg.get("use_position", True)),
            coarse_scale=self.coarse_scale,
        )
        self.rec_weight = float(cfg.get("rec_weight", 0.0))
        self.rec_head = nn.Linear(dim, self.length_vocab) if self.rec_weight > 0 else None

        # -- byte branch ---------------------------------------------------------
        self.byte_stem = BytePatchStem(
            dim=dim,
            byte_embed_dim=int(cfg.get("byte_embed_dim", 24)),
            max_bytes=int(cfg["max_bytes_per_direction"]),
            patch_size=int(cfg.get("byte_patch_size", 8)),
            dropout=dropout,
        )

        # -- curvature (shared) + per-branch lift scales -------------------------
        self.curvature = Curvature(
            float(cfg.get("curvature", 1.0)),
            bool(cfg.get("learnable_curvature", False)),
            float(cfg.get("min_curvature", 0.05)),
            float(cfg.get("max_curvature", 2.0)),
        )
        lift = min(max(float(cfg.get("lift_scale", 0.1)), 1e-4), 1 - 1e-4)
        lift_raw = math.log(lift / (1 - lift))
        self.packet_lift_raw = nn.Parameter(torch.tensor(lift_raw, dtype=torch.float32))
        self.byte_lift_raw = nn.Parameter(torch.tensor(lift_raw, dtype=torch.float32))

        # -- per-view dilated convolution stacks ----------------------------------
        dilations = list(cfg.get("dilation_cycle", [1]))
        if self.euclidean:
            block_cls = EuclideanTemporalBlock
        else:
            block_cls = OriginTangentTemporalBlock

        def _stack() -> nn.ModuleList:
            return nn.ModuleList(
                [
                    (block_cls(dim, int(cfg["kernel_size"]), dropout, self.max_norm, int(dilations[i % len(dilations)]))
                     if block_cls is OriginTangentTemporalBlock
                     else block_cls(dim, int(cfg["kernel_size"]), int(dilations[i % len(dilations)]), dropout))
                    for i in range(int(cfg["depth"]))
                ]
            )

        self.packet_blocks = _stack()
        self.byte_blocks = _stack()
        self.packet_norm = nn.LayerNorm(dim)
        self.byte_norm = nn.LayerNorm(dim)

        # -- fusion + classifier --------------------------------------------------
        self.fusion = nn.Sequential(nn.Linear(2 * dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout))
        if self.euclidean:
            self.classifier = nn.Linear(dim, num_classes)
        else:
            self.classifier = LorentzPrototypeClassifier(
                dim, num_classes, float(cfg.get("prototype_temperature", 1.0)), self.max_norm
            )

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        fwd_bytes: torch.Tensor,
        bwd_bytes: torch.Tensor,
        return_aux: bool = False,
    ):
        """Classify one batch of multi-view flows.

        Args:
            x: Packet features ``[B, T, 2]``
                (``[signed_log1p_packet_size, log1p_iat]``).
            mask: Packet validity ``[B, T]``.
            fwd_bytes: Forward byte tokens ``[B, max_bytes]``.
            bwd_bytes: Backward byte tokens ``[B, max_bytes]``.
            return_aux: If ``True`` return ``(logits, aux)``.  ``aux`` always
                carries the pooled embedding, the learned curvature and the two
                per-branch lift scales; ``rec_loss`` is added when the model has
                a reconstruction head, and ``manifold_error_mean`` /
                ``manifold_error_max`` only under hyperbolic geometry.

        Returns:
            Logits ``[B, num_classes]``, or ``(logits, aux)`` when
            ``return_aux=True``.
        """
        c = self.curvature()
        aux: dict[str, Any] = {"curvature": c.detach()}
        with torch.autocast(device_type=x.device.type, enabled=False):
            # ---- packet / length branch ---------------------------------------
            packet_h = self.packet_stem(x, mask)
            packet_lift = torch.sigmoid(self.packet_lift_raw.float())
            if self.euclidean:
                packet_tangent = packet_h.float()
                for block in self.packet_blocks:
                    packet_tangent = block(packet_tangent, mask)
                packet_tangent = self.packet_norm(packet_tangent)
            else:
                packet_m = reset_masked_to_origin(
                    expmap_origin(packet_lift * packet_h.float(), c, self.max_norm), mask, c
                )
                for block in self.packet_blocks:
                    packet_m = block(packet_m, mask, c)
                packet_tangent = self.packet_norm(logmap_origin(packet_m, c))
            if self.packet_pooling == "attention":
                scores = torch.einsum("bld,d->bl", packet_tangent, self.packet_query)
                scores = scores.masked_fill(~mask, float("-inf"))
                weights = torch.softmax(scores, dim=-1)
                packet_z = torch.einsum("bl,bld->bd", weights, packet_tangent)
            else:
                packet_z = masked_mean(packet_tangent, mask)
            if self.rec_head is not None:
                length_token, _ = discretize_length(
                    x[..., 0], self.length_block, self.length_vocab, self.coarse_scale
                )
                rec_logits = self.rec_head(packet_tangent)
                ce = F.cross_entropy(
                    rec_logits.reshape(-1, self.length_vocab),
                    length_token.reshape(-1),
                    reduction="none",
                )
                rec_loss = (ce.reshape(length_token.shape) * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
                aux["rec_loss"] = self.rec_weight * rec_loss
            aux["packet_lift_scale"] = packet_lift.detach()

            # ---- byte branch ---------------------------------------------------
            byte_h, byte_mask = self.byte_stem(fwd_bytes, bwd_bytes)
            byte_lift = torch.sigmoid(self.byte_lift_raw.float())
            if self.euclidean:
                byte_feat = byte_h.float()
                for block in self.byte_blocks:
                    byte_feat = block(byte_feat, byte_mask)
                byte_z = masked_mean(self.byte_norm(byte_feat), byte_mask)
            else:
                byte_m = reset_masked_to_origin(
                    expmap_origin(byte_lift * byte_h.float(), c, self.max_norm), byte_mask, c
                )
                for block in self.byte_blocks:
                    byte_m = block(byte_m, byte_mask, c)
                byte_z = masked_mean(self.byte_norm(logmap_origin(byte_m, c)), byte_mask)
            aux["byte_lift_scale"] = byte_lift.detach()

            # ---- fusion + readout ----------------------------------------------
            pooled_tangent = self.fusion(torch.cat([packet_z, byte_z], dim=-1))
            if self.euclidean:
                logits = self.classifier(pooled_tangent)
            else:
                errors = torch.cat(
                    [manifold_error(packet_m, c).flatten(), manifold_error(byte_m, c).flatten()]
                )
                aux["manifold_error_mean"] = errors.mean().detach()
                aux["manifold_error_max"] = errors.max().detach()
                pooled = expmap_origin(pooled_tangent, c, self.max_norm)
                logits = self.classifier(pooled, c)
            aux["embedding"] = pooled_tangent
        return (logits, aux) if return_aux else logits


def build_model(model_cfg: dict[str, Any], num_classes: int, max_packets: int) -> nn.Module:
    """Build a CURV-TAIL model from a model-config mapping.

    Args:
        model_cfg: The ``model`` section of a YAML config (see
            :class:`MultiViewLorentzTrafficCNN` for the supported keys).
        num_classes: Number of classes.
        max_packets: Maximum packet sequence length.

    Returns:
        An initialized :class:`MultiViewLorentzTrafficCNN`.

    Raises:
        KeyError: If ``model_cfg`` has no ``family`` key.
        ValueError: If ``family`` is not ``lorentz_origin_multiview``.
    """
    if str(model_cfg["family"]) != "lorentz_origin_multiview":
        raise ValueError(
            f"unsupported model family for CURV-TAIL: {model_cfg['family']} "
            "(expected lorentz_origin_multiview)"
        )
    return MultiViewLorentzTrafficCNN(model_cfg, num_classes, max_packets)
