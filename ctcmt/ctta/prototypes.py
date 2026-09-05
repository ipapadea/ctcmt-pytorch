"""V4 per-class feature-prototype anchor for CT-CMT-Seg.

Matches ``CTCMT_Seg._proto_loss`` in the reference detectron2 meta-arch.

Per adapt step:
  1. Interpolate teacher softmax down to the feature resolution.
  2. For each class c: pixels where teacher confidence for c >= τ contribute
     to a class-c centroid ``z_c`` in the current student features.
  3. EMA-update ``proto[c]`` with ``z_c``.
  4. Loss = Σ_c (1 − cos(z_c, proto[c].detach())), pulling current features
     back toward the stored prototype.

Optional "source-init" mode: on the first call, initialize prototypes from
the frozen anchor's own features + confidence map, then freeze them. Used
by the ``source_proto`` ablation.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeAnchor(nn.Module):
    def __init__(
        self,
        num_classes: int,
        proto_ema: float = 0.999,
        proto_conf_thresh: float = 0.9,
        min_pixels: int = 4,
        source_init: bool = False,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.proto_ema = float(proto_ema)
        self.proto_conf_thresh = float(proto_conf_thresh)
        self.min_pixels = int(min_pixels)
        self.source_init = bool(source_init)
        self._prototypes: Dict[int, torch.Tensor] = {}
        self._source_frozen = False

    @torch.no_grad()
    def _init_from_anchor(
        self,
        anchor_probs_full: torch.Tensor,
        anchor_features: torch.Tensor,
    ) -> None:
        H, W = anchor_features.shape[-2:]
        probs_ds = F.interpolate(
            anchor_probs_full, size=(H, W), mode="bilinear", align_corners=False
        )
        for c in range(self.num_classes):
            mask = probs_ds[0, c] >= self.proto_conf_thresh
            if int(mask.sum().item()) < self.min_pixels:
                continue
            z = anchor_features[0, :, mask].mean(dim=1)
            self._prototypes[c] = F.normalize(z, dim=0).detach().clone()
        self._source_frozen = True

    def maybe_init(
        self,
        anchor_probs_full: Optional[torch.Tensor],
        anchor_features: Optional[torch.Tensor],
    ) -> None:
        if not self.source_init or self._source_frozen:
            return
        if anchor_probs_full is None or anchor_features is None:
            return
        self._init_from_anchor(anchor_probs_full, anchor_features)

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_probs_full: torch.Tensor,
    ) -> torch.Tensor:
        feat = student_features  # (1, C, H, W)
        H, W = feat.shape[-2:]
        probs_ds = F.interpolate(
            teacher_probs_full, size=(H, W), mode="bilinear", align_corners=False
        )
        loss = feat.new_zeros(())
        for c in range(self.num_classes):
            conf_mask = probs_ds[0, c] >= self.proto_conf_thresh
            if int(conf_mask.sum().item()) < self.min_pixels:
                continue
            z = feat[0, :, conf_mask].mean(dim=1)
            z_n = F.normalize(z, dim=0)
            with torch.no_grad():
                if c not in self._prototypes:
                    self._prototypes[c] = z_n.detach().clone()
                elif not (self.source_init and self._source_frozen):
                    a = self.proto_ema
                    self._prototypes[c] = a * self._prototypes[c] + (1 - a) * z_n.detach()
            proto = self._prototypes[c].to(feat.device)
            loss = loss + (1.0 - (z_n * proto.detach()).sum())
        return loss
