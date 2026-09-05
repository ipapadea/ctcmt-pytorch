"""Stochastic parameter restore (CoTTA-style) with V2 shared-trunk factor.

Matches ``CTCMT_MTL._stochastic_restore``:
- Snapshot source weight/bias params from the anchor once at construction.
- Each step: Bernoulli restore mask with per-parameter probability ``rst``.
  For shared backbone/FPN params the probability is scaled by
  ``backbone_rst_factor`` (defaults to 1.0 = disabled; V2 configs use 0.1).
- Adaptive-STR / drift-scaling variants live in ``restore_extra.py`` if
  needed later; this module implements only the confirmed reference logic.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch


_SHARED_TRUNK_PREFIXES: Tuple[str, ...] = (
    "backbone.",
    "fpn.",
    "proposal_generator.anchor_generator.",
)


def _is_shared_trunk(fq_name: str) -> bool:
    return any(fq_name.startswith(pfx) for pfx in _SHARED_TRUNK_PREFIXES)


class StochasticRestore:
    def __init__(
        self,
        anchor_named_wb: Iterable[Tuple[str, torch.Tensor]],
        rst_prob: float,
        cross_task_fisher: bool = False,
        backbone_rst_factor: float = 1.0,
    ):
        self.rst_prob = float(rst_prob)
        self.cross_task_fisher = bool(cross_task_fisher)
        self.backbone_rst_factor = float(backbone_rst_factor)
        self._source: Dict[str, torch.Tensor] = {}
        for name, p in anchor_named_wb:
            self._source[name] = p.detach().clone()

    @torch.no_grad()
    def apply(self, student_named_wb: Iterable[Tuple[str, torch.Tensor]]) -> None:
        if self.rst_prob <= 0.0:
            return
        for name, p in student_named_wb:
            src = self._source.get(name)
            if src is None:
                continue
            rst = self.rst_prob
            if self.cross_task_fisher and _is_shared_trunk(name):
                rst = rst * self.backbone_rst_factor
            if rst <= 0.0:
                continue
            mask = (torch.rand_like(p) < rst).float()
            src_dev = src.to(p.device, non_blocking=True)
            p.data.mul_(1.0 - mask).add_(src_dev * mask)
