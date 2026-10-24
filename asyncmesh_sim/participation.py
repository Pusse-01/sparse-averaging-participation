"""
Partial participation in the two phases of a sparse all-reduce.

Agora (arXiv:2607.13332, section 5.7.1) instruments the two data-movement
phases of its sparse all-reduce separately, and they fail independently:

    gather  ("reduce success")     the reducer only receives slices from some
                                   replicas; the rest are "effectively lost and
                                   excluded from the global average"
    scatter ("averaging success")  only some replicas receive the averaged
                                   slice back; a node that misses it "retains
                                   its own values for those parameters"

Every published guarantee for sparse averaging assumes both are complete and
that the mask is common to all replicas -- LOSCAR-SGD states it outright ("the
same mask S^r is used by all workers"). This module is what breaks that
assumption in a controlled way, so the two paths can be varied independently.

Two granularities:

    mode="replica"      a replica succeeds or fails for the whole round. This
                        is what Theorems 1-2 of PROPOSAL.md assume exactly, so
                        it is the setting in which the closed forms are
                        predictions rather than approximations. Default.
    mode="coordinate"   independent Bernoulli per (replica, coordinate),
                        modelling a node that delivers only part of its slice.
                        Agora's banding at 94% and 75% is closer to this.

The window schedule exists for E-D: switch failures on for a stretch of
training and back off again, to show that participation failure produces a
*recoverable* stall, unlike the Pi instability which does not recover.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ParticipationConfig:
    phi_g: float = 1.0          # gather-side participation (contributing)
    phi_s: float = 1.0          # scatter-side participation (receiving)
    mode: str = "replica"       # replica | coordinate
    window: tuple | None = None # (start, end) steps; None => always active

    @property
    def enabled(self) -> bool:
        return self.phi_g < 1.0 or self.phi_s < 1.0


class ParticipationSampler:
    """
    Draws the gather and scatter sets for one averaging round.

    Returns float/bool tensors broadcastable against `own` of shape [m, k]:
    shape [m, 1] in replica mode, [m, k] in coordinate mode. Callers do not
    need to branch on the mode.
    """

    def __init__(self, cfg: ParticipationConfig, m: int, seed: int = 0):
        self.cfg = cfg
        self.m = m
        self.gen = torch.Generator().manual_seed(seed)

    # ------------------------------------------------------------------ util
    def active(self, step: int) -> bool:
        """Is participation failure switched on at this step?"""
        if not self.cfg.enabled:
            return False
        if self.cfg.window is None:
            return True
        start, end = self.cfg.window
        return start <= step < end

    def _n_active(self, phi: float) -> int:
        """
        Fixed subset size, which is what the finite-population variance
        identities behind Theorems 1-2 assume. At least one replica always
        participates, otherwise the round is undefined.
        """
        return max(1, min(self.m, int(round(phi * self.m))))

    # ---------------------------------------------------------------- sample
    def _replica_mask(self, phi: float, device) -> torch.Tensor:
        k = self._n_active(phi)
        idx = torch.randperm(self.m, generator=self.gen)[:k]
        mask = torch.zeros(self.m, dtype=torch.bool)
        mask[idx] = True
        return mask.to(device).unsqueeze(1)            # [m, 1]

    def _coordinate_mask(self, phi: float, k: int, device) -> torch.Tensor:
        probs = torch.full((self.m, k), phi)
        return torch.bernoulli(probs, generator=self.gen).bool().to(device)  # [m, k]

    def draw(self, phi: float, k: int, device) -> torch.Tensor:
        if self.cfg.mode == "replica":
            return self._replica_mask(phi, device)
        if self.cfg.mode == "coordinate":
            return self._coordinate_mask(phi, k, device)
        raise ValueError(f"unknown participation mode {self.cfg.mode!r}")

    def gather(self, step: int, k: int, device) -> torch.Tensor:
        if not self.active(step):
            return torch.ones(self.m, 1, dtype=torch.bool, device=device)
        return self.draw(self.cfg.phi_g, k, device)

    def scatter(self, step: int, k: int, device) -> torch.Tensor:
        if not self.active(step):
            return torch.ones(self.m, 1, dtype=torch.bool, device=device)
        return self.draw(self.cfg.phi_s, k, device)


def partial_average(own: torch.Tensor, gmask: torch.Tensor):
    """
    The reducer's view: average over contributing replicas only.

    own    [m, k] the replicas' values on the masked coordinates
    gmask  [m, 1] or [m, k] bool, who contributed

    Returns (avg [k], have [k] bool). `have` is False on coordinates where
    nobody contributed, for which no average exists and the write must be
    skipped entirely -- otherwise a coordinate with an empty gather set would
    be overwritten with garbage.
    """
    g = gmask.to(own.dtype)
    denom = (g * torch.ones_like(own)).sum(dim=0)
    total = (own * g).sum(dim=0)
    have = denom > 0
    avg = torch.where(have, total / denom.clamp(min=1.0), own.mean(dim=0))
    return avg, have
