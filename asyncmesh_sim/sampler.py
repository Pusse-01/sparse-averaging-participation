"""
Index selectors, ported from PluralisResearch/AsyncMesh sparta/sparta.py.

PartitionedIndexSelector is the one AsyncMesh actually uses (SpartaInterpolator
hardcodes it). It is *not* Bernoulli(p) sampling: it splits each parameter into
ceil(1/p) disjoint partitions and walks them in order, reshuffling after a full
sweep. Two consequences drive this whole study:

  1. The realised averaging fraction is p_eff = 1/ceil(1/p), not p. Nominal
     p = 0.34 and p = 0.45 produce *the same three partitions* and therefore
     bit-identical training runs.
  2. Every coordinate is written exactly once per T_cycle = ceil(1/p) averaging
     operations, so with buffer depth tau there are Pi = tau/T_cycle stale
     writes to the same coordinate in flight at once.

RandomIndexSelector (present upstream but unused) gives true Bernoulli(p) with
p_eff = p and no cycle structure; it is the control condition that separates
"the sampler quantises p" from "large p is intrinsically unstable".
"""

from __future__ import annotations

import math

import torch


class IndexSelector:
    def __init__(self, p: float):
        self.state: dict = {}
        self.p = p

    def get_indices(self, param: torch.Tensor) -> torch.Tensor:
        return torch.ones(param.shape, dtype=torch.bool, device=param.device)

    @property
    def p_effective(self) -> float:
        return 1.0

    @property
    def t_cycle(self) -> int:
        return 1


class RandomIndexSelector(IndexSelector):
    """True Bernoulli(p) masks: p_eff == p, no cycle structure."""

    def get_indices(self, param: torch.Tensor) -> torch.Tensor:
        return torch.bernoulli(
            torch.full(param.shape, self.p, device=param.device)
        ).bool()

    @property
    def p_effective(self) -> float:
        return self.p

    @property
    def t_cycle(self) -> float:
        return 1.0 / self.p


class PartitionedIndexSelector(IndexSelector):
    """Verbatim port of the selector AsyncMesh uses."""

    def __init__(self, p: float):
        super().__init__(p)

    def _set_partition(self, param: torch.Tensor) -> None:
        param_state = self.state[param]
        param_state["curr_partition"] = 0
        param_state["num_partitions"] = min(math.ceil(1 / self.p), param.numel())
        param_state["partitions"] = (
            torch.rand(param.numel(), device=param.device)
            .argsort()
            .view(param.shape)
            % param_state["num_partitions"]
        )

    def get_indices(self, param: torch.Tensor) -> torch.Tensor:
        if param not in self.state:
            self.state[param] = {}
            self._set_partition(param)
        elif self.state[param]["curr_partition"] >= self.state[param]["num_partitions"]:
            self._set_partition(param)

        indices = (
            self.state[param]["partitions"] == self.state[param]["curr_partition"]
        ).bool()
        self.state[param]["curr_partition"] += 1
        return indices

    @property
    def num_partitions(self) -> int:
        return math.ceil(1 / self.p)

    @property
    def p_effective(self) -> float:
        return 1.0 / self.num_partitions

    @property
    def t_cycle(self) -> int:
        return self.num_partitions


class TopKIndexSelector(IndexSelector):
    """Biased selector; the paper reports it fails to converge. Kept for E3."""

    def get_indices(self, param: torch.Tensor) -> torch.Tensor:
        flat = param.view(-1)
        k = max(1, int(self.p * flat.numel()))
        topk = flat.abs().topk(k, largest=True).indices
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask[topk] = True
        return mask.view(param.shape)

    @property
    def p_effective(self) -> float:
        return self.p

    @property
    def t_cycle(self) -> float:
        return 1.0 / self.p


def build_selector(kind: str, p: float) -> IndexSelector:
    kinds = {
        "partitioned": PartitionedIndexSelector,
        "random": RandomIndexSelector,
        "topk": TopKIndexSelector,
    }
    if kind not in kinds:
        raise ValueError(f"unknown sampler {kind!r}, expected one of {list(kinds)}")
    return kinds[kind](p)
