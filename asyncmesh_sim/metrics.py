"""
Divergence detection and the early-warning signal.

The divergence criterion is fixed here rather than eyeballed per run, because
the whole study is a phase diagram: every run must be labelled stable/unstable
by the same rule, decided before the sweep is launched.

A run is DIVERGED if any of:
  D1  any loss is NaN or Inf;
  D2  after step `grace`, the consensus-model validation loss is at or above
      the loss of an untrained model, ln(vocab_size);
  D3  the final validation loss exceeds the best seen by more than `regress`.

D1 and D2 are hard blow-ups. D3 catches the slower "climbs back out" failure,
which is what the paper's >=40% curves look like.

The early-warning signal is the geometric growth rate of the consensus error,
fitted over a trailing window. The linear model predicts it equals the
monodromy spectral radius rho, and that rho > 1 exactly when the run will
diverge -- so this can be checked long before the loss reacts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DivergenceMonitor:
    vocab_size: int
    grace: int = 500
    regress: float = 0.5

    diverged: bool = False
    reason: str = ""
    step: int | None = None
    best_val: float = field(default=float("inf"))

    @property
    def chance_loss(self) -> float:
        return math.log(self.vocab_size)

    def update_train(self, step: int, loss: float) -> bool:
        if not math.isfinite(loss):
            self._flag(step, "D1: non-finite training loss")
        return self.diverged

    def update_val(self, step: int, val_loss: float) -> bool:
        if not math.isfinite(val_loss):
            self._flag(step, "D1: non-finite validation loss")
            return self.diverged
        if step >= self.grace and val_loss >= self.chance_loss:
            self._flag(step, f"D2: val loss {val_loss:.3f} >= chance "
                             f"{self.chance_loss:.3f}")
            return self.diverged
        self.best_val = min(self.best_val, val_loss)
        if step >= self.grace and val_loss > self.best_val + self.regress:
            self._flag(step, f"D3: val loss {val_loss:.3f} regressed "
                             f">{self.regress} above best {self.best_val:.3f}")
        return self.diverged

    def _flag(self, step: int, reason: str) -> None:
        if not self.diverged:
            self.diverged = True
            self.reason = reason
            self.step = step


class GrowthRateEstimator:
    """
    Trailing-window fit of the consensus error's geometric growth rate.

    Returns rho_hat such that ||Delta||^2 ~ rho_hat^(2t); rho_hat > 1 predicts
    divergence. Compare directly against
    theory.linear_model.monodromy_spectral_radius.
    """

    def __init__(self, window: int = 400):
        self.window = window
        self.buf: list[tuple[int, float]] = []

    def update(self, step: int, consensus_error: float) -> float | None:
        if not math.isfinite(consensus_error) or consensus_error <= 0:
            return float("inf")
        self.buf.append((step, consensus_error))
        if len(self.buf) > self.window:
            self.buf.pop(0)
        if len(self.buf) < max(20, self.window // 4):
            return None
        steps = np.array([s for s, _ in self.buf], dtype=float)
        vals = np.log(np.array([v for _, v in self.buf], dtype=float))
        slope = np.polyfit(steps, vals, 1)[0]
        return float(np.exp(slope / 2.0))


def summarize(monitor: DivergenceMonitor, rho_hat: float | None,
              pi: float, p_eff: float, t_cycle: float) -> dict:
    return {
        "diverged": int(monitor.diverged),
        "divergence_reason": monitor.reason,
        "divergence_step": monitor.step if monitor.step is not None else -1,
        "best_val_loss": monitor.best_val,
        "rho_hat": rho_hat if rho_hat is not None else float("nan"),
        "Pi": pi,
        "p_effective": p_eff,
        "T_cycle": t_cycle,
    }
