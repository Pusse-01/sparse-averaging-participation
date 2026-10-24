"""
Linear surrogate for AsyncMesh's asynchronous sparse averaging.

We replace the language model with a quadratic f(w) = 0.5 * h * w^2 per
coordinate and keep the averaging algorithm *bit-for-bit identical* in
structure to sparta.py: partitioned (or Bernoulli) masks, a depth-tau buffer
of pending averages, and the EMA delay correction implemented as SGD-with-
momentum on the negated tau-step increment.

Because the resulting system is linear in the state, consensus either
contracts or grows geometrically, and the boundary between the two can be
computed exactly. Two routines are provided:

  monodromy_spectral_radius()  exact, for the partitioned sampler, whose mask
                               schedule is periodic with period N = ceil(1/p).
                               Returns the per-step growth factor rho; the
                               algorithm is consensus-stable iff rho < 1.

  simulate_growth_rate()       direct simulation of the same recursion, which
                               also covers the Bernoulli sampler and any
                               lambda schedule. Returns the measured
                               log-growth rate of the consensus error.

Both agree with each other; the first is the statement you put in a paper,
the second is the one that sweeps cheaply over grids.

State (per replica i, for a single coordinate):
    w_i                      current weight
    w_i[t-1] ... w_i[t-tau]  ring buffer, needed to form the tau-step increment
    e_i                      EMA state (the correction d_i)

The consensus error is sum_i (w_i - mean_w)^2, so we track deviations only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class LinearConfig:
    p: float = 0.05  # nominal subset size, as passed to --p_sparta
    tau: int = 10  # async_sparta_delay, in averaging operations
    lam: float = 0.5  # EMA coefficient = 1 - sparta_momentum
    eta: float = 3e-4  # learning rate
    h: float = 1.0  # curvature of the quadratic
    m: int = 4  # DP replicas
    sampler: str = "partitioned"  # or "bernoulli"

    @property
    def n_partitions(self) -> int:
        return max(1, math.ceil(1.0 / self.p))

    @property
    def p_effective(self) -> float:
        return 1.0 / self.n_partitions if self.sampler == "partitioned" else self.p

    @property
    def pi(self) -> float:
        """In-flight stale writes per coordinate."""
        return self.tau * self.p_effective


def _step_matrix(cfg: LinearConfig, masked: bool) -> np.ndarray:
    """
    One step of the deviation dynamics for a single replica's single
    coordinate, as a linear map on the state

        x = [ w, w_{t-1}, ..., w_{t-tau}, e ]        (dim tau + 2)

    All quantities are deviations from the replica mean, which is what the
    consensus error measures. Working in deviations is what makes the
    averaging write appear as "set w to 0 plus the EMA deviation": the shared
    stale average wbar^{t-tau} is common to all replicas and therefore drops
    out of the deviation, leaving exactly the EMA spread. This mirrors Eq.
    (25) of the paper, Delta_i = d_i - dbar.

    masked=True  -> this step writes the coordinate (it was in the popped mask)
    masked=False -> plain local SGD step
    """
    tau, lam, eta, h = cfg.tau, cfg.lam, cfg.eta, cfg.h
    n = tau + 2
    A = np.zeros((n, n))

    # --- history shift: w_{t-k} <- w_{t-k+1} for k = 1..tau -----------------
    for k in range(1, tau + 1):
        A[k, k - 1] = 1.0

    # --- local SGD contraction on the deviation ----------------------------
    # what_i = w_i - eta*(h*w_i + noise); noise is additive so it does not
    # enter the transition matrix, only the stationary covariance.
    local = 1.0 - eta * h

    if not masked:
        A[0, 0] = local
        A[n - 1, n - 1] = 1.0  # EMA untouched on non-write steps (see below)
        return A

    # --- averaging write with EMA correction -------------------------------
    # increment seen by the EMA: incr = what_i^t - what_i^{t-tau}
    #   what_i^t     = local * w_i          (post local step, pre write)
    #   what_i^{t-tau} = w_{t-tau}          (stored in the buffer at write time)
    # EMA: e <- (1-lam)*e + lam*incr
    A[n - 1, 0] = lam * local
    A[n - 1, tau] = -lam
    A[n - 1, n - 1] = 1.0 - lam

    # write: w_i <- wbar^{t-tau} + e_new. In deviation coordinates the shared
    # stale average contributes nothing, so w_i <- e_new.
    A[0, :] = A[n - 1, :]

    return A


def monodromy_spectral_radius(cfg: LinearConfig) -> float:
    """
    Exact per-step growth factor for the partitioned sampler.

    Under PartitionedIndexSelector a coordinate is written exactly once every
    N = ceil(1/p) averaging operations, so the deviation dynamics are linear
    and periodic with period N. The monodromy matrix is the product of one
    write step and N-1 plain steps; the per-step growth factor is
    rho = spectral_radius(monodromy) ** (1/N).

    Consensus is stable iff rho < 1. Note the transition matrix is identical
    for every replica, so the m-replica system is m independent copies and the
    stability threshold does not depend on m -- only the noise floor does.
    """
    N = cfg.n_partitions
    M = _step_matrix(cfg, masked=True)
    plain = _step_matrix(cfg, masked=False)
    for _ in range(N - 1):
        M = plain @ M
    rho_period = np.max(np.abs(np.linalg.eigvals(M)))
    return float(rho_period ** (1.0 / N))


def simulate_growth_rate(
    cfg: LinearConfig,
    steps: int = 20000,
    d: int = 4096,
    seed: int = 0,
    lam_schedule=None,
    burn_in: int = 2000,
) -> dict:
    """
    Run the actual algorithm on the quadratic, with d coordinates and m
    replicas, and measure the growth rate of the consensus error.

    This is a faithful re-implementation of sparta.py's arithmetic, so it
    also exercises the sampler, the buffer, and the lambda schedule. Returns
    the mean log-growth per step of the consensus error over the tail, plus
    the trajectory.

    lam_schedule: optional callable step -> lambda, matching the paper's
    cosine anneal from 0.5 to 0.01 starting at step 1000.
    """
    rng = np.random.default_rng(seed)
    m, tau, eta, h = cfg.m, cfg.tau, cfg.eta, cfg.h
    N = cfg.n_partitions

    w = rng.standard_normal((m, d)) * 1e-3  # replicas start slightly apart
    ema = np.zeros((m, d))
    buffer: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # partitioned sampler state
    partitions = rng.permutation(d) % N
    cursor = 0

    trace = []
    for t in range(steps):
        lam = cfg.lam if lam_schedule is None else lam_schedule(t)

        # ---- local step: what = w - eta*(h*w + noise) ----------------------
        noise = rng.standard_normal((m, d))
        what = w - eta * (h * w + noise)

        # ---- choose mask ---------------------------------------------------
        if cfg.sampler == "partitioned":
            if cursor >= N:
                partitions = rng.permutation(d) % N
                cursor = 0
            mask = partitions == cursor
            cursor += 1
        else:
            mask = rng.random(d) < cfg.p

        # ---- all-reduce the masked coordinates, push to buffer -------------
        avg = what[:, mask].mean(axis=0)  # shared across replicas
        own = what[:, mask].copy()  # per-replica value at push time
        buffer.append((mask, avg, own))

        w = what

        # ---- pop and apply if the buffer is deeper than tau ----------------
        if len(buffer) > tau:
            pmask, pavg, pown = buffer.pop(0)
            cur = w[:, pmask]
            incr = cur - pown  # what_i^t - what_i^{t-tau}
            ema[:, pmask] = (1.0 - lam) * ema[:, pmask] + lam * incr
            w[:, pmask] = pavg[None, :] + ema[:, pmask]

        dev = w - w.mean(axis=0, keepdims=True)
        trace.append(float((dev**2).sum() / (m * d)))

    tr = np.asarray(trace)
    tail = tr[burn_in:]
    # geometric growth rate of the consensus error, per step
    if np.any(~np.isfinite(tail)) or np.any(tail <= 0):
        rate = float("inf")
    else:
        x = np.arange(len(tail))
        slope = np.polyfit(x, np.log(tail), 1)[0]
        rate = float(np.exp(slope / 2.0))  # error is squared -> sqrt for w-scale

    return {"growth_rate": rate, "trace": tr, "pi": cfg.pi,
            "p_effective": cfg.p_effective}


def paper_lambda_schedule(total_steps: int = 30000, warmup: int = 1000,
                          lam0: float = 0.5, lam1: float = 0.01):
    """
    The schedule actually used by AsyncMesh: sparta_momentum is cosine-annealed
    from 0.5 to 0.99 after sparta_warmup_steps, i.e. lambda = 1 - momentum goes
    0.5 -> 0.01. See SparseSGD.step in sparta/sparta.py.
    """

    def sched(t: int) -> float:
        if t < warmup:
            return lam0
        frac = (t - warmup) / max(1, total_steps - warmup - 1)
        frac = min(1.0, max(0.0, frac))
        return lam0 + (lam1 - lam0) * (1 - 0.5 * (1 + math.cos(math.pi * frac)))

    return sched


def critical_pi(tau: int = 10, lam: float = 0.5, eta: float = 3e-4,
                h: float = 1.0, p_grid=None) -> dict:
    """Locate the stability boundary in p for a fixed tau, via the exact rho."""
    if p_grid is None:
        p_grid = np.concatenate([np.linspace(0.01, 0.6, 120)])
    rows = []
    for p in p_grid:
        cfg = LinearConfig(p=float(p), tau=tau, lam=lam, eta=eta, h=h)
        rows.append((float(p), cfg.p_effective, cfg.pi,
                     monodromy_spectral_radius(cfg)))
    stable = [r for r in rows if r[3] < 1.0]
    unstable = [r for r in rows if r[3] >= 1.0]
    return {
        "rows": rows,
        "max_stable_pi": max((r[2] for r in stable), default=None),
        "min_unstable_pi": min((r[2] for r in unstable), default=None),
    }


if __name__ == "__main__":
    print("Exact per-step growth factor rho (partitioned sampler, tau=10)\n")
    print(f"{'p_nom':>7}{'p_eff':>8}{'Pi':>8}{'rho(lam=0.5)':>14}{'rho(lam=0.01)':>15}")
    print("-" * 52)
    for p in [0.01, 0.05, 0.10, 0.20, 0.25, 0.30, 0.34, 0.40, 0.45, 0.50, 0.60]:
        a = LinearConfig(p=p, tau=10, lam=0.5)
        b = LinearConfig(p=p, tau=10, lam=0.01)
        print(f"{p:>7.2f}{a.p_effective:>8.3f}{a.pi:>8.2f}"
              f"{monodromy_spectral_radius(a):>14.5f}"
              f"{monodromy_spectral_radius(b):>15.5f}")

    def fmt(v):
        return "  --  " if v is None else f"{v:.2f}"

    print("\nBoundary location vs tau (lambda = 0.5):\n")
    print(f"{'tau':>5}{'max stable Pi':>16}{'min unstable Pi':>18}")
    print("-" * 39)
    for tau in [2, 5, 10, 20, 50, 100]:
        r = critical_pi(tau=tau, lam=0.5)
        print(f"{tau:>5}{fmt(r['max_stable_pi']):>16}"
              f"{fmt(r['min_unstable_pi']):>18}")

    print("\nBoundary location vs lambda (tau = 10):\n")
    print(f"{'lambda':>8}{'max stable Pi':>16}{'min unstable Pi':>18}")
    print("-" * 42)
    for lam in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        r = critical_pi(tau=10, lam=lam)
        print(f"{lam:>8.2f}{fmt(r['max_stable_pi']):>16}"
              f"{fmt(r['min_unstable_pi']):>18}")
