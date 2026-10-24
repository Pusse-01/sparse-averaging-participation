"""
Does the DP consensus boundary move with PP stage delay?

Adds AsyncPP-style gradient delay and Nesterov lookahead to the local step of
the linear model, then recomputes the monodromy spectral radius. Tests the
claim that the subset-size cliff is a cross-axis phenomenon with
p* = p*(tau, P, lambda).

Result (see FINDINGS.md F5): it is not. Stage delay 1 -> 8 moves rho in the
4th-5th decimal place; the boundary stays where tau * p_eff puts it.
"""

from __future__ import annotations

import math

import numpy as np

from .linear_model import LinearConfig


def step_matrix_pp(cfg: LinearConfig, masked: bool, pp_delay: int,
                   beta: float) -> np.ndarray:
    """
    Local update with stage-dependent gradient delay and lookahead, on a
    quadratic:  w^{t+1} = w^t - eta*h*w^{t-d} + beta*(w^t - w^{t-1})
    """
    tau, lam, eta, h = cfg.tau, cfg.lam, cfg.eta, cfg.h
    n = tau + 2
    A = np.zeros((n, n))
    for k in range(1, tau + 1):
        A[k, k - 1] = 1.0

    d = min(pp_delay, tau)
    A[0, 0] = 1.0 + beta
    if tau >= 1:
        A[0, 1] -= beta
    A[0, d] -= eta * h
    local_row = A[0, :].copy()

    if not masked:
        A[n - 1, n - 1] = 1.0
        return A

    A[n - 1, :] = lam * local_row
    A[n - 1, tau] += -lam
    A[n - 1, n - 1] = 1.0 - lam
    A[0, :] = A[n - 1, :]
    return A


def rho_pp(p: float, tau: int, lam: float, pp_delay: int, beta: float) -> float:
    cfg = LinearConfig(p=p, tau=tau, lam=lam)
    N = cfg.n_partitions
    M = step_matrix_pp(cfg, True, pp_delay, beta)
    P = step_matrix_pp(cfg, False, pp_delay, beta)
    for _ in range(N - 1):
        M = P @ M
    return float(np.max(np.abs(np.linalg.eigvals(M))) ** (1.0 / N))


if __name__ == "__main__":
    ps = [0.20, 0.25, 0.30, 0.34, 0.40, 0.50]
    print("Does the DP consensus boundary move with PP stage delay?\n")
    print(f"{'pp_delay':>9}{'beta':>6} |" + "".join(f"{'p='+str(p):>10}" for p in ps))
    print(f"{'':>9}{'':>6} |"
          + "".join(f"{'Pi='+f'{10/math.ceil(1/p):.1f}':>10}" for p in ps))
    print("-" * 76)
    for ppd, beta in [(0, 0.0), (1, 0.0), (2, 0.9), (4, 0.9), (8, 0.9), (8, 0.99)]:
        row = f"{ppd:>9}{beta:>6.2f} |"
        for p in ps:
            row += f"{rho_pp(p, 10, 0.5, ppd, beta):>10.5f}"
        print(row)
    print("\nStage delay 1 -> 8 changes rho in the 4th-5th decimal:")
    print("the boundary is set by tau * p_eff regardless of P.")
