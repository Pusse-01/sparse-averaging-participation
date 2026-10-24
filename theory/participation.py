"""
Exact one-round dynamics of sparse averaging under partial participation.

Backs Theorems 1 and 2 of PROPOSAL.md. Two independent participation sets per
averaging round, matching the two data-movement phases Agora instruments
separately (arXiv:2607.13332, section 5.7.1):

    gather set  G, |G| = phi_g * m   the reducer only sees these replicas
                                     ("reduce success")
    scatter set R, |R| = phi_s * m   only these replicas apply the average
                                     ("averaging success"); the rest keep
                                     their own values

Every published guarantee for sparse averaging assumes phi_g = phi_s = 1 and a
mask common to all replicas (SPARTA; AsyncMesh Thm 1-2; LOSCAR-SGD Thm 4.1).
The closed forms below are exact for an *arbitrary fixed* deviation vector --
only G and R are random -- so they need no distributional assumption on the
weights and are unaffected by data heterogeneity.

Run directly to check both closed forms against Monte Carlo.
"""

from __future__ import annotations

import numpy as np


def contraction_retained(m: int, phi_g: float, phi_s: float) -> float:
    """
    Fraction of consensus error surviving one averaged coordinate.

    Theorem 1:  E[S'] / S = (1 - phi_s)
                            + phi_s(1 - phi_s)/(m - 1) * [(1 - phi_g)/phi_g - 1]

    phi_g = phi_s = 1 gives 0, recovering AsyncMesh Lemma 1's factor (1 - p).
    phi_s = 1 gives 0 for *any* phi_g > 0: agreement is reached even when
    almost nobody contributed to the average.
    """
    return (1.0 - phi_s) + phi_s * (1.0 - phi_s) / (m - 1) * ((1.0 - phi_g) / phi_g - 1.0)


def psi(m: int, phi_g: float, phi_s: float) -> float:
    """
    Per-coordinate contraction rate, so the whole-model recursion is

        E||Delta'||^2 = (1 - p * psi) * ||Delta||^2

    To leading order in m this is just phi_s: the gather path does not appear
    in the consensus recursion at all.
    """
    return 1.0 - contraction_retained(m, phi_g, phi_s)


def mean_displacement(m: int, phi_g: float, phi_s: float) -> float:
    """
    E[(mean shift)^2] / v for one averaged coordinate, where v = S / m.

    Theorem 2:  = 1/(m-1) * [ phi_s^2 (1 - phi_g)/phi_g + phi_s (1 - phi_s) ]

    Two channels. The gather term diverges as phi_g -> 0 and vanishes at
    phi_g = 1; the scatter term is maximal at phi_s = 1/2 and vanishes at both
    ends. Note the 1/(m-1): this channel is O(1/m) while the contraction
    channel is O(1), which is what produces the predicted crossover in which
    failure path dominates.
    """
    return (phi_s**2 * (1.0 - phi_g) / phi_g + phi_s * (1.0 - phi_s)) / (m - 1)


def monte_carlo(m: int, g: int, r: int, trials: int = 60000, seed: int = 0):
    """Direct simulation of the write, for validation. Returns (retained, meanpert)."""
    rng = np.random.default_rng(seed)
    num = den = mp = 0.0
    for _ in range(trials):
        d = rng.standard_normal(m)
        d -= d.mean()                     # deviations, sum zero
        G = rng.choice(m, g, replace=False)
        R = rng.choice(m, r, replace=False)
        b = d[G].mean()                   # reducer sees only G
        u = d.copy()
        u[R] = b                          # only R writes the average back
        mp += u.mean() ** 2               # the consensus mean has moved
        u = u - u.mean()
        num += (u**2).sum()
        den += (d**2).sum()
    return num / den, mp / trials / (den / trials / m)


def _check() -> None:
    cases = [(16, 16, 12), (16, 12, 12), (16, 8, 8), (16, 4, 4), (16, 16, 8),
             (16, 8, 16), (16, 4, 16), (8, 6, 6), (4, 3, 3), (16, 14, 16)]
    print(f"{'m':>3}{'phi_g':>7}{'phi_s':>7} |"
          f"{'retain MC':>11}{'retain CF':>11} |{'mean MC':>10}{'mean CF':>10}")
    worst = 0.0
    for m, g, r in cases:
        pg, ps = g / m, r / m
        rm, mm = monte_carlo(m, g, r)
        rc, mc_ = contraction_retained(m, pg, ps), mean_displacement(m, pg, ps)
        worst = max(worst, abs(rm - rc), abs(mm - mc_))
        print(f"{m:>3}{pg:>7.3f}{ps:>7.3f} |{rm:>11.5f}{rc:>11.5f} |{mm:>10.5f}{mc_:>10.5f}")
    print(f"\nmax |MC - closed form| = {worst:.2e}  (Monte Carlo s.e. ~1e-3)")

    print("\nSplitting a fixed 25% failure budget between the two paths (m=16):")
    print(f"{'gather fail':>12}{'scatter fail':>13}{'psi':>10}{'mean-noise':>12}")
    for fg, fs in [(0.25, 0.0), (0.1875, 0.0625), (0.125, 0.125),
                   (0.0625, 0.1875), (0.0, 0.25)]:
        pg, ps = 1 - fg, 1 - fs
        print(f"{fg:>12.4f}{fs:>13.4f}{psi(16, pg, ps):>10.5f}"
              f"{mean_displacement(16, pg, ps):>12.5f}")
    print("\nConsensus degrades toward the scatter end; mean noise moves the other way.")


if __name__ == "__main__":
    _check()
