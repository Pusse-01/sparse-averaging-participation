"""
Every convergence/divergence outcome reported in the AsyncMesh paper, scored
against the in-flight staleness multiplicity

    Pi = tau / T_cycle,     T_cycle = avg_interval * ceil(1 / p_nominal)

T_cycle is how many *optimizer steps* pass between two consecutive averaging
writes to any one coordinate, and tau is the buffer depth in averaging
operations, so Pi counts how many stale writes to the same coordinate are
simultaneously in flight.

The claim under test: divergence is governed by Pi alone, not by p or tau
separately. Run this file to print the table.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PaperRun:
    label: str
    p_nominal: float
    tau: int  # async_sparta_delay, in averaging operations
    avg_interval: int  # sparta_interval
    converged: bool
    source: str

    @property
    def n_partitions(self) -> int:
        """PartitionedIndexSelector uses ceil(1/p) disjoint partitions."""
        return math.ceil(1.0 / self.p_nominal)

    @property
    def p_effective(self) -> float:
        """Realised averaging fraction per operation -- NOT p_nominal."""
        return 1.0 / self.n_partitions

    @property
    def t_cycle(self) -> int:
        """Optimizer steps between two writes to the same coordinate."""
        return self.avg_interval * self.n_partitions

    @property
    def pi(self) -> float:
        """In-flight stale writes per coordinate."""
        return (self.tau * self.avg_interval) / self.t_cycle


# Sources are the arXiv v1 PDF (2601.22442): Sec 5.3 / Fig 4 for the
# configuration sweep, Sec 5.3 "Robustness of Sparse Averaging" for the
# subset-size cliff, and the Fig 4 caption for the "100*" contrast.
PAPER_RUNS = [
    PaperRun("subset 1%",       0.01,  10,  1, True,  "Fig 4 (subset-size sweep)"),
    PaperRun("subset 5% (dflt)",0.05,  10,  1, True,  "Sec 5.2 main results"),
    PaperRun("subset 30%",      0.30,  10,  1, True,  "Sec 5.3 '5%-30% similar'"),
    PaperRun("subset 40%",      0.40,  10,  1, False, "Sec 5.3 'diverges >= 40%'"),
    PaperRun("subset 50%",      0.50,  10,  1, False, "Sec 5.3 'diverges >= 40%'"),
    PaperRun("delay 50",        0.05,  50,  1, True,  "Fig 4 'tolerates delay up to 50'"),
    PaperRun("delay 100",       0.05, 100,  1, False, "Fig 4 caption 'did not converge'"),
    PaperRun("delay 10 x K=10", 0.05,  10, 10, True,  "Fig 4 caption '100*'"),
]


def bracket():
    """Tightest interval for the critical Pi consistent with all paper runs."""
    stable = [r.pi for r in PAPER_RUNS if r.converged]
    diverged = [r.pi for r in PAPER_RUNS if not r.converged]
    return max(stable), min(diverged)


def main():
    hdr = (f"{'run':<18}{'p_nom':>7}{'p_eff':>8}{'tau':>5}{'K':>4}"
           f"{'T_cyc':>7}{'Pi':>8}{'outcome':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(PAPER_RUNS, key=lambda r: r.pi):
        print(f"{r.label:<18}{r.p_nominal:>7.2f}{r.p_effective:>8.3f}{r.tau:>5}"
              f"{r.avg_interval:>4}{r.t_cycle:>7}{r.pi:>8.2f}"
              f"{'converged' if r.converged else 'DIVERGED':>11}")

    lo, hi = bracket()
    print("-" * len(hdr))
    print(f"\nAll {len(PAPER_RUNS)} reported outcomes are separated by Pi alone.")
    print(f"  highest Pi that converged : {lo:.2f}")
    print(f"  lowest  Pi that diverged  : {hi:.2f}")
    print(f"  => critical Pi* in ({lo:.2f}, {hi:.2f}]")

    print("\nNote the two contrasts this resolves, which p and tau alone cannot:")
    print("  * p=0.30 (Pi=2.50) converges but p=0.40 (Pi=3.33) diverges, even")
    print("    though both are 'moderate' subset sizes.")
    print("  * tau=100 at K=1 (Pi=5.00) diverges while tau=10 at K=10 (Pi=0.50)")
    print("    converges -- the paper calls both a '100-step effective delay'.")


if __name__ == "__main__":
    main()
