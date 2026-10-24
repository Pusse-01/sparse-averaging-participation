"""
Load-bearing checks for the partial-participation claims (PROPOSAL.md Thms 1-2).

Run before launching the sweep. If these fail, E-A through E-F are measuring
something other than what the theory predicts.

    python -m tests.test_participation

The important one is test_operator_matches_theory: it drives the *real*
averaging code, not a reimplementation, and checks the measured one-round
contraction and mean displacement against the closed forms. That is the
experiment E-A performs at model scale, done here in seconds.
"""

from __future__ import annotations

import torch

from asyncmesh_sim.averaging import AsyncSparseAverager, AveragingConfig
from asyncmesh_sim.participation import (
    ParticipationConfig,
    ParticipationSampler,
    partial_average,
)
from theory.participation import contraction_retained, mean_displacement, psi


def _fail(msg):
    raise AssertionError(msg)


def _drive(m, phi_g, phi_s, rounds=400, d=4096, p=0.25, seed=0, mode="replica"):
    """
    Run the plain averaging operator (no corrector, no delay) on random
    replica states, re-randomising the deviations every round so each round is
    an independent draw of the operator. Returns the run-averaged measured
    retained fraction and mean displacement.
    """
    torch.manual_seed(seed)
    cfg = AveragingConfig(
        p=p, tau=0, method="avg", sampler="random",
        phi_g=phi_g, phi_s=phi_s, part_mode=mode,
    )
    ref = [torch.zeros(d)]
    avgr = AsyncSparseAverager(cfg, m, ref, part_seed=seed)

    ret, shift = [], []
    for t in range(rounds):
        # fresh, zero-mean deviations: the theory holds for an arbitrary fixed
        # deviation vector, so any draw is a valid test
        dev = torch.randn(m, d)
        dev -= dev.mean(dim=0, keepdim=True)
        params = [[dev[i].clone()] for i in range(m)]
        diag = avgr.step(params, t, compute_diag=True)
        if "part/retained_measured" in diag:
            ret.append(diag["part/retained_measured"])
            shift.append(diag["part/meanshift_measured"])
    return sum(ret) / len(ret), sum(shift) / len(shift)


def test_closed_forms_self_consistent():
    """Full participation must reproduce AsyncMesh Lemma 1 exactly."""
    for m in [2, 4, 8, 16]:
        if abs(contraction_retained(m, 1.0, 1.0)) > 1e-12:
            _fail(f"m={m}: full participation should annihilate consensus error")
        if abs(mean_displacement(m, 1.0, 1.0)) > 1e-12:
            _fail(f"m={m}: full participation should not move the mean")
        if abs(psi(m, 1.0, 1.0) - 1.0) > 1e-12:
            _fail(f"m={m}: psi should be 1 at full participation")
    print("  ok  closed forms recover AsyncMesh Lemma 1 at phi_g = phi_s = 1")


def test_gather_does_not_affect_consensus():
    """
    The central structural claim: with phi_s = 1 the consensus error is
    annihilated for ANY phi_g > 0. Agreement without average consensus.
    """
    for phi_g in [1.0, 0.75, 0.5, 0.25]:
        r = contraction_retained(16, phi_g, 1.0)
        if abs(r) > 1e-12:
            _fail(f"phi_g={phi_g}: expected exact consensus, got retained={r}")
        if mean_displacement(16, phi_g, 1.0) <= 0 and phi_g < 1.0:
            _fail(f"phi_g={phi_g}: mean should be displaced")
    print("  ok  gather path is absent from the consensus recursion")


def test_operator_matches_theory():
    """
    The real averaging code, measured against Thms 1-2. This is E-A.
    """
    cases = [
        (16, 1.0, 0.75), (16, 0.75, 0.75), (16, 0.5, 0.5),
        (16, 1.0, 0.5), (16, 0.5, 1.0), (8, 0.75, 0.75), (4, 0.5, 0.5),
    ]
    worst_r = worst_s = 0.0
    print(f"  {'m':>3}{'phi_g':>7}{'phi_s':>7} |{'retain':>19} |{'meanshift':>19}")
    for m, pg, ps in cases:
        rm, sm = _drive(m, pg, ps)
        rp, sp = contraction_retained(m, pg, ps), mean_displacement(m, pg, ps)
        worst_r = max(worst_r, abs(rm - rp))
        worst_s = max(worst_s, abs(sm - sp))
        print(f"  {m:>3}{pg:>7.2f}{ps:>7.2f} |{rm:>9.5f}{rp:>10.5f} "
              f"|{sm:>9.5f}{sp:>10.5f}")
    if worst_r > 0.02:
        _fail(f"retained deviates from Thm 1 by {worst_r:.4f} (tol 0.02)")
    if worst_s > 0.02:
        _fail(f"mean displacement deviates from Thm 2 by {worst_s:.4f} (tol 0.02)")
    print(f"  ok  real operator matches Thms 1-2 "
          f"(max err {max(worst_r, worst_s):.4f})")


def test_scatter_preserves_nonreceivers():
    """
    A replica that does not receive must keep its value bit-exactly. If this
    fails the simulator is quietly averaging through the failure and every
    downstream result is wrong.
    """
    torch.manual_seed(0)
    m, d = 8, 512
    cfg = AveragingConfig(p=1.0, tau=0, method="avg", sampler="random",
                          phi_g=1.0, phi_s=0.5, part_mode="replica")
    avgr = AsyncSparseAverager(cfg, m, [torch.zeros(d)], part_seed=3)
    dev = torch.randn(m, d)
    params = [[dev[i].clone()] for i in range(m)]
    before = [p[0].clone() for p in params]
    avgr.step(params, 0, compute_diag=False)

    unchanged = sum(
        1 for i in range(m) if torch.equal(params[i][0], before[i])
    )
    changed = m - unchanged
    if unchanged == 0:
        _fail("phi_s=0.5: no replica kept its own values")
    if changed == 0:
        _fail("phi_s=0.5: no replica received the average")
    print(f"  ok  scatter failure is a true no-op "
          f"({unchanged}/{m} replicas kept their values)")


def test_empty_gather_skips_write():
    """
    In coordinate mode a coordinate can have an empty gather set. No average
    exists for it, so the write must be skipped rather than filled with a
    degenerate value.
    """
    m, k = 4, 2000
    own = torch.randn(m, k)
    gmask = torch.zeros(m, k, dtype=torch.bool)
    gmask[:, : k // 2] = True  # second half has nobody contributing
    avg, have = partial_average(own, gmask)
    if not bool(have[: k // 2].all()) or bool(have[k // 2:].any()):
        _fail("`have` does not mark the empty-gather coordinates")
    if not torch.allclose(avg[: k // 2], own[:, : k // 2].mean(dim=0)):
        _fail("average is wrong on the coordinates that did have contributors")
    print("  ok  empty gather sets are detected and skipped")


def test_window_schedule():
    """E-D needs failures confined to a step range."""
    s = ParticipationSampler(
        ParticipationConfig(phi_g=1.0, phi_s=0.5, window=(100, 200)), m=8
    )
    if s.active(50) or s.active(250):
        _fail("failures active outside the window")
    if not s.active(150):
        _fail("failures inactive inside the window")
    full = s.scatter(50, 4, torch.device("cpu"))
    if not bool(full.all()):
        _fail("outside the window the scatter set must be complete")
    print("  ok  window schedule gates failures correctly")


def test_effective_sparsity():
    """
    Cor. 3a: the effective sparsity is p * phi_s, not p. Check the recursion
    factor (1 - p*psi) tracks p*phi_s to O(1/m).
    """
    m, p = 32, 0.05
    for phi_s in [1.0, 0.9, 0.75, 0.5]:
        got = p * psi(m, 1.0, phi_s)
        want = p * phi_s
        if abs(got - want) > p * 0.05:
            _fail(f"phi_s={phi_s}: p*psi={got:.5f} far from p*phi_s={want:.5f}")
    print("  ok  effective sparsity is p * phi_s to O(1/m)")


TESTS = [
    test_closed_forms_self_consistent,
    test_gather_does_not_affect_consensus,
    test_scatter_preserves_nonreceivers,
    test_empty_gather_skips_write,
    test_window_schedule,
    test_effective_sparsity,
    test_operator_matches_theory,
]


def main():
    print("partial-participation claims\n")
    for t in TESTS:
        t()
    print("\nall participation checks passed")


if __name__ == "__main__":
    main()
