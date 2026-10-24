"""
Checks on the claims the experiment design rests on. Run before launching the
sweep: if any of these fail, the sweep is measuring something else.

    python -m tests.test_claims
"""

from __future__ import annotations

import math

import torch

from asyncmesh_sim.averaging import AsyncSparseAverager, AveragingConfig
from asyncmesh_sim.sampler import build_selector
from theory.linear_model import LinearConfig, monodromy_spectral_radius


def _fail(msg):
    raise AssertionError(msg)


def test_partition_quantisation():
    """p_eff = 1/ceil(1/p): nominal p in (1/3, 1/2] all realise 1/3."""
    for p, expect_n in [(0.34, 3), (0.35, 3), (0.40, 3), (0.45, 3),
                        (0.50, 2), (0.30, 4), (0.25, 4), (0.05, 20)]:
        sel = build_selector("partitioned", p)
        if sel.num_partitions != expect_n:
            _fail(f"p={p}: expected {expect_n} partitions, got {sel.num_partitions}")
        param = torch.zeros(10000)
        mask = sel.get_indices(param)
        frac = mask.float().mean().item()
        if abs(frac - 1.0 / expect_n) > 0.01:
            _fail(f"p={p}: realised fraction {frac:.4f} != {1/expect_n:.4f}")
    print("  ok  partition quantisation: p_eff = 1/ceil(1/p)")


def test_identical_runs_within_bucket():
    """
    The sharpest prediction: nominal p=0.34 and p=0.45 are the same algorithm,
    so with the same seed they must produce bit-identical masks and weights.
    If a real sweep shows them differing in loss, the mechanism is not the
    sampler.
    """

    def run(p, steps=40):
        torch.manual_seed(0)
        params = [[torch.randn(2000)] for _ in range(3)]
        cfg = AveragingConfig(p=p, tau=5, method="ema", sampler="partitioned")
        torch.manual_seed(123)
        avg = AsyncSparseAverager(cfg, 3, params[0])
        for t in range(steps):
            for i in range(3):
                params[i][0] += 0.01 * torch.randn_like(params[i][0])
            avg.step(params, t)
        return torch.stack([params[i][0] for i in range(3)])

    a, b = run(0.34), run(0.45)
    if not torch.equal(a, b):
        _fail(f"p=0.34 and p=0.45 diverged (max abs diff "
              f"{(a-b).abs().max().item():.3e}) -- expected bit-identical")

    c = run(0.30)
    if torch.equal(a, c):
        _fail("p=0.34 and p=0.30 were identical -- expected different "
              "(3 vs 4 partitions)")
    print("  ok  p=0.34 and p=0.45 are bit-identical; p=0.30 differs")


def test_bernoulli_is_not_quantised():
    """The control sampler realises p exactly, with no cycle structure."""
    for p in [0.34, 0.40, 0.45]:
        sel = build_selector("random", p)
        frac = sel.get_indices(torch.zeros(200000)).float().mean().item()
        if abs(frac - p) > 0.01:
            _fail(f"bernoulli p={p}: realised {frac:.4f}")
        if abs(sel.p_effective - p) > 1e-9:
            _fail(f"bernoulli p_effective wrong for p={p}")
    print("  ok  bernoulli sampler realises p exactly")


def test_lambda_schedule():
    """lambda runs 0.5 -> 0.01, held at 0.5 through the warmup."""
    cfg = AveragingConfig(lam0=0.5, lam1=0.01, lam_warmup=1000, total_steps=6000)
    avg = AsyncSparseAverager(cfg, 2, [torch.zeros(4)])
    avg.n_avg_ops = 0
    if abs(avg.lam() - 0.5) > 1e-9:
        _fail(f"lambda at step 0 = {avg.lam()}")
    avg.n_avg_ops = 999
    if abs(avg.lam() - 0.5) > 1e-9:
        _fail(f"lambda at step 999 = {avg.lam()}")
    avg.n_avg_ops = 5999
    if abs(avg.lam() - 0.01) > 1e-3:
        _fail(f"lambda at final step = {avg.lam()}, expected ~0.01")
    print("  ok  lambda schedule 0.5 -> 0.01 after warmup")


def test_ema_reduces_to_paper_form():
    """
    With lam=1 the EMA correction collapses to the single-sample form
    w_i <- wbar^{t-tau} + (w_i^t - w_i^{t-tau}), which is 'weight_update'.
    """
    def run(method, lam):
        torch.manual_seed(0)
        params = [[torch.randn(500)] for _ in range(2)]
        cfg = AveragingConfig(p=0.5, tau=2, method=method, lam0=lam,
                              anneal_lam=False)
        torch.manual_seed(9)
        avg = AsyncSparseAverager(cfg, 2, params[0])
        for t in range(12):
            for i in range(2):
                params[i][0] += 0.01 * torch.randn_like(params[i][0])
            avg.step(params, t)
        return torch.stack([p[0] for p in params])

    a = run("ema", 1.0)
    b = run("weight_update", 1.0)
    if not torch.allclose(a, b, atol=1e-6):
        _fail(f"ema(lam=1) != weight_update, max diff {(a-b).abs().max():.3e}")
    print("  ok  ema with lambda=1 reduces to the single-sample correction")


def test_pi_matches_paper_bracket():
    """The exact linear model must put the boundary inside (2.5, 3.33]."""
    lam = 0.5
    stable = LinearConfig(p=0.30, tau=10, lam=lam)  # Pi = 2.5, paper: converges
    unstable = LinearConfig(p=0.40, tau=10, lam=lam)  # Pi = 3.33, paper: diverges
    rs = monodromy_spectral_radius(stable)
    ru = monodromy_spectral_radius(unstable)
    if not (rs < 1.0 < ru):
        _fail(f"linear model does not bracket the paper's data: "
              f"rho(Pi=2.5)={rs:.5f}, rho(Pi=3.33)={ru:.5f}")
    print(f"  ok  linear model brackets the paper: rho(Pi=2.5)={rs:.5f} < 1 "
          f"< rho(Pi=3.33)={ru:.5f}")


def test_low_lambda_is_stable():
    """The predicted remedy: at small lambda the instability disappears."""
    bad = monodromy_spectral_radius(LinearConfig(p=0.50, tau=10, lam=0.5))
    fixed = monodromy_spectral_radius(LinearConfig(p=0.50, tau=10, lam=0.1))
    if not (bad > 1.0 > fixed):
        _fail(f"lambda remedy not predicted: rho(lam=0.5)={bad:.5f}, "
              f"rho(lam=0.1)={fixed:.5f}")
    print(f"  ok  lowering lambda stabilises Pi=5: {bad:.5f} -> {fixed:.5f}")


def _quadratic_growth_rate(p, tau, lam=0.5, steps=4000, m=4, d=8000,
                           eta=3e-4, h=1.0, seed=0):
    """
    Drive the *shipped* averaging code with a quadratic and measure the
    geometric growth rate of the consensus error. This exercises the real
    torch implementation, not the numpy surrogate.
    """
    import numpy as np

    torch.manual_seed(seed)
    params = [[torch.randn(d) * 1e-3] for _ in range(m)]
    cfg = AveragingConfig(p=p, tau=tau, method="ema", sampler="partitioned",
                          lam0=lam, anneal_lam=False)
    torch.manual_seed(seed + 1)
    avg = AsyncSparseAverager(cfg, m, params[0])

    trace = []
    for t in range(steps):
        for i in range(m):
            g = h * params[i][0] + torch.randn(d)
            params[i][0] -= eta * g
        diag = avg.step(params, t, compute_diag=(t % 20 == 0))
        if diag:
            trace.append(diag["consensus/error"])

    a = np.array(trace[len(trace) // 2:])
    if not np.all(np.isfinite(a)) or np.any(a <= 0):
        return float("inf"), avg.pi
    x = np.arange(len(a)) * 20
    slope = np.polyfit(x, np.log(a), 1)[0]
    return float(np.exp(slope / 2)), avg.pi


def test_shipped_code_reproduces_instability():
    """
    End-to-end: the real averaging implementation must be stable at Pi = 2.5
    and unstable at Pi = 3.33, and its measured growth rate must match the
    exact monodromy eigenvalue.
    """
    rho_stable, pi_s = _quadratic_growth_rate(0.30, 10)
    rho_unstable, pi_u = _quadratic_growth_rate(0.34, 10)

    if rho_stable > 1.0005:
        _fail(f"Pi={pi_s:.2f} should be stable, measured rho={rho_stable:.5f}")
    if not rho_unstable > 1.0005:
        _fail(f"Pi={pi_u:.2f} should be unstable, measured rho={rho_unstable:.5f}")

    predicted = monodromy_spectral_radius(LinearConfig(p=0.34, tau=10, lam=0.5))
    if abs(rho_unstable - predicted) > 5e-3:
        _fail(f"measured rho {rho_unstable:.5f} vs predicted {predicted:.5f} "
              "-- disagreement larger than tolerance")
    print(f"  ok  shipped code: stable at Pi={pi_s:.2f} (rho={rho_stable:.5f}), "
          f"unstable at Pi={pi_u:.2f} (rho={rho_unstable:.5f} vs "
          f"predicted {predicted:.5f})")


def main():
    print("checks on the experiment's load-bearing claims\n")
    for fn in [
        test_partition_quantisation,
        test_identical_runs_within_bucket,
        test_bernoulli_is_not_quantised,
        test_lambda_schedule,
        test_ema_reduces_to_paper_form,
        test_pi_matches_paper_bracket,
        test_low_lambda_is_stable,
        test_shipped_code_reproduces_instability,
    ]:
        fn()
    print("\nall checks passed")


if __name__ == "__main__":
    main()
