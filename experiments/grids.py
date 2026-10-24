"""
Generates the run manifests consumed by slurm/run_array.sbatch.

    python -m experiments.grids --all --out slurm/manifests

Each manifest is a text file with one full `python -m asyncmesh_sim.train ...`
argument string per line; the SLURM array index selects the line. Manifests are
deterministic, so regenerating them gives the same run-to-index mapping.
"""

from __future__ import annotations

import argparse
import math
import os

COMMON = (
    "--preset small --steps 6000 --batch-size 8 --lr 3e-4 --warmup 600 "
    "--eval-interval 250 --diag-interval 25 --lam-warmup 1000"
)


def pi_of(p: float, tau: int, sampler: str, avg_interval: int = 1) -> float:
    p_eff = 1.0 / math.ceil(1.0 / p) if sampler == "partitioned" else p
    return tau * p_eff


def _run(project: str, group: str, tags: str, common: str = COMMON, **kw) -> str:
    parts = [f"--wandb-project {project}", f"--wandb-group {group}"]
    if tags:
        parts.append(f"--wandb-tags {tags}")
    for k, v in kw.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                parts.append(flag)
        else:
            parts.append(f"{flag} {v}")
    return common + " " + " ".join(parts)


# --------------------------------------------------------------------------
# E1  Is the cliff a property of p, or of the sampler's quantisation of p?
#
# Prediction if the mechanism is Pi = tau * p_eff with p_eff = 1/ceil(1/p):
#   * p in {0.26, 0.30, 0.33} give runs identical to p = 0.25  (4 partitions)
#   * p in {0.35, 0.40, 0.45} give runs identical to p = 0.34  (3 partitions)
#   * the transition sits exactly at p = 1/3, not "near 40%"
#   * with the Bernoulli sampler the staircase vanishes and the boundary moves
#     to p = Pi*/tau ~ 0.3
# Falsified if loss varies smoothly across 0.34 -> 0.45.
# --------------------------------------------------------------------------
E1_P = [0.20, 0.25, 0.26, 0.30, 0.33, 0.34, 0.35, 0.40, 0.45, 0.50, 0.55]


def e1(project: str) -> list[str]:
    runs = []
    for sampler in ["partitioned", "random"]:
        for p in E1_P:
            for seed in [0, 1, 2]:
                runs.append(
                    _run(project, "e1_quantisation", f"e1,{sampler}",
                         p=p, tau=10, m=4, method="ema", sampler=sampler,
                         seed=seed)
                )
    return runs


# --------------------------------------------------------------------------
# E2  The phase boundary in (p, tau).
#
# Prediction: divergence iff tau * p_eff >~ 3, i.e. the boundary is a
# hyperbola, not a vertical line in p or in tau. This is the experiment that
# unifies the paper's "diverges at p >= 40%" and "diverges at tau = 100".
# --------------------------------------------------------------------------
E2_P = [0.02, 0.05, 0.10, 0.20, 0.25, 0.34, 0.50]
E2_TAU = [1, 3, 5, 10, 20, 50, 100]


def e2(project: str) -> list[str]:
    runs = []
    for p in E2_P:
        for tau in E2_TAU:
            runs.append(
                _run(project, "e2_phase_boundary", "e2",
                     p=p, tau=tau, m=4, method="ema", sampler="partitioned",
                     seed=0)
            )
    return runs


# --------------------------------------------------------------------------
# E3  Which ingredient causes it? Run at one unstable point (Pi = 3.33) and
#     one stable control (Pi = 0.5), varying one thing at a time.
#
#  a) method: is the EMA feedback necessary?
#       H_overlap predicts 'avg' (AsyncSPARTA, no correction) degrades but does
#       not blow up, while 'ema' does -- the instability is *caused by* the
#       correction, not merely uncorrected by it.
#  b) lambda: the predicted remedy. rho < 1 for lam <~ 0.2 at Pi = 5, so
#       lam0 = 0.1 should remove the cliff entirely at no other cost.
#  c) optimizer: adamw vs sgd separates the EMA loop from NAdamW second-moment
#       mismatch (GAP 6). H_overlap predicts the cliff survives under SGD.
#  d) lr: separates the mechanism from "the learning rate is simply too large".
#       H_overlap predicts the cliff survives at lr/3, since rho depends on
#       eta only weakly.
#  e) m: the paper argues more replicas improve the estimate. The linear model
#       predicts the *threshold* is m-independent; only the noise floor moves.
# --------------------------------------------------------------------------
def e3(project: str) -> list[str]:
    runs = []
    points = [("unstable", 0.40), ("stable", 0.05)]

    for label, p in points:
        for method in ["ema", "weight_update", "avg"]:
            for seed in [0, 1]:
                runs.append(_run(project, "e3a_method", f"e3,{label}",
                                 p=p, tau=10, m=4, method=method,
                                 sampler="partitioned", seed=seed))

    for label, p in points:
        for lam0 in [0.5, 0.3, 0.1, 0.05]:
            for seed in [0, 1]:
                runs.append(_run(project, "e3b_lambda", f"e3,{label}",
                                 p=p, tau=10, m=4, method="ema",
                                 sampler="partitioned", lam0=lam0,
                                 no_anneal_lam=True, seed=seed))

    for label, p in points:
        for opt in ["adamw", "sgd"]:
            for seed in [0, 1]:
                runs.append(_run(project, "e3c_optimizer", f"e3,{label}",
                                 p=p, tau=10, m=4, method="ema",
                                 sampler="partitioned", optimizer=opt,
                                 seed=seed))

    for label, p in points:
        for lr in [3e-4, 1e-4]:
            for seed in [0, 1]:
                runs.append(_run(project, "e3d_lr", f"e3,{label}",
                                 p=p, tau=10, m=4, method="ema",
                                 sampler="partitioned", lr=lr, seed=seed))

    for label, p in points:
        for m in [2, 4, 8]:
            runs.append(_run(project, "e3e_replicas", f"e3,{label}",
                             p=p, tau=10, m=m, method="ema",
                             sampler="partitioned", seed=0))
    return runs


# --------------------------------------------------------------------------
# E4  Confirmation at the paper's own scale and mesh width.
#
# 163M base model, m = 2 (the 4x2 mesh's DP width), 15k steps. Reproduces the
# paper's two reported failures side by side and checks that the predicted
# lambda remedy rescues both.
# --------------------------------------------------------------------------
E4_COMMON = (
    "--preset base --steps 15000 --batch-size 8 --lr 3e-4 --warmup 1500 "
    "--eval-interval 500 --diag-interval 50 --lam-warmup 1000"
)


def e4(project: str) -> list[str]:
    base = dict(m=2, method="ema", sampler="partitioned")
    runs = []
    for p, tau in [(0.05, 10), (0.30, 10), (0.40, 10), (0.05, 50), (0.05, 100)]:
        runs.append(_run(project, "e4_base_model", "e4", common=E4_COMMON,
                         p=p, tau=tau, seed=0, **base))
    # the predicted remedy, at the two failing points
    for p, tau in [(0.40, 10), (0.05, 100)]:
        runs.append(_run(project, "e4_base_model", "e4,remedy", common=E4_COMMON,
                         p=p, tau=tau, seed=0, lam0=0.1, no_anneal_lam=True,
                         **base))
    return runs


# ==========================================================================
# E-A .. E-F  Partial participation.  See PROPOSAL.md and EXPERIMENTS.md.
#
# All existing theory assumes the averaging operator is complete (phi_g =
# phi_s = 1) and common-mask. These grids break that in a controlled way.
# Throughout: phi_g is gather-side participation (Agora "reduce success"),
# phi_s is scatter-side (Agora "averaging success").
# ==========================================================================

# Participation runs use m = 16 by default so the O(1/m) mean-displacement
# channel is well separated from the O(1) contraction channel; E-C varies m
# deliberately, since the crossover between the two is the point.
# --dtype bf16 measured at 1115 ms/step vs 1840 for fp32 on the A40s
# (scripts/calibrate.py, m=16). Set explicitly so the manifests do not depend
# on the default, and re-run calibrate if you move to another card.
PART_COMMON = (
    "--preset small --steps 6000 --batch-size 8 --lr 3e-4 --warmup 600 "
    "--eval-interval 250 --diag-interval 25 --lam-warmup 1000 --dtype bf16"
)


# --------------------------------------------------------------------------
# E-A  Operator exactness. Does the real averaging code obey Thms 1-2?
#
# Run the plain write (method=avg, tau=0) so the operator is tested in
# isolation, then re-run with a delay and with the EMA corrector to see how
# far each moves the measured contraction off the closed form. The first block
# is a prediction; the last two are measurements of the departure.
#
# Predicted: retained_measured == contraction_retained(m, phi_g, phi_s) and
# meanshift_measured == mean_displacement(m, phi_g, phi_s), to within the
# tolerance already demonstrated in tests/test_participation.py (~0.003).
# Falsified if the plain-write block departs by more than 0.02.
# --------------------------------------------------------------------------
EA_POINTS = [(1.0, 1.0), (1.0, 0.75), (1.0, 0.5),
             (0.75, 1.0), (0.5, 1.0), (0.75, 0.75), (0.5, 0.5)]

# E-A measures a per-round operator statistic, not a converged loss, and it is
# averaged over every diagnostic point in the run. 1500 steps at
# diag-interval 25 gives ~60 samples, which is already far past where the mean
# settles -- tests/test_participation.py matches the closed forms to 3e-3 on
# 400 rounds. No reason to pay for 6000.
EA_COMMON = (
    "--preset small --steps 1500 --batch-size 8 --lr 3e-4 --warmup 200 "
    "--eval-interval 250 --diag-interval 25 --lam-warmup 1000 --dtype bf16"
)


def ea(project: str) -> list[str]:
    runs = []
    for phi_g, phi_s in EA_POINTS:
        runs.append(_run(project, "ea_operator", "ea,plain", common=EA_COMMON,
                         p=0.25, tau=0, m=16, method="avg", sampler="random",
                         phi_g=phi_g, phi_s=phi_s, seed=0))
    # how much does delay, then correction, move the operator off the theory?
    for phi_g, phi_s in [(1.0, 0.75), (0.75, 0.75)]:
        runs.append(_run(project, "ea_operator", "ea,delayed", common=EA_COMMON,
                         p=0.25, tau=10, m=16, method="avg", sampler="random",
                         phi_g=phi_g, phi_s=phi_s, seed=0))
        runs.append(_run(project, "ea_operator", "ea,corrected", common=EA_COMMON,
                         p=0.25, tau=10, m=16, method="ema", sampler="random",
                         phi_g=phi_g, phi_s=phi_s, seed=0))
    return runs


# --------------------------------------------------------------------------
# E-B  The asymmetry. Fixed total failure budget, swept across the two paths.
#
# This is the core experiment. Total failure 25% throughout, so the only thing
# changing is *which path* it lands on. Nobody has measured these separately;
# Agora injects gather-side failures only.
#
# Predicted (m = 16, from theory/participation.py):
#   (f_g, f_s)     psi      mean-noise
#   (.250, .000)   1.000    0.0222
#   (.188, .062)   0.941    0.0174
#   (.125, .125)   0.881    0.0146
#   (.062, .188)   0.822    0.0131
#   (.000, .250)   0.763    0.0125
# So consensus error must worsen monotonically toward the scatter end while
# mean displacement moves the other way. Falsified if the two move together,
# or if neither moves.
# --------------------------------------------------------------------------
EB_SPLITS = [(0.25, 0.0), (0.1875, 0.0625), (0.125, 0.125),
             (0.0625, 0.1875), (0.0, 0.25)]


def eb(project: str) -> list[str]:
    runs = []
    for f_g, f_s in EB_SPLITS:
        for seed in [0, 1, 2]:
            runs.append(_run(project, "eb_asymmetry", "eb", common=PART_COMMON,
                             p=0.05, tau=10, m=16, method="ema",
                             sampler="partitioned",
                             phi_g=round(1 - f_g, 4), phi_s=round(1 - f_s, 4),
                             seed=seed))
    return runs


# --------------------------------------------------------------------------
# E-C  The crossover. Which path dominates, as a function of m?
#
# The headline prediction. The contraction channel is O(1) in m; the
# mean-displacement channel is O(1/m). So at small m the gather path should
# cost more, and at large m the scatter path should. A crossover in *which
# sweep is worse* as m grows cannot be produced by any single-parameter
# theory, because a single-parameter theory has only one channel.
#
# Falsified if the same path dominates at both m, or if neither sweep
# separates from the phi = 1 baseline.
# --------------------------------------------------------------------------
EC_PHI = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def ec(project: str) -> list[str]:
    runs = []
    for m in [4, 16]:
        for phi in EC_PHI:
            for seed in [0, 1]:
                # gather-only sweep
                runs.append(_run(project, "ec_crossover", f"ec,gather,m{m}",
                                 common=PART_COMMON,
                                 p=0.05, tau=10, m=m, method="ema",
                                 sampler="partitioned",
                                 phi_g=phi, phi_s=1.0, seed=seed))
                # scatter-only sweep, same total failure rate
                runs.append(_run(project, "ec_crossover", f"ec,scatter,m{m}",
                                 common=PART_COMMON,
                                 p=0.05, tau=10, m=m, method="ema",
                                 sampler="partitioned",
                                 phi_g=1.0, phi_s=phi, seed=seed))
    return runs


# --------------------------------------------------------------------------
# E-D  Stall versus divergence. The two failure modes are different objects.
#
# Participation failure raises a noise floor, so it should produce a stall
# that *recovers* when participation is restored -- which is exactly what
# Agora Fig. 45 reports. The Pi instability is a spectral blow-up and must not
# recover. Run both with the same restore schedule, side by side.
#
# Falsified if participation failure diverges (D1/D2 fire), or if the Pi run
# recovers after the window closes.
# --------------------------------------------------------------------------
ED_WINDOW = "2000:4000"


def ed(project: str) -> list[str]:
    runs = []
    # participation stall, switched on for [2000, 4000) then released
    for phi_s in [0.6, 0.4]:
        runs.append(_run(project, "ed_stall_vs_divergence", "ed,participation",
                         common=PART_COMMON,
                         p=0.05, tau=10, m=16, method="ema",
                         sampler="partitioned", phi_g=1.0, phi_s=phi_s,
                         part_window=ED_WINDOW, seed=0))
    # gather-side equivalent: same failure rate, other path
    runs.append(_run(project, "ed_stall_vs_divergence", "ed,participation",
                     common=PART_COMMON,
                     p=0.05, tau=10, m=16, method="ema", sampler="partitioned",
                     phi_g=0.6, phi_s=1.0, part_window=ED_WINDOW, seed=0))
    # clean control
    runs.append(_run(project, "ed_stall_vs_divergence", "ed,control",
                     common=PART_COMMON,
                     p=0.05, tau=10, m=16, method="ema", sampler="partitioned",
                     seed=0))
    # the Pi instability, for contrast: full participation, unstable Pi
    for p in [0.40, 0.05]:
        tau = 10 if p == 0.40 else 100
        runs.append(_run(project, "ed_stall_vs_divergence", "ed,pi",
                         common=PART_COMMON,
                         p=p, tau=tau, m=16, method="ema",
                         sampler="partitioned", seed=0))
    return runs


# --------------------------------------------------------------------------
# E-E  Does it survive at LM scale? 163M, m = 8.
#
# Guards against "this is an artifact of a 45M toy". One gather-only and one
# scatter-only point at the same 25% failure rate, against the complete
# baseline.
# --------------------------------------------------------------------------
EE_COMMON = (
    "--preset base --steps 15000 --batch-size 8 --lr 3e-4 --warmup 1500 "
    "--eval-interval 500 --diag-interval 50 --lam-warmup 1000 --dtype bf16"
)


def ee(project: str) -> list[str]:
    base = dict(p=0.05, tau=10, m=8, method="ema", sampler="partitioned", seed=0)
    return [
        _run(project, "ee_lm_scale", "ee,baseline", common=EE_COMMON,
             phi_g=1.0, phi_s=1.0, **base),
        _run(project, "ee_lm_scale", "ee,gather", common=EE_COMMON,
             phi_g=0.75, phi_s=1.0, **base),
        _run(project, "ee_lm_scale", "ee,scatter", common=EE_COMMON,
             phi_g=1.0, phi_s=0.75, **base),
    ]


# --------------------------------------------------------------------------
# E-F  Does the design rule pay?
#
# Given a fixed reliability budget -- you can protect one path but not both --
# Cor. 3 says which to protect, and says the answer flips with m. Spend the
# budget on the return path at m = 16 and on the gather path at m = 4, and
# check that the theory-directed allocation beats the other one at each m.
# --------------------------------------------------------------------------
def ef(project: str) -> list[str]:
    runs = []
    for m in [4, 16]:
        # budget spent protecting the return path: failures pushed to gather
        runs.append(_run(project, "ef_budget", f"ef,protect_scatter,m{m}",
                         common=PART_COMMON,
                         p=0.05, tau=10, m=m, method="ema",
                         sampler="partitioned", phi_g=0.75, phi_s=1.0, seed=0))
        # budget spent protecting the gather path: failures pushed to scatter
        runs.append(_run(project, "ef_budget", f"ef,protect_gather,m{m}",
                         common=PART_COMMON,
                         p=0.05, tau=10, m=m, method="ema",
                         sampler="partitioned", phi_g=1.0, phi_s=0.75, seed=0))
    return runs


# --------------------------------------------------------------------------
# E-G  The headline practical experiment: one reliability budget, two ways to
#      spend it, at three budget sizes.
#
# E-B already contains this at a single 25% budget, where the gap is 0.0256 --
# real and monotone in every seed, but small enough for a reader to shrug at.
# The point of sweeping the budget is that Theorem 1 makes a quantitative
# prediction about how the gap should grow, not merely that it is positive:
# spending the budget on the send path leaves psi = 1, while spending it on
# the receive path leaves psi = 1 - b. So the predicted contraction penalty is
# linear in the budget b, and the loss gap should widen accordingly.
#
# Two arms per budget, five seeds, m = 16:
#   PROTECT RECEIVE -> all failure on the send path    (phi_g = 1-b, phi_s = 1)
#   PROTECT SEND    -> all failure on the receive path (phi_g = 1,   phi_s = 1-b)
#
# Predicted: PROTECT RECEIVE wins at every budget, by a margin that grows
# roughly linearly in b. Falsified if the gap fails to grow, or reverses.
# --------------------------------------------------------------------------
EG_BUDGETS = [0.25, 0.40, 0.50]
EG_SEEDS = [0, 1, 2, 3, 4]


def eg(project: str) -> list[str]:
    runs = []
    for b in EG_BUDGETS:
        for seed in EG_SEEDS:
            runs.append(_run(project, "eg_budget_sweep",
                             f"eg,protect_receive,b{b}", common=PART_COMMON,
                             p=0.05, tau=10, m=16, method="ema",
                             sampler="partitioned",
                             phi_g=round(1 - b, 4), phi_s=1.0, seed=seed))
            runs.append(_run(project, "eg_budget_sweep",
                             f"eg,protect_send,b{b}", common=PART_COMMON,
                             p=0.05, tau=10, m=16, method="ema",
                             sampler="partitioned",
                             phi_g=1.0, phi_s=round(1 - b, 4), seed=seed))
    return runs


# --------------------------------------------------------------------------
# E-H  The result in Agora's actual operating point.
#
# Everything above averages every step (avg_interval = 1), which is the regime
# where losing returns hurts LEAST: a coordinate missed this round is picked
# up next round. Agora averages every 20 steps and tiles the parameters into
# 20 segments, so each parameter is averaged once per 400 steps. There, a
# missed return is not retried for another 400 steps.
#
# Theorem 1 says the effective sparsity is p*phi_s, so Agora's observed 75%
# return-path banding turns "averaged once per 400 steps" into "once per ~533
# steps". Their Fig. 25 shows inter-replica variance rising linearly with time
# since last averaging, so that is ~33% more variance than the cadence implies
# -- and their Fig. 27 already reports replicas diverging faster than the
# consensus mean moves.
#
# Three arms, matched total failure, at their cadence:
#   complete      phi_g = 1.00, phi_s = 1.00   what the cadence assumes
#   lose returns  phi_g = 1.00, phi_s = 0.75   what Agora measures
#   lose sends    phi_g = 0.75, phi_s = 1.00   same failure rate, other path
#
# Predicted: "lose returns" is clearly worse than both others, and "lose
# sends" is indistinguishable from complete. Reported as steps-to-target-loss,
# which is the quantity a practitioner budgets in, not final loss.
# --------------------------------------------------------------------------
EH_COMMON = (
    "--preset small --steps 12000 --batch-size 8 --lr 3e-4 --warmup 600 "
    "--eval-interval 250 --diag-interval 50 --lam-warmup 1000 --dtype bf16"
)


def eh(project: str) -> list[str]:
    runs = []
    arms = [("complete", 1.0, 1.0),
            ("lose_returns", 1.0, 0.75),
            ("lose_sends", 0.75, 1.0)]
    for name, g, s in arms:
        for seed in [0, 1, 2]:
            runs.append(_run(project, "eh_agora_cadence", f"eh,{name}",
                             common=EH_COMMON,
                             p=0.05, tau=10, avg_interval=20, m=16,
                             method="ema", sampler="partitioned",
                             phi_g=g, phi_s=s, seed=seed))
    return runs


# --------------------------------------------------------------------------
# E-I  DECISIVE. What does the delta-rule corrector cost in consensus, as a
#      function of the in-flight multiplicity Pi = tau * p_eff?
#
# E-A found the EMA corrector leaves 0.731 of the consensus error standing
# where the plain write leaves 0.235 -- a 2.8x reduction in contraction rate.
# If that holds at deployed operating points it is a far bigger effect than
# partial participation, and it targets the central contribution of
# AsyncMesh (arXiv:2601.22442), whose Theorem 2 claims the EMA correction
# *ensures* consensus.
#
# BUT E-A's corrected runs sit at p=0.25, tau=10, i.e. Pi = 2.5 -- just below
# the Pi ~ 3 instability boundary this repo established earlier. So the
# measured penalty may be a Pi effect rather than a generic property of the
# corrector. Agora runs p=0.05 and averages every 5-20 steps with a delay of
# a few steps, putting it at Pi well under 0.1.
#
# This sweep settles it: hold p=0.05 (Agora's sparsity), vary tau to sweep Pi
# across two decades, and measure the corrector's marginal cost against the
# uncorrected write at each point.
#
#   Pi = tau * p = 0.05, 0.25, 0.5, 1.0, 2.5   for tau = 1, 5, 10, 20, 50
#
# If the corrector's penalty vanishes as Pi -> 0, deployed systems are fine
# and there is no paper in this direction -- report it and move on. If the
# penalty is already substantial at Pi ~ 0.1, it applies to every deployment
# using the delta rule, and it explains why Agora observes replicas diverging
# faster than the consensus mean moves.
#
# Short runs: this measures a per-round operator statistic, not a converged
# loss, exactly as E-A does.
# --------------------------------------------------------------------------
EI_COMMON = (
    "--preset small --steps 1500 --batch-size 8 --lr 3e-4 --warmup 200 "
    "--eval-interval 500 --diag-interval 25 --lam-warmup 1000 --dtype bf16"
)
EI_TAU = [1, 5, 10, 20, 50]


def ei(project: str) -> list[str]:
    runs = []
    for tau in EI_TAU:
        for method in ["avg", "ema"]:
            for seed in [0, 1]:
                runs.append(_run(project, "ei_corrector_cost",
                                 f"ei,{method},tau{tau}", common=EI_COMMON,
                                 p=0.05, tau=tau, m=16, method=method,
                                 sampler="random", phi_g=1.0, phi_s=1.0,
                                 seed=seed))
    return runs


GRIDS = {"e1": e1, "e2": e2, "e3": e3, "e4": e4,
         "ea": ea, "eb": eb, "ec": ec, "ed": ed, "ee": ee, "ef": ef,
         "eg": eg, "eh": eh, "ei": ei}

# The participation study on its own, for `--which part`.
PART_GRIDS = ["ea", "eb", "ec", "ed", "ee", "ef", "eg", "eh", "ei"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--part", action="store_true",
                    help="just the participation study (ea..ef)")
    ap.add_argument("--which", nargs="*", default=[])
    ap.add_argument("--project", default="asyncmesh-failure")
    ap.add_argument("--out", default="slurm/manifests")
    args = ap.parse_args()

    if args.all:
        which = list(GRIDS)
    elif args.part:
        which = list(PART_GRIDS)
    else:
        which = args.which
    if not which:
        ap.error("pass --all, --part, or --which ea eb ...")

    os.makedirs(args.out, exist_ok=True)
    total = 0
    for name in which:
        runs = GRIDS[name](args.project)
        path = os.path.join(args.out, f"{name}.txt")
        with open(path, "w") as f:
            f.write("\n".join(runs) + "\n")
        total += len(runs)
        print(f"{name:>4}: {len(runs):>4} runs -> {path}  "
              f"(array 1-{len(runs)})")
    print(f"\ntotal {total} runs")


if __name__ == "__main__":
    main()
