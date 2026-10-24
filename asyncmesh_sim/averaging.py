"""
Asynchronous sparse parameter averaging with EMA delay correction.

Faithful single-process port of SpartaInterpolator._interpolate_models and
SparseSGD.step from PluralisResearch/AsyncMesh (sparta/sparta.py). The m DP
replicas live in one process so that masks, buffers and RNG are exactly shared,
which makes runs reproducible to the bit -- necessary for the E1 test that
nominal p = 0.34 and p = 0.45 are the same algorithm.

Arithmetic being reproduced, for a masked coordinate popped at step t after a
delay of tau averaging operations:

    incr    = w_i^t - w_i^{t-tau}                       (per replica)
    d_i     = (1 - lam_t) * d_i + lam_t * incr          (EMA, masked coords only)
    w_i^t   = wbar^{t-tau} + sparta_lr * d_i

Upstream expresses this as SGD-with-momentum on grad = -incr with lr = 1.0 and
momentum = 1 - lam_t, which is algebraically the same thing. lam_t follows the
paper's schedule: constant 0.5 for sparta_warmup_steps, then cosine-annealed to
0.01 (upstream anneals the momentum 0.5 -> 0.99).

Methods, matching --sparta_method upstream:
    "ema"           the full method (delay correction via EMA)
    "weight_update" single-sample correction, d_i = incr (no EMA)
    "avg"           no correction at all -- this is AsyncSPARTA
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from theory.participation import contraction_retained, mean_displacement, psi

from .participation import ParticipationConfig, ParticipationSampler, partial_average
from .sampler import build_selector


@dataclass
class AveragingConfig:
    p: float = 0.05
    tau: int = 10  # async_sparta_delay, in averaging operations
    avg_interval: int = 1  # sparta_interval
    method: str = "ema"  # ema | weight_update | avg | shared_extrap
    sampler: str = "partitioned"  # partitioned | random | topk
    sparta_lr: float = 1.0  # sparta_lambda upstream
    lam0: float = 0.5  # 1 - sparta_momentum
    lam1: float = 0.01  # 1 - 0.99
    anneal_lam: bool = True  # sparta_adaptive_momentum
    lam_warmup: int = 1000  # sparta_warmup_steps
    total_steps: int = 30000
    extrap_gain: float | None = None  # shared_extrap; None => tau/T_cycle
    # --- partial participation (PROPOSAL.md Thms 1-2) ---------------------
    # phi_g = 1, phi_s = 1 is the complete, common-mask operator that every
    # published guarantee assumes. Anything else is outside all of them.
    phi_g: float = 1.0
    phi_s: float = 1.0
    part_mode: str = "replica"  # replica | coordinate
    part_window: tuple | None = None  # (start, end) steps, for E-D


@dataclass
class ReplicaState:
    """Per-replica EMA buffers, keyed by parameter index."""

    ema: dict = field(default_factory=dict)


class AsyncSparseAverager:
    """
    Holds the delay buffer and EMA state for m replicas.

    Call step(replica_params, global_step) once per optimizer step. Replica
    params are given as a list (over replicas) of lists (over parameters) of
    tensors, all sharing the same shapes and order.
    """

    def __init__(self, cfg: AveragingConfig, m: int, ref_params: list[torch.Tensor],
                 part_seed: int = 0):
        self.cfg = cfg
        self.m = m
        self.selector = build_selector(cfg.sampler, cfg.p)
        self.participation = ParticipationSampler(
            ParticipationConfig(
                phi_g=cfg.phi_g, phi_s=cfg.phi_s,
                mode=cfg.part_mode, window=cfg.part_window,
            ),
            m,
            seed=part_seed,
        )
        self.n_params = len(ref_params)
        # buffer[j] is a FIFO of pending averages for parameter j
        self.buffer: list[list[tuple]] = [[] for _ in range(self.n_params)]
        self.replicas = [ReplicaState() for _ in range(m)]
        self.shared_prev: dict = {}   # last averaged value per coordinate
        self.shared_seen: dict = {}   # has this coordinate been averaged yet
        self.n_avg_ops = 0  # counts calls that performed an averaging op
        self._last_diag: dict = {}

    # ---------------------------------------------------------------- lambda
    def lam(self) -> float:
        """
        EMA coefficient at the current averaging-operation count.

        Mirrors SparseSGD.step: momentum is held at (1 - lam0) until
        warmup_steps, then cosine-annealed to 0.99. current_step there
        increments once per _interpolate_models call, which is what
        self.n_avg_ops counts.
        """
        c = self.cfg
        if not c.anneal_lam or self.n_avg_ops < c.lam_warmup:
            return c.lam0
        denom = max(1, c.total_steps - c.lam_warmup - 1)
        t = (self.n_avg_ops - c.lam_warmup) / denom
        t = min(1.0, max(0.0, t))
        mom0 = 1.0 - c.lam0
        mom1 = 1.0 - c.lam1
        mom = mom0 + (mom1 - mom0) * (1 - 0.5 * (1 + math.cos(math.pi * t)))
        return 1.0 - mom

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, replica_params: list[list[torch.Tensor]], global_step: int,
             compute_diag: bool = True) -> dict:
        """
        One averaging operation. Returns a diagnostics dict (may be empty on
        steps where nothing was popped).

        compute_diag=False skips the whole-model consensus and EMA statistics,
        which sweep over every parameter of every replica and otherwise cost
        more than the averaging itself. The averaging arithmetic is identical
        either way.
        """
        if self.cfg.avg_interval > 1 and global_step % self.cfg.avg_interval != 0:
            return {}

        cfg = self.cfg
        lam = self.lam()
        diag = {"lambda": lam}

        pushed_masked = 0
        popped_masked = 0
        sq_dev_masked = 0.0
        sq_dev_all = 0.0
        n_all = 0
        ema_var_sum = 0.0
        ema_var_max = 0.0
        ema_bias_sq = 0.0
        ema_n = 0
        # one-round operator measurement (E-A)
        part_S_pre = 0.0
        part_S_post = 0.0
        part_shift = 0.0
        g_frac_sum = s_frac_sum = 0.0
        g_frac_n = s_frac_n = 0

        for j in range(self.n_params):
            ref = replica_params[0][j]

            # ---- sample mask (shared across replicas, as dist.broadcast does)
            mask = self.selector.get_indices(ref)
            if compute_diag:
                pushed_masked += int(mask.sum().item())

            # ---- all-reduce the masked coordinates ------------------------
            # Gather side: the reducer only sees the contributing replicas.
            # A replica that fails here is excluded from the average, but it
            # still keeps its own snapshot locally, so the corrector's
            # increment (cur - pown) is unaffected -- as in the real system.
            own = torch.stack([replica_params[i][j][mask] for i in range(self.m)])
            gmask = self.participation.gather(global_step, own.shape[1], own.device)
            avg, have = partial_average(own, gmask)
            self.buffer[j].append((mask, avg, own, have))
            if compute_diag:
                g_frac_sum += float(
                    gmask.to(own.dtype).expand(self.m, own.shape[1]).mean()
                )
                g_frac_n += 1

            # ---- pop and apply if deeper than tau -------------------------
            if len(self.buffer[j]) > cfg.tau:
                pmask, pavg, pown, phave = self.buffer[j].pop(0)
                if compute_diag:
                    popped_masked += int(pmask.sum().item())

                cur = torch.stack(
                    [replica_params[i][j][pmask] for i in range(self.m)]
                )

                # Scatter side: only these replicas receive the average back.
                # The rest "retain their own values for those parameters"
                # (Agora s5.7.1), so for them this round is a no-op -- the
                # write is skipped and their corrector state does not tick.
                # `phave` masks out coordinates for which the gather set was
                # empty, so no average exists to apply.
                smask = self.participation.scatter(
                    global_step, cur.shape[1], cur.device
                )
                write = smask & phave.unsqueeze(0)

                if compute_diag:
                    # One-round operator measurement, for E-A. Everything is
                    # on the popped mask and compares immediately pre- to
                    # immediately post-write, which is exactly the quantity
                    # Theorems 1-2 predict.
                    pre_mean = cur.mean(dim=0)
                    part_S_pre += float(((cur - pre_mean) ** 2).sum())
                    s_frac_sum += float(
                        smask.to(cur.dtype).expand(self.m, cur.shape[1]).mean()
                    )
                    s_frac_n += 1

                if cfg.method == "avg":
                    # AsyncSPARTA: write the stale average, no correction
                    corrected = pavg.unsqueeze(0).expand_as(cur)
                elif cfg.method == "shared_extrap":
                    # Estimate the shared target from shared data only.
                    #
                    # For coordinate mu, the previous average was written
                    # T_cycle steps earlier, so (pavg - prev)/T_cycle is the
                    # observed drift rate of the *average* trajectory. Scaling
                    # by tau extrapolates it across the delay gap:
                    #
                    #   d = (tau / T_cycle) * (wbar^{t-tau} - wbar^{prev})
                    #
                    # Every term is identical across replicas, so d carries no
                    # cross-replica spread and the write injects exactly zero
                    # consensus error -- for any tau.
                    prev = self._get_shared_prev(j, pmask, ref, pavg)
                    gain = cfg.extrap_gain if cfg.extrap_gain is not None else self.pi
                    d_shared = gain * (pavg - prev)
                    self._set_shared_prev(j, pmask, pavg)
                    corrected = (pavg + d_shared).unsqueeze(0).expand_as(cur)

                    if compute_diag:
                        true_fresh = cur.mean(dim=0)
                        est_fresh = pavg + d_shared
                        ema_bias_sq += float(((est_fresh - true_fresh) ** 2).sum())
                        ema_n += d_shared.shape[0]
                else:
                    incr = cur - pown  # w_i^t - w_i^{t-tau}
                    if cfg.method == "weight_update":
                        d = incr
                    elif cfg.method == "ema":
                        prev = self._get_ema(j, pmask, ref)
                        d = (1.0 - lam) * prev + lam * incr
                        # A replica that never received the average did not
                        # complete this round, so its estimator must not tick.
                        d = torch.where(write, d, prev)
                        self._set_ema(j, pmask, d)
                    else:
                        raise ValueError(f"unknown method {cfg.method!r}")
                    corrected = pavg.unsqueeze(0) + cfg.sparta_lr * d

                    if compute_diag:
                        # how far is the corrected estimate from the true fresh
                        # average it is trying to reconstruct?
                        true_fresh = cur.mean(dim=0)
                        est_fresh = pavg + cfg.sparta_lr * d.mean(dim=0)
                        ema_bias_sq += float(((est_fresh - true_fresh) ** 2).sum())
                        dvar = d.var(dim=0, unbiased=False)
                        ema_var_sum += float(dvar.sum())
                        ema_var_max = max(
                            ema_var_max, float(dvar.max()) if dvar.numel() else 0.0
                        )
                        ema_n += d.shape[1]

                new_vals = torch.where(write, corrected, cur)
                for i in range(self.m):
                    replica_params[i][j][pmask] = new_vals[i]

                if compute_diag:
                    post_mean = new_vals.mean(dim=0)
                    dev = new_vals - post_mean
                    sq_dev_masked += float((dev**2).sum())
                    part_S_post += float((dev**2).sum())
                    part_shift += float(((post_mean - pre_mean) ** 2).sum())

            # ---- global consensus error -----------------------------------
            if compute_diag:
                stacked = torch.stack([replica_params[i][j] for i in range(self.m)])
                dev_all = stacked - stacked.mean(dim=0, keepdim=True)
                sq_dev_all += float((dev_all**2).sum())
                n_all += ref.numel()

        self.n_avg_ops += 1

        if not compute_diag:
            return {}

        diag.update(
            {
                "consensus/error": sq_dev_all / (self.m * max(1, n_all)),
                "consensus/error_masked_after_write": sq_dev_masked
                / max(1, self.m * max(1, popped_masked)),
                "diag/masked_frac_pushed": pushed_masked / max(1, n_all),
                "diag/masked_frac_popped": popped_masked / max(1, n_all),
            }
        )
        if ema_n:
            diag.update(
                {
                    "ema/var_across_replicas_mean": ema_var_sum / ema_n,
                    "ema/var_across_replicas_max": ema_var_max,
                    "ema/reconstruction_error": ema_bias_sq / ema_n,
                }
            )

        # ---- the one-round operator, measured against Theorems 1-2 --------
        # `retained` is E[S']/S and `meanshift` is E[(mean shift)^2]/v with
        # v = S/m, which are exactly the two closed forms in
        # theory.participation. They are predictions only for the plain write
        # (method="avg", tau=0); with a corrector or a delay in play the
        # measured values are expected to depart, and by how much is itself
        # the interesting quantity.
        if part_S_pre > 0:
            diag.update(
                {
                    "part/retained_measured": part_S_post / part_S_pre,
                    "part/meanshift_measured": part_shift / (part_S_pre / self.m),
                    "part/retained_predicted": contraction_retained(
                        self.m, cfg.phi_g, cfg.phi_s
                    ),
                    "part/meanshift_predicted": mean_displacement(
                        self.m, cfg.phi_g, cfg.phi_s
                    ),
                    "part/psi_predicted": psi(self.m, cfg.phi_g, cfg.phi_s),
                }
            )
        if g_frac_n:
            diag["part/phi_g_realised"] = g_frac_sum / g_frac_n
        if s_frac_n:
            diag["part/phi_s_realised"] = s_frac_sum / s_frac_n

        self._last_diag = diag
        return diag

    # ------------------------------------------------------------------ EMA
    def _get_ema(self, j: int, mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        out = []
        for i in range(self.m):
            store = self.replicas[i].ema
            if j not in store:
                store[j] = torch.zeros_like(ref)
            out.append(store[j][mask])
        return torch.stack(out)

    def _set_ema(self, j: int, mask: torch.Tensor, d: torch.Tensor) -> None:
        for i in range(self.m):
            self.replicas[i].ema[j][mask] = d[i]

    # --------------------------------------------------- shared-signal state
    # One tensor per parameter, holding the last averaged value of each
    # coordinate. Replaces both the per-replica EMA state (m x d) and the
    # per-replica `own` copies in the buffer (m x tau x p x d), so the
    # shared-signal corrector is strictly cheaper in memory than the EMA one.
    def _get_shared_prev(self, j: int, mask: torch.Tensor, ref: torch.Tensor,
                         current_avg: torch.Tensor) -> torch.Tensor:
        if j not in self.shared_prev:
            self.shared_prev[j] = torch.zeros_like(ref)
            self.shared_seen[j] = torch.zeros_like(ref, dtype=torch.bool)
        prev = self.shared_prev[j][mask].clone()
        # A coordinate averaged for the first time has no previous average to
        # difference against. Seed it with the current one so its first
        # correction is exactly zero rather than a spurious full-magnitude jump.
        unseen = ~self.shared_seen[j][mask]
        if unseen.any():
            prev[unseen] = current_avg[unseen]
        return prev

    def _set_shared_prev(self, j: int, mask: torch.Tensor,
                         avg: torch.Tensor) -> None:
        self.shared_prev[j][mask] = avg
        self.shared_seen[j][mask] = True

    # ------------------------------------------------------------- reporting
    @property
    def p_effective(self) -> float:
        return self.selector.p_effective

    @property
    def t_cycle(self) -> float:
        return self.selector.t_cycle * self.cfg.avg_interval

    @property
    def pi(self) -> float:
        """In-flight stale writes per coordinate: the order parameter."""
        return (self.cfg.tau * self.cfg.avg_interval) / self.t_cycle
