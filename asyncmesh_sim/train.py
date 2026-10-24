"""
Runner for the AsyncMesh DP-axis failure-mode experiments.

The DP axis is simulated in a single process: m replicas, each with its own
model copy, optimizer and data shard, stepped in turn, with asynchronous sparse
averaging between them through a depth-tau buffer. This is exactly the regime
the paper's own theory covers ("We consider the DP setup without PP for
simplified theoretical analysis", Appendix A.1), so it is the faithful test of
Theorems 1-2 and it removes AsyncPP as a confound.

Everything goes to Weights & Biases. On compute nodes without egress set
WANDB_MODE=offline and sync afterwards with slurm/sync_wandb.sh.

    python -m asyncmesh_sim.train --p 0.40 --tau 10 --m 4 --steps 6000 \
        --wandb-project asyncmesh-failure --wandb-group e1_quantization
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from .averaging import AsyncSparseAverager, AveragingConfig
from .data import ShardedTokenStream, SyntheticStream, ValStream
from theory.participation import contraction_retained, mean_displacement, psi

from .metrics import DivergenceMonitor, GrowthRateEstimator, summarize
from .model import PRESETS, build_model


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    # --- the two axes of the phase diagram ------------------------------
    ap.add_argument("--p", type=float, default=0.05, help="nominal subset size")
    ap.add_argument("--tau", type=int, default=10, help="async delay, in avg ops")
    ap.add_argument("--avg-interval", type=int, default=1)
    # --- mechanism ablations --------------------------------------------
    ap.add_argument("--method", default="ema", choices=["ema", "weight_update", "avg"])
    ap.add_argument("--sampler", default="partitioned",
                    choices=["partitioned", "random", "topk"])
    ap.add_argument("--lam0", type=float, default=0.5)
    ap.add_argument("--lam1", type=float, default=0.01)
    ap.add_argument("--no-anneal-lam", action="store_true")
    ap.add_argument("--lam-warmup", type=int, default=1000)
    ap.add_argument("--sparta-lr", type=float, default=1.0)
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    # --- partial participation (PROPOSAL.md Thms 1-2) --------------------
    ap.add_argument("--phi-g", type=float, default=1.0,
                    help="gather-side participation: fraction of replicas "
                         "whose slice reaches the reducer (Agora 'reduce "
                         "success'). 1.0 is what all existing theory assumes.")
    ap.add_argument("--phi-s", type=float, default=1.0,
                    help="scatter-side participation: fraction of replicas "
                         "that receive the average back (Agora 'averaging "
                         "success'). Non-receivers keep their own values.")
    ap.add_argument("--part-mode", default="replica",
                    choices=["replica", "coordinate"])
    ap.add_argument("--part-window", default="",
                    help="'start:end' -- apply failures only in this step "
                         "range, for the stall-and-recover test (E-D)")
    ap.add_argument("--part-seed", type=int, default=0,
                    help="separate stream for participation draws, so the "
                         "failure pattern can be reseeded independently")
    # --- scale ------------------------------------------------------------
    ap.add_argument("--m", type=int, default=4, help="DP replicas")
    ap.add_argument("--preset", default="small", choices=list(PRESETS))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    # --- data -------------------------------------------------------------
    ap.add_argument("--train-bin", default="data/wikitext103_train.bin")
    ap.add_argument("--val-bin", default="data/wikitext103_val.bin")
    ap.add_argument("--synthetic", action="store_true",
                    help="skip real data; smoke tests only")
    ap.add_argument("--vocab-size", type=int, default=50257)
    # --- logging ----------------------------------------------------------
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--diag-interval", type=int, default=25)
    ap.add_argument("--wandb-project", default="asyncmesh-failure")
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-tags", default="")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--stop-on-diverge", action="store_true",
                    help="end the run once the divergence criterion fires")
    return ap.parse_args(argv)


def lr_at(step: int, args) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    t = min(1.0, max(0.0, t))
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))


def build_optimizer(model, args):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if args.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2))
    return torch.optim.SGD(groups, lr=args.lr, momentum=0.9)


@torch.no_grad()
def evaluate(models, val_stream, device, autocast_ctx) -> float:
    """
    Validation loss of the *consensus model* (the average across replicas),
    which is what AsyncMesh reports: "we compute the validation loss on the
    model averaged across all DP replicas, for all the methods".
    """
    ref = models[0]
    backup = [p.detach().clone() for p in ref.parameters()]
    with torch.no_grad():
        for j, p in enumerate(ref.parameters()):
            stacked = torch.stack([list(mm.parameters())[j].detach() for mm in models])
            p.copy_(stacked.mean(dim=0))

    ref.eval()
    total, n = 0.0, 0
    for x, y in val_stream:
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            _, loss = ref(x, y)
        total += float(loss)
        n += 1
    ref.train()

    with torch.no_grad():
        for p, b in zip(ref.parameters(), backup):
            p.copy_(b)
    return total / max(1, n)


def main(argv=None):
    args = get_args(argv)
    torch.manual_seed(args.seed)
    device = args.device

    # Print before anything that can block, and again around each step that
    # can, so a stalled job says exactly where it stalled rather than leaving
    # a log containing only the SLURM preamble.
    print(f"starting: preset={args.preset} m={args.m} p={args.p} tau={args.tau} "
          f"phi_g={args.phi_g} phi_s={args.phi_s} device={device}", flush=True)

    use_wandb = not args.no_wandb
    if use_wandb:
        print("importing wandb ...", flush=True)
        import wandb
        print("wandb imported", flush=True)

    # ---- models, one per DP replica, identically initialised -------------
    torch.manual_seed(args.seed)
    models = []
    for i in range(args.m):
        torch.manual_seed(args.seed)  # identical init, per the homogeneous setup
        models.append(build_model(args.preset, args.vocab_size).to(device))
    torch.manual_seed(args.seed + 7)
    print(f"built {args.m} replicas of {models[0].num_params():,} params",
          flush=True)

    params = [[p for p in mm.parameters()] for mm in models]
    optims = [build_optimizer(mm, args) for mm in models]
    block_size = models[0].cfg.block_size

    # ---- data -------------------------------------------------------------
    if args.synthetic:
        train_stream = SyntheticStream(args.vocab_size, args.m, block_size, args.seed)
        val_stream = [
            (
                torch.randint(0, args.vocab_size, (args.batch_size, block_size)),
                torch.randint(0, args.vocab_size, (args.batch_size, block_size)),
            )
            for _ in range(4)
        ]
    else:
        train_stream = ShardedTokenStream(
            args.train_bin, args.m, block_size, args.seed
        )
        val_stream = ValStream(
            args.val_bin, block_size, args.batch_size, args.eval_batches
        )

    # ---- averaging --------------------------------------------------------
    part_window = None
    if args.part_window:
        lo, hi = args.part_window.split(":")
        part_window = (int(lo), int(hi))

    avg_cfg = AveragingConfig(
        p=args.p,
        tau=args.tau,
        avg_interval=args.avg_interval,
        method=args.method,
        sampler=args.sampler,
        sparta_lr=args.sparta_lr,
        lam0=args.lam0,
        lam1=args.lam1,
        anneal_lam=not args.no_anneal_lam,
        lam_warmup=args.lam_warmup,
        total_steps=args.steps,
        phi_g=args.phi_g,
        phi_s=args.phi_s,
        part_mode=args.part_mode,
        part_window=part_window,
    )
    averager = AsyncSparseAverager(
        avg_cfg, args.m, params[0], part_seed=args.part_seed
    )

    monitor = DivergenceMonitor(vocab_size=args.vocab_size)
    growth = GrowthRateEstimator(window=400)

    # bfloat16 tensor cores need Ampere (sm_80) or newer. Note that
    # torch.cuda.is_bf16_supported() returns True on Turing as well, because
    # it counts the emulated path -- so check the compute capability directly.
    # Emulated bf16 is correct but has no tensor-core backing and can be
    # slower than plain fp32, which is a per-device question, not a universal
    # one. Warn rather than override: the answer comes from CLUSTER_SETUP.md
    # section 5, and then --dtype goes in the grid explicitly.
    use_bf16 = args.dtype == "bf16" and device.startswith("cuda")
    if use_bf16 and torch.cuda.get_device_capability()[0] < 8:
        cap = torch.cuda.get_device_capability()
        print(f"WARNING: {torch.cuda.get_device_name(0)} is sm_{cap[0]}{cap[1]}; "
              f"bf16 is emulated (no tensor cores) and may be slower than "
              f"--dtype fp32. Benchmark both before submitting an array.")
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else torch.autocast(device_type="cpu", enabled=False)
    )

    run_cfg = vars(args) | {
        "Pi": averager.pi,
        "p_effective": averager.p_effective,
        "T_cycle": averager.t_cycle,
        "n_params": models[0].num_params(),
        # effective sparsity is p_eff * phi_s, not p_eff (Cor. 3a)
        "p_effective_participating": averager.p_effective * args.phi_s,
        "psi_predicted": psi(args.m, args.phi_g, args.phi_s),
        "retained_predicted": contraction_retained(args.m, args.phi_g, args.phi_s),
        "meanshift_predicted": mean_displacement(args.m, args.phi_g, args.phi_s),
    }
    name = args.wandb_name or (
        f"p{args.p}_tau{args.tau}_{args.method}_{args.sampler}"
        f"_lam{args.lam0}_m{args.m}_g{args.phi_g}_s{args.phi_s}_s{args.seed}"
    )

    if use_wandb:
        # W&B is for convenience, not for correctness: every run also writes
        # its config and summary to results/<name>.json, and the analysis can
        # be rebuilt from those. So a broken W&B must not take the run with
        # it -- on a compute node with no egress this otherwise burns the
        # whole allocation and produces nothing.
        try:
            wandb.init(
                project=args.wandb_project,
                group=args.wandb_group,
                name=name,
                tags=[t for t in args.wandb_tags.split(",") if t],
                config=run_cfg,
            )
            wandb.define_metric("step")
            wandb.define_metric("*", step_metric="step")
            print(f"wandb: {wandb.run.mode if hasattr(wandb.run, 'mode') else 'init'} "
                  f"ok ({name})", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"wandb init failed ({type(exc).__name__}: {exc}); "
                  f"continuing without it -- results still go to "
                  f"{args.out_dir}/{name}.json", flush=True)
            use_wandb = False

    print(json.dumps({k: v for k, v in run_cfg.items()}, indent=2, default=str))
    print(f"Pi = {averager.pi:.3f}  p_eff = {averager.p_effective:.4f}  "
          f"T_cycle = {averager.t_cycle}")

    rho_hat = None
    op_retained: list[float] = []
    op_meanshift: list[float] = []
    t0 = time.time()
    for step in range(args.steps):
        lr = lr_at(step, args)
        for opt in optims:
            for g in opt.param_groups:
                g["lr"] = lr

        losses = []
        for i in range(args.m):
            x, y = train_stream.batch(i, args.batch_size, device)
            with autocast_ctx:
                _, loss = models[i](x, y)
            optims[i].zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(models[i].parameters(), args.grad_clip)
            optims[i].step()
            losses.append(float(loss.detach()))

        train_loss = sum(losses) / len(losses)
        monitor.update_train(step, train_loss)

        diag = {}
        if args.m > 1 and args.p > 0:
            want_diag = (step % args.diag_interval == 0)
            diag = averager.step(params, step, compute_diag=want_diag)

        if diag and "consensus/error" in diag:
            r = growth.update(step, diag["consensus/error"])
            if r is not None:
                rho_hat = r
        if diag and "part/retained_measured" in diag:
            # Averaged over the run, this is the E-A statistic: it must equal
            # theory.participation.contraction_retained for the plain write.
            op_retained.append(diag["part/retained_measured"])
            op_meanshift.append(diag["part/meanshift_measured"])

        if use_wandb and step % args.diag_interval == 0:
            payload = {"step": step, "train/loss": train_loss, "train/lr": lr}
            payload.update(diag)
            if rho_hat is not None:
                payload["consensus/rho_hat"] = rho_hat
            wandb.log(payload)

        if step % args.eval_interval == 0 or step == args.steps - 1:
            val = evaluate(models, val_stream, device, autocast_ctx)
            monitor.update_val(step, val)
            msg = (f"step {step:6d}  train {train_loss:7.4f}  val {val:7.4f}  "
                   f"rho_hat {rho_hat if rho_hat else float('nan'):.5f}  "
                   f"{(time.time()-t0)/max(1,step+1)*1000:.0f} ms/step")
            print(msg, flush=True)
            if use_wandb:
                wandb.log(
                    {
                        "step": step,
                        "val/loss": val,
                        "val/ppl": math.exp(min(20.0, val)),
                        "val/diverged": int(monitor.diverged),
                    }
                )
            if monitor.diverged and args.stop_on_diverge:
                print(f"DIVERGED at step {step}: {monitor.reason}")
                break

    summary = summarize(
        monitor, rho_hat, averager.pi, averager.p_effective, averager.t_cycle
    )
    summary["wall_clock_s"] = time.time() - t0
    summary.update(
        {
            "phi_g": args.phi_g,
            "phi_s": args.phi_s,
            "psi_predicted": run_cfg["psi_predicted"],
            "retained_predicted": run_cfg["retained_predicted"],
            "meanshift_predicted": run_cfg["meanshift_predicted"],
        }
    )
    if op_retained:
        # Trim the first few rounds: the corrector state and the partition
        # sweep are still warming up and the operator is not in steady state.
        warm = min(len(op_retained) // 5, 20)
        summary["retained_measured"] = float(sum(op_retained[warm:])
                                             / max(1, len(op_retained[warm:])))
        summary["meanshift_measured"] = float(sum(op_meanshift[warm:])
                                              / max(1, len(op_meanshift[warm:])))
        summary["retained_abs_err"] = abs(
            summary["retained_measured"] - summary["retained_predicted"]
        )
    print(json.dumps(summary, indent=2))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{name}.json"), "w") as f:
        json.dump({"config": run_cfg, "summary": summary}, f, indent=2, default=str)

    if use_wandb:
        for k, v in summary.items():
            wandb.run.summary[k] = v
        wandb.finish()

    return summary


if __name__ == "__main__":
    main()
