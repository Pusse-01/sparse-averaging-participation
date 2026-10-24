"""
Pick --dtype and --batch-size for this GPU before submitting an array.

    python scripts/calibrate.py                      # defaults for the E-B/E-C shape
    python scripts/calibrate.py --m 8 --preset base  # the E-E shape

Answers two questions that are per-device and cannot be guessed:

  1. bf16 or fp32? torch.cuda.is_bf16_supported() returns True on Turing
     (sm_75) because it counts the emulated path, but emulated bf16 has no
     tensor-core backing and is often slower than fp32. Measure it.
  2. Is GPU utilisation above the site's floor? FITcluster cancels A100 jobs
     under 60%. This workload steps m replicas sequentially in one process, so
     utilisation is a real risk at small model sizes.

Prints the projected wall clock for one 6000-step run, which is what sets the
array throttles in EXPERIMENTS.md.
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import threading
import time

# invoked as `python scripts/calibrate.py`, so put the repo root on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from asyncmesh_sim.averaging import AsyncSparseAverager, AveragingConfig
from asyncmesh_sim.model import PRESETS, build_model


class GpuSampler(threading.Thread):
    """Poll nvidia-smi in the background; returns mean/min utilisation."""

    # NB: do not name the event `_stop` -- threading.Thread._stop is an
    # internal method that Thread.join() calls, and shadowing it makes join()
    # fail with "'Event' object is not callable".
    def __init__(self, interval: float = 0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[float] = []
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits", "-i", "0"],
                    capture_output=True, text=True, timeout=5,
                )
                self.samples.append(float(out.stdout.strip().split("\n")[0]))
            except Exception:  # noqa: BLE001
                pass
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        if self.is_alive():
            self.join(timeout=3)


def bench(dtype: str, args) -> tuple[float, float, float]:
    """Returns (median ms/step, mean GPU util, peak GB)."""
    device = "cuda"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(0)
    models = []
    for _ in range(args.m):
        torch.manual_seed(0)
        models.append(build_model(args.preset, args.vocab_size).to(device))
    params = [[p for p in mm.parameters()] for mm in models]
    optims = [torch.optim.AdamW(mm.parameters(), lr=3e-4) for mm in models]
    block = models[0].cfg.block_size

    avgr = AsyncSparseAverager(
        AveragingConfig(p=args.p, tau=args.tau, method="ema",
                        phi_g=1.0, phi_s=0.75),
        args.m, params[0],
    )

    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if dtype == "bf16" else torch.autocast(device_type="cpu", enabled=False))

    times: list[float] = []
    sampler = GpuSampler()
    for step in range(args.steps):
        if step == args.warmup:          # start sampling after warm-up
            sampler.start()
        t0 = time.perf_counter()
        for i in range(args.m):
            x = torch.randint(0, args.vocab_size, (args.batch_size, block),
                              device=device)
            with ctx:
                _, loss = models[i](x, x)
            optims[i].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(models[i].parameters(), 1.0)
            optims[i].step()
        avgr.step(params, step, compute_diag=(step % args.diag_interval == 0))
        torch.cuda.synchronize()
        if step >= args.warmup:
            times.append((time.perf_counter() - t0) * 1000)
    sampler.stop()

    util = statistics.mean(sampler.samples) if sampler.samples else float("nan")
    peak = torch.cuda.max_memory_allocated() / 1e9
    del models, optims, params, avgr
    torch.cuda.empty_cache()
    return statistics.median(times), util, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small", choices=list(PRESETS))
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--tau", type=int, default=10)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--diag-interval", type=int, default=25)
    ap.add_argument("--vocab-size", type=int, default=50257)
    ap.add_argument("--target-steps", type=int, default=6000,
                    help="steps in a real run, for the wall-clock projection")
    ap.add_argument("--util-floor", type=float, default=60.0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible -- run this inside an srun allocation")

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability()
    print(f"device      : {name}  sm_{cap[0]}{cap[1]}")
    print(f"shape       : preset={args.preset} m={args.m} "
          f"batch={args.batch_size} p={args.p} tau={args.tau}")
    native_bf16 = cap[0] >= 8
    print(f"native bf16 : {native_bf16}"
          f"{'' if native_bf16 else '  (emulated; expect fp32 to compete)'}\n")

    rows = []
    for dtype in ["fp32", "bf16"]:
        ms, util, peak = bench(dtype, args)
        hrs = ms * args.target_steps / 1000 / 3600
        rows.append((dtype, ms, util, peak, hrs))
        print(f"  {dtype:>5}  {ms:7.1f} ms/step   util {util:5.1f}%   "
              f"peak {peak:5.2f} GB   {hrs:4.2f} h per {args.target_steps}-step run")

    best = min(rows, key=lambda r: r[1])
    print(f"\nrecommend   : --dtype {best[0]}  ({best[1]:.0f} ms/step, "
          f"{best[4]:.2f} h per run)")

    if best[2] == best[2] and best[2] < args.util_floor:
        print(f"\nWARNING: utilisation {best[2]:.1f}% is below the {args.util_floor:.0f}% "
              f"floor.\n  On a policed partition these jobs can be cancelled. Raise "
              f"--batch-size (16, then 32)\n  and re-run; it changes no prediction, since the "
              f"theory concerns the averaging\n  operator, not the batch. Or submit to a "
              f"partition without a utilisation policy.")
    else:
        print(f"utilisation : {best[2]:.1f}%  -- clear of the "
              f"{args.util_floor:.0f}% floor")


if __name__ == "__main__":
    main()
