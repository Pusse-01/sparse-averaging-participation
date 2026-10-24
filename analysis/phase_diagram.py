"""
Pulls finished runs from W&B and produces the two figures the study turns on.

    python -m analysis.phase_diagram --entity YOUR_ENTITY \
        --project asyncmesh-failure --out figures/

Figure 1 (E2): the (p_eff, tau) plane with each run marked converged/diverged,
    overlaid with the predicted boundary tau * p_eff = Pi*. If the mechanism is
    right, the empirical boundary is the hyperbola and not a vertical line.

Figure 2 (E1): validation loss against nominal p, with the ceil(1/p) buckets
    shaded. If the sampler is responsible, the curve is a staircase whose
    risers sit exactly at p = 1/2, 1/3, 1/4, ... and runs inside one bucket are
    indistinguishable.

Also writes a table comparing the measured growth rate rho_hat against the
exact monodromy prediction, which is the quantitative test of the mechanism.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

from theory.linear_model import LinearConfig, monodromy_spectral_radius


def fetch(entity: str, project: str, group: str | None = None) -> list[dict]:
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    filters = {"group": group} if group else None
    rows = []
    for r in api.runs(path, filters=filters):
        if r.state not in ("finished", "crashed", "failed"):
            continue
        s, c = dict(r.summary), dict(r.config)
        rows.append(
            {
                "name": r.name,
                "group": r.group,
                "state": r.state,
                "p": c.get("p"),
                "tau": c.get("tau"),
                "m": c.get("m"),
                "method": c.get("method"),
                "sampler": c.get("sampler"),
                "lam0": c.get("lam0"),
                "seed": c.get("seed"),
                "optimizer": c.get("optimizer"),
                "lr": c.get("lr"),
                "Pi": s.get("Pi", c.get("Pi")),
                "p_effective": s.get("p_effective", c.get("p_effective")),
                "T_cycle": s.get("T_cycle", c.get("T_cycle")),
                # a crashed run is a diverged run: NaN loss kills the process
                "diverged": int(s.get("diverged", 1 if r.state != "finished" else 0)),
                "best_val_loss": s.get("best_val_loss", float("nan")),
                "rho_hat": s.get("rho_hat", float("nan")),
                "divergence_step": s.get("divergence_step", -1),
            }
        )
    return rows


def fig_phase_boundary(rows, out, pi_star=3.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r["p_effective"] and r["tau"]]
    if not rows:
        print("no E2 rows found; skipping phase-boundary figure")
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for r in rows:
        ok = not r["diverged"]
        ax.scatter(
            r["p_effective"], r["tau"],
            marker="o" if ok else "X",
            s=70 if ok else 110,
            facecolor="white" if ok else "#c1121f",
            edgecolor="#264653" if ok else "#780000",
            linewidths=1.4, zorder=3,
        )

    xs = np.linspace(0.01, 0.6, 300)
    ax.plot(xs, pi_star / xs, color="#264653", lw=2,
            label=rf"predicted boundary $\tau\,p_{{\rm eff}}={pi_star:g}$")
    ax.fill_between(xs, pi_star / xs, 200, color="#c1121f", alpha=0.08)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.015, 0.62)
    ax.set_ylim(0.8, 150)
    ax.set_xlabel(r"realised averaging fraction $p_{\rm eff}=1/\lceil 1/p\rceil$")
    ax.set_ylabel(r"delay $\tau$ (averaging operations)")
    ax.set_title("AsyncMesh consensus stability: the boundary is a hyperbola")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(alpha=0.25, which="both", lw=0.5)

    from matplotlib.lines import Line2D
    ax.add_artist(ax.legend(handles=[
        Line2D([], [], marker="o", ls="", mfc="white", mec="#264653",
               label="converged"),
        Line2D([], [], marker="X", ls="", mfc="#c1121f", mec="#780000",
               label="diverged"),
    ], loc="lower left", frameon=False, fontsize=9))

    fig.tight_layout()
    path = os.path.join(out, "fig1_phase_boundary.png")
    fig.savefig(path, dpi=200)
    print(f"wrote {path}")


def fig_quantisation(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    for ax, sampler in zip(axes, ["partitioned", "random"]):
        sub = [r for r in rows if r["sampler"] == sampler and r["p"]]
        if not sub:
            ax.set_title(f"{sampler} (no runs)")
            continue
        by_p: dict[float, list[float]] = {}
        for r in sub:
            v = r["best_val_loss"]
            by_p.setdefault(r["p"], []).append(
                float("nan") if r["diverged"] else v
            )
        ps = sorted(by_p)
        mean = [np.nanmean(by_p[p]) if np.any(~np.isnan(by_p[p])) else np.nan
                for p in ps]
        div = [np.mean([math.isnan(x) for x in by_p[p]]) for p in ps]

        if sampler == "partitioned":
            for k in range(2, 7):  # risers at p = 1/2, 1/3, 1/4, ...
                ax.axvline(1.0 / k, color="#6c757d", ls=":", lw=1)
                ax.text(1.0 / k, 0.02, f"1/{k}", rotation=90, fontsize=8,
                        color="#6c757d", transform=ax.get_xaxis_transform(),
                        ha="right", va="bottom")

        ax.plot(ps, mean, "o-", color="#264653", label="best val loss")
        for p, d in zip(ps, div):
            if d > 0:
                ax.scatter([p], [ax.get_ylim()[1]], marker="X", s=90,
                           color="#c1121f", clip_on=False, zorder=5)
        ax.set_xlabel("nominal subset size $p$")
        ax.set_title(f"{sampler} sampler")
        ax.grid(alpha=0.25, lw=0.5)

    axes[0].set_ylabel("best validation loss (X = diverged)")
    fig.suptitle(r"Does the cliff track $p$, or $1/\lceil 1/p\rceil$?", y=1.0)
    fig.tight_layout()
    path = os.path.join(out, "fig2_quantisation.png")
    fig.savefig(path, dpi=200)
    print(f"wrote {path}")


def table_growth_rates(rows, lam=0.5):
    print("\nmeasured vs predicted consensus growth rate\n")
    print(f"{'p':>6}{'tau':>5}{'Pi':>7}{'rho_pred':>11}{'rho_hat':>11}"
          f"{'diverged':>10}")
    print("-" * 50)
    seen = set()
    for r in sorted(rows, key=lambda r: (r["Pi"] or 0)):
        key = (r["p"], r["tau"], r["sampler"])
        if key in seen or r["p"] is None:
            continue
        seen.add(key)
        cfg = LinearConfig(p=r["p"], tau=r["tau"], lam=lam,
                           sampler=r["sampler"] or "partitioned")
        pred = monodromy_spectral_radius(cfg)
        hat = r["rho_hat"]
        hat_s = "  n/a" if hat is None or (isinstance(hat, float) and math.isnan(hat)) \
            else f"{hat:.5f}"
        print(f"{r['p']:>6.2f}{r['tau']:>5}{r['Pi']:>7.2f}{pred:>11.5f}"
              f"{hat_s:>11}{'yes' if r['diverged'] else 'no':>10}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="")
    ap.add_argument("--project", default="asyncmesh-failure")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--pi-star", type=float, default=3.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = fetch(args.entity, args.project)
    print(f"fetched {len(rows)} runs")
    if not rows:
        return

    e2 = [r for r in rows if r["group"] == "e2_phase_boundary"]
    e1 = [r for r in rows if r["group"] == "e1_quantisation"]

    fig_phase_boundary(e2 or rows, args.out, args.pi_star)
    fig_quantisation(e1 or rows, args.out)
    table_growth_rates(e2 or rows)


if __name__ == "__main__":
    main()
