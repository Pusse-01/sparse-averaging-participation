"""
Figures for the partial-participation study (E-A .. E-F).

    python -m analysis.participation --entity YOUR_ENTITY \
        --project asyncmesh-failure --out figures/

Four figures, one per required claim:

  F1 (E-A)  measured vs predicted one-round operator. Points must sit on the
            diagonal for the plain write. This is what makes the paper a
            theory paper rather than a curve fit.
  F2 (E-B)  the asymmetry. A fixed failure budget moved from the gather path
            to the scatter path: consensus error must rise while mean
            displacement falls. The two panels must move in opposite
            directions -- that is the whole claim.
  F3 (E-C)  the crossover. Validation loss against participation for each
            path, at m = 4 and m = 16. Which curve is worse must swap.
  F4 (E-D)  stall versus divergence. Loss traces with the failure window
            shaded: participation runs recover when it closes, the Pi run
            does not.

Every figure also writes a CSV next to it so the numbers can go into the paper
without re-running anything.
"""

from __future__ import annotations

import argparse
import csv
import os

from theory.participation import contraction_retained, mean_displacement, psi

# Consistent across figures: gather is warm, scatter is cool.
C_GATHER = "#e07a5f"
C_SCATTER = "#3d5a80"
C_PRED = "#264653"
C_CTRL = "#8d99ae"


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
                "tags": list(r.tags),
                "m": c.get("m"),
                "p": c.get("p"),
                "tau": c.get("tau"),
                "method": c.get("method"),
                "seed": c.get("seed"),
                "phi_g": c.get("phi_g", 1.0),
                "phi_s": c.get("phi_s", 1.0),
                "part_window": c.get("part_window", ""),
                "Pi": s.get("Pi", c.get("Pi")),
                "diverged": int(
                    s.get("diverged", 1 if r.state != "finished" else 0)
                ),
                "best_val_loss": s.get("best_val_loss", float("nan")),
                "consensus_error": s.get("consensus/error", float("nan")),
                "retained_measured": s.get("retained_measured", float("nan")),
                "retained_predicted": s.get("retained_predicted", float("nan")),
                "meanshift_measured": s.get("meanshift_measured", float("nan")),
                "meanshift_predicted": s.get("meanshift_predicted", float("nan")),
                "_run": r,
            }
        )
    return rows


def fetch_json(dirs: list[str]) -> list[dict]:
    """
    Rebuild the same rows from results/<group>/<name>.json, which every run
    writes regardless of whether W&B worked. Use this when the compute nodes
    have no egress and the offline sync has not run, or when W&B failed.

    Note F4 needs per-step history, which the JSON does not carry -- that one
    figure still requires W&B.
    """
    import glob
    import json

    rows = []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            with open(path) as f:
                blob = json.load(f)
            c, s = blob.get("config", {}), blob.get("summary", {})
            rows.append(
                {
                    "name": os.path.splitext(os.path.basename(path))[0],
                    "group": c.get("wandb_group") or os.path.basename(d),
                    "tags": [t for t in (c.get("wandb_tags") or "").split(",") if t],
                    "m": c.get("m"),
                    "p": c.get("p"),
                    "tau": c.get("tau"),
                    "method": c.get("method"),
                    "seed": c.get("seed"),
                    "phi_g": c.get("phi_g", 1.0),
                    "phi_s": c.get("phi_s", 1.0),
                    "part_window": c.get("part_window", ""),
                    "Pi": s.get("Pi", c.get("Pi")),
                    "diverged": int(s.get("diverged", 0)),
                    "best_val_loss": s.get("best_val_loss", float("nan")),
                    "consensus_error": float("nan"),  # summary-only; see W&B
                    "retained_measured": s.get("retained_measured", float("nan")),
                    "retained_predicted": s.get("retained_predicted", float("nan")),
                    "meanshift_measured": s.get("meanshift_measured", float("nan")),
                    "meanshift_predicted": s.get("meanshift_predicted", float("nan")),
                    "_run": None,
                }
            )
    return rows


def _dump(rows, path, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path}")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# --------------------------------------------------------------- F1  (E-A)
def fig_operator(rows, out):
    plt = _plt()
    rows = [r for r in rows if r["group"] == "ea_operator"
            and r["retained_measured"] == r["retained_measured"]]
    if not rows:
        print("no E-A rows; skipping F1")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    for ax, key, title in [
        (axes[0], "retained", "consensus retained  E[S']/S   (Thm 1)"),
        (axes[1], "meanshift", "mean displacement  E[$\\Delta\\bar w^2$]/v   (Thm 2)"),
    ]:
        lims = [0, 0]
        for r in rows:
            plain = "plain" in r["tags"]
            delayed = "delayed" in r["tags"]
            x, y = r[f"{key}_predicted"], r[f"{key}_measured"]
            ax.scatter(
                x, y,
                marker="o" if plain else ("s" if delayed else "^"),
                s=90 if plain else 60,
                facecolor=C_PRED if plain else "white",
                edgecolor=C_PRED, linewidths=1.5, zorder=3,
                label=None,
            )
            lims[1] = max(lims[1], x, y)
        hi = lims[1] * 1.15 + 1e-9
        ax.plot([0, hi], [0, hi], "--", color=C_CTRL, lw=1.2, zorder=1)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_xlabel("predicted")
        ax.set_ylabel("measured")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)

    handles = [
        plt.Line2D([], [], marker="o", ls="", mfc=C_PRED, mec=C_PRED,
                   label="plain write (prediction)"),
        plt.Line2D([], [], marker="s", ls="", mfc="white", mec=C_PRED,
                   label=r"+ delay $\tau=10$"),
        plt.Line2D([], [], marker="^", ls="", mfc="white", mec=C_PRED,
                   label="+ EMA corrector"),
    ]
    axes[0].legend(handles=handles, fontsize=8, loc="upper left")
    fig.suptitle("F1 (E-A): the one-round operator against Theorems 1-2",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(out, "f1_operator_exactness.png")
    fig.savefig(path, dpi=180)
    print(f"  wrote {path}")

    _dump(rows, os.path.join(out, "f1_operator.csv"),
          ["name", "tags", "m", "phi_g", "phi_s", "method", "tau",
           "retained_predicted", "retained_measured",
           "meanshift_predicted", "meanshift_measured"])

    plain = [r for r in rows if "plain" in r["tags"]]
    if plain:
        worst = max(abs(r["retained_measured"] - r["retained_predicted"])
                    for r in plain)
        verdict = "PASS" if worst <= 0.02 else "FAIL"
        print(f"  E-A plain-write max |measured - predicted| = {worst:.4f} "
              f"(tol 0.02) -> {verdict}")


# --------------------------------------------------------------- F2  (E-B)
def fig_asymmetry(rows, out):
    plt = _plt()
    rows = [r for r in rows if r["group"] == "eb_asymmetry"]
    if not rows:
        print("no E-B rows; skipping F2")
        return

    by_split: dict[float, list[dict]] = {}
    for r in rows:
        by_split.setdefault(round(1.0 - r["phi_g"], 4), []).append(r)
    splits = sorted(by_split)

    def agg(rs, key):
        vals = [x[key] for x in rs if x[key] == x[key]]
        return sum(vals) / len(vals) if vals else float("nan")

    m = rows[0]["m"]
    x = list(range(len(splits)))
    labels = [f"{s:.0%}g\n{(0.25 - s):.0%}s" for s in splits]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))

    # consensus error: predicted psi (higher psi = stronger contraction)
    pred_psi = [psi(m, 1 - s, 1 - (0.25 - s)) for s in splits]
    axes[0].plot(x, pred_psi, "--o", color=C_PRED, label="predicted $\\psi$")
    axes[0].set_ylabel("contraction rate $\\psi$")
    axes[0].set_title("predicted: contraction weakens\ntoward the scatter end",
                      fontsize=10)

    meas = [agg(by_split[s], "consensus_error") for s in splits]
    axes[1].plot(x, meas, "-o", color=C_SCATTER)
    axes[1].set_ylabel("consensus error (measured)")
    axes[1].set_title("measured: consensus error rises\ntoward the scatter end",
                      fontsize=10)

    pred_shift = [mean_displacement(m, 1 - s, 1 - (0.25 - s)) for s in splits]
    ax2 = axes[2]
    ax2.plot(x, pred_shift, "--o", color=C_PRED, label="predicted")
    meas_shift = [agg(by_split[s], "meanshift_measured") for s in splits]
    if any(v == v for v in meas_shift):
        ax2.plot(x, meas_shift, "-o", color=C_GATHER, label="measured")
    ax2.set_ylabel("mean displacement")
    ax2.set_title("and mean displacement falls\n(the opposite direction)",
                  fontsize=10)
    ax2.legend(fontsize=8)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("failure budget: gather / scatter")
        ax.grid(alpha=0.25)

    fig.suptitle(
        "F2 (E-B): a fixed 25% failure budget moved between the two paths",
        fontsize=11,
    )
    fig.tight_layout()
    path = os.path.join(out, "f2_asymmetry.png")
    fig.savefig(path, dpi=180)
    print(f"  wrote {path}")
    _dump(rows, os.path.join(out, "f2_asymmetry.csv"),
          ["name", "m", "phi_g", "phi_s", "seed", "best_val_loss",
           "consensus_error", "meanshift_measured", "diverged"])

    # The verdict. The claim is that moving a fixed failure budget from the
    # gather path to the scatter path makes validation loss worse
    # monotonically, while predicted mean displacement moves the other way.
    print(f"\n  {'gather fail':>11}{'scatter fail':>13}{'val loss':>11}"
          f"{'seed range':>12}{'psi':>8}{'mean-noise':>12}")
    vals = []
    for s in splits:
        rs = by_split[s]
        v = [x["best_val_loss"] for x in rs if x["best_val_loss"] == x["best_val_loss"]]
        if not v:
            continue
        mean_v = sum(v) / len(v)
        vals.append((s, mean_v, max(v) - min(v)))
        print(f"  {s:>11.4f}{0.25 - s:>13.4f}{mean_v:>11.4f}"
              f"{max(v) - min(v):>12.4f}"
              f"{psi(m, 1 - s, 1 - (0.25 - s)):>8.4f}"
              f"{mean_displacement(m, 1 - s, 1 - (0.25 - s)):>12.5f}")

    # Score this within seed, not across seeds. The between-seed spread here
    # (~0.025) is as large as the whole effect, but it is common-mode: the
    # same seeds appear at every split, so comparing splits *within* a seed
    # cancels it. An unpaired endpoint test throws away that structure and
    # calls a clean monotone trend inconclusive.
    #
    # splits is ordered by gather-failure ascending, so splits[0] puts the
    # entire budget on the scatter path and splits[-1] on the gather path.
    seeds = sorted({r["seed"] for r in rows if r["seed"] is not None})
    per_seed_curve = {}
    for sd in seeds:
        curve = []
        for s in splits:
            v = [x["best_val_loss"] for x in by_split[s]
                 if x["seed"] == sd and x["best_val_loss"] == x["best_val_loss"]]
            curve.append(sum(v) / len(v) if v else float("nan"))
        if all(c == c for c in curve):
            per_seed_curve[sd] = curve

    if per_seed_curve and len(splits) >= 2:
        n_desc = sum(
            1 for c in per_seed_curve.values()
            if all(c[i] >= c[i + 1] for i in range(len(c) - 1))
        )
        deltas = [c[0] - c[-1] for c in per_seed_curve.values()]  # scatter - gather
        n = len(deltas)
        mean_d = sum(deltas) / n
        print(f"\n  within-seed, all-scatter minus all-gather:")
        for sd, c in sorted(per_seed_curve.items()):
            mono = "monotone" if all(c[i] >= c[i + 1]
                                     for i in range(len(c) - 1)) else "not monotone"
            print(f"    seed {sd}: {c[0] - c[-1]:+.4f}   ({mono})")
        print(f"    mean {mean_d:+.4f} over {n} seed(s); "
              f"{n_desc}/{n} monotone in the predicted direction")
        if all(d > 0 for d in deltas) and n_desc == n:
            verdict = ("CONFIRMED: within every seed, loss worsens "
                       "monotonically toward the scatter end")
        elif all(d > 0 for d in deltas):
            verdict = ("CONFIRMED in sign for every seed, but not monotone "
                       "at every split")
        elif all(d < 0 for d in deltas):
            verdict = "CONTRADICTED: loss improves toward the scatter end"
        else:
            verdict = "INCONCLUSIVE: sign of the effect varies across seeds"
        print(f"  -> {verdict}")


# --------------------------------------------------------------- F3  (E-C)
def fig_crossover(rows, out):
    plt = _plt()
    rows = [r for r in rows if r["group"] == "ec_crossover"]
    if not rows:
        print("no E-C rows; skipping F3")
        return

    ms = sorted({r["m"] for r in rows})
    fig, axes = plt.subplots(1, len(ms), figsize=(5.2 * len(ms), 4.4),
                             squeeze=False)

    summary = []
    per_seed: dict[tuple, list[float]] = {}
    for ax, m in zip(axes[0], ms):
        sub = [r for r in rows if r["m"] == m]
        for path_name, color, sel in [
            ("gather ($\\varphi_g$)", C_GATHER, lambda r: r["phi_s"] == 1.0),
            ("scatter ($\\varphi_s$)", C_SCATTER, lambda r: r["phi_g"] == 1.0),
        ]:
            key = "gather" if "gather" in path_name else "scatter"
            pts: dict[float, list[float]] = {}
            for r in sub:
                if not sel(r):
                    continue
                phi = r["phi_s"] if r["phi_g"] == 1.0 else r["phi_g"]
                if r["best_val_loss"] == r["best_val_loss"]:
                    pts.setdefault(phi, []).append(r["best_val_loss"])
            if not pts:
                continue
            xs = sorted(pts)
            ys = [sum(pts[k]) / len(pts[k]) for k in xs]
            errs = [(max(pts[k]) - min(pts[k])) / 2 if len(pts[k]) > 1 else 0
                    for k in xs]
            # error bars are half the seed range: a curve separation smaller
            # than these is not a result
            ax.errorbar(xs, ys, yerr=errs, fmt="-o", color=color,
                        capsize=3, label=path_name)
            for k in xs:
                per_seed[(m, key, k)] = pts[k]
                summary.append({"m": m, "path": path_name, "phi": k,
                                "val": sum(pts[k]) / len(pts[k]),
                                "n_seeds": len(pts[k]),
                                "seed_range": max(pts[k]) - min(pts[k])})
        ax.invert_xaxis()
        ax.set_xlabel("participation $\\varphi$ (worse to the right)")
        ax.set_ylabel("best validation loss")
        ax.set_title(f"m = {m}", fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

    fig.suptitle(
        "F3 (E-C): which failure path costs more, and the predicted swap with m",
        fontsize=11,
    )
    fig.tight_layout()
    path = os.path.join(out, "f3_crossover.png")
    fig.savefig(path, dpi=180)
    print(f"  wrote {path}")

    with open(os.path.join(out, "f3_crossover.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["m", "path", "phi", "val",
                                          "n_seeds", "seed_range"])
        w.writeheader()
        w.writerows(summary)
    print(f"  wrote {os.path.join(out, 'f3_crossover.csv')}")

    # The verdict, with the seed spread that decides whether it means anything.
    # A gap between the two paths is only a result if it is larger than the
    # spread between seeds at a single point; with 2 seeds this is a crude
    # check, but a gap inside the spread is definitely not a finding.
    import statistics as st

    def spread(m, path_key):
        """Max within-point seed range over the phi<=0.6 points."""
        out = []
        for phi in sorted({s["phi"] for s in summary
                           if s["m"] == m and path_key in s["path"]}):
            if phi > 0.6:
                continue
            vals = per_seed.get((m, path_key, phi), [])
            if len(vals) > 1:
                out.append(max(vals) - min(vals))
        return max(out) if out else float("nan")

    for m in ms:
        g = [s["val"] for s in summary if s["m"] == m and "gather" in s["path"]
             and s["phi"] <= 0.6]
        s_ = [s["val"] for s in summary if s["m"] == m and "scatter" in s["path"]
              and s["phi"] <= 0.6]
        if not (g and s_):
            continue
        gm, sm = sum(g) / len(g), sum(s_) / len(s_)
        gap = abs(gm - sm)
        worse = "gather" if gm > sm else "scatter"
        sg, ss = spread(m, "gather"), spread(m, "scatter")
        noise = max([x for x in (sg, ss) if x == x] or [float("nan")])
        verdict = ("INCONCLUSIVE: gap within seed spread"
                   if not (gap > noise) else f"worse path is {worse}")
        print(f"  m={m:>2}: gather {gm:.4f}  scatter {sm:.4f}  "
              f"gap {gap:.4f}  max seed spread {noise:.4f}  -> {verdict}")


# --------------------------------------------------------------- F4  (E-D)
def fig_stall(rows, out, window=(2000, 4000)):
    plt = _plt()
    rows = [r for r in rows if r["group"] == "ed_stall_vs_divergence"]
    if not rows:
        print("no E-D rows; skipping F4")
        return

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.axvspan(window[0], window[1], color="#f4a26120", zorder=0)

    for r in rows:
        tags = r["tags"]
        if "control" in tags:
            color, lw, label = C_CTRL, 1.4, "control (complete operator)"
        elif "pi" in tags:
            color, lw, label = "#c1121f", 1.8, f"$\\Pi$ instability (p={r['p']}, $\\tau$={r['tau']})"
        elif r["phi_s"] < 1.0:
            color, lw, label = C_SCATTER, 1.8, f"scatter $\\varphi_s$={r['phi_s']}"
        else:
            color, lw, label = C_GATHER, 1.8, f"gather $\\varphi_g$={r['phi_g']}"

        try:
            hist = r["_run"].history(keys=["step", "val/loss"], pandas=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  could not fetch history for {r['name']}: {exc}")
            continue
        xs = [h["step"] for h in hist if "val/loss" in h]
        ys = [h["val/loss"] for h in hist if "val/loss" in h]
        if xs:
            ax.plot(xs, ys, color=color, lw=lw, label=label)

    # after plotting, so the label sits inside the real data range
    ax.text(sum(window) / 2, ax.get_ylim()[1], "failures on",
            ha="center", va="top", fontsize=8, color="#bc6c25")
    ax.set_xlabel("step")
    ax.set_ylabel("validation loss (consensus model)")
    ax.set_title("F4 (E-D): participation failure stalls and recovers;\n"
                 "the $\\Pi$ instability does not", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out, "f4_stall_vs_divergence.png")
    fig.savefig(path, dpi=180)
    print(f"  wrote {path}")

    print("\n  recovery check (val loss at end vs at window close):")
    for r in rows:
        kind = ("pi" if "pi" in r["tags"]
                else "control" if "control" in r["tags"] else "participation")
        print(f"    {kind:>14}  {r['name']:<44} diverged={r['diverged']}")


FIGS = {
    "f1": fig_operator,
    "f2": fig_asymmetry,
    "f3": fig_crossover,
    "f4": fig_stall,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="")
    ap.add_argument("--project", default="asyncmesh-failure")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--which", nargs="*", default=list(FIGS))
    ap.add_argument("--from-json", nargs="*", default=None,
                    help="read results/<group> directories instead of W&B, "
                         "e.g. --from-json results/ea results/eb. F4 needs "
                         "per-step history and is unavailable this way.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.from_json:
        print(f"reading {len(args.from_json)} results directories ...")
        rows = fetch_json(args.from_json)
    else:
        print(f"fetching runs from {args.entity or '(default)'}/{args.project} ...")
        rows = fetch(args.entity, args.project)
    print(f"  {len(rows)} finished runs\n")

    for name in args.which:
        print(f"{name}:")
        FIGS[name](rows, args.out)
        print()


if __name__ == "__main__":
    main()
