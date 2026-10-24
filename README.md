# Which Half of Participation Does the Sparse-Averaging Contraction Need?

Code, data and paper for the CODEC-FM 2026 (NeurIPS workshop) tiny paper.

**[Paper (PDF)](paper/main.pdf)** · **[Project page](https://pusse-01.github.io/sparse-averaging-participation/)** · [OpenReview](https://openreview.net/forum?id=MTzRGBVAfc)

---

## The result in one paragraph

Decentralized training synchronizes replicas by periodically averaging a random
subset of their weights. Existing guarantees assume every replica both
**contributes** to the average and **receives** the result. In deployment these
are two network events that fail independently, and they are already logged as
separate counters. We compute one averaging round exactly, for an arbitrary
weight configuration, with the two fractions separated: `φ_g` (send/contribute)
and `φ_s` (receive/adopt).

The two halves are not interchangeable.

| | contraction of disagreement | shift in the agreed value |
|---|---|---|
| receive failure (`φ_s < 1`) | reduces it **linearly** | — |
| send failure (`φ_g < 1`) | absent at leading order | **displaces the mean**, diverging as `φ_g → 0` |

So the contraction needs only the receive half: replicas still agree exactly
when every replica receives, however few contribute. Send failures instead act
on a second channel that doubly stochastic mixing *cannot represent*, because
double stochasticity is exactly the statement that a round preserves the mean.

Two consequences: the effective averaging rate is `p·φ_s`, not `p`, which
narrows the admissible step size; and a staleness corrector is revealed as a
finite-time **transfer** of error between the two channels rather than a removal
of it.

## Quick start — the whole argument, on a laptop, in seconds

The central claim is a closed-form identity, so it needs no GPU and no data:

```bash
pip install numpy
python -m theory.participation
```

This prints the closed form, checks it against Monte Carlo, and reproduces
**Table 1 of the paper** — a fixed 25% failure budget shifted from the receive
path to the send path:

```
 gather fail scatter fail       psi  mean-noise
      0.2500       0.0000   1.00000     0.02222
      0.1875       0.0625   0.94050     0.01743
      0.1250       0.1250   0.88125     0.01458
      0.0625       0.1875   0.82198     0.01309
      0.0000       0.2500   0.76250     0.01250
```

`psi` is the effective receive fraction (`ψ = φ_s + O(1/m)`); it falls as the
budget moves to the receive path, while the predicted mean displacement moves
the opposite way. That opposition is the paper's whole point, and the loss
column in Table 1 follows `ψ` monotonically.

## What reproduces what

| Paper | Experiment | Data shipped here | Command |
|---|---|---|---|
| Table 1, Appendix F | E-F, E-G | `results/ef`, `results/eg_a100` | `python -m analysis.participation --from-json results/ef results/eg_a100` |
| Appendix C, Table 2 — operator exactness, 11 configs, agreement to `3.2e-3` | E-A | `results/ea` | `python -m analysis.participation --from-json results/ea --which f1` |
| Appendix C, Table 3 — the corrector's two channels | E-B | `results/eb` | `python -m analysis.participation --from-json results/eb --which f2` |
| Appendix C, Table 4 — delay sweep, `Π = τp` | E-I | `results/ei` | `python -m analysis.participation --from-json results/ei` |
| Appendix E + `figures/f3_crossover.png` — the pre-registered crossover, **falsified** | E-C | `results/ec` | `python -m analysis.participation --from-json results/ec --which f3` |
| Appendix H — stall and recovery | E-D | `results/ed` | `python -m analysis.participation --from-json results/ed --which f4` |
| LM scale | E-E | `results/ee` | — |
| Agora's averaging cadence | E-H | `results/eh` | — |

`--from-json` reads the run summaries in `results/` directly, so none of the
table reproduction needs Weights & Biases. Figure rendering needs `matplotlib`.

## Reproducing the training runs from scratch

```bash
pip install -r requirements.txt
python scripts/prepare_data.py          # WikiText-103, needs internet
python -m experiments.grids --all --out slurm/manifests
sbatch --array=1-11 slurm/run_array.sbatch slurm/manifests/e1.txt
```

One run is one `python -m asyncmesh_sim.train ...` line in a manifest; the SLURM
array index selects the line, so manifests are the unit of reproduction. The
flags that matter here are `--phi-g` and `--phi-s` (the two participation
fractions: each round draws a uniform random subset of exactly that many
replicas, independently for the two phases), `--tau` (delay),
`--sampler` (`partitioned` or `bernoulli`), and `--part-window` (failures active
only in a step range, used for the stall-and-recovery experiment).

## Layout

```
paper/              camera-ready LaTeX source and PDF
theory/             closed-form operator; numpy only, no GPU
  participation.py    the two-channel identity + Monte Carlo check  <- start here
  linear_model.py     quadratic surrogate for the stability boundary
  pp_coupling.py      pipeline-delay cross-axis check (negative result)
  paper_datapoints.py numbers quoted in the text
asyncmesh_sim/      the training implementation the theory is tested against
  averaging.py        sparse averaging with both failure channels
  participation.py    per-replica send/receive failure injection
  sampler.py          partitioned (cycling) and Bernoulli coordinate selectors
experiments/grids.py  generates the run manifests
analysis/           tables and figures from results/ or W&B
tests/              the load-bearing claims, as assertions
slurm/              cluster runners and the generated manifests
results/            per-run JSON summaries backing every table (E-A … E-I)
figures/            the paper's figure and its source CSV
docs/               project page
```

## Experimental configuration

All runs use the 45M `small` preset (6 layers, d=512, block 512, GPT-2
tokenizer, tied embeddings) on WikiText-103, batch 8, AdamW at 3e-4 with 600
warmup steps and cosine decay, weight decay 0.01, gradient clipping 1.0, bf16
autocast. The `m` replicas are stepped **sequentially in a single process**,
which makes a run bit-reproducible given its seed — a requirement for the
operator-exactness test, where predicted and measured quantities are compared on
the same weights. Sparse averaging uses the partitioned index selector, which
splits each parameter into `ceil(1/p)` disjoint parts and cycles them, so the
realised fraction is `1/ceil(1/p)`; every `p` quoted is a realised fraction.

Divergence is decided by a rule fixed before any run (non-finite loss;
validation loss at or above `ln V` after step 500; or validation loss regressing
more than 0.5 above its best), never by inspection.

## Scope

The identity is exact for arbitrary replica weights and arbitrary `m`, with no
distributional assumption and no replica homogeneity. It is a **one-round**
statement, not a convergence guarantee under heterogeneous objectives. The
empirical work is at 45M parameters, `m ≤ 16`, with replicas simulated
sequentially; the deployment recommendation should be read at that scale.

## Tests

```bash
python -m tests.test_participation   # needs torch: drives the real averaging code
python -m tests.test_claims          # needs torch
python -m theory.participation       # numpy only
```

## Citation

```bibtex
@inproceedings{pussadeniya2026participation,
  title     = {Which Half of Participation Does the Sparse-Averaging
               Contraction Need?},
  author    = {Pussadeniya, Nirodya},
  booktitle = {NeurIPS 2026 Workshop on Collaborative, Open and Decentralized
               Training of Foundation Models (CODEC-FM)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=MTzRGBVAfc}
}
```

## License

MIT, see [LICENSE](LICENSE).
