"""
Pre-tokenise a corpus into flat uint16 .bin files (nanoGPT convention).

Run on a machine with internet -- the login node, not a compute node.

    python scripts/prepare_data.py --dataset wikitext103 --out-dir data

WikiText-103 is the default because AsyncMesh's own run.bash uses
wikitext-103-v1 and its headline ablations (subset size, delay) are all on
WikiText.
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

# Fully-qualified `namespace/name` repo ids. The old bare canonical names
# ("wikitext", "bookcorpus") were resolvable when datasets shipped loading
# scripts, but huggingface_hub >= 1.0 parses them as hf:// URIs and rejects
# anything without a namespace:
#
#   HfUriError: Repository id must be 'namespace/name', got 'wikitext'.
#
# These are the canonical datasets' current homes, all parquet, no scripts.
DATASETS = {
    "wikitext103": ("Salesforce/wikitext", "wikitext-103-raw-v1", "text"),
    "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1", "text"),
    "bookcorpus": ("bookcorpus/bookcorpus", None, "text"),
    "openwebtext": ("Skylion007/openwebtext", None, "text"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wikitext103", choices=list(DATASETS))
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--num-proc", type=int, default=8)
    ap.add_argument("--val-fraction", type=float, default=0.01,
                    help="used only when the dataset has no validation split")
    ap.add_argument("--force", action="store_true",
                    help="rebuild .bin files that already exist")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    name, config, col = DATASETS[args.dataset]
    print(f"loading {name} {config or ''}")
    ds = load_dataset(name, config) if config else load_dataset(name)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eot = tok.eos_token_id
    print(f"tokenizer {args.tokenizer}, vocab {tok.vocab_size}, eot {eot}")

    if "validation" not in ds:
        split = ds["train"].train_test_split(test_size=args.val_fraction, seed=1337)
        ds = {"train": split["train"], "validation": split["test"]}

    os.makedirs(args.out_dir, exist_ok=True)

    def encode(example):
        ids = tok(example[col])["input_ids"]
        return {"ids": [i + [eot] for i in ids],
                "len": [len(i) + 1 for i in ids]}

    for split in ["train", "validation"]:
        suffix = "train" if split == "train" else "val"
        path = os.path.join(args.out_dir, f"{args.dataset}_{suffix}.bin")
        if os.path.exists(path) and not args.force:
            print(f"{path} exists, skipping (use --force to rebuild)")
            continue

        data = ds[split]
        enc = data.map(
            encode,
            batched=True,
            remove_columns=data.column_names,
            num_proc=args.num_proc,
            desc=f"tokenising {split}",
        )

        # Sum the length column through Arrow rather than enc["len"], which
        # materialises all 1.8M values as Python ints first.
        total = int(sum(
            c.to_numpy(zero_copy_only=False).astype(np.uint64).sum()
            for c in enc.data.column("len").chunks
        ))
        print(f"{split}: {len(enc):,} rows, {total:,} tokens "
              f"({total * 2 / 1e9:.2f} GB)", flush=True)

        # Write to a temp file and rename at the end, so an interrupted run
        # never leaves a short .bin that later looks complete. np.memmap
        # allocates the full size up front, so a partial file is full-length
        # and indistinguishable from a good one by `ls`.
        tmp = path + ".partial"
        arr = np.memmap(tmp, dtype=np.uint16, mode="w+", shape=(total,))

        idx = 0
        rows_per_shard = 1000
        n_shards = max(1, math.ceil(len(enc) / rows_per_shard))
        for b in range(n_shards):
            shard = enc.shard(num_shards=n_shards, index=b, contiguous=True)
            flat = np.concatenate([np.asarray(x, dtype=np.uint16)
                                   for x in shard["ids"]])
            arr[idx: idx + len(flat)] = flat
            idx += len(flat)
            if b % 200 == 0 or b == n_shards - 1:
                print(f"  shard {b + 1}/{n_shards}  {idx:,}/{total:,} tokens",
                      flush=True)
        arr.flush()
        del arr

        if idx != total:
            raise SystemExit(
                f"wrote {idx:,} tokens but expected {total:,} -- refusing to "
                f"rename {tmp}. This means the shard loop missed rows."
            )
        os.replace(tmp, path)
        print(f"wrote {path}: {total:,} tokens ({total * 2 / 1e9:.2f} GB)",
              flush=True)

    print("\ndone. Point --train-bin / --val-bin at these files.")


if __name__ == "__main__":
    main()
