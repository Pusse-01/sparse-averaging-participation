"""
Token stream loading.

Data is pre-tokenised to flat uint16 .bin files (nanoGPT convention) by
scripts/prepare_data.py, which must be run somewhere with internet access --
compute nodes on the cluster generally have none.

Each DP replica draws from a disjoint shard of the training stream, matching
AsyncMesh's i.i.d. data split (D_i in Eq. 2). For the heterogeneity extension
in E4, pass different .bin files per replica instead.
"""

from __future__ import annotations

import os

import numpy as np
import torch


class ShardedTokenStream:
    """Contiguous-shard sampler: replica i draws only from its own shard."""

    def __init__(self, bin_path: str, m: int, block_size: int, seed: int = 0):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(
                f"{bin_path} not found. Run scripts/prepare_data.py first "
                "(on a node with internet access)."
            )
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.m = m
        self.block_size = block_size
        self.shard_len = len(self.data) // m
        if self.shard_len <= block_size + 1:
            raise ValueError(
                f"{bin_path} has {len(self.data)} tokens, too few for m={m} "
                f"replicas at block_size={block_size}"
            )
        self.rngs = [np.random.default_rng(seed * 1000 + i) for i in range(m)]

    def batch(self, replica: int, batch_size: int, device: str):
        lo = replica * self.shard_len
        hi = lo + self.shard_len - self.block_size - 1
        ix = self.rngs[replica].integers(lo, hi, size=batch_size)
        x = np.stack([self.data[i : i + self.block_size].astype(np.int64) for i in ix])
        y = np.stack(
            [self.data[i + 1 : i + 1 + self.block_size].astype(np.int64) for i in ix]
        )
        x = torch.from_numpy(x).to(device, non_blocking=True)
        y = torch.from_numpy(y).to(device, non_blocking=True)
        return x, y


class ValStream:
    """Deterministic validation batches, identical for every run."""

    def __init__(self, bin_path: str, block_size: int, batch_size: int,
                 n_batches: int = 40, seed: int = 1234):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        rng = np.random.default_rng(seed)
        hi = len(self.data) - block_size - 1
        self.starts = rng.integers(0, hi, size=n_batches * batch_size)
        self.block_size = block_size
        self.batch_size = batch_size
        self.n_batches = n_batches

    def __iter__(self):
        for b in range(self.n_batches):
            ix = self.starts[b * self.batch_size : (b + 1) * self.batch_size]
            x = np.stack(
                [self.data[i : i + self.block_size].astype(np.int64) for i in ix]
            )
            y = np.stack(
                [
                    self.data[i + 1 : i + 1 + self.block_size].astype(np.int64)
                    for i in ix
                ]
            )
            yield torch.from_numpy(x), torch.from_numpy(y)


class SyntheticStream:
    """
    Deterministic pseudo-text for smoke tests and for cluster nodes with no
    prepared data. Not a substitute for real corpora in reported results.
    """

    def __init__(self, vocab_size: int, m: int, block_size: int, seed: int = 0):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.rngs = [np.random.default_rng(seed * 1000 + i) for i in range(m)]

    def batch(self, replica: int, batch_size: int, device: str):
        r = self.rngs[replica]
        # low-entropy Markov-ish stream so the loss actually moves
        base = r.integers(0, self.vocab_size, size=(batch_size, 1))
        offs = np.cumsum(
            r.integers(0, 3, size=(batch_size, self.block_size + 1)), axis=1
        )
        toks = (base + offs) % self.vocab_size
        x = torch.from_numpy(toks[:, :-1].astype(np.int64)).to(device)
        y = torch.from_numpy(toks[:, 1:].astype(np.int64)).to(device)
        return x, y
