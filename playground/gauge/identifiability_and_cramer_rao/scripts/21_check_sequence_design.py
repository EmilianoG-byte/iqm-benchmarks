"""Check of the depth fix in mGST.additional_fns.random_seq_design (sequences longer than 23 gates).

1) Reproducibility: for every design that the old implementation could produce, the new one returns the identical
   array. The old implementation is taken from git (commit 5a12696, before the fix) and executed in its own namespace.
2) Long sequences (L_max = 28, 48): correct number of sequences per length, no duplicates, indices in [0, d),
   padding with -1 only at the end. Also generate_sequence_indices, which the scripts and notebooks call.

Usage: python 21_check_sequence_design.py      (a few seconds)
"""
import random
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from mGST import additional_fns
from mGST.low_level_jit import local_basis
from mGST.simulate import generate_sequence_indices

REPO = Path(__file__).resolve().parents[4]
OLD_COMMIT = "5a12696"

# 1) old implementation, verbatim from git
source = subprocess.run(["git", "show", f"{OLD_COMMIT}:src/mGST/additional_fns.py"], cwd=REPO, capture_output=True, text=True, check=True).stdout
start = source.index("def random_seq_design(")
end = source.index("\ndef ", start + 1)
namespace = {"np": np, "random": random, "local_basis": local_basis}
exec(source[start:end], namespace)
old_random_seq_design = namespace["random_seq_design"]


def design(fn, d, L_max, N, seed):
    # the call made by generate_sequence_indices with seq_len_list = [1, L_max // 2 + 1, L_max]
    return fn(d, 1, L_max // 2 + 1, L_max, int(np.ceil(N / 2)), int(np.floor(N / 2)), seed=seed)


for d, L_max, N in [(6, 6, 400), (6, 14, 400), (6, 14, 1000), (6, 23, 1000), (9, 19, 2000)]:
    for seed in [0, 1, 2, 12345]:
        old, new = design(old_random_seq_design, d, L_max, N, seed), design(additional_fns.random_seq_design, d, L_max, N, seed)
        assert old.shape == new.shape and np.array_equal(old, new), (d, L_max, N, seed)
print("1) identical to the old implementation for d = 6 (L_max = 6, 14, 23) and d = 9 (L_max = 19), seeds 0, 1, 2, 12345")


# 2) long sequences
def check(J, d, L_max):
    rows = J[1:]  # the first row of random_seq_design is a dummy row of -1
    lengths = (rows >= 0).sum(axis=1)
    for row, l in zip(rows, lengths):
        assert np.all(row[:l] >= 0) and np.all(row[:l] < d) and np.all(row[l:] == -1)
    assert len({tuple(r) for r in rows}) == len(rows), "duplicate sequences"
    return Counter(lengths.tolist())


for d, L_max, N in [(6, 28, 1000), (6, 48, 1000), (6, 48, 400), (9, 24, 2000), (9, 48, 2000)]:
    counts = check(design(additional_fns.random_seq_design, d, L_max, N, 0), d, L_max)
    long_counts = [counts[l] for l in range(L_max // 2 + 2, L_max + 1)]
    assert sum(counts.values()) == N - 1, (sum(counts.values()), N)  # the old code also returns N - 1 sequences
    assert max(long_counts) - min(long_counts) <= 1, long_counts
    rerun = design(additional_fns.random_seq_design, d, L_max, N, 0)
    assert np.array_equal(rerun, design(additional_fns.random_seq_design, d, L_max, N, 0)), "not reproducible"
    print(f"2) d = {d}, L_max = {L_max}, N = {N}: {sum(counts.values())} distinct sequences, "
          f"{min(long_counts)}-{max(long_counts)} per long length, reproducible with the seed")

indices = generate_sequence_indices(num_gates=6, num_circuits=1000, seq_len_list=[1, 25, 48], seed=0)["gate_indices"]
assert max(len(s) for s in indices) == 48 and all(0 <= g < 6 for s in indices for g in s)
print(f"   generate_sequence_indices(L_max = 48): {len(indices)} sequences, longest {max(len(s) for s in indices)}")
print("all checks passed")
