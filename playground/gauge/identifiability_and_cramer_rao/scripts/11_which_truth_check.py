"""Which true gate set generated the saved estimates in data/crb_estimates?

The noise generator `randKrausSet` was changed during the session from one seed for all gates (all six gates get the
same noise channel) to `seed + i` (a different channel per gate). Compares the saved sampled frequencies of one
realization with the probabilities of both versions of the true gate set: the right one sits at the shot-noise level.
"""
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from mGST.utility_functions_comparisons import get_target_operators_gate_set_name, get_compressed_rep_from_mgst_output
from mGST.simulate import apply_noisy_kraus_tensors_to_target_gate_set, get_random_noisy_kraus_tensors_for_gate_set, generate_sequence_indices
from mGST import gauge_analysis as ga

DATA = Path(__file__).resolve().parent.parent / "data"
target_operators, _ = get_target_operators_gate_set_name(gate_set_name="2QXYICZ")
noise = get_random_noisy_kraus_tensors_for_gate_set(num_gates=6, dim=4, rank_kraus=4, noise_strength=5e-2, seed=42)
noise_old = {k: (np.repeat(np.asarray(v)[:1], 6, axis=0) if np.ndim(v) == 4 else v) for k, v in noise.items()}  # gate 0 has seed + 0
seqs = generate_sequence_indices(num_gates=6, num_circuits=400, seq_len_list=[1, 8, 14], seed=12345)["gate_indices"]
gates = np.full((len(seqs), 14), -1, dtype=np.int32)
for r, s in enumerate(seqs):
    gates[r, :len(s)] = np.asarray(s, dtype=np.int32)[::-1]
sampled = np.asarray(np.load(DATA / "crb_estimates" / "default_0.npz")["sampled"])
for name, nz in [("current code (seed + i)", noise), ("old code (same noise for all gates)", noise_old)]:
    sup = apply_noisy_kraus_tensors_to_target_gate_set(kraus_tensors_noisy=nz, target_gate_set=target_operators, is_target_vectorized=True)
    K, A, B = (jnp.asarray(x) for x in get_compressed_rep_from_mgst_output(*sup.values(), kraus_rank=4, state_rank=4, povm_rank=4))
    p = np.asarray(ga._circuit_probabilities(K, A, B, jnp.asarray(gates)))
    f = sampled if sampled.shape == p.shape else sampled.T
    choi = np.linalg.svd(np.asarray(K).reshape(6, 4, -1), compute_uv=False) ** 2
    choi /= choi.sum(axis=1, keepdims=True)
    print(f"{name}: mean total variation distance(sampled, truth) = {np.mean(0.5 * np.abs(f - p).sum(axis=1)):.2e}; "
          f"smallest Choi eigenvalue over gates {choi[:, -1].min():.1e}; gate-0 Choi spectrum {choi[0].round(5)}")
print("expected shot-noise level of that distance, ~", f"{np.mean(0.5 * np.sqrt(2 / np.pi) * np.sqrt(p * (1 - p) / 1000).sum(axis=1)):.2e}")
