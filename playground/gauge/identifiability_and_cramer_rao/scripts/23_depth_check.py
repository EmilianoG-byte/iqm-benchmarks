"""Depth check for the notebook depth_check_2q.ipynb: does a larger maximum sequence length L_max still lower the
shots needed, and where does the signal of long circuits collapse?

Three evaluation points (gate sets), each at one N and one design seed, L_max in {14, 18, 23, 28, 32, 40, 48}:
- "noise 0.05 prior": target plus random noise of strength 0.05, noise seed 7 (hardware-like; regime A), N = 600;
- "noise 0.3 truth":  strength 0.3, noise seed 42 (the truth of ptm_precision_design_2q.ipynb; regime B), N = 1000;
- "IQMFakeApollo":    the fake backend's noise on qubits (0, 1) (ptm_design.fake_backend_gate_set), N = 1000.
Per design: ptm_design.depth_check (widths with and without near-GL-gauge directions, outcome probabilities).
The van Trees width uses tau = 2 tau_0 per point, tau_0 from the reference design (N = 400, L_max = 14, m = 1000) as in
the old notebook. Writes data/depth_check_2q.npz.
Usage: python 23_depth_check.py      (about 20 min; prints one line per design)
"""
import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp

import ptm_design as pdz
from mGST import gauge_analysis as ga
from mGST.utility_functions_comparisons import get_target_operators_gate_set_name, get_compressed_rep_from_mgst_output
from mGST.simulate import apply_noisy_kraus_tensors_to_target_gate_set, get_random_noisy_kraus_tensors_for_gate_set, generate_sequence_indices
from iqm.qiskit_iqm.fake_backends import IQMFakeApollo

OUT = Path(__file__).resolve().parent.parent / "data" / "depth_check_2q.npz"
L_LIST = [14, 18, 23, 28, 32, 40, 48]
SEED = 0
M_GRID = np.logspace(1.5, 6.5, 101)

target, _ = get_target_operators_gate_set_name(gate_set_name="2QXYICZ")
target_ptms = pdz.pauli_transfer_matrices(jnp.asarray(target["kraus"]))


def random_noise_gate_set(noise_strength, seed):
    noise = get_random_noisy_kraus_tensors_for_gate_set(num_gates=6, dim=4, rank_kraus=4, noise_strength=noise_strength, seed=seed)
    superop = apply_noisy_kraus_tensors_to_target_gate_set(kraus_tensors_noisy=noise, target_gate_set=target, is_target_vectorized=True)
    return tuple(jnp.asarray(x) for x in get_compressed_rep_from_mgst_output(*superop.values(), kraus_rank=4, state_rank=4, povm_rank=4))


POINTS = {
    "noise 0.05 prior": (lambda: random_noise_gate_set(0.05, 7), 600),
    "noise 0.3 truth": (lambda: random_noise_gate_set(0.3, 42), 1000),
    "IQMFakeApollo": (lambda: pdz.fake_backend_gate_set(IQMFakeApollo(), [0, 1], target["kraus"]), 1000),
}

results = {}
for name, (make, N) in POINTS.items():
    t0 = time.time()
    point = pdz.prepare_point(make(), target_ptms)
    kraus, povm, state = point["operators"]

    # the Pauli representation used by the GL-gauge test must reproduce the model probabilities (one-gate circuits)
    R, r, e = (np.asarray(x) for x in pdz.pauli_representation(pdz.to_vec(kraus, povm, state), pdz._shapes(kraus, povm, state)))
    p_model = np.asarray(ga._circuit_probabilities(kraus, povm, state, pdz.pad_sequences([[g] for g in range(6)])))
    assert np.allclose(p_model, np.einsum("ip,gpq,q->gi", e, R, r), atol=1e-10), "Pauli representation inconsistent"

    ref = generate_sequence_indices(num_gates=6, num_circuits=400, seq_len_list=pdz.seq_len_list(14), seed=0)["gate_indices"]
    tau0 = pdz.prior_width(kraus, pdz.fisher_decomposition(kraus, povm, state, ref, basis=point["T"]))
    mu = np.asarray(pdz.choi_eigenvalues(kraus))
    print(f"=== {name}: N = {N}, tau0 = {tau0:.3e}, max(1 - mu_1) = {(1 - mu[:, 0]).max():.2e}, gauge fixing {point['gauge_info']}", flush=True)
    rows = pdz.depth_check(point, N, L_LIST, SEED, M_GRID, 2 * tau0)
    results[name] = dict(rows=rows, N=N, tau=2 * tau0, tau0=tau0, mu=mu, mu_small=point["mu_small"], n_elements=int(point["keep"].sum()),
                         gate=np.where(point["keep"])[0] // 256)
    print(f"    {time.time() - t0:.0f}s", flush=True)
    np.savez(OUT, results=np.array(results, dtype=object), L_list=L_LIST, seed=SEED, m_grid=M_GRID)
print(f"saved {OUT}")
