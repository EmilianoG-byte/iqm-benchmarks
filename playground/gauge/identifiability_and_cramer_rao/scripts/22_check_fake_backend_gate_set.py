"""Check of ptm_design.fake_backend_gate_set: does the gate set reproduce the fake backend?

Compares the exact outcome probabilities of the (uncompressed, Kraus rank up to 16) gate set with Aer sampling on
IQMFakeApollo, qubits (0, 1), for random 2QXYICZ sequences without the Idle gate (the backend rejects delays). The two
additions that the Aer model does not have are switched off (spectator_idle=False, excited_population=0).

Conventions, as in the GST benchmark (src/iqm/benchmarks/compressive_gst):
- sequence order: the circuit applies the gates of J in order, while mGST uses J[:, ::-1] (compressive_gst.py, `self.J`);
- outcome index: the qiskit bit string reversed, so qubit 0 is the most significant bit
  (gst_analysis.dataset_counts_to_mgst_format).
Pass: every probability within 4 shot-noise standard deviations. Also prints the same statistic against the ideal
gates (with the noisy readout), which must be large, to show that the test is sensitive.
Usage: python 22_check_fake_backend_gate_set.py      (about 15 s)
"""
import numpy as np
import jax.numpy as jnp
from qiskit import transpile

import ptm_design as pdz
from mGST.utility_functions_comparisons import get_target_operators_gate_set_name
from mGST.simulate import compute_probability_matrices
from mGST.qiskit_interface import get_qiskit_circuits
from iqm.benchmarks.compressive_gst.compressive_gst import create_predefined_gate_set
from iqm.qiskit_iqm.fake_backends import IQMFakeApollo

SHOTS = 100_000
target, _ = get_target_operators_gate_set_name(gate_set_name="2QXYICZ")
backend = IQMFakeApollo()
_, full, table = pdz.fake_backend_gate_set(backend, [0, 1], target["kraus"], spectator_idle=False, excited_population=0.0, return_full=True)

rng = np.random.default_rng(3)
seqs = [list(rng.integers(1, 6, size=n)) for n in [2, 3, 3, 4, 5, 6, 8, 10, 14, 20]]
mgst_seqs = [s[::-1] for s in seqs]
p_model = np.asarray(compute_probability_matrices(*full, gate_indices=mgst_seqs, num_shots=10)["exact"])
p_ideal = np.asarray(compute_probability_matrices(jnp.asarray(target["kraus"]), *full[1:], gate_indices=mgst_seqs, num_shots=10)["exact"])

gates, _, _ = create_predefined_gate_set("2QXYICZ", 2, [[0, 1]])
circuits = transpile(get_qiskit_circuits(seqs, gates, 2, [0, 1]), backend, initial_layout=[0, 1], optimization_level=0)
counts = backend.run(circuits, shots=SHOTS).result().get_counts()
freq = np.array([[sum(v for k, v in c.items() if int(k.replace(" ", "")[::-1], 2) == i) for c in counts] for i in range(4)]) / SHOTS

z = lambda p: np.abs(freq - p) / np.sqrt(np.maximum(p * (1 - p), 1e-12) / SHOTS)
print(f"{len(seqs)} sequences of length 2-20, {SHOTS} shots each")
print(f"max |frequency - p| / shot-noise sd: model {z(p_model).max():.2f}, ideal gates {z(p_ideal).max():.1f}")
print(f"max |frequency - p|: model {np.abs(freq - p_model).max():.1e}, ideal gates {np.abs(freq - p_ideal).max():.1e}")
assert z(p_model).max() < 4, "the gate set does not reproduce the fake backend"
assert z(p_ideal).max() > 10, "the test is not sensitive to the gate noise"
print("passed")
