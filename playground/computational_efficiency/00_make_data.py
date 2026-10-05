"""Generate the GST datasets used by the computational-efficiency benchmarks and pickle them into ./data.

All benchmarks load the same pickled data, so that different code versions (e.g. an old commit checked out with
`git worktree`) are compared on identical inputs.

Usage:
    python 00_make_data.py 1q        # 1 qubit,  1QXYI,  100 circuits  (the setting of 1q_mgst_vs_current_state.ipynb)
    python 00_make_data.py 2q        # 2 qubits, 2QXYICZ, 1000 circuits
    python 00_make_data.py 3q        # 3 qubits, 3QXYCZ,  1000 circuits (the setting of time_complexity/linearization.ipynb)
"""
import pickle
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from mGST.utility_functions_comparisons import GSTConfiguration, get_full_mgst_parameters_from_configuration, get_initial_gate_set_for_optimization

DATA_DIR = Path(__file__).parent / "data"
SETTINGS = {"1q": ([[0]], "1QXYI", 100), "2q": ([[0, 1]], "2QXYICZ", 1000), "3q": ([[0, 1, 3]], "3QXYCZ", 1000)}
KRAUS_RANK, NUM_SHOTS, BACKEND = 4, 1000, "iqmfakeapollo"

if __name__ == "__main__":
    name = sys.argv[1]
    layout, gate_set, num_circuits = SETTINGS[name]
    dim = 2 ** len(layout[0])
    configuration = GSTConfiguration(qubit_layouts=layout, gate_set=gate_set, num_circuits=num_circuits, shots=NUM_SHOTS, rank=KRAUS_RANK)
    _, kraus_mgst, povm_mgst, state_mgst, prob_matrix, indices_list, _ = get_full_mgst_parameters_from_configuration(configuration, BACKEND, only_jax_variables=False)
    init = get_initial_gate_set_for_optimization(
        target_superops={"kraus": kraus_mgst, "povm": povm_mgst, "state": state_mgst},
        kraus_rank=KRAUS_RANK, state_rank=dim, povm_rank=dim)
    data = dict(init={k: np.asarray(v) for k, v in init.items()}, prob_matrix=np.asarray(prob_matrix),
                indices_list=[np.asarray(i) for i in indices_list], num_shots=NUM_SHOTS)
    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / f"data_{name}.pkl", "wb") as f:
        pickle.dump(data, f)
    print(f"saved data/data_{name}.pkl:", {k: v.shape for k, v in data["init"].items()}, f"{len(indices_list)} sequences")
