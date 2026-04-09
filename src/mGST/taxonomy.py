"""Module containing taxonomy and other processing functions to interpret the results of mGST"""

import jax.numpy as jnp
from mGST.typing import Matrix, Vector, Tensor
import matplotlib.pyplot as plt

def walsh_hadamard_matrix(n:int)->Matrix:
    """
    Generate the Walsh-Hadamard matrix of order n.

    Args:
        n: Order of the Walsh-Hadamard matrix (must be a power of 2)
    Returns:
        Walsh-Hadamard matrix of shape (n, n)
    """
    power = jnp.log2(n)
    if power % 1 != 0:
        raise ValueError("n must be a power of 2")
    hadamard_matrix_2 = jnp.array([[1, 1], [1, -1]])
    hadamard_matrix = hadamard_matrix_2
    for _ in range(int(power) - 1):
        hadamard_matrix = jnp.kron(hadamard_matrix, hadamard_matrix_2) 
    return hadamard_matrix * (1 / n)

def walsh_hadamard_transform_2n_basis(eigenvalues:Vector)->Vector:
    """
    Apply the Walsh-Hadamard transform to a matrix by matrix multiplication.

    Here, we generate the full Walsh-Hadamard matrix and apply it to the input matrix.
    Expensive for high hilber space dimension.

    Args:
        eigenvalues: vector of eigenvalues of the matrix. Shape (n,)
    Returns:
        Transformed matrix of shape (n, n)
    """
    n = eigenvalues.shape[0]
    hadamard_matrix = walsh_hadamard_matrix(n)
    return hadamard_matrix @ eigenvalues

# Symplectic map: each single-qubit Pauli → (x_bit, z_bit)
_PAULI_TO_SYMP = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}

def pauli_to_symplectic(pauli_str: str) -> Vector:
    """Convert a multi-qubit Pauli string to its binary symplectic vector.

    For an n-qubit Pauli P = P_1 P_2 ... P_n the symplectic vector is
        v = (x_1, ..., x_n | z_1, ..., z_n)  ∈ GF(2)^{2n}
    where X → (1,0), Z → (0,1), Y → (1,1), I → (0,0).

    Args:
        pauli: A string of I/X/Y/Z characters, e.g. "IXZY".

    Returns:
        Symplectic representation of the pauli as a binary vector of shape (2n,).
    """
    x_bits = []
    z_bits = []
    for pauli in pauli_str:
        x, z = _PAULI_TO_SYMP[pauli]
        x_bits.append(x)
        z_bits.append(z)
    return jnp.concatenate([jnp.array(x_bits, dtype=int), jnp.array(z_bits, dtype=int)])

def symplectic_inner_product(v: Vector, w: Vector) -> int:
    """Symplectic inner product of two binary symplectic vectors mod 2.

    For v = (a | b) and w = (c | d) (each of length 2n):
        [v, w] = a·d + b·c  (mod 2)

    Args:
        v: symplectic vector of shape (2n,)
        w: symplectic vector of shape (2n,)

    Returns:
        0 or 1 indicating whether the corresponding Paulis commute or anticommute, respectively.
     """
    n = len(v) // 2
    a, b = v[:n], v[n:]
    c, d = w[:n], w[n:]
    return int((a @ d + b @ c) % 2)

def pauli_to_bitmasks(pauli_str: str) -> tuple[int, int]:
    """
    Convert a Pauli string like 'IXYZ' into two bitmasks (x, z).

    Bit convention:
    - leftmost qubit is the most significant bit
    - x has 1 where the Pauli has X or Y
    - z has 1 where the Pauli has Z or Y
    
    Args:
        pauli_str: A string of I/X/Y/Z characters, e.g. "IXZY".
    Returns:
        A tuple (x, z) where x and z are integers representing the bitmasks for the X and Z components of the Pauli operator, respectively.
    """
    x = 0
    z = 0
    for pauli in pauli_str:
        x <<= 1
        z <<= 1
        if pauli in ("X", "Y"):
            x |= 1
        if pauli in ("Z", "Y"):
            z |= 1
    return x, z

def symplect_inner_product_bitmasks(s1:tuple[int, int], s2:tuple[int, int]) -> int:
    """
    Compute the symplectic inner product of two Pauli operators given by their bitmasks.

    The symplectic inner product is defined as:
        [P1, P2] = (x1 & z2) ^ (z1 & x2)
    where & is bitwise AND and ^ is bitwise XOR.
    Args:
        s1: tuple (x1, z1) for the first Pauli operator
        s2: tuple (x2, z2) for the second Pauli operator

    Returns:
        0 or 1 indicating whether the corresponding Paulis commute or anticommute, respectively.
    """
    x1, z1 = s1
    x2, z2 = s2
    return ((x1 & z2) ^ (z1 & x2)).bit_count() % 2

def commutation_relation(pauli_str_1: str, pauli_str_2: str) -> int:
    """
    Determine the commutation relation between two Pauli operators given by their labels.

    Args:
        pauli_str_1: String label of the first Pauli operator (e.g., "IXYZ")
        pauli_str_2: String label of the second Pauli operator (e.g., "ZIXY")

    Returns:
        0 if the operators commute, 1 if they anticommute.
    """
    s1 = pauli_to_bitmasks(pauli_str_1)
    s2 = pauli_to_bitmasks(pauli_str_2)
    return symplect_inner_product_bitmasks(s1, s2)

def walsh_hadamard_transform_symplectic(eigenvalues: Vector, pauli_labels: list[str]) -> Vector:
    """
    Apply the Walsh-Hadamard transform to a vector of eigenvalues, using the symplectic formulation.

    Args:
        eigenvalues: Array of shape (n,) containing the eigenvalues corresponding to the Pauli operators.
        labels: List of length n containing the string labels of the Pauli operators (e.g., "IXYZ").
    Returns:
        Pauli probabilities.
    """
    dim_sqrd = len(eigenvalues) # should be 4^n for n qubits
    probabilities = {}
    for pauli_prob in pauli_labels:
        pa = 0.0
        for eigval, pauli_eigval in zip(eigenvalues, pauli_labels):
            symplectic_prod = commutation_relation(pauli_prob, pauli_eigval)
            pa += eigval * (-1)**symplectic_prod
        probabilities[pauli_prob] = pa / dim_sqrd
    return probabilities

# 1-qubit transform in the [I, X, Y, Z] basis.
# lambda = H4 @ p
# p      = (1/4) H4 @ lambda
H4 = jnp.array([
    [ 1,  1,  1,  1],   # I
    [ 1,  1, -1, -1],   # X
    [ 1, -1,  1, -1],   # Y
    [ 1, -1, -1,  1],   # Z
], dtype=float)

def _apply_local_hadamard(tensor: Tensor, axis: int) -> Tensor:
    """
    Apply H4 along one tensor axis.
    """
    tensor = jnp.moveaxis(tensor, axis, 0)          # bring target axis to front
    tensor = jnp.tensordot(H4, tensor, axes=(1, 0)) # apply 4x4 transform
    tensor = jnp.moveaxis(tensor, 0, axis)          # restore axis position
    return tensor

def walsh_hadamard_transform_local_basis(
    eigenvalues: Vector,
    pauli_labels: list[str],
    *,
    check_normalization: bool = True,
    atol: float = 1e-10,
) -> dict[str, float]:
    """
    Convert the diagonal of a twirled PTM into Pauli probabilities.

    Args:
        eigenvalues: Length-4^n vector of PTM diagonal entries, ordered as
            [III..., III...X, ...] in your lexicographic Pauli basis.
        pauli_labels: List of length 4^n containing the string labels of the Pauli operators (e.g., "IXYZ").
        check_normalization: If True, check that the output sums to 1.
        atol: Tolerance for the normalization check.

    Returns:
        Dictionary mapping Pauli string labels to their corresponding probabilities.
    """
    d = eigenvalues.size
    n = int(jnp.log2(d) / 2)
    if 2 ** (2 * n) != d:
        raise ValueError(f"Input length {d} is not a power of 4.")

    # Reshape into an n-way tensor with local dimension 4.
    tensor = eigenvalues.reshape((4,) * n)

    # Apply H4 on each qubit axis.
    for axis in range(n):
        tensor = _apply_local_hadamard(tensor, axis)

    probs = tensor.ravel() / (4 ** n)

    result = dict(zip(pauli_labels, probs))

    if check_normalization:
        s = probs.sum()
        if not jnp.isclose(s, 1.0, atol=atol):
            raise ValueError(f"Probabilities do not sum to 1 (sum={s}).")

    return result

def pauli_weight(pauli_str: str) -> int:
    """
    Compute the Pauli weight of a given Pauli string.

    The Pauli weight is defined as the number of non-identity characters in the string.
    
    Args:
        pauli_str: A string of I/X/Y/Z characters, e.g. "IXZY".
    Returns:
        The Pauli weight (number of non-identity characters) as an integer.
    """
    return sum(1 for p in pauli_str if p != 'I')

symbols = ["o", "x", "s", "d", "^", "v", "<", ">", "p", "*"]

def plot_pauli_probabilities_indexed(probabilities: dict[str, float], title:str = None, sort: bool = False, ylabel: str = "Pauli Probability") -> None:

    plt.figure(figsize=(8,4), dpi=200)

    sort = False

    if sort:
        probabilites_sorted = [jnp.sort(probs, descending=True) for probs in probabilities.values()]
        probabilities = dict(zip(probabilities.keys(), probabilites_sorted))

    for i, (label, probs) in enumerate(probabilities.items()):
        plt.semilogy(probs, symbols[i % len(symbols)] + "-", label=label)
    plt.xlabel(f"Pauli Index (sorted = {sort})")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()
    return plt.figure()


def plot_pauli_probabilities_labeled(probabilities: dict[str, float], max_num_labels: int = None, safe: bool = True , log_scale: bool = False, title: str = None) -> None:
    """
    Plot the probabilities of Pauli operators as a bar chart.

    Args:
        probabilities: Dictionary mapping Pauli string labels to their corresponding probabilities.
        max_num_labels: Maximum number of labels to display on the x-axis.
    """
    if max_num_labels is not None and max_num_labels > len(probabilities):
        raise ValueError(f"max_num_labels ({max_num_labels}) exceeds the number of probabilities ({len(probabilities)}).")
    if max_num_labels is None:
        max_num_labels = len(probabilities)
        if max_num_labels > 20 and safe:
            print(f"Warning: Displaying all {max_num_labels} labels may result in a cluttered plot. Defaulting to 20 labels. To forcefully display all labels, set safe=False to disable this warning.")
            max_num_labels = 20
    
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    # Sort probabilities in descending order and keep only the top max_num_labels
    sorted_probs = sorted_probs[:max_num_labels]
    probabilities = dict(sorted_probs)
    labels = list(probabilities.keys())
    probs = list(probabilities.values())

    plt.figure(figsize=(10, 6), dpi=200)
    plt.bar(labels, probs)
    if log_scale:
        plt.yscale('log')
    plt.xlabel('Pauli Operators')
    plt.ylabel('Probability')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def weight_histogram(probabilities: dict[str, float], normalize: bool = False, exclude_identity: bool = False) -> dict[int, float]:
    """
    Compute the histogram of probabilities by Pauli weight.

    Args:
        probabilities: Dictionary mapping Pauli string labels to their corresponding probabilities.
        exclude_identity: If True, exclude the identity operator from the histogram.
    Returns:
        Dictionary mapping Pauli weight (number of non-identity characters) to the total probability mass of all Paulis with that weight.
    """
    histogram = {}
    for pauli_str, prob in probabilities.items():
        weight = pauli_weight(pauli_str)
        histogram[weight] = histogram.get(weight, 0.0) + prob
        
    if exclude_identity:
        histogram.pop(0, None)  # Remove the identity weight if it exists
        normalize = True  # If we exclude the identity, we should normalize the remaining probabilities
        
    if normalize:
        total = sum(histogram.values())
        print(f"Total probability: {total}")
        histogram = {w: p / total for w, p in histogram.items()}
        
    return histogram

def plot_weight_histogram(histogram: dict[int, float], ascending: bool = True, title: str = None, log_scale: bool = True) -> None:
    """
    Plot the histogram of probabilities by Pauli weight.

    Args:
        histogram: Dictionary mapping Pauli weight to total probability.
        title: Title of the plot.
    """

    weights = sorted(histogram.keys(), reverse=not ascending)
    probabilities = [histogram[w] for w in weights]

    plt.figure(figsize=(6, 4), dpi=250)
    plt.bar(weights, probabilities)
    plt.xlabel('Pauli Weight')
    plt.ylabel('Total Probability')
    plt.title(title)
    plt.xticks(weights)
    if log_scale:
        plt.yscale('log')
    plt.grid(axis='y')
    plt.show()
    return plt.figure()