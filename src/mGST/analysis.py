"""Utility functions to analyze GST results, e.g. MVE, average, std's, etc."""

import jax.numpy as jnp
import numpy as np

from qiskit.quantum_info import SuperOp
from qiskit.quantum_info.operators.measures import diamond_norm
from mGST.additional_fns import MVE
from mGST.reporting.reporting import MVE_data
from typing import Any

def compute_weighted_linear_fit_of_exponential_data(
    x: jnp.ndarray,
    y_avg_std: list[tuple[float, float]],
    base: float = 10.0,
    eps: float = 1e-30,
):
    """
    Weighted fit of log_b(y) = slope * log_b(x) + intercept,
    where weights come from y_std.

    Uses propagated uncertainty:
        sigma_logy = y_std / (y * ln(base))
        w = 1 / sigma_logy^2
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray([val for val, _ in y_avg_std], dtype=jnp.float64)
    y_std = jnp.asarray([std for _, std in y_avg_std], dtype=jnp.float64)

    # Keep only physically valid points for log transform and weighting
    valid = (x > 0.0) & (y > 0.0) & (y_std > 0.0)
    x = x[valid]
    y = y[valid]
    y_std = y_std[valid]

    if x.size < 2:
        raise ValueError("Need at least 2 valid points with x>0, y>0, y_std>0.")

    ln_base = jnp.log(base)
    log_x = jnp.log(x) / ln_base
    log_y = jnp.log(y) / ln_base

    sigma_log_y = y_std / (y * ln_base)
    sigma_log_y = jnp.maximum(sigma_log_y, eps)
    w = 1.0 / (sigma_log_y ** 2)

    # Weighted least squares closed form
    S = jnp.sum(w)
    Sx = jnp.sum(w * log_x)
    Sy = jnp.sum(w * log_y)
    Sxx = jnp.sum(w * log_x * log_x)
    Sxy = jnp.sum(w * log_x * log_y)

    Delta = S * Sxx - Sx * Sx
    if jnp.abs(Delta) < eps:
        raise ValueError("Degenerate fit: check x values and uncertainties.")

    slope = (S * Sxy - Sx * Sy) / Delta
    intercept = (Sxx * Sy - Sx * Sxy) / Delta

    # Optional 1-sigma parameter uncertainties and covariance
    slope_std = jnp.sqrt(S / Delta)
    intercept_std = jnp.sqrt(Sxx / Delta)
    cov_slope_intercept = -Sx / Delta

    return slope, intercept, slope_std, intercept_std, cov_slope_intercept

def generate_weighted_fitted_values(
    x: jnp.ndarray,
    y_avg_std: list[tuple[float, float]],
    base: float = 10.0,
):
    slope, intercept, slope_std, intercept_std, cov = (
        compute_weighted_linear_fit_of_exponential_data(x, y_avg_std, base=base)
    )
    log_x = jnp.log(x) / jnp.log(base)
    log_y_fit = slope * log_x + intercept
    y_fit = base ** log_y_fit
    return y_fit, slope, intercept, slope_std, intercept_std, cov

def compute_mean_and_std(array:jnp.ndarray)->tuple[float, float]:
    """
    Compute the mean and standard deviation of a 1D array.
    """
    mean = jnp.mean(array)
    std = jnp.std(array)
    return mean, std

def diamond_norm_distances(superop1, superop2):
    """
    Compute the diamond norm distance between two superoperators.
    """
    assert superop1.ndim == superop2.ndim == 3, "Superoperators must have the same number of dimensions."
    assert superop1.shape == superop2.shape, "Superoperators must have the same shape."
    num_gates = superop1.shape[0]
    return [diamond_norm(SuperOp(superop1[i]) - SuperOp(superop2[i])) / 2 for i in range(num_gates)]

def MVE_from_superops(superops_gauged:dict[str, jnp.ndarray], indices_dict:dict[str,Any], prob_matrix_dict:dict[str, jnp.ndarray]):
    """
    Compute the Mean Variational Error (MVE) from superoperators.
    """
    kraus = superops_gauged["kraus"]
    povm = superops_gauged["povm"]
    state = superops_gauged["state"]
    indices_list = indices_dict['gate_indices_with_negs']
    probability_matrix = prob_matrix_dict["sampled"]
    return MVE_data(X=kraus, E=povm, rho=state, J=indices_list, y=probability_matrix)[0]

def MVE_from_true_and_gst_superops(true_superops:dict[str, jnp.ndarray], gst_superops:dict[str, jnp.ndarray], length:int = 14, samples:int|str=1000)->tuple[float, float]:
    """
    Compute the Mean Variational Error (MVE) from target and GST superoperators.

    
    Args:
        true_superops: A dictionary containing the true superoperators with keys "kraus", "povm", and "state".
        gst_superops: A dictionary containing the GST superoperators with keys "kraus", "povm", and "state".
        length: The length of the sequences to consider for MVE computation. Default is 14.
        samples: The number of samples to use for MVE computation. Can be an integer or "all" to use all samples. Default is 1000.
    Returns:
        A tuple containing the lower bound of the mean value error and the maximum distance.
    """
    # convert the superoperators to numpy since these need to be passed to numba functions
    
    kraus_true = np.array(true_superops["kraus"])
    povm_true = np.array(true_superops["povm"])
    state_true = np.array(true_superops["state"])
    
    kraus_gst = np.array(gst_superops["kraus"])
    povm_gst = np.array(gst_superops["povm"])
    state_gst = np.array(gst_superops["state"])
    
    num_gates = len(kraus_gst)
    num_povm = len(povm_gst)
    
    
    return MVE(X_true=kraus_true,
               E_true=povm_true,
               rho_true=state_true,
               X=kraus_gst,
               E=povm_gst,
               rho=state_gst,
               d=num_gates,
               n_povm=num_povm,
               length=length,
               samples=samples)
    
def average_and_std_mve(true_superops:dict[str, jnp.ndarray], gst_superops:dict[str, jnp.ndarray], length:int = 14, samples:int|str=1000, num_runs:int=10)->tuple[float, float]:
    """
    Compute the average and standard deviation of the Mean Variational Error (MVE) over multiple runs.
    
    Args:
        true_superops: A dictionary containing the true superoperators with keys "kraus", "povm", and "state".
        gst_superops: A dictionary containing the GST superoperators with keys "kraus", "povm", and "state".
        length: The length of the sequences to consider for MVE computation. Default is 14.
        samples: The number of samples to use for MVE computation. Can be an integer or "all" to use all samples. Default is 1000.
        num_runs: The number of runs to average over. Default is 10.
        
    Returns:
        A tuple containing the average and standard deviation of the mean value error over the specified number of runs.
    """
    mve_values = []
    for _ in range(num_runs):
        mve, _ = MVE_from_true_and_gst_superops(true_superops=true_superops, gst_superops=gst_superops, length=length, samples=samples)
        mve_values.append(mve)
    
    return jnp.mean(jnp.array(mve_values)), jnp.std(jnp.array(mve_values))
    
def mve_avg_and_std_from_list_of_superops(num_runs:int, true_superops:dict[str, jnp.ndarray], gst_superops:list[dict[str, jnp.ndarray]], length:int=14, samples:int|str=1000)->list[tuple[float, float]]:
    """
    Compute the Mean Variational Error (MVE) for a list of GST superoperators against the true superoperators.
    
    Args:
        true_superops: A dictionary containing the true superoperators with keys "kraus", "povm", and "state".
        length: The length of the sequences to consider for MVE computation. Default is 14.
        samples: The number of samples to use for MVE computation. Can be an integer or "all" to use all samples. Default is 1000.
        gst_superops: A list of dictionaries, each containing GST superoperators with keys "kraus", "povm", and "state".
        
    Returns:
        A list of floats containing the lower bound of the mean value error for each GST superoperator in the order they were provided.
    """
    mve_results = []
    for idx, superop in enumerate(gst_superops):
        print(f"Processing superop no.: {idx}")
        mve_results.append(average_and_std_mve(true_superops=true_superops, gst_superops=superop, length=length, samples=samples, num_runs=num_runs))
    return mve_results

def compute_linear_fit_of_exponential_data(x:jnp.ndarray, y:jnp.ndarray, base:float=10.0):
    """Compute the linear fit of exponential data by linearizing the data and performing a linear regression."""
    # linearize
    log_y = jnp.log(y) / jnp.log(base)
    log_x = jnp.log(x) / jnp.log(base)
    
    # linear fit
    coeffs = jnp.polyfit(log_x, log_y, deg=1)
    slope, intercept = coeffs
    return slope, intercept
    
def generate_fitted_values(x:jnp.ndarray, y:jnp.ndarray, base:float=10.0):
    """Generate fitted values for exponential data by linearizing the data, performing a linear regression, and then exponentiating the fitted values."""
    slope, intercept = compute_linear_fit_of_exponential_data(x, y, base)
    log_x = jnp.log(x) / jnp.log(base)
    log_y_fit = slope * log_x + intercept
    y_fit = base ** log_y_fit
    return y_fit, slope, intercept