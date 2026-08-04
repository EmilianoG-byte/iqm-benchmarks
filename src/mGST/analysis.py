"""Utility functions to analyze GST results, e.g. MVE, average, std's, etc."""

import jax.numpy as jnp
import numpy as np

from qiskit.quantum_info import SuperOp
from qiskit.quantum_info.operators.measures import diamond_norm
from mGST.additional_fns import MVE
from mGST.reporting.reporting import MVE_data
from mGST.typing import Tensor
from typing import Any
from typing import Literal, Callable

from mGST.utility_functions_comparisons import get_mgst_tensors_from_psd_representation

WhichData = Literal["superops_gauged", "optimized_operators"]


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

def diamond_norm_distances(superop1:Tensor, superop2:Tensor)->list[float]:
    """
    Compute the diamond norm distance between two superoperators.
    """
    # convert both to numpy arrays if they are not already
    superop1 = np.array(superop1)
    superop2 = np.array(superop2)
    
    assert superop1.ndim == superop2.ndim == 3, "Superoperators must have the same number of dimensions."
    assert superop1.shape == superop2.shape, "Superoperators must have the same shape."
    num_gates = superop1.shape[0]
    return [diamond_norm(SuperOp(superop1[i]) - SuperOp(superop2[i])) / 2 for i in range(num_gates)]

def compute_diamond_distances_from_list_of_superops(true_superops:dict[str, Tensor], gst_superops:list[dict[str, Tensor]], operator_type:str="kraus", report:str="mean")->list[float]:
    """Compute the diamond norm distances for each list of superoperators against the true superoperators."""
    if operator_type != "kraus":
        raise NotImplementedError("Currently only 'kraus' operator type is supported for diamond norm distance computation.")
    
    if report == "mean":
        function = jnp.mean
    elif report == "max":
        function = jnp.max
    elif report == "min":
        function = jnp.min
    else:
        raise ValueError("Report must be either 'mean', 'min' or 'max'.")    

    true_kraus = true_superops["kraus"]
    diamond_values_per_superop = []
    for idx, gst_dict in enumerate(gst_superops):
        print(f"💎 Computing diamond norm for superoperator {idx} 💎")
        gst_kraus = gst_dict["kraus"]
        distances = jnp.array(diamond_norm_distances(true_kraus, gst_kraus))
        value = function(distances)
        diamond_values_per_superop.append((value, jnp.std(distances)))
    return diamond_values_per_superop

def variational_error_from_superops(superops_gauged:dict[str, jnp.ndarray], indices_dict:dict[str,Any], prob_matrix_dict:dict[str, jnp.ndarray]):
    """
    Compute the Variational Error from superoperators for both the mean (M) and worse-case (W) errors.
    """
    kraus = superops_gauged["kraus"]
    povm = superops_gauged["povm"]
    state = superops_gauged["state"]
    indices_list = indices_dict['gate_indices_with_negs']
    probability_matrix = prob_matrix_dict["sampled"]
    return MVE_data(X=kraus, E=povm, rho=state, J=indices_list, y=probability_matrix)[0]

def variational_error_from_true_and_gst_superops(true_superops:dict[str, jnp.ndarray], gst_superops:dict[str, jnp.ndarray], length:int = 14, samples:int|str=1000)->tuple[float, float]:
    """
    Compute the Variational Error (MVE) from target and GST superoperators for both the mean (MVE) and worst-case (WVE) errors.

    Args:
        true_superops: A dictionary containing the true superoperators with keys "kraus", "povm", and "state".
        gst_superops: A dictionary containing the GST superoperators with keys "kraus", "povm", and "state".
        length: The length of the sequences to consider for MVE computation. Default is 14.
        samples: The number of samples to use for MVE computation. Can be an integer or "all" to use all samples. Default is 1000.
    Returns:
        A tuple containing the mean variational error (MVE) and the worst-case variational error (WVE).
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
    
def compute_average_and_std_variational_error(true_superops:dict[str, jnp.ndarray], gst_superops:dict[str, jnp.ndarray], length:int = 14, samples:int|str=1000, num_runs:int=10, error_type:str="MVE")->tuple[float, float]:
    """
    Compute the average and standard deviation of the Variational Error (VE) for the Mean/Worst case (MVE/WVE) over multiple runs.
    
    Args:
        true_superops: A dictionary containing the true superoperators with keys "kraus", "povm", and "state".
        gst_superops: A dictionary containing the GST superoperators with keys "kraus", "povm", and "state".
        length: The length of the sequences to consider for VE computation. Default is 14.
        samples: The number of samples to use for VE computation. Can be an integer or "all" to use all samples. Default is 1000.
        num_runs: The number of runs to average over. Default is 10.
        error_type: type of error to compute on each run. Can be either mean (MVE) or worst-case error (WVE).
        
    Returns:
        A tuple containing the average and standard deviation of the chosen type Variational Errors (VE) over the specified number of runs.
    """
    
    if error_type not in ["MVE", "WVE"]:
        raise ValueError("error_type must be either 'MVE' or 'WVE'.")
    
    ve_values = []
    for _ in range(num_runs):
        mve, wve = variational_error_from_true_and_gst_superops(true_superops=true_superops, gst_superops=gst_superops, length=length, samples=samples) # mean variational error and worst variational error
        if error_type == "MVE":
            ve_values.append(mve)
        elif error_type == "WVE":
            ve_values.append(wve)
    
    return jnp.mean(jnp.array(ve_values)), jnp.std(jnp.array(ve_values))
    
def variational_error_avg_and_std_from_list_of_superops(num_runs:int, true_superops:dict[str, jnp.ndarray], gst_superops:list[dict[str, jnp.ndarray]], length:int=14, samples:int|str=1000, error_type:str="MVE", verbose:bool=False)->list[tuple[float, float]]:
    """
    Compute the Variational Error (VE) for a list of GST superoperators against the true superoperators.
    
    Args:
        num_runs: The number of runs to average over for each GST superoperator. Default is 10.
        true_superops: A dictionary containing the true superoperators with keys "kraus", "povm", and "state".
        gst_superops: A list of dictionaries, each containing GST superoperators with keys "kraus", "povm", and "state".
        length: The length of the sequences to consider for VE computation. Default is 14.
        samples: Number of random gate sequences over which the variation error is computed. Can be an integer or "all" to use all samples. Default is 1000.
        error_type: type of error to compute on each run. Can be either mean (MVE) or worst-case error (WVE).
        
    Returns:
        A list of tuples containing the average and standard deviation of the chosen type Variational Errors (VE) for each GST superoperator in the order they were provided.
    """
    mve_results = []
    for idx, superop in enumerate(gst_superops):
        if verbose:
            print(f"Processing superop no.: {idx}")
        mve_results.append(compute_average_and_std_variational_error(true_superops=true_superops, gst_superops=superop, length=length, samples=samples, num_runs=num_runs, error_type=error_type))
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

def compute_avg_and_std_along_keys(list_of_errors:list[dict[str, dict[str, float]]]):
    """
    Compute the average and standard deviation of errors along the keys of a list of dictionaries.
    
    Args:
        list_of_errors: A list of dictionaries containing error metrics for different keys.
    Returns:
        A dictionary containing the mean and standard deviation of errors for each key.
    """
    keys = list_of_errors[0].keys()
    avg_std_dict = {}
    for key in keys:
        errors_at_key = jnp.array([error[key]["mean"] for error in list_of_errors])
        mean = jnp.mean(errors_at_key)
        std = jnp.std(errors_at_key)
        avg_std_dict[key] = {"mean": mean, "std": std}
    return avg_std_dict

def compute_mve_and_wve_for_all_realizations(optimization_results: list[list[dict[str, Any]]], expected_keys: list, true_superops: dict[str, jnp.ndarray], sequence_length:int=14, num_circuits_sampled:int=1000, num_repetitions:int=10, which_data:str="superops_gauged",verbose:bool=False)-> tuple[list[dict[str, dict[str, float]]], list[dict[str, dict[str, float]]]]:
    """
    Compute the Mean Variational Error (MVE) and Mean Worst-case Variational Error (WVE) for all realizations of optimization results.
    
    Args:
        optimization_results: A list of lists of result dictionaries returned by `run_optimization_workflow`. The outer list corresponds to different realizations, and the inner list corresponds to different optimization results for each realization, where each of these has an associated expected key (e.g. the number of sequences ran).
        expected_keys: A list of expected keys in the optimization results.
        true_superops: The true superoperators (kraus, povm, state) used to generate the probability matrices.
        sequence_length: The length of the sequences to use for computing MVE and WVE.
        num_circuits_sampled: The number of circuits to sample for computing MVE and WVE.
        num_repetitions: The number of repetitions to perform for computing MVE and WVE.

    Returns:
        The MVE and WVE results for all realizations, where each realization's results are stored in a dictionary with the expected keys, and each key maps to a dictionary containing the mean and standard deviation of the MVE and WVE.
    """
    mve_results_all_realizations = []
    wve_results_all_realizations = []

    for realization_results in optimization_results:
        # create the dictionary from where to compute mve and wve
        if not len(realization_results) == len(expected_keys):
            raise ValueError(f"Length of realization results ({len(realization_results)}) does not match length of expected keys ({len(expected_keys)}).")
        
        optimization_result_idx = {str(key): res for key, res in zip(expected_keys, realization_results)} 

        mve_results_dict, wve_results_dict = compute_mve_and_wve_from_optimization_results(
            optimization_results=optimization_result_idx,
            true_superops=true_superops,
            sequence_length=sequence_length,
            num_circuits_sampled=num_circuits_sampled,
            num_repetitions=num_repetitions,
            verbose=verbose,
            which_data=which_data,
        )

        mve_results_all_realizations.append(mve_results_dict)
        wve_results_all_realizations.append(wve_results_dict)

    return mve_results_all_realizations, wve_results_all_realizations

def compute_mve_for_gauged_and_non_gauged_all_realizations(expected_keys:list, optimization_results: list[list[dict[str, dict[str, Any]]]], sequence_length:int=14, num_circuits_sampled:int=1000, num_repetitions:int=10)->list[tuple[float, float]]:
    """Compute the Mean Variational Error (MVE) for both gauged and non-gauged superoperators from optimization results for all realizations."""
    mve_results_all_realizations = []
    for realization_results in optimization_results:
        mve_results = compute_mve_for_gauged_and_non_gauged_ops(
            expected_keys=expected_keys,
            optimization_results=realization_results,
            sequence_length=sequence_length,
            num_circuits_sampled=num_circuits_sampled,
            num_repetitions=num_repetitions
        )
        mve_results_all_realizations.append(mve_results)
    return mve_results_all_realizations

def compute_mve_for_gauged_and_non_gauged_ops(expected_keys:list, optimization_results: list[dict[str, dict[str, Any]]], sequence_length:int=14, num_circuits_sampled:int=1000, num_repetitions:int=10)->list[tuple[float, float]]:
    """Compute the Mean Variational Error (MVE) for both gauged and non-gauged superoperators from optimization results."""

    opt_results_dict = {str(key): res for key, res in zip(expected_keys, optimization_results)}
    superops_gauged = _extract_gst_superops_list(opt_results_dict, which_data="superops_gauged")
    superops_non_gauged = _extract_gst_superops_list(opt_results_dict, which_data="optimized_operators")

    mve_list = [
        compute_average_and_std_variational_error(
        true_superops=ops_non_gauged,
        gst_superops=ops_gauged,
        length=sequence_length,
        samples=num_circuits_sampled,
        num_runs=num_repetitions,
        error_type="MVE"
        )
    for ops_gauged, ops_non_gauged in zip(superops_gauged, superops_non_gauged)]
    
    return {str(key): {"mean": mean, "std": std} for key, (mean, std) in zip(expected_keys, mve_list)}

def _to_superops_from_optimized(result: dict[str, Any]) -> dict[str, jnp.ndarray]:
    ops = result["optimized_operators"]
    # Explicit key access avoids relying on dict value order.
    return get_mgst_tensors_from_psd_representation(
        ops["kraus"],
        ops["povm"],
        ops["state"],
    )

def _extract_gst_superops_list(
    optimization_results: dict[str, dict[str, Any]],
    which_data: WhichData,
) -> list[dict[str, jnp.ndarray]]:
    extractors: dict[WhichData, Callable[[dict[str, Any]], dict[str, jnp.ndarray]]] = {
        "superops_gauged": lambda result: result["superops_gauged"],
        "optimized_operators": _to_superops_from_optimized,
    }
    return [extractors[which_data](result) for result in optimization_results.values()]


def compute_mve_and_wve_from_optimization_results(optimization_results: dict[str, dict[str, Any]], true_superops: dict[str, jnp.ndarray], sequence_length:int=14, num_circuits_sampled:int=1000, num_repetitions:int=10, which_data:str = "superops_gauged",verbose:bool=False)-> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """
    Compute the Mean Variational Error (MVE) and Mean Worst-case Variational Error (WVE) from a dictionary of optimization results.
    
    Args:
        optimization_results: A dictionary of result dictionaries returned by `run_optimization_workflow`.
        true_superops: The true superoperators (kraus, povm, state) used to generate the probability matrices.
        sequence_length: The length of the sequences to use for computing MVE and WVE.
        num_circuits_sampled: The number of circuits to sample for computing MVE and WVE.
        num_repetitions: The number of repetitions to perform for computing MVE and WVE.

    Returns:
        The MVE and WVE results, where each result is stored in a dictionary with the expected keys, and each key maps to a dictionary containing the mean and standard deviation of the MVE and WVE.
    """
    keys = list(optimization_results.keys())
    
    if which_data not in ["superops_gauged", "optimized_operators"]:
        raise ValueError(f"which_data must be either 'superops_gauged' or 'optimized_operators', got {which_data}.")
        
    gst_superops_list = _extract_gst_superops_list(optimization_results, which_data=which_data)

    mve_results_over_sequences = variational_error_avg_and_std_from_list_of_superops(num_runs=num_repetitions, true_superops=true_superops, gst_superops=gst_superops_list, length=sequence_length, samples=num_circuits_sampled, error_type="MVE", verbose=verbose)
    
    mve_results_dict = {str(key): {"mean": mean, "std": std} for key, (mean, std) in zip(keys, mve_results_over_sequences)}
    
    wve_results_over_sequences = variational_error_avg_and_std_from_list_of_superops(num_runs=num_repetitions, true_superops=true_superops, gst_superops=gst_superops_list, length=sequence_length, samples=num_circuits_sampled, error_type="WVE", verbose=verbose)
    
    wve_results_dict = {str(key): {"mean": mean, "std": std} for key, (mean, std) in zip(keys, wve_results_over_sequences)}

    return mve_results_dict, wve_results_dict


def combine_error_results(*error_results_list):
    """Combine multiple error result lists by merging per-realization dicts.
        
    Args:
        *error_results_list: Two or more lists of the form list[dict[str, dict[str, float]]],
            where each outer list has the same number of realizations and the dicts
            at each index have disjoint keys (sequence counts).
    
    Returns:
        A combined list with merged dicts per realization, sorted by numeric key.
    """
    n_realizations = len(error_results_list[0])
    combined = []
    for i in range(n_realizations):
        merged = {}
        for error_results in error_results_list:
            #check they all have the same lenghth
            if len(error_results) != n_realizations:
                raise ValueError(f"All error result lists must have the same number of realizations. Found {len(error_results)} and {n_realizations}.")
            
            # check for disjoint keys before merging
            if not merged.keys().isdisjoint(error_results[i].keys()):
                raise ValueError(f"Error result dicts at index {i} have overlapping keys: {merged.keys() & error_results[i].keys()}")
            
            merged.update(error_results[i])
        # Sort keys numerically so plots come out in the right order
        merged = dict(sorted(merged.items(), key=lambda kv: float(kv[0])))
        combined.append(merged)
    return combined

def to_json_serializable(error_results):
    """Convert list[dict[str, dict[str, float]]] with JAX arrays to JSON-serializable form."""
    return [
        {key: {k: float(v) for k, v in inner.items()} for key, inner in realization.items()}
        for realization in error_results
    ]