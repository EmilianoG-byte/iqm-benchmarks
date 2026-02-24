"""
The main algorithm and functions that perform iteration steps
"""

from decimal import Decimal
import time
from warnings import warn

import numpy as np
import numpy.linalg as la
from scipy.linalg import eig, eigh
from scipy.optimize import minimize
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

from iqm.benchmarks.logging_config import qcvv_logger
from mGST.additional_fns import batch, random_gs, transp
from mGST.low_level_jit import ddA_derivs, ddB_derivs, ddM, dK, dK_dMdM, objf
from mGST.optimization import (
    lineobjf_A_geodesic,
    lineobjf_B_geodesic,
    lineobjf_isom_geodesic,
    tangent_proj,
    update_A_geodesic,
    update_B_geodesic,
    update_K_geodesic,
)
from mGST.reporting.figure_gen import plot_objf


def A_SFN_riem_Hess(K, A, B, y, J, d, r, n_povm, lam=1e-3, mle=False):
    """Riemannian saddle free Newton step on the POVM parametrization

    Parameters
    ----------
    K : numpy array
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.
    A : numpy array
        Current POVM parametrization
    B : numpy array
        Current initial state parametrization
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    length : int
        Length of the test sequences
    d : int
        Number of different gates in the gate set
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Target Kraus rank
    n_povm : int
        Number of POVM-Elements
    lam : float
        Damping parameter for dampled Newton method; Default: 1e-3

    Returns
    -------
    A_new : numpy array
        Updated POVM parametrization
    """
    pdim = int(np.sqrt(r))
    n = n_povm * pdim
    nt = n_povm * r
    rho = (B @ B.T.conj()).reshape(-1)
    H = np.zeros((2, nt, 2, nt)).astype(np.complex128)
    P_T = np.zeros((2, nt, 2, nt)).astype(np.complex128) # superoperor projection acting on vectorized arbitrary elements and producing vectorized tangent vectors
    Fyconjy = np.zeros((n_povm, r, n_povm, r)).astype(np.complex128) # second derivative, first by conjugate then by y
    Fyy = np.zeros((n_povm, r, n_povm, r)).astype(np.complex128) # second derivative, first by y then by y

    X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((d, r, r))
    # Euclidean derivatives
    dA_, dMdM, dMconjdM, dconjdA = ddA_derivs(X, A, B, J, y, r, pdim, n_povm)
    # dA_: first derivative wrt to A
    # dconjA: second derivative first by A* and then by A

    # Second derivatives
    for i in range(n_povm):
        # sum in equation of derivatives
        Fyconjy[i, :, i, :] = dMconjdM[i] + dconjdA[i]
        Fyy[i, :, i, :] = dMdM[i]

    # derivative
    Fy = dA_.reshape(n, pdim) # converting into isometry
    Y = A.reshape(n, pdim) # converting into isometry
    rGrad = 2 * (Fy.conj() - Y @ Fy.T @ Y) # riemannian gradient
    G = np.array([rGrad, rGrad.conj()]).reshape(-1) # vectorized form of gradient

    P = np.eye(n) - Y @ Y.T.conj()
    T = transp(n, pdim) # transpose superoperator

    # Hessian assembly
    # See theorem 1, equation A27
    # This is already the riemannian hessian elements
    H00 = (
        -(np.kron(Y, Y.T)) @ T @ Fyy.reshape(nt, nt).T
        + Fyconjy.reshape(nt, nt).T.conj()
        - (np.kron(np.eye(n), Y.T @ Fy)) / 2
        - (np.kron(Y @ Fy.T, np.eye(pdim))) / 2
        - (np.kron(P, Fy.T.conj() @ Y.conj())) / 2
    )
    H01 = (
        Fyy.reshape(nt, nt).T.conj()
        - np.kron(Y, Y.T) @ T @ Fyconjy.reshape(nt, nt).T
        + (np.kron(Fy.conj(), Y.T) @ T) / 2
        + (np.kron(Y, Fy.T.conj()) @ T) / 2
    )
    # Full hessian on Z and Z*
    # See relation in equation A37 of individual hessian submatrices
    H[0, :, 0, :] = H00
    H[0, :, 1, :] = H01
    H[1, :, 0, :] = H01.conj()
    H[1, :, 1, :] = H00.conj()

    P_T[0, :, 0, :] = np.eye(nt) - np.kron(Y @ Y.T.conj(), np.eye(pdim)) / 2 # projection on x, component x
    P_T[0, :, 1, :] = -np.kron(Y, Y.T) @ T / 2 # projection on x, component x*
    P_T[1, :, 0, :] = P_T[0, :, 1, :].conj() # projection on x*, component x
    P_T[1, :, 1, :] = P_T[0, :, 0, :].conj() # projection on x*, component x*

    # We compose the projection onto the tangent space with the actual riemannian hessian
    # since we are accepting any input Y
    # A38
    H = H.reshape(2 * nt, 2 * nt) @ P_T.reshape(2 * nt, 2 * nt)

    # saddle free newton method
    # See equation A39
    H = (H + H.T.conj()) / 2
    evals, U = eigh(H)

    # Disregarding gauge directions (zero eigenvalues of the Hessian)
    # inv_diag = evals.copy()
    # inv_diag[np.abs(evals)<1e-14] = 1
    # inv_diag[np.abs(evals)>1e-14] = np.abs(inv_diag[np.abs(evals)>1e-14]) + lam
    # H_abs_inv = U@np.diag(1/inv_diag)@U.T.conj()

    # Damping all eigenvalues
    H_abs_inv = U @ np.diag(1 / (np.abs(evals) + lam)) @ U.T.conj()
    # At this point we take only the real variable part
    # since we are updating only A and not A*
    Delta_A = ((H_abs_inv @ G)[:nt]).reshape(n, pdim)
    # numerical accuracy
    # maybe undoing the symmetrization
    Delta = tangent_proj(A, Delta_A, 1, n_povm)[0]

    a = minimize(lineobjf_A_geodesic, 1e-9, args=(Delta, X, A, rho, J, y, mle), method="COBYLA").x
    A_new = update_A_geodesic(A, Delta, a)
    return A_new

def riemannian_hessian_povm(K:np.ndarray, A:np.ndarray, B:np.ndarray, y:np.ndarray, J:list[list[int]], return_gradient:bool=True, euclidean:bool=False, hermitian_order:bool=True)-> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the Riemannian Hessian of the objective function
    
    Args:
        K: Kraus tensor of dimensions (num_gates, kraus_rank, dim_out, dim_in)
        A: POVM factorization of dimensions (num_povm, povm_rank, dim)
        B: State factorization of dimensions (dim, state_rank)
        y: Measurement outcomes of dimensions (num_povm, num_gate_sequences)
        J: List of gate sequences, where each sequence is a list of gate indices.
    """
    
    num_povm, dim, dim = A.shape
    num_gates = K.shape[0]
    dim_squared = dim ** 2
    n = num_povm * dim
    nt = num_povm * dim_squared
    H = np.zeros((2, nt, 2, nt)).astype(np.complex128)
    Fyconjy = np.zeros((num_povm, dim_squared, num_povm, dim_squared)).astype(np.complex128) # second derivative, first by conjugate then by y
    Fyy = np.zeros((num_povm, dim_squared, num_povm, dim_squared)).astype(np.complex128) # second derivative, first by y then by y

    X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((num_gates, dim_squared, dim_squared))
    # Euclidean derivatives
    dA_, dMdM, dMconjdM, dconjdA = ddA_derivs(X, A, B, J, y, dim_squared, dim, num_povm)
    # dA_: first derivative wrt to A
    # dconjA: second derivative first by A* and then by A

    # Second derivatives
    for i in range(num_povm):
        # sum in equation of derivatives
        Fyconjy[i, :, i, :] = dMconjdM[i] + dconjdA[i]
        Fyy[i, :, i, :] = dMdM[i]

    if euclidean:
        # NOTE: this is not the same convention as for the riemannian Hessian.
        # In the riemannian Hessian we have:
        # [[F_yy, F_yconjy], [F_conjyy, F_conjyconjy]]
        if hermitian_order:
            H[0, :, 0, :] = Fyconjy.reshape(nt, nt)
            H[0, :, 1, :] = Fyy.reshape(nt, nt)
        else:
            H[0, :, 0, :] = Fyy.reshape(nt, nt)
            H[0, :, 1, :] = Fyconjy.reshape(nt, nt)
        H[1, :, 0, :] = H[0, :, 1, :].conj()
        H[1, :, 1, :] = H[0, :, 0, :].conj()
        return H
    # derivative
    Fy = dA_.reshape(n, dim) # converting into isometry
    Y = A.reshape(n, dim) # converting into isometry

    P = np.eye(n) - Y @ Y.T.conj() # projector onto orthogonal complement of stiefel manifold. I = X X^dag + X_perp X_perp^dag
    T = transp(n, dim) # transpose superoperator

    # Hessian assembly
    # See theorem 1, equation A27
    # This is already the riemannian hessian elements
    H00 = (
        -(np.kron(Y, Y.T)) @ T @ Fyy.reshape(nt, nt).T
        + Fyconjy.reshape(nt, nt).T.conj()
        - (np.kron(np.eye(n), Y.T @ Fy)) / 2
        - (np.kron(Y @ Fy.T, np.eye(dim))) / 2
        - (np.kron(P, Fy.T.conj() @ Y.conj())) / 2
    )
    H01 = (
        Fyy.reshape(nt, nt).T.conj()
        - np.kron(Y, Y.T) @ T @ Fyconjy.reshape(nt, nt).T
        + (np.kron(Fy.conj(), Y.T) @ T) / 2
        + (np.kron(Y, Fy.T.conj()) @ T) / 2
    )
    # Full hessian on Z and Z*
    # See relation in equation A37 of individual hessian submatrices
    if hermitian_order:
        H[0, :, 0, :] = H01
        H[0, :, 1, :] = H00
    else:
        H[0, :, 0, :] = H00
        H[0, :, 1, :] = H01
    H[1, :, 0, :] = H[0, :, 1, :].conj()
    H[1, :, 1, :] = H[0, :, 0, :].conj()
    
    if return_gradient:
        # derivative
        Fy = dA_.reshape(n, dim) # converting into isometry
        Y = A.reshape(n, dim) # converting into isometry
        rGrad = Fy.conj() - Y @ Fy.T @ Y # riemannian gradient
        G = np.array([rGrad, rGrad.conj()]) # vectorized form of gradient
        return H, G
    return H

def B_SFN_riem_Hess(K, A, B, y, J, d, r, n_povm, lam=1e-3, mle=False):
    """Riemannian saddle free Newton step on the initial state parametrization

    Parameters
    ----------
    K : numpy array
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.
    A : numpy array
        Current POVM parametrization
    B : numpy array
        Current initial state parametrization
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    length : int
        Length of the test sequences
    d : int
        Number of different gates in the gate set
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Target Kraus rank
    n_povm : int
        Number of POVM-Elements
    lam : float
        Damping parameter for dampled Newton method; Default: 1e-3

    Returns
    -------
    B_new : numpy array
        Updated initial state parametrization
    """

    pdim = int(np.sqrt(r))
    n = r
    nt = r
    E = np.array([(A[i].T.conj() @ A[i]).reshape(-1) for i in range(n_povm)])
    H = np.zeros((2, nt, 2, nt)).astype(np.complex128)
    P_T = np.zeros((2, nt, 2, nt)).astype(np.complex128)

    X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((d, r, r))
    dB_, dMdM, dMconjdM, dconjdB = ddB_derivs(X, A, B, J, y, r, pdim, mle=mle)

    # Second derivatives
    Fyconjy = dMconjdM + dconjdB
    Fyy = dMdM

    # derivative
    Fy = dB_.reshape(n)
    Y = B.reshape(n)
    rGrad = 2 * (Fy.conj() - Y * (Fy.T @ Y))
    G = np.array([rGrad, rGrad.conj()]).reshape(-1)

    P = np.eye(n) - np.outer(Y, Y.T.conj())

    # Hessian assembly
    H00 = (
        -(np.outer(Y, Y.T)) @ Fyy.reshape(nt, nt).T
        + Fyconjy.reshape(nt, nt).T.conj()
        - np.eye(n) * (Y.T @ Fy) / 2
        - np.outer(Y, Fy.T) / 2
        - P * (Fy.T.conj() @ Y.conj()) / 2
    )
    H01 = (
        Fyy.reshape(nt, nt).T.conj()
        - np.outer(Y, Y.T) @ Fyconjy.reshape(nt, nt).T
        + np.outer(Fy.conj(), Y.T) / 2
        + np.outer(Y, Fy.T.conj()) / 2
    )

    H[0, :, 0, :] = H00
    H[0, :, 1, :] = H01
    H[1, :, 0, :] = H01.conj()
    H[1, :, 1, :] = H00.conj()

    P_T[0, :, 0, :] = np.eye(nt) - np.outer(Y, Y.T.conj()) / 2
    P_T[0, :, 1, :] = -np.outer(Y, Y.T) / 2
    P_T[1, :, 0, :] = P_T[0, :, 1, :].conj()
    P_T[1, :, 1, :] = P_T[0, :, 0, :].conj()

    H = H.reshape(2 * nt, 2 * nt) @ P_T.reshape(2 * nt, 2 * nt)

    # saddle free newton method
    H = (H + H.T.conj()) / 2
    evals, U = eigh(H)

    # Disregarding gauge directions (zero eigenvalues of the Hessian)
    # inv_diag = evals.copy()
    # inv_diag[np.abs(evals)<1e-14] = 1
    # inv_diag[np.abs(evals)>1e-14] = np.abs(inv_diag[np.abs(evals)>1e-14]) + lam
    # H_abs_inv = U@np.diag(1/inv_diag)@U.T.conj()

    # Damping all eigenvalues
    H_abs_inv = U @ np.diag(1 / (np.abs(evals) + lam)) @ U.T.conj()

    Delta = (H_abs_inv @ G)[:nt]
    # Projection onto tangent space
    Delta = Delta - Y * (Y.T.conj() @ Delta + Delta.T.conj() @ Y) / 2
    res = minimize(
        lineobjf_B_geodesic, 1e-9, args=(Delta, X, E, B, J, y, mle), method="COBYLA", options={"maxiter": 20}
    )
    a = res.x
    B_new = update_B_geodesic(B, Delta, a)
    return B_new

from mGST.low_level_jit import dK_jax

def gd(K, E, rho, y, J, d, r, rK, fixed_gates, ls="COBYLA", mle=False,
       use_jax:bool=False, 
       conjugate:bool=False,
       optimize_step:bool=True,
       step_size:float=1,
       verbose:bool=False,
       ):
    """Do Riemannian gradient descent optimization step on gates

    Parameters
    ----------
    K : numpy array
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.
    E : numpy array
        Current POVM estimate
    rho : numpy array
        Current initial state estimate
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    length : int
        Length of the test sequences
    d : int
        Number of different gates in the gate set
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Target Kraus rank
    ls : {"COBYLA", ...}
        Line search method, takes "method" arguments of scipy.optimize.minimize

    Returns
    -------
    K_new : numpy array
        Updated Kraus parametrizations

    Notes:
        Gradient descent using the Riemannian gradient and updating along the geodesic.
        The step size is determined by minimizing the objective function in the step size parameter.
    """
    # setup
    pdim = int(np.sqrt(r))
    n = rK * pdim
    Delta = np.zeros((d, n, pdim)).astype(np.complex128)
    
    if not use_jax:
        X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((d, r, r))
        dK_ = dK(X, K, E, rho, J, y, d, r, rK, mle=mle)
    else:
        dK_ = dK_jax(K, E, rho, J, y)
        
    if conjugate:
        dK_ = dK_.conj()
    
    for k in np.where(~fixed_gates)[0]:
        # derivative
        Fy = dK_[k].reshape(n, pdim)
        Y = K[k].reshape(n, pdim)
        # Riem. gradient taken from conjugate derivative
        rGrad = 2 * (Fy.conj() - Y @ Fy.T @ Y)
        Delta[k] = rGrad

    Delta = tangent_proj(K, Delta, d, rK)
    
    if optimize_step:
        res = minimize(
            lineobjf_isom_geodesic, 1e-8, args=(Delta, K, E, rho, J, y, mle), method=ls, options={"maxiter": 200}
            )
        a = res.x
        if verbose:
            print('optimized step size: ', a)
    else:
        a = step_size
    K_new = update_K_geodesic(K, Delta, a)

    return K_new


def SFN_riem_Hess(K, E, rho, y, J, d, r, rK, lam=1e-3, ls="COBYLA", fixed_gates=None, mle=False):
    """Riemannian saddle free Newton step on each gate individually

    Parameters
    ----------
    K : numpy array
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.
    E : numpy array
        Current POVM estimate
    rho : numpy array
        Current initial state estimate
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    length : int
        Length of the test sequences
    d : int
        Number of different gates in the gate set
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Target Kraus rank
    lam : float
        Damping parameter for dampled Newton method; Default: 1e-3
    ls : {"COBYLA", ...}
        Line search method, takes "method" arguments of scipy.optimize.minimize
    fixed_gates : List
        List of gate indices which are not optimized over and assumed as fixed

    Returns
    -------
    K_new : numpy array
        Updated Kraus parametrizations
    """
    # setup
    pdim = int(np.sqrt(r))
    n = rK * pdim
    nt = rK * r
    H = np.zeros((2 * nt, 2 * nt)).astype(np.complex128)
    P_T = np.zeros((2 * nt, 2 * nt)).astype(np.complex128)
    Delta_K = np.zeros((d, rK, pdim, pdim)).astype(np.complex128)
    X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((d, r, r))
    if not fixed_gates:
        fixed_gates = []

    # compute derivatives
    dK_, dM10, dM11 = dK_dMdM(X, K, E, rho, J, y, d, r, rK, mle=mle)
    dd, dconjd = ddM(X, K, E, rho, J, y, d, r, rK, mle=mle)

    # Second derivatives
    Fyconjy = dM11.reshape(d, nt, d, nt) + np.einsum("ijklmnop->ikmojlnp", dconjd).reshape((d, nt, d, nt))
    Fyy = dM10.reshape(d, nt, d, nt) + np.einsum("ijklmnop->ikmojlnp", dd).reshape((d, nt, d, nt))

    for k in np.where(~fixed_gates)[0]:
        Fy = dK_[k].reshape(n, pdim)
        Y = K[k].reshape(n, pdim)
        # riemannian gradient, taken from conjugate derivative
        rGrad = 2 * (Fy.conj() - Y @ Fy.T @ Y)
        G = np.array([rGrad, rGrad.conj()]).reshape(-1)

        P = np.eye(n) - Y @ Y.T.conj()
        T = transp(n, pdim)

        # Riemannian Hessian with correction terms
        H00 = (
            -(np.kron(Y, Y.T)) @ T @ Fyy[k, :, k, :].T
            + Fyconjy[k, :, k, :].T.conj()
            - (np.kron(np.eye(n), Y.T @ Fy)) / 2
            - (np.kron(Y @ Fy.T, np.eye(pdim))) / 2
            - (np.kron(P, Fy.T.conj() @ Y.conj())) / 2
        )
        H01 = (
            Fyy[k, :, k, :].T.conj()
            - np.kron(Y, Y.T) @ T @ Fyconjy[k, :, k, :].T
            + (np.kron(Fy.conj(), Y.T) @ T) / 2
            + (np.kron(Y, Fy.T.conj()) @ T) / 2
        )

        H[:nt, :nt] = H00
        H[:nt, nt:] = H01
        H[nt:, :nt] = H[:nt, nt:].conj()
        H[nt:, nt:] = H[:nt, :nt].conj()

        # Tangent space projection
        P_T[:nt, :nt] = np.eye(nt) - np.kron(Y @ Y.T.conj(), np.eye(pdim)) / 2
        P_T[:nt, nt:] = -np.kron(Y, Y.T) @ T / 2
        P_T[nt:, :nt] = P_T[:nt, nt:].conj()
        P_T[nt:, nt:] = P_T[:nt, :nt].conj()

        H = H @ P_T

        # saddle free newton method
        evals, S = eig(H)

        H_abs_inv = S @ np.diag(1 / (np.abs(evals) + lam)) @ la.inv(S)
        Delta_K[k] = ((H_abs_inv @ G)[:nt]).reshape(rK, pdim, pdim)

    Delta = tangent_proj(K, Delta_K, d, rK)

    res = minimize(
        lineobjf_isom_geodesic, 1e-8, args=(Delta, K, E, rho, J, y, mle), method=ls, options={"maxiter": 200}
    )
    a = res.x
    K_new = update_K_geodesic(K, Delta, a)

    return K_new


def SFN_riem_Hess_full(K, E, rho, y, J, d, r, rK, lam=1e-3, ls="COBYLA", mle=False):
    """Riemannian saddle free Newton step on product manifold of all gates

    Parameters
    ----------
    K : numpy array
        Each subarray along the first axis contains a set of Kraus operators.
        The second axis enumerates Kraus operators for a gate specified by the first axis.
    E : numpy array
        Current POVM estimate
    rho : numpy array
        Current initial state estimate
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    length : int
        Length of the test sequences
    d : int
        Number of different gates in the gate set
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Target Kraus rank
    lam : float
        Damping parameter for dampled Newton method; Default: 1e-3
    ls : {"COBYLA", ...}
        Line search method, takes "method" arguments of scipy.optimize.minimize

    Returns
    -------
    K_new : numpy array
        Updated Kraus parametrizations
    """
    pdim = int(np.sqrt(r))
    n = rK * pdim
    nt = rK * r
    H = np.zeros((2, d, nt, 2, d, nt)).astype(np.complex128)
    P_T = np.zeros((2, d, nt, 2, d, nt)).astype(np.complex128)
    G = np.zeros((2, d, nt)).astype(np.complex128)
    X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((d, r, r))

    # compute derivatives
    dK_, dM10, dM11 = dK_dMdM(X, K, E, rho, J, y, d, r, rK, mle=mle)
    dd, dconjd = ddM(X, K, E, rho, J, y, d, r, rK, mle=mle)

    # Second derivatives
    Fyconjy = dM11.reshape(d, nt, d, nt) + np.einsum("ijklmnop->ikmojlnp", dconjd).reshape((d, nt, d, nt))
    Fyy = dM10.reshape(d, nt, d, nt) + np.einsum("ijklmnop->ikmojlnp", dd).reshape((d, nt, d, nt))

    for k in range(d):
        # Reshaping into isometry
        Fy = dK_[k].reshape((n, pdim))
        Y = K[k].reshape((n, pdim))
        # riemannian gradient under canonical metric
        rGrad = 2 * (Fy.conj() - Y @ Fy.T @ Y)

        # saving rgrad for gate k as a vector
        G[0, k, :] = rGrad.reshape(-1)
        # saving (rgrad)* for gate k as a vector
        G[1, k, :] = rGrad.conj().reshape(-1)

        # projector onto orthogonal complement of stiefel manifold. I = X X^dag + X_perp X_perp^dag
        P = np.eye(n) - Y @ Y.T.conj()
        # transpose superoperator
        T = transp(n, pdim)
        # Hessian assembly
        # See theorem 1, equation A27
        # This is already the riemannian hessian elements
        H00 = (
            -(np.kron(Y, Y.T)) @ T @ Fyy[k, :, k, :].T
            + Fyconjy[k, :, k, :].T.conj()
            - (np.kron(np.eye(n), Y.T @ Fy)) / 2
            - (np.kron(Y @ Fy.T, np.eye(pdim))) / 2
            - (np.kron(P, Fy.T.conj() @ Y.conj())) / 2
        )
        H01 = (
            Fyy[k, :, k, :].T.conj()
            - np.kron(Y, Y.T) @ T @ Fyconjy[k, :, k, :].T
            + (np.kron(Fy.conj(), Y.T) @ T) / 2
            + (np.kron(Y, Fy.T.conj()) @ T) / 2
        )

        # Full hessian on Z and Z*
        # See relation in equation A37 of individual hessian submatrices
        # Riemannian Hessian with correction terms
        H[0, k, :, 0, k, :] = H00
        H[0, k, :, 1, k, :] = H01
        H[1, k, :, 0, k, :] = H01.conj()
        H[1, k, :, 1, k, :] = H00.conj()

        # Tangent space projection
        P_T[0, k, :, 0, k, :] = np.eye(nt) - np.kron(Y @ Y.T.conj(), np.eye(pdim)) / 2
        P_T[0, k, :, 1, k, :] = -np.kron(Y, Y.T) @ T / 2
        P_T[1, k, :, 0, k, :] = P_T[0, k, :, 1, k, :].conj()
        P_T[1, k, :, 1, k, :] = P_T[0, k, :, 0, k, :].conj()

        # These are the cross terms of the Hessian between different gates.
        for k2 in range(d):
            if k2 != k:
                Yk2 = K[k2].reshape(n, pdim)
                H[0, k2, :, 0, k, :] = Fyconjy[k, :, k2, :].T.conj() - np.kron(Yk2, Yk2.T) @ T @ Fyy[k, :, k2, :].T
                H[0, k2, :, 1, k, :] = Fyy[k, :, k2, :].T.conj() - np.kron(Yk2, Yk2.T) @ T @ Fyconjy[k, :, k2, :].T
                H[1, k2, :, 0, k, :] = H[0, k2, :, 1, k, :].conj()
                H[1, k2, :, 1, k, :] = H[0, k2, :, 0, k, :].conj()

    H = H.reshape(2 * d * nt, -1) @ P_T.reshape((2 * d * nt, -1))

    # application of saddle free newton method
    H = (H + H.T.conj()) / 2
    evals, U = eigh(H)

    # Damping all eigenvalues
    H_abs_inv = U @ np.diag(1 / (np.abs(evals) + lam)) @ U.T.conj()
    Delta_K = ((H_abs_inv @ G.reshape(-1))[: d * nt]).reshape((d, rK, pdim, pdim))

    # Delta_K is already in tangent space but not to sufficient numerical accuracy
    Delta = tangent_proj(K, Delta_K, d, rK)
    res = minimize(lineobjf_isom_geodesic, 1e-8, args=(Delta, K, E, rho, J, y, mle), method=ls, options={"maxiter": 20})
    a = res.x
    K_new = update_K_geodesic(K, Delta, a)
    return K_new


def optimize(y, J, d, r, rK, n_povm, method, K, rho, A, B, fixed_elements, mle=False):
    """Full gate set optimization update alternating on E, K and rho

    Parameters
    ----------
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    d : int
        Number of different gates in the gate set
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Target Kraus rank
    n_povm : int
        Number of POVM-Elements
    method : {"SFN", "GD"}
        Optimization method, Default: "SFN"
    K : numpy array
        Current estimates of Kraus operators
    E : numpy array
        Current POVM estimate
    rho : numpy array
        Current initial state estimate
    A : numpy array
        Current POVM parametrization
    B : numpy array
        Current initial state parametrization

    Returns
    -------
    K_new : numpy array
        Updated estimates of Kraus operators
    X_new : numpy array
        Updated estimates of superoperatos corresponding to K_new
    E_new : numpy array
        Updated POVM estimate
    rho_new : numpy array
        Updated initial state estimate
    A_new : numpy array
        Updated POVM parametrization
    B_new : numpy array
        Updated initial state parametrization
    """
    if "E" in fixed_elements:
        A_new = A
        E_new = np.array([(A_new[i].T.conj() @ A_new[i]).reshape(-1) for i in range(n_povm)])
    else:
        A_new = A_SFN_riem_Hess(K, A, B, y, J, d, r, n_povm, mle=mle)
        E_new = np.array([(A_new[i].T.conj() @ A_new[i]).reshape(-1) for i in range(n_povm)])
    if any(((f"G%i" % i in fixed_elements) for i in range(d))):
        fixed_gates = np.array([(f"G%i" % i in fixed_elements) for i in range(d)])
        if method == "SFN":
            K_new = SFN_riem_Hess(
                K, E_new, rho, y, J, d, r, rK, lam=1e-3, ls="COBYLA", fixed_gates=fixed_gates, mle=mle
            )
        else:
            K_new = gd(K, E_new, rho, y, J, d, r, rK, ls="COBYLA", fixed_gates=fixed_gates, mle=mle)
    else:
        if method == "SFN":
            K_new = SFN_riem_Hess_full(K, E_new, rho, y, J, d, r, rK, lam=1e-3, ls="COBYLA", mle=mle)
        else:
            fixed_gates = np.array([(f"G%i" % i in fixed_elements) for i in range(d)])
            K_new = gd(K, E_new, rho, y, J, d, r, rK, fixed_gates=fixed_gates, ls="COBYLA", mle=mle)
    if "rho" in fixed_elements:
        rho_new = rho
        B_new = B
    else:
        B_new = B_SFN_riem_Hess(K_new, A_new, B, y, J, d, r, n_povm, lam=1e-3, mle=mle)
        rho_new = (B_new @ B_new.T.conj()).reshape(-1)
    X_new = np.einsum("ijkl,ijnm -> iknlm", K_new, K_new.conj()).reshape((d, r, r))
    return K_new, X_new, E_new, rho_new, A_new, B_new


def run_mGST(
    *args,
    method="SFN",
    max_inits=10,
    max_iter=200,
    final_iter=120,
    target_rel_prec=1e-5,
    threshold_multiplier=5,
    fixed_elements=None,
    init=None,
    verbose_level=0,
    return_operators_list:bool=False,
):  # pylint: disable=too-many-branches, too-many-statements
    """Main mGST routine

    Parameters
    ----------
    y : numpy array
        2D array of measurement outcomes for sequences in J;
        Each column contains the outcome probabilities for a fixed sequence
    J : numpy array
        2D array where each row contains the gate indices of a gate sequence
    length : int
        Length of the test sequences
    d : int
        Number of different gates in the gate set
    r : int
        Superoperator dimension of the gates given by the square of the physical dimension
    rK : int
        Target Kraus rank
    n_povm : int
        Number of POVM-Elements
    bsize : int
        Size of the batch (number of sequences)
    meas_samples : int
        Number of samples taken per gate sequence to obtain measurement array y
    method : {"SFN", "GD"}
        Optimization method, Default: "SFN"
    max_iter : int
        Maximum number of iterations on batches; Default: 200
    final_iter : int
        Maximum number of iterations on full data set; Default: 70
    target_rel_prec : float
        Target precision relative to stopping value at which the final iteration loop breaks
    init : [ , , ]
        List of 3 numpy arrays in the format [K, E, rho], that can be used as an initialization;
        If no initialization is given a random initialization is used

    Returns
    -------
    K: numpy array
        Updated estimates of Kraus operators
    X: numpy array
        Updated estimates of superoperatos corresponding to K_new
    E: numpy array
        Updated POVM estimate
    rho : numpy array
        Updated initial state estimate
    res_list : list
        Collected objective function values after each iteration
    """
    y, J, _, d, r, rK, n_povm, bsize, meas_samples = args
    t0 = time.time()
    pdim = int(np.sqrt(r))
    # stopping criterion (Factor 3 can be increased if model mismatch is high)
    delta = threshold_multiplier * (1 - y.reshape(-1)) @ y.reshape(-1) / len(J) / n_povm / meas_samples

    if not fixed_elements:
        fixed_elements = []

    if any(((f"G%i" % i in fixed_elements) for i in range(d))) and method == "SFN":
        warn(
            f"The SFN method with fixed gates is currently only implemented via \n"
            f"iterative updates over individual gates and might lead to a slower converges \n"
            f"compared to the default SFN method.",
            stacklevel=2,
        )

    success = False
    if verbose_level > 0:
        qcvv_logger.info(f"Starting mGST optimization...")

    if init:
        K, E = (init[0], init[1])
        # offset small negative eigenvalues for stability
        rho = init[2] + 1e-14 * np.eye(pdim).reshape(-1)
        A = np.array([la.cholesky(E[k].reshape(pdim, pdim) + 1e-14 * np.eye(pdim)).T.conj() for k in range(n_povm)])
        B = la.cholesky(rho.reshape(pdim, pdim))
        X = np.einsum("ijkl,ijnm -> iknlm", K, K.conj()).reshape((d, r, r))
        res_list = [objf(X, E, rho, J, y)]
    else:
        for i in range(max_inits):
            K, X, E, rho = random_gs(d, r, rK, n_povm)
            A = np.array([la.cholesky(E[k].reshape(pdim, pdim) + 1e-14 * np.eye(pdim)).T.conj() for k in range(n_povm)])
            B = la.cholesky(rho.reshape(pdim, pdim))
            res_list = [objf(X, E, rho, J, y)]
            with logging_redirect_tqdm(loggers=[qcvv_logger] if verbose_level > 0 else None):
                for _ in trange(max_iter, disable=verbose_level == 0):
                    yb, Jb = batch(y, J, bsize)
                    K, X, E, rho, A, B = optimize(yb, Jb, d, r, rK, n_povm, method, K, rho, A, B, fixed_elements)
                    res_list.append(objf(X, E, rho, J, y))
                    if res_list[-1] < delta:
                        qcvv_logger.info(f"Batch optimization successful, improving estimate over full data....")
                        success = True
                        break
            if verbose_level == 2:
                plot_objf(res_list, f"Objective function for batch optimization", delta=delta)
            if success:
                break
            if verbose_level > 0:
                qcvv_logger.info(f"Run {i+1}/{max_inits} failed, trying new initialization...")

    if not success and init is None and verbose_level > 0:
        qcvv_logger.info(f"Success threshold not reached, attempting optimization over full data set...")
        
    kraus_i = [K]
    povm_i = [E]
    state_i = [rho]
    kraus_full_i = [X]        

    with logging_redirect_tqdm(loggers=[qcvv_logger] if verbose_level > 0 else None):
        res_list_mle = []
        for _ in trange(final_iter, disable=verbose_level == 0):
            K, X, E, rho, A, B = optimize(y, J, d, r, rK, n_povm, method, K, rho, A, B, fixed_elements, mle=True)
            
            kraus_i.append(K)
            povm_i.append(E)
            state_i.append(rho)
            kraus_full_i.append(X)
            
            res_list.append(objf(X, E, rho, J, y))
            res_list_mle.append(objf(X, E, rho, J, y, mle=True))
            if (
                len(res_list_mle) >= 2
                and np.abs(res_list_mle[-2] - res_list_mle[-1]) < res_list_mle[-1] * target_rel_prec
            ):
                break
    if verbose_level == 2:
        plot_objf(res_list, f"Least squares error over batches and full data", delta=delta)
        plot_objf(res_list_mle, f"Negative log-likelihood over full data")
    if verbose_level > 0:
        if success or (res_list[-1] < delta):
            qcvv_logger.info(f"Convergence criterion satisfied")
        else:
            qcvv_logger.warning(
                f"Convergence criterion not satisfied. Potential causes include too low max_iterations, bad initialization or model mismatch."
            )
        qcvv_logger.info(
            f"Final objective {Decimal(res_list[-1]):.2e} in time {(time.time() - t0):.2f}s",
        )
        
    if return_operators_list:
        K = kraus_i
        E = povm_i
        rho = state_i
        X = kraus_full_i
    return K, X, E, rho, res_list
