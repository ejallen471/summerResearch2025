#############################################################################

# Calculate the moments using the 2D KDE PDF reconstruction and empirical distribution

#############################################################################

import numba
import pickle
import warnings
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

#############################################################################
#############################################################################

# --- Load serialised data
def read_in_data():
    """
    Load flavour- and evolution-basis replica data from the shared 00_data.

    Expects ``flavour_basis.pkl`` and ``evolution_basis.pkl`` in the
    ``00_data`` directory one level above this folder.
    """

    data_dir = Path(__file__).resolve().parent.parent / "00_data"
    paths = [data_dir / "flavour_basis.pkl", data_dir / "evolution_basis.pkl"]
    return (pickle.load(open(p, "rb")) for p in paths)

# --- Extract 2D data at fixed index
def prepare_2d_data(res, key_x, key_y, index=25):
    return np.array([[r[key_x][index], r[key_y][index]] for r in res])

# --- Get data and transform into one single array
def prepare_data(res, keys, indices=None):
    """
    Prepare data from res list of dicts.

    Parameters:
    - res: list of replicas (each replica is a dict of arrays)
    - keys: list of keys to extract
    - indices: None, int, or list/array of ints

    Returns:
    - np.ndarray of shape:
        (num_replicas, num_keys) if indices is int or None
        (num_replicas, num_keys, len(indices)) if indices is list/array
    """
    num_replicas = len(res)
    num_keys = len(keys)

    if indices is None:
        indices = np.arange(50)
    
    indices = np.array(indices)
    data_array = np.empty((num_replicas, num_keys, len(indices)), dtype=float)
    for i, replica in enumerate(res):
        for j, key in enumerate(keys):
            data_array[i, j, :] = replica[key][indices]

    return data_array

#############################################################################
### BUILD KDE MODEL
#############################################################################

# --- Mahalanobis squared 2x2 ---
@numba.njit(fastmath=True)
def _mahalanobis2_2x2(dx, dy, invH00, invH01, invH11):
    return invH00*dx*dx + 2*invH01*dx*dy + invH11*dy*dy

# --- SCV objective, fully log-space ---
@numba.njit(fastmath=True)
def _scv_objective_numba(L_flat, data):
    n = data.shape[0]
    L00, L10, L11 = L_flat[0], L_flat[1], L_flat[2]

    # regularized H
    H00 = max(L00*L00, 1e-10)
    H01 = L00*L10
    H11 = max(L10*L10 + L11*L11, 1e-10)

    detH = H00*H11 - H01*H01
    if detH <= 0.0:
        print('detH < 0, returned inf')
        return np.inf

    invH00 =  H11 / detH
    invH01 = -H01 / detH
    invH11 =  H00 / detH

    log_norm_const = -np.log(2*np.pi) - 0.5*np.log(detH)
    total_log = 0.0

    for i in range(n):
        xi0, xi1 = data[i,0], data[i,1]

        # leave-one-out Mahalanobis in log-space (basically leave out the reference point which you calculate the distance to other points from)
        m2_max = -1e20
        m2_vals = np.empty(n-1, dtype=np.float64)
        idx = 0
        for j in range(n):
            if i == j:
                continue
            dx = xi0 - data[j,0]
            dy = xi1 - data[j,1]
            m2 = -0.5 * _mahalanobis2_2x2(dx, dy, invH00, invH01, invH11)
            m2_vals[idx] = m2
            if m2 > m2_max:
                m2_max = m2
            idx += 1

        # log-sum-exp trick - rewrite it by factoring out the maximum value to stabilise it
        s = 0.0
        for k in range(n-1):
            s += np.exp(m2_vals[k] - m2_max)
        p_i = (s / (n-1)) * np.exp(log_norm_const + m2_max)

        if p_i <= 0.0 or not np.isfinite(p_i):
            return np.inf
        total_log += np.log(p_i)

    return - total_log / n

def scv_objective(params, data):
    try:
        val = _scv_objective_numba(params, data)
        return val if np.isfinite(val) else np.inf
    except:
        return np.inf

# --- Ensure SPD 2x2 ---
def _make_positive_definite_2x2(M, eps=1e-5):
    M = 0.5*(M + M.T)
    det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    if det <= eps:
        M[0,0] += eps
        M[1,1] += eps
    return M

# --- Robust SCV bandwidth for 2x2 ---
def estimate_bandwidth_matrix_scv(data, maxiter=200):
    n, d = data.shape
    assert d == 2, "Only 2x2 supported"

    Sigma = np.cov(data, rowvar=False)
    Sigma = _make_positive_definite_2x2(Sigma)

    c = (4 / (d + 2))**(1/(d+4)) * n**(-1/(d+4))
    L0 = np.linalg.cholesky(Sigma) * c
    initial_params = L0[np.tril_indices(d)]

    # reasonable bounds on L
    bounds = [(1e-6, 1e2), (-1e2, 1e2), (1e-6, 1e2)]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = minimize(
                scv_objective,
                initial_params,
                args=(data,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-6}
            )
    except:
        result = None

    if result is None or not result.success or not np.isfinite(result.fun):
        print('Result Inf, using Backup')
        return Sigma * c**2

    L_opt = np.zeros((2,2))
    L_opt[np.tril_indices(2)] = result.x
    H = L_opt @ L_opt.T
    
    return _make_positive_definite_2x2(H)

#############################################################################

# ---------------------------------------------
# --- KDE PDF Evaluation Function ---
# ---------------------------------------------

# --- Estimate KDE at given points using batching 
def calc_kdeGaussianEstimate_nD(points, data, bandwidth, batch_size=50):
    n, d = data.shape
    m = points.shape[0]

    bandwidth_inv = np.linalg.inv(bandwidth)
    det_bandwidth = np.linalg.det(bandwidth)
    norm_const = 1.0 / ((2 * np.pi) ** (d / 2) * np.sqrt(det_bandwidth))

    densities = np.empty(m, dtype=np.float32)

    for start in range(0, m, batch_size): # calculate in batches to reduce computational load
        end = min(start + batch_size, m)
        batch = points[start:end].astype(np.float32) # slices to get a subset of points (exclusive slicing)
        diffs = batch[:, np.newaxis, :] - data  # (b, n, d)
        D2 = np.einsum('bnd,dd,bnd->bn', diffs, bandwidth_inv, diffs) # einsum - sinstein summation, general syntax is np.einsum(subscripts, *operands), these are the input subscripts bnd,dd,bnd and the output subscript is bn and summuation is over d because it doesnt appear in the outputs
        kernel_vals = norm_const * np.exp(-0.5 * D2)
        densities[start:end] = np.mean(kernel_vals, axis=1)

    return densities

# --- calculate the pdf estimate for a single point - NOT USED
def calc_pdf_pointwise(point, data, bandwidthMatrix):
    d = data.shape[1] if data.ndim > 1 else 1
    n = data.shape[0]

    if d == 1:
        diffs = data - point
        D2 = (diffs / bandwidthMatrix) ** 2
        norm_const = 1.0 / ((2 * np.pi) ** (d / 2) * bandwidthMatrix)
        kernel_vals = np.exp(-0.5 * D2)
        return (1.0 / n) * np.sum(norm_const * kernel_vals)

    else:
        H_inv = np.linalg.inv(bandwidthMatrix)
        det_H = np.linalg.det(bandwidthMatrix)
        diffs = data - point
        u = diffs @ H_inv.T
        D2 = np.sum(u ** 2, axis=1)
        norm_const = 1.0 / ((2 * np.pi) ** (d / 2) * (det_H))
        kernel_vals = np.exp(-0.5 * D2)
        return (1.0 / n) * np.sum(norm_const * kernel_vals)

@numba.njit(parallel=True, fastmath=True)
def calc_pdf_batch(points, data, H_inv, norm_const):
    m, d = points.shape
    n = data.shape[0]
    pdf_vals = np.zeros(m)

    for i in numba.prange(m):
        diff = points[i] - data  # shape (n, d)
        # Mahalanobis distance for all n data points using dot product
        maha = np.sum(diff @ H_inv * diff, axis=1)  # shape (n,)
        pdf_vals[i] = norm_const * np.mean(np.exp(-0.5 * maha))

    return pdf_vals

# --------------------------------------------------------
# --- Monte Carlo moment integration (Importance Sampling) 
# --------------------------------------------------------

def calc_covariance_KDE(data, bandwidthMatrix, n_samplesMC, d=2):

    # --- empirical mean and covariance for proposal distribution
    mu = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)

    # --- draw samples from proposal
    try:
        samples = np.random.multivariate_normal(mu, cov, size=n_samplesMC)
    except np.linalg.LinAlgError:
        return np.nan, np.full(d, np.nan), np.full(d, np.nan), np.full((d, d), np.nan)

    # --- 2x2 determinant and inverse
    a, b = bandwidthMatrix[0,0], bandwidthMatrix[0,1]
    c = bandwidthMatrix[1,1]
    det_H = a*c - b**2
    if det_H <= 0:
        return np.nan, np.full(d, np.nan), np.full(d, np.nan), np.full((d, d), np.nan)
    
    logdet_H = np.log(det_H) # numerical stability
    H_inv = np.array([[ c, -b],
                      [-b,  a]]) / det_H

    # --- normalisation constant
    norm_const = np.exp(-np.log(2*np.pi) - 0.5*logdet_H)  # (2*pi)^(d/2) with d=2

    # --- evaluate KDE density at samples
    p_vals = calc_pdf_batch(samples, data, H_inv, norm_const)  

    # --- proposal density
    q_pdf = multivariate_normal(mean=mu, cov=cov, allow_singular=True)
    q_vals = q_pdf.pdf(samples)

    # --- importance weights
    weights = np.divide(p_vals, q_vals, out=np.zeros_like(p_vals), where=(q_vals > 0))
    weight_sum = np.sum(weights)
    if weight_sum <= 1e-10 or not np.isfinite(weight_sum):
        return np.nan, np.full(d, np.nan), np.full(d, np.nan), np.full((d, d), np.nan)

    # -----------------------------
    # Moments
    # -----------------------------
    
    zerothMoment = np.mean(weights)
    firstMoment = (weights @ samples) / weight_sum

    centred = samples - firstMoment
    variance = np.sum(weights[:, None] * centred**2, axis=0) / weight_sum
    covarianceMatrix = (centred.T * weights) @ centred / weight_sum

    return zerothMoment, firstMoment, variance, covarianceMatrix

# ---------------------------------------------
# --- Errors 
# ---------------------------------------------

# --- calculate errors via bootstrap for importance method 
def calc_bootstrap_error_mc_importance(data, bandwidthMatrix, n_bootstrap, n_samplesBootStrap):
    """
    Bootstrap standard error for importance sampling KDE moments.
    Extends to calculate covariance matrix error too.
    """

    n, d = data.shape
    zeroth_moments = np.zeros(n_bootstrap)
    first_moments = np.zeros((n_bootstrap, d))
    variances = np.zeros((n_bootstrap, d))
    covariances = np.zeros((n_bootstrap, d, d)) 

    for i in range(n_bootstrap):
        idxs = np.random.choice(n, size=n, replace=True)
        resampled_data = data[idxs]

        zerothMoment, firstMomentVec, varianceVec, covarianceMatrix = calc_covariance_KDE(resampled_data, bandwidthMatrix, n_samplesMC=n_samplesBootStrap)

        zeroth_moments[i] = zerothMoment
        first_moments[i] = firstMomentVec
        variances[i] = varianceVec
        covariances[i] = covarianceMatrix

    std_zeroth = np.std(zeroth_moments, ddof=1)
    std_first = np.std(first_moments, axis=0, ddof=1)
    std_variance = np.std(variances, axis=0, ddof=1)
    std_covariance = np.std(covariances, axis=0, ddof=1)

    return std_zeroth, std_first, std_variance, std_covariance

# --- calculate bootstrap errors for empirical distributions
def calc_bootstrapErrorEmpirical_all(data, n_bootstrap):
    """
    Returns bootstrap standard errors for mean, variance, and covariance.
    
    Parameters:
    - data: ndarray of shape (n_samples, n_features)
    - n_bootstrap: number of bootstrap resamples

    Returns:
    - dict with keys 'mean', 'variance', 'covariance', each containing bootstrap standard error
    """
    
    # Store bootstrap results
    mean_samples = []
    var_samples = []
    cov_samples = []

    n = data.shape[0]

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        resample = data[indices]

        # Compute statistics
        mean_samples.append(np.mean(resample, axis=0))
        var_samples.append(np.var(resample, axis=0, ddof=1))
        cov_samples.append(np.cov(resample, rowvar=False))

    # Convert lists to arrays for std calculation
    mean_samples = np.stack(mean_samples)
    var_samples = np.stack(var_samples)
    cov_samples = np.stack(cov_samples)

    # Compute standard errors
    mean_error = np.std(mean_samples, axis=0, ddof=1)
    var_error = np.std(var_samples, axis=0, ddof=1)
    cov_error = np.std(cov_samples, axis=0, ddof=1)

    return mean_error, var_error, cov_error

# ---------------------------------------------
# --- Run Moments Code 
# ---------------------------------------------

# --- Run all the moment calculations
def run_2D_momentCalculations(data, bandwidthMatrix, n_bootstrap=100, n_samplesMC=10000):
    """
    Compute and print 2D KDE and empirical means and covariances, with bootstrap errors.

    Parameters
    ----------
    data : ndarray
        Array of shape ``(n_samples, 2)`` containing the 2D samples.
    bandwidthMatrix : ndarray
        2x2 bandwidth matrix used for the Gaussian KDE.
    n_bootstrap : int, optional
        Number of bootstrap resamples.
    n_samplesMC : int, optional
        Number of Monte Carlo samples used in the KDE importance sampling.
    """

    zerothMoment, firstMomentVec, varianceVec, covarianceMatrix = calc_covariance_KDE(
        data, bandwidthMatrix, n_samplesMC=n_samplesMC
    )
    _, bootstrapError_mean_is, _, bootstrap_covariance_is = calc_bootstrap_error_mc_importance(
        data, bandwidthMatrix, n_bootstrap, n_samplesMC
    )

    # Empirical mean and covariance
    empirical_mean = np.mean(data, axis=0)
    empirical_covariance = np.cov(data, rowvar=False)
    bootstrapError_mean, _, bootstrap_covariance = calc_bootstrapErrorEmpirical_all(
        data, n_bootstrap
    )

    print("\n--- 2D Moments ---")
    print("KDE-based:")
    print(f"  Mean: {firstMomentVec} +/- {bootstrapError_mean_is}")
    print(f"  Covariance:\n{covarianceMatrix}\n  +/-\n{bootstrap_covariance_is}")

    print("Empirical:")
    print(f"  Mean: {empirical_mean} +/- {bootstrapError_mean}")
    print(f"  Covariance:\n{empirical_covariance}\n  +/-\n{bootstrap_covariance}")

#############################################################################
#############################################################################

# --- Main function 
def main():

    res_flav, res_ev = read_in_data()

    keys_ev = ['Sigma', 'V', 'V3', 'V8', 'T3', 'T8', 'c+', 'g', 'V15']
    keys_flav = ['d', 'u', 's', 'c', 'dbar', 'ubar', 'sbar', 'cbar', 'g']
    
    # choose flavours and index here
    keys_test = ['u', 'g']
    index = 42
    
    # calculate the bandwidth matrix - through SCV 
    data = prepare_data(res_flav, keys_test, index)
    bandwidthMatrix = estimate_bandwidth_matrix_scv(data)
    print(bandwidthMatrix)

    # Calculate and print KDE and empirical means and covariances
    run_2D_momentCalculations(data, bandwidthMatrix)


if __name__ == "__main__":
    main()


