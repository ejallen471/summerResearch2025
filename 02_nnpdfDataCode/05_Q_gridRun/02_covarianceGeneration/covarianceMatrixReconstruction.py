"""
covarianceMatrixReconstruction.py

Construct covariance and correlation matrices from NNDPF replica data.

This module provides functions to:
- read serialised replica data
- prepare data for pairwise 2D KDE and empirical calculations
- estimate 2x2 KDE bandwidth matrices via smooth cross-validation (SCV)
- compute KDE-based covariance estimates using importance-sampling Monte Carlo
- assemble full covariance and correlation matrices in parallel and store
    intermediate results in memory-mapped files and final CSV outputs

Outputs
-------
- CSVs: 
    `covariance_kde.csv`, 
    `correlation_kde.csv`,
    `covariance_empirical.csv`, 
    `correlation_empirical.csv`.
- Optional memmap files: 
    `<tmp_prefix>_cov_kde.dat`,
    `<tmp_prefix>_cov_emp.dat`, 
    `<tmp_prefix>_count_kde.dat`,
    `<tmp_prefix>_count_emp.dat`.

"""

import pickle
import warnings
import numba
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from joblib import Parallel, delayed
from tqdm import tqdm

np.set_printoptions(threshold=np.inf, linewidth=np.inf) 

#############################################################################
#############################################################################

# --- Load serialised data
def read_in_data():
    """
    Load serialised replica data from working directory.

    Expects files matching "flavour_basis*" and "evolution_basis*" in the
    current working directory and returns two deserialised objects (flavour
    and evolution replicas).

    Returns
    -------
    tuple
        A generator yielding the unpickled objects in the order defined above.
    """
    paths = [Path("./flavour_basis.pkl"), Path("./evolution_basis.pkl")]
    return (pickle.load(open(p, 'rb')) for p in paths)

# --- Get 2D data at fixed index
def prepare_data_fixedIndex(res, key1, key2, idx1, idx2):
    """
    Extract paired 2D data at specific grid indices from replicas.

    Parameters
    ----------
    res : list
        List of replica dicts where each dict maps flavour keys to arrays.
    key1, key2 : str
        Keys to extract from each replica.
    idx1, idx2 : int
        Grid indices for the two keys.

    Returns
    -------
    ndarray
        Array of shape (n_replicas, 2) where each row contains
        [replica[key1][idx1], replica[key2][idx2]].
    """
    data = np.array([[r[key1][idx1], r[key2][idx2]] for r in res])
    return data

# --- Get the data and bring into lists etc
def prepare_data_rangeIndices(res, keys, indices=None):
    """
    Prepare flattened data for empirical covariance over ranges of indices.

    This flattens selected grid points for multiple flavours so that each
    replica yields a single long vector: [f1[idxs], f2[idxs], ...].

    Parameters
    ----------
    res : list
        List of replica dicts.
    keys : list of str
        Flavour keys to extract.
    indices : None, int, or array-like
        If None, uses all 50 grid points. If int, treats it as the number
        of grid points and uses indices in ``range(indices)``.

    Returns
    -------
    ndarray
        Array of shape (n_replicas, num_keys * len(indices)).
    """
    num_replicas = len(res)
    num_keys = len(keys)

    # Default: use all 50 grid points
    if indices is None:
        indices = np.arange(50)

    # If single int, treat it as the number of grid points to select.
    if isinstance(indices, int):
        indices = np.arange(indices)

    # Final shape = (num_replicas, num_keys * len(indices))
    data_array = np.empty((num_replicas, num_keys * len(indices)), dtype=float)

    for i, replica in enumerate(res):
        values = []
        for key in keys:
            values.append(replica[key][indices])  # shape: (len(indices),)
        data_array[i] = np.concatenate(values)    # flatten: (num_keys * len(indices),)

    return data_array

#############################################################################
### BUILDING KDE STUFF
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
def _make_positive_definite_2x2(M, eps=1e-4):
    M = 0.5*(M + M.T)
    det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    if det <= eps:
        M[0,0] += eps
        M[1,1] += eps
    return M

# --- Robust SCV bandwidth for 2x2 ---
def estimate_bandwidth_matrix_scv(data, maxiter=200):
    """
    Estimate a 2x2 symmetric positive-definite bandwidth matrix via SCV.

    Uses a Cholesky parameterisation and L-BFGS-B optimisation of a
    numba-accelerated SCV objective. Returns a 2x2 bandwidth matrix H.

    Parameters
    ----------
    data : ndarray
        Input data of shape (n_samples, 2).
    maxiter : int
        Maximum iterations passed to the optimiser.

    Returns
    -------
    ndarray
        2x2 positive-definite bandwidth matrix.
    """
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
### MOMENTS CALCULATION STUFF
#############################################################################

# ---------------------------------------------
# --- KDE PDF Evaluation Function 
# ---------------------------------------------

@numba.njit(parallel=True, fastmath=True)
def calc_pdf_batch_numba(points, data, H_inv, norm_const):
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

def calc_covariance_KDE(data, bandwidthMatrix, n_samplesMC):
    """
    Estimate covariance matrix from KDE via importance-sampling MC.

    Samples from a Gaussian proposal (mean and empirical covariance of the
    data) are reweighted by KDE values to compute the KDE mean and covariance.

    Parameters
    ----------
    data : ndarray
        Array of shape (n_samples, d).
    bandwidthMatrix : ndarray
        dxd bandwidth matrix for KDE.
    n_samplesMC : int
        Number of Monte Carlo samples to draw for importance sampling.

    Returns
    -------
    ndarray
        Estimated covariance matrix of shape (d, d) or NaNs on failure.
    """
    d = data.shape[1]
    mu = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    
    try:
        samples = np.random.multivariate_normal(mu, cov, size=n_samplesMC)
    except np.linalg.LinAlgError:
        return np.full((d, d), np.nan)

    try:
        H_inv = np.linalg.inv(bandwidthMatrix)
        det_H = np.linalg.det(bandwidthMatrix)
        norm_const = 1.0 / (np.sqrt((2 * np.pi)**d) * np.sqrt(det_H))
    except Exception:
        return np.full((d, d), np.nan)

    p_vals = calc_pdf_batch_numba(samples, data, H_inv, norm_const)
    q_pdf = multivariate_normal(mean=mu, cov=cov, allow_singular=True)
    q_vals = q_pdf.pdf(samples)

    with np.errstate(divide='ignore', invalid='ignore'):
        weights = np.where(q_vals > 0, p_vals / q_vals, 0.0)

    weight_sum = np.sum(weights)
    if weight_sum <= 1e-10 or not np.isfinite(weight_sum):
        return np.full((d, d), np.nan)
    
    # --- KDE mean (first moment)
    firstMomentVec = np.sum(weights[:, None] * samples, axis=0) / weight_sum

    # --- KDE covariance
    weighted_outer = np.einsum('i,ij,ik->jk', weights, samples, samples)
    covarianceMatrix = weighted_outer / weight_sum - np.outer(firstMomentVec, firstMomentVec)
    
    return covarianceMatrix

# ---------------------------------------------
# --- Run Moments Code 
# ---------------------------------------------

# --- Run all the moment calculations
def run_2D_momentCalculations(data, bandwidthMatrix, n_bootstrap=100, n_samplesMC=10000):
    """
    Run KDE-based and empirical covariance calculations for 2D data.

    Returns a tuple (kde_covariance, empirical_covariance).
    """
    covarianceMatrix = calc_covariance_KDE(data, bandwidthMatrix, n_samplesMC=n_samplesMC)    
    empiricalCovarianceMatrix = np.cov(data, rowvar=False)
    
    return covarianceMatrix, empiricalCovarianceMatrix

#############################################################################
### Matrix Construction
#############################################################################

def _single_covariance_task(idx1, idx2, flav1, flav2, flav_to_index, res_flav, numberOfGridPoints):
    """
    Compute 2x2 covariance for a single pair of grid indices/flavours.

    Returns (i_pos, j_pos, cov2x2, cov2x2_empirical) or None on failure.
    """
    if idx1 == idx2 and flav1 == flav2:
        return None

    data = prepare_data_fixedIndex(res_flav, flav1, flav2, idx1, idx2)
    bandwidth_matrix = estimate_bandwidth_matrix_scv(data)
    if bandwidth_matrix is None or np.any(np.diag(bandwidth_matrix) < 1e-6):
        return None

    covMatrix, covMatrix_empirical = run_2D_momentCalculations(data, bandwidth_matrix)
    i_pos = flav_to_index[flav1] * numberOfGridPoints + idx1
    j_pos = flav_to_index[flav2] * numberOfGridPoints + idx2

    return i_pos, j_pos, covMatrix, covMatrix_empirical

def _single_covariance_task_chunk(tasks, flav_to_index, res_flav, numberOfGridPoints, cov_full, cov_full_empirical, count_matrix, count_matrix_empirical):
    """
    Process a chunk of tasks, accumulating results into memmaps.

    Each task is a tuple (idx1, idx2, flav1, flav2). This function updates
    the provided memmapped accumulation arrays in-place.
    """
    for idx1, idx2, flav1, flav2 in tasks:
        res = _single_covariance_task(idx1, idx2, flav1, flav2, flav_to_index, res_flav, numberOfGridPoints)
        if res is None:
            continue

        i, j, cov2x2, cov2x2_emp = res

        # --- KDE accumulation
        cov_full[i, i] += cov2x2[0, 0]
        cov_full[j, j] += cov2x2[1, 1]
        cov_full[i, j] += cov2x2[0, 1]
        cov_full[j, i] = cov_full[i, j]

        count_matrix[i, i] += 1
        count_matrix[j, j] += 1
        count_matrix[i, j] += 1
        count_matrix[j, i] += 1

        # --- Empirical accumulation
        cov_full_empirical[i, i] += cov2x2_emp[0, 0]
        cov_full_empirical[j, j] += cov2x2_emp[1, 1]
        cov_full_empirical[i, j] += cov2x2_emp[0, 1]
        cov_full_empirical[j, i] = cov_full_empirical[i, j]

        count_matrix_empirical[i, i] += 1
        count_matrix_empirical[j, j] += 1
        count_matrix_empirical[i, j] += 1
        count_matrix_empirical[j, i] += 1

    return True  

def construct_covariance_matrix_parallel(keys_flav, res_flav, numberOfGridPoints, n_jobs=-1, tmp_prefix="covariance_tmp", flush_every=10):
    """
    Construct the full covariance matrix in parallel and return results.

    Writes memmap temporary files named with `tmp_prefix` and returns the
    normalised KDE and empirical covariance matrices as numpy arrays.
    """
    n_flav = len(keys_flav)
    dim = n_flav * numberOfGridPoints
    flav_to_index = {flav: i for i, flav in enumerate(keys_flav)}

    # --- memmaps
    cov_full = np.memmap(f"{tmp_prefix}_cov_kde.dat", dtype=np.float64, mode="w+", shape=(dim, dim))
    cov_full_empirical = np.memmap(f"{tmp_prefix}_cov_emp.dat", dtype=np.float64, mode="w+", shape=(dim, dim))
    count_matrix = np.memmap(f"{tmp_prefix}_count_kde.dat", dtype=np.int32, mode="w+", shape=(dim, dim))
    count_matrix_empirical = np.memmap(f"{tmp_prefix}_count_emp.dat", dtype=np.int32, mode="w+", shape=(dim, dim))

    cov_full[:] = 0.0
    cov_full_empirical[:] = 0.0
    count_matrix[:] = 0
    count_matrix_empirical[:] = 0

    # --- tasks
    tasks = [
        (idx1, idx2, flav1, flav2)
        for idx1 in range(numberOfGridPoints)
        for idx2 in range(numberOfGridPoints)
        for i_f1, flav1 in enumerate(keys_flav)
        for i_f2, flav2 in enumerate(keys_flav)
        if i_f1 < i_f2 or (i_f1 == i_f2 and idx1 <= idx2)
    ]

    chunk_size = max(1, len(tasks) // (n_jobs * 4))
    task_chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]

    for batch_start in tqdm(range(0, len(task_chunks), flush_every), desc="Covariance Batches"):
        batch_chunks = task_chunks[batch_start:batch_start + flush_every]

        Parallel(n_jobs=n_jobs)(
            delayed(_single_covariance_task_chunk)(
                chunk, flav_to_index, res_flav, numberOfGridPoints,
                cov_full, cov_full_empirical, count_matrix, count_matrix_empirical
            ) for chunk in batch_chunks
        )

        cov_full.flush()
        cov_full_empirical.flush()
        count_matrix.flush()
        count_matrix_empirical.flush()

    # --- normalise
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised_cov_kde = np.divide(cov_full, count_matrix, where=count_matrix > 0)
        normalised_cov_emp = np.divide(cov_full_empirical, count_matrix_empirical, where=count_matrix_empirical > 0)

    return normalised_cov_kde, normalised_cov_emp

def covariance_to_correlation(cov_matrix, eps=1e-12):
    """
    Convert covariance matrix to correlation matrix safely.

    A small `eps` prevents division by zero for zero-variance entries.
    """
    
    diag = np.sqrt(np.diag(cov_matrix))
    diag = np.maximum(diag, eps)
    inv_diag = 1.0 / diag
    corr_matrix = cov_matrix * np.outer(inv_diag, inv_diag)
    np.fill_diagonal(corr_matrix, 1.0)
    
    return corr_matrix

#############################################################################
### MAIN FUNCTION
#############################################################################

def main(empiricalReconstruction=False, integralReconstructionFull=True, cleanup_memmaps=False):
    
    path = next(Path(".").glob("flavour_basis*"))
    res_flav = pickle.load(open(path, 'rb'))
    keys_flav = ['d', 'u', 's', 'c', 'dbar', 'ubar', 'sbar', 'cbar', 'g']

    numberOfGridPoints = 45

# ---------------------------------------------
# --- Matrix reconstruction (Empricial)  
# ---------------------------------------------

    def empirical_cov_corr_with_error(dataMatrix):
        """
        Compute empirical covariance and correlation matrices.

        Returns
        -------
        covarianceMatrix : ndarray
            (d, d) empirical covariance matrix
        correlationMatrix : ndarray
            (d, d) empirical correlation matrix
        """
        n_samples, d = dataMatrix.shape

        # Empirical covariance
        covarianceMatrix = np.cov(dataMatrix, rowvar=False)

        # Empirical correlation
        diag_std = np.sqrt(np.diag(covarianceMatrix))
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_diag_std = 1.0 / diag_std
            correlationMatrix = covarianceMatrix * np.outer(inv_diag_std, inv_diag_std)
            correlationMatrix[~np.isfinite(correlationMatrix)] = 0.0

        return covarianceMatrix, correlationMatrix

    if empiricalReconstruction:
        dataMatrix = prepare_data_rangeIndices(res_flav, keys_flav, numberOfGridPoints)

        covMatrix, corrMatrix = empirical_cov_corr_with_error(dataMatrix)

        np.savetxt("covariance_empirical.csv", covMatrix, delimiter=",", fmt="%.6e")
        np.savetxt("correlation_empirical.csv", corrMatrix, delimiter=",", fmt="%.6e")

# ---------------------------------------------
# --- Matrix reconstruction (KDE)
# ---------------------------------------------

    if integralReconstructionFull:
        
        cov_full_kde, cov_full_emp = construct_covariance_matrix_parallel(keys_flav, res_flav, numberOfGridPoints=numberOfGridPoints, n_jobs=-1)

        print('Covariance Calculations Complete, Moving onto Correlation Calculations')

        np.savetxt("covariance_kde.csv", cov_full_kde, delimiter=",", fmt="%.6e")
        np.savetxt("covariance_empirical.csv", cov_full_emp, delimiter=",", fmt="%.6e")

        corr_full_kde  = covariance_to_correlation(cov_matrix=cov_full_kde)
        corr_full_emp = covariance_to_correlation(cov_matrix=cov_full_emp)

        print('Correlation Calculations Complete')

        np.savetxt("correlation_kde.csv", corr_full_kde, delimiter=",", fmt="%.6e")
        np.savetxt("correlation_empirical.csv", corr_full_emp, delimiter=",", fmt="%.6e")

# ---------------------------------------------
# --- Cleanup memmaps
# ---------------------------------------------

    if cleanup_memmaps:
        tmp_files = [
            "covariance_tmp_cov_kde.dat",
            "covariance_tmp_cov_emp.dat",
            "covariance_tmp_count_kde.dat",
            "covariance_tmp_count_emp.dat"
        ]
        
        for f in tmp_files:
            try:
                Path(f).unlink()
                print(f"Deleted temporary file: {f}")
            except Exception as e:
                print(f"Error deleting file {f}: {e}")

if __name__ == "__main__":
    main()
