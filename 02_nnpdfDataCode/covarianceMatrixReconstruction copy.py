import pickle
import warnings
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from joblib import Parallel, delayed
from tqdm import tqdm

# If running on separate laptop/computer, this will need commenting out 
plt.style.use('pythonStyle')
import pythonStyle as ed

# np.random.seed(4) # run code with the same random seed each time
np.set_printoptions(threshold=np.inf, linewidth=np.inf) # printing settings

#############################################################################
#############################################################################

# --- Load serialised data
def read_in_data():
    paths = [Path("./flavour_basis.pkl"), Path("./evolution_basis.pkl")]
    return (pickle.load(open(p, 'rb')) for p in paths)

# --- Get 2D data at fixed index
def prepare_data_fixedIndex(res, key1, key2, idx1, idx2):
    """
    Extracts data from replicas for:
      - key1 at idx1
      - key2 at idx2

    Returns:
    - np.ndarray of shape (num_replicas, 2)
      where each row = [key1[idx1], key2[idx2]] for one replica
    """
    data = np.array([[r[key1][idx1], r[key2][idx2]] for r in res])
    return data

# --- Get the data and bring into lists etc
def prepare_data_rangeIndices(res, keys, indices=None):
    """
    Prepare flattened data from res list of dicts.

    Parameters:
    - res: list of replicas (each replica is a dict of arrays)
    - keys: list of keys (flavours) to extract
    - indices: None, int, or list/array of indices

    Returns:
    - np.ndarray of shape (num_replicas, num_keys * len(indices))
    """
    num_replicas = len(res)
    num_keys = len(keys)

    # Default: use all 50 grid points
    if indices is None:
        indices = np.arange(50)

    # If single int, treat as "0 to index" range
    if isinstance(indices, int):
        indices = np.arange(indices + 1)

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

# --- calculate the bandwidth matrix - diagonal, ignore covariance, much quicker - NOT USED
def calc_bandwidthMatrix(data, n=10000):
    
    # Calculate Silverman bandwidth vector
    n, d = data.shape
    sigma = np.std(data, axis=0, ddof=1)
    h_p = (4 / (d + 2)) ** (1 / (d + 4)) * n ** (-1 / (d + 4)) * sigma
    print(f'Initial h_p: {h_p}')

    # Create candidate bandwidth matrices
    scaling_factors = np.linspace(0.5, 2.0, 10)

    H_Matrix_candidateLst = []
    hLst = []
    for s in scaling_factors:
        H_diag = (s * h_p) ** 2 
        H_matrix = np.diag(H_diag)
        H_Matrix_candidateLst.append(H_matrix)
        hLst.append(H_diag)

    # Cross-validation
    bandwidthMatrix, _ = calc_kdeCrossValidation_nD(data, H_Matrix_candidateLst, k=5, subsample_size=10000)

    return bandwidthMatrix

# --- Helper function for estimate_bandwidth_matrix_scv
def scv_objective(params, data):
    n, d = data.shape

    # Build lower-triangular matrix L from params
    lowerTriangularMatrix = np.zeros((d, d))
    tril_indices = np.tril_indices(d)
    lowerTriangularMatrix[tril_indices] = params

    # Bandwidth matrix H = L L^T + epsilon * I
    choleskyMatrix = lowerTriangularMatrix @ lowerTriangularMatrix.T

    try:
        H_inv = np.linalg.inv(choleskyMatrix)
        det_H = np.linalg.det(choleskyMatrix)
    except np.linalg.LinAlgError:
        return None

    norm_const = 1.0 / ((2 * np.pi) ** (d / 2) * np.sqrt(det_H))

    diffs = data[:, np.newaxis, :] - data[np.newaxis, :, :]  # (n, n, d)

    dists = np.einsum('ijk,kl,ijl->ij', diffs, H_inv, diffs)

    np.fill_diagonal(dists, np.inf)

    kernels = np.exp(-0.5 * dists)

    estimate = norm_const * np.mean(kernels, axis=1)  # (n,)

    # Suppress divide by zero warnings for log(estimate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        score = -np.mean(np.log(estimate))

    return score

# --- calculate the bandwidth matrix with the covariance terms
def estimate_bandwidth_matrix_scv(data, initial_scale=1.0):
    n, d = data.shape

    initialCholeskyMatrix = np.zeros((d, d))
    np.fill_diagonal(initialCholeskyMatrix, initial_scale * np.std(data, axis=0))
    initial_params = initialCholeskyMatrix[np.tril_indices(d)]

    # Suppress runtime warnings during minimise
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = minimize(scv_objective, initial_params, args=(data,), method='L-BFGS-B', options={'maxiter': 500})

    if not result.success:
        return None

    L_opt = np.zeros((d, d))
    L_opt[np.tril_indices(d)] = result.x
    H_opt = L_opt @ L_opt.T

    # Final checks
    if not np.all(np.isfinite(H_opt)):
        return None
    if np.any(np.diag(H_opt) <= 0):
        return None

    return H_opt

#############################################################################

# --- Cross-validation of KDE bandwidth matrix- using subsampling for speed
def calc_kdeCrossValidation_nD(data, H_Matrix_candidateLst, k=5, subsample_size=10000):
    n, d = data.shape
    kf = KFold(n_splits=k, shuffle=True, random_state=42)  # shuffle the data before splitting into k groups
    mean_logLikelihoodLst = []

    # Use subsampling to reduce computation
    if n > subsample_size:
        indices = np.random.choice(n, subsample_size, replace=False)
        data_sub = data[indices]
    else:
        data_sub = data

    for H in H_Matrix_candidateLst:
        H = np.array(H)
        H_inv = np.linalg.inv(H)
        det_H = np.linalg.det(H)

        norm_const = 1.0 / ((2 * np.pi) ** (d / 2) * np.sqrt(det_H))

        fold_log_likelihoods = []

        for train_idx, val_idx in kf.split(data_sub):
            X_train = data_sub[train_idx]
            X_val = data_sub[val_idx]

            diffs = X_val[:, np.newaxis, :] - X_train[np.newaxis, :, :]
            dists = np.einsum('mnd,dd,mnd->mn', diffs, H_inv, diffs)
            K = norm_const * np.exp(-0.5 * dists)

            f_vals = np.mean(K, axis=1)
            f_vals = np.clip(f_vals, 1e-300, None)
            fold_log_likelihoods.append(np.mean(np.log(f_vals)))

        mean_logLikelihoodLst.append(np.mean(fold_log_likelihoods))

    mean_logLikelihoodLst = np.array(mean_logLikelihoodLst)
    optimal_idx = np.argmax(mean_logLikelihoodLst)
    optimalBandwidthMatrix = H_Matrix_candidateLst[optimal_idx]

    return optimalBandwidthMatrix, mean_logLikelihoodLst

# --- calcualte the PDF estimate and KDE estimate
def calc_pdf_and_kde_values(data, bandwidthMatrix, dim):

    # Get scalar bandwidth h = sqrt of diagonal element from bandwidthMatrix
    data_1d = data[:, dim]
    h = np.sqrt(bandwidthMatrix[dim, dim])

    # Compute mean and std for that dimension
    mean_x = np.mean(data_1d)
    std_x = np.std(data_1d, ddof=0)

    print(f"Using bandwidth h extracted from bandwidthMatrix diagonal: h = {h:.5f}")
    print(f"Mean = {round(mean_x, 5)}, Std = {round(std_x, 5)}")

    # Generate x values for plotting KDE and PDF
    x_vals = np.linspace(np.min(data_1d), np.max(data_1d), 500)

    # Empirical PDF assuming Gaussian distribution
    pdf_vals = norm.pdf(x_vals, loc=mean_x, scale=std_x)

    # calculate KDE values
    kde_vals = np.zeros_like(x_vals)
    diff = (x_vals[:, None] - data_1d[None, :]) / h
    kde_vals = np.mean(np.exp(-0.5 * diff**2), axis=1) / (np.sqrt(2 * np.pi) * h)

    return x_vals, pdf_vals, kde_vals

# --- Plot histogram
def plot_1D_histogram(data, x_vals, pdf_vals, kde_vals, dim, bins=50):

    data_1d = data[:, dim]

    # Plot histogram, empirical PDF, and KDE estimate
    plt.figure(figsize=(8, 5))
    plt.hist(data_1d, bins=bins, density=True, color='#68A5A1', edgecolor='black', alpha=0.6)
    plt.plot(x_vals, pdf_vals, lw=2, label="Empirical PDF")
    plt.plot(x_vals, kde_vals, '--', lw=2, label="KDE Estimate PDF")
    plt.xlabel(f"Dimension {dim}", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title("1D Histogram with Empirical PDF and KDE Estimate", fontsize=14)
    plt.legend(fontsize=12)
    plt.show()

# --- Plot histogram with scatter graph as well
def plot_1D_histogram_withScatter(data, x_vals, pdf_vals, kde_vals, dim, bins=50):
    data_1d = data[:, dim]

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=(4, 2), wspace=0.05)

    # Left: Scatter plot
    ax_main = fig.add_subplot(gs[0])
    ax_main.scatter(np.arange(len(data_1d)), data_1d, color='#68A5A1', s=3)
    ax_main.set_ylabel(f"Dimension {dim}", fontsize=20)
    ax_main.set_xlabel(f"Replica Index", fontsize=20)
    ax_main.set_title("1D Histogram with Empirical PDF and KDE Estimate", fontsize=14)
    ax_main.set_title("1D Scatter Plot and Histogram with Empirical PDF and KDE Estimate", fontsize=16)
    ax_main.tick_params(axis='both', labelsize=12)
    ax_main.grid(True)

    # Right: Rotated histogram (density on x, value on y)
    ax_hist = fig.add_subplot(gs[1], sharey=ax_main)
    hist, bin_edges = np.histogram(data_1d, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bar_width = bin_edges[1] - bin_edges[0]
    ax_hist.barh(
        bin_centers,
        hist,
        height=bar_width,
        color="#68A5A1",
        edgecolor='black',
    )

    # Overlay: empirical PDF (solid line) and KDE estimate (dashed line)
    ax_hist.plot(pdf_vals, x_vals, lw=2, label='Empirical PDF')
    ax_hist.plot(kde_vals, x_vals, lw=2, linestyle='--', label='KDE Estimate')

    ax_hist.set_xlabel('Probability Density', fontsize=12, labelpad=12)
    ax_hist.tick_params(axis='x', labelsize=12)
    ax_hist.tick_params(axis='y', left=False, labelleft=False)
    ax_hist.set_xlim(left=0)
    ax_hist.legend(fontsize=10, loc='lower right')

    plt.tight_layout()
    plt.show()


#############################################################################
### MOMENTS CALCULATION STUFF
#############################################################################

# ---------------------------------------------
# --- Grid Evaulation Moment Integration 
# ---------------------------------------------

# --- generate grid for integration - NOT USED
def generate_nd_grid(data, num_points_per_dim=20):
    """
    Generates an n-dimensional grid covering the range of the data.

    Parameters:
        data: (n_samples, d) array
        num_points_per_dim: number of grid points per dimension

    Returns:
        grid_points: (total_points, d) array of grid points
        mesh: list of n-dimensional meshgrid arrays (X, Y, Z, etc.)
    """
    d = data.shape[1]
    grid_axes = []

    for dim in range(d):
        min_val = np.min(data[:, dim])
        max_val = np.max(data[:, dim])
        axis = np.linspace(min_val, max_val, num_points_per_dim)
        grid_axes.append(axis)

    mesh = np.meshgrid(*grid_axes, indexing='ij')
    grid_points = np.vstack([m.ravel() for m in mesh]).T
    mesh_shape = mesh[0].shape

    return grid_points, mesh, mesh_shape

# ---------------------------------------------
# --- KDE PDF Evaluation Function 
# ---------------------------------------------

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

# --- calculate the pdf estimate for whole sample
def calc_pdf_batch(points, data, bandwidthMatrix):
    """
    Vectorised KDE PDF evaluation at multiple points.

    points: (m, d) array of evaluation points
    data: (n, d) input data points
    bandwidth: (d, d) bandwidth matrix

    Returns:
        pdf_vals: (m,) KDE density values at each point
    """
    d = data.shape[1] if data.ndim > 1 else 1
    n = data.shape[0]
    m = points.shape[0]

    H_inv = np.linalg.inv(bandwidthMatrix)
    det_H = np.linalg.det(bandwidthMatrix)

    # Compute differences between each sample point and each data point:
    # Result shape (m, n, d): broadcast points and data
    diffs = points[:, np.newaxis, :] - data[np.newaxis, :, :]  # shape (m, n, d)

    # Apply bandwidth inverse: (m, n, d) @ (d, d)T -> (m, n, d)
    u = np.einsum('mnd,dk->mnk', diffs, H_inv.T)  # shape (m, n, d)

    # Squared Mahalanobis distances: sum over d
    D2 = np.sum(u**2, axis=2)  # shape (m, n)

    # Kernel values
    kernel_vals = np.exp(-0.5 * D2)  # shape (m, n)

    norm_const = 1.0 / (np.sqrt((2 * np.pi)**d) * det_H)

    # Sum over data points axis (n), average, and multiply by norm_const
    result = (norm_const / n) * np.sum(kernel_vals, axis=1)  # shape (m,)

    return result

# ---------------------------------------------
# --- Monte Carlo moment integration (Importance Sampling) 
# ---------------------------------------------

def calc_moments_importanceSampling_ALL(data, bandwidthMatrix, n_samplesMC):
    """
    Efficient calculation of moments via Monte Carlo importance sampling of KDE.

    Returns:
    - zerothMoment: scalar
    - firstMomentVec: vector of means (shape d,)
    - varianceVec: vector of variances (shape d,)
    - covarianceMatrix: full covariance matrix (shape d x d)
    """
    d = data.shape[1]
    mu = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)

    try:
        samples = np.random.multivariate_normal(mu, cov, size=n_samplesMC)
    except np.linalg.LinAlgError:
        # Return NaNs if sampling fails
        return np.nan, np.full(d, np.nan), np.full(d, np.nan), np.full((d, d), np.nan)

    q_pdf = multivariate_normal(mean=mu, cov=cov, allow_singular=True)
    q_vals = q_pdf.pdf(samples)
    p_vals = calc_pdf_batch(samples, data, bandwidthMatrix)

    # Add guard to prevent division by zero or invalid weights
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = np.where(q_vals > 0, p_vals / q_vals, 0.0)

    weight_sum = np.sum(weights)
    if weight_sum <= 1e-14 or not np.isfinite(weight_sum):
        # Graceful fallback
        return np.nan, np.full(d, np.nan), np.full(d, np.nan), np.full((d, d), np.nan)

    zerothMoment = weight_sum / n_samplesMC
    weighted_samples = weights[:, None] * samples
    firstMomentVec = np.sum(weighted_samples, axis=0) / weight_sum

    weighted_outer = np.einsum('i,ij,ik->jk', weights, samples, samples)
    secondMomentMatrix = weighted_outer / weight_sum

    covarianceMatrix = secondMomentMatrix - np.outer(firstMomentVec, firstMomentVec)
    varianceVec = np.diag(covarianceMatrix)

    return zerothMoment, firstMomentVec, varianceVec, covarianceMatrix


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

        zerothMoment, firstMomentVec, varianceVec, covarianceMatrix = calc_moments_importanceSampling_ALL(
            resampled_data, bandwidthMatrix, n_samplesMC=n_samplesBootStrap)

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
    
    zerothMoment, firstMomentVec, varianceVec, covarianceMatrix = calc_moments_importanceSampling_ALL(data, bandwidthMatrix, n_samplesMC=n_samplesMC)
    # error_zeroth_is, bootstrapError_mean_is, bootstrapError_variance_is, bootstrap_covariance_is = calc_bootstrap_error_mc_importance(data, bandwidthMatrix, n_bootstrap, n_samplesMC)

    # --- Empirical moments - calc moments + covariance
    # empirical_mean = np.mean(data, axis=0)
    # empirical_variance = np.var(data, axis=0, ddof=1)
    empirical_covariance = np.cov(data, rowvar=False)

    # bootstrapError_mean, bootstrapError_variance, bootstrap_covariance = calc_bootstrapErrorEmpirical_all(data, n_bootstrap)

    # print("\n--- Moments ---")
    # print(f"Zeroth (KDE): {zerothMoment} with error {error_zeroth_is}\n")
    
    # print(f"Mean (KDE): {firstMomentVec} with error {bootstrapError_mean_is}")
    # print(f"Mean (Empirical): {empirical_mean} with error {bootstrapError_mean}\n")

    # print(f"Variance (KDE): {varianceVec} with error {bootstrapError_variance_is}")
    # print(f"Variance (Empirical): {empirical_variance} with error {bootstrapError_variance}\n")

    # print(f"Covariance (KDE): {covarianceMatrix} with error {bootstrap_covariance_is}\n")
    # print(f"Covariance (Empirical): {empirical_covariance} with error {bootstrap_covariance}\n")

    # print("\n--- Moments ---")
    # print(f"Zeroth (KDE): {zerothMoment}\n")
    
    # print(f"Mean (KDE): {firstMomentVec}")
    # print(f"Mean (Empirical): {empirical_mean}\n")

    # print(f"Variance (KDE): {varianceVec}")
    # print(f"Variance (Empirical): {empirical_variance}\n")

    # print(f"Covariance (KDE): {covarianceMatrix}\n")
    # print(f"Covariance (Empirical): {empirical_covariance}\n")

    return covarianceMatrix, empirical_covariance # firstMomentVec, varianceVec, covarianceMatrix # , bootstrapError_mean_is, bootstrapError_variance_is, bootstrap_covariance


#############################################################################
### Matrix Construction
#############################################################################

# --- helper function for parallel tasks
def _single_covariance_task(idx1, idx2, flav1, flav2, flav_to_index, res_flav, numberOfGridPoints):
    if idx1 == idx2 and flav1 == flav2:
        return None

    data = prepare_data_fixedIndex(res_flav, flav1, flav2, idx1, idx2)
    bandwidth_matrix = estimate_bandwidth_matrix_scv(data)

    if bandwidth_matrix is None or np.any(np.diag(bandwidth_matrix) < 1e-9):
        return None

    covMatrix, covMatrix_empirical = run_2D_momentCalculations(data, bandwidth_matrix)

    i_pos = flav_to_index[flav1] * numberOfGridPoints + idx1
    j_pos = flav_to_index[flav2] * numberOfGridPoints + idx2

    return (i_pos, j_pos, covMatrix, covMatrix_empirical)

# --- construct the covariance matrix using parallel computing - speedier
def construct_covariance_matrix_parallel(keys_flav, res_flav, numberOfGridPoints, n_jobs=-1):
    n_flav = len(keys_flav)
    dim = n_flav * numberOfGridPoints

    flav_to_index = {flav: i for i, flav in enumerate(keys_flav)}

    # Generate all tasks
    tasks = [(idx1, idx2, flav1, flav2)
             for idx1 in range(numberOfGridPoints)
             for idx2 in range(numberOfGridPoints)
             for flav1 in keys_flav
             for flav2 in keys_flav]

    # Use tqdm to wrap the tasks iterator for a progress bar in the main thread
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_single_covariance_task)(idx1, idx2, flav1, flav2, flav_to_index, res_flav, numberOfGridPoints)
        for idx1, idx2, flav1, flav2 in tqdm(tasks, desc="Covariance Tasks", total=len(tasks))
    )

    # Initialise matrices
    cov_full = np.zeros((dim, dim))
    count_matrix = np.zeros((dim, dim))
    cov_full_empirical = np.zeros((dim, dim))
    count_matrix_empirical = np.zeros((dim, dim))

    # Accumulate results
    for res in results:
        if res is None:
            continue
        
        i_pos, j_pos, covMatrix, covMatrix_empirical = res

        # update the bigger matrix with the KDE results
        cov_full[i_pos, j_pos] += covMatrix[0, 1]
        count_matrix[i_pos, j_pos] += 1

        cov_full[j_pos, i_pos] += covMatrix[1, 0]
        count_matrix[j_pos, i_pos] += 1

        cov_full[i_pos, i_pos] += covMatrix[0, 0]
        count_matrix[i_pos, i_pos] += 1

        cov_full[j_pos, j_pos] += covMatrix[1, 1]
        count_matrix[j_pos, j_pos] += 1

        # update the bigger matrix with the empirical results
        cov_full_empirical[i_pos, j_pos] += covMatrix_empirical[0, 1]
        count_matrix_empirical[i_pos, j_pos] += 1

        cov_full_empirical[j_pos, i_pos] += covMatrix_empirical[1, 0]
        count_matrix_empirical[j_pos, i_pos] += 1

        cov_full_empirical[i_pos, i_pos] += covMatrix_empirical[0, 0]
        count_matrix_empirical[i_pos, i_pos] += 1

        cov_full_empirical[j_pos, j_pos] += covMatrix_empirical[1, 1]
        count_matrix_empirical[j_pos, j_pos] += 1

    # Normalise
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised_cov_kde = np.divide(cov_full, count_matrix, where=count_matrix > 0)
        normalised_covEmpirical = np.divide(cov_full_empirical, count_matrix_empirical, where=count_matrix_empirical > 0)

    # Correlation matrices
    std_kde = np.sqrt(np.diag(normalised_cov_kde))
    std_empirical = np.sqrt(np.diag(normalised_covEmpirical))

    denom_kde = np.outer(std_kde, std_kde)
    denom_empirical = np.outer(std_empirical, std_empirical)

    with np.errstate(divide='ignore', invalid='ignore'):
        correlation_kde = np.divide(normalised_cov_kde, denom_kde, where=denom_kde > 0)
        correlation_empirical = np.divide(normalised_covEmpirical, denom_empirical, where=denom_empirical > 0)

    return normalised_cov_kde, correlation_kde, normalised_covEmpirical, correlation_empirical

# ---------------------------------------------
# --- Plot Matrix Code
# ---------------------------------------------

# plot the matrices
def plot_matrix_comparison(matrix1, matrix2, title1, title2, cbar_label, save_filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot matrix 1
    im0 = axes[0].imshow(matrix1, aspect='equal')
    axes[0].set_title(title1, fontsize=12)
    axes[0].invert_yaxis()
    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])
    axes[0].grid(False)

    # Plot matrix 2
    im1 = axes[1].imshow(matrix2, aspect='equal')
    axes[1].set_title(title2, fontsize=12)
    axes[1].invert_yaxis()
    axes[1].set_xticklabels([])
    axes[1].set_yticklabels([])
    axes[1].grid(False)

    # Add colorbar to the second plot
    cbar = fig.colorbar(im1)
    cbar.set_label(cbar_label, fontsize=12)

    plt.subplots_adjust(wspace=0.3)
    plt.savefig(save_filename, bbox_inches='tight')
    plt.show()

#############################################################################
### MAIN FUNCTION
#############################################################################

def main(plotting1D=False, empiricalReconstruction=True, integralReconstructionFull=True):
    res_flav, res_ev = read_in_data()
    keys_ev = ['Sigma', 'V', 'V3', 'V8', 'T3', 'T8', 'c+', 'g', 'V15']
    keys_flav = ['d', 'u', 's', 'c', 'dbar', 'ubar', 'sbar', 'cbar', 'g']
    # keys_flav = ['d', 'u', 's', 'c']
    numberOfGridPoints = 30  # number of indices to include for integralReconstructionFull and empiricalReconstruction options
    
# ---------------------------------------------
# --- Matrix reconstruction (Empricial)  
# ---------------------------------------------

    # --- Construct the co-variance matrix from the data empirically 
    if empiricalReconstruction == True:

        dataMatrix = prepare_data_rangeIndices(res_flav, keys_flav, numberOfGridPoints)

        # calculate the covariance 
        covarianceMatrix = np.cov(dataMatrix, rowvar=False)

        plt.figure(figsize=(10, 8))
        im = plt.imshow(covarianceMatrix, aspect='auto') 
        plt.title("PDF Covariance Matrix - Full np.cov", fontsize=12)

        cbar = plt.colorbar(im)
        cbar.set_label("Covariance", fontsize=12)

        plt.gca().invert_yaxis()
        plt.grid(False)
        plt.savefig('reconstructedMatrixFull.png')
        plt.show()

# ---------------------------------------------
# --- Matrix reconstruction (KDE)
# ---------------------------------------------

    if integralReconstructionFull:
        normalised_cov_kde, correlation_kde, normalised_covEmpirical, correlation_empirical = construct_covariance_matrix_parallel(keys_flav, res_flav, numberOfGridPoints=numberOfGridPoints, n_jobs=-1)

        plot_matrix_comparison(normalised_cov_kde,normalised_covEmpirical,"KDE Reconstructed Covariance","Empirical Covariance (reconstructed)", "Covariance", "reconstructedMatrix_Covariance.png")
        plot_matrix_comparison(correlation_kde, correlation_empirical, "KDE Reconstructed Correlation", "Empirical Correlation (reconstructed)", "Correlation", "reconstructedMatrix_Correlation.png")


# ---------------------------------------------
# --- Plot distributions in 1D
# ---------------------------------------------
    
    if plotting1D == True:
        idx1, idx2 = 28, 28
        key1 = 'u'
        key2 = 'g'
        data = prepare_data_fixedIndex(res_flav, key1, key2, idx1, idx2)
        bandwidthMatrix = estimate_bandwidth_matrix_scv(data)
        run_2D_momentCalculations(data, bandwidthMatrix)

        d = data.shape[1]
        for dim in range(0, d):      
            x_vals, pdf_vals, kde_vals = calc_pdf_and_kde_values(data, bandwidthMatrix, dim)
            plot_1D_histogram(data, x_vals, pdf_vals, kde_vals, dim)
            plot_1D_histogram_withScatter(data, x_vals, pdf_vals, kde_vals, dim)

if __name__ == "__main__":
    main()
