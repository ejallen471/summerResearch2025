import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize


# If running on separate laptop/computer, this will need commenting out 
plt.style.use('pythonStyle')
import pythonStyle as ed

np.random.seed(4)

#############################################################################
#############################################################################

# --- Load serialised data
def read_in_data():
    paths = [Path("./flavour_basis.pkl"), Path("./evolution_basis.pkl")]
    return (pickle.load(open(p, 'rb')) for p in paths)

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
        # Use all indices (0..49)
        indices = np.arange(50)
    
    if isinstance(indices, int):
        # Single index case, output 2D (num_replicas, num_keys)
        data_array = np.empty((num_replicas, num_keys), dtype=float)
        for i, replica in enumerate(res):
            for j, key in enumerate(keys):
                data_array[i, j] = replica[key][indices]
    else:
        # Multiple indices case, output 3D (num_replicas, num_keys, len(indices))
        indices = np.array(indices)
        data_array = np.empty((num_replicas, num_keys, len(indices)), dtype=float)
        for i, replica in enumerate(res):
            for j, key in enumerate(keys):
                data_array[i, j, :] = replica[key][indices]

    return data_array

#############################################################################
### BUILD KDE MODEL
#############################################################################

# --- Calculate bandwidth Matrix - diagonal matrix 
def calc_bandwidthMatrix(data, keys_test, x_idx, y_idx, plotting1D=False):
    
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
    bandwidthMatrix, mean_logLikelihoodLst = calc_kdeCrossValidation_nD(data, H_Matrix_candidateLst, k=5, subsample_size=10000)
    plot_KDE_CV(mean_logLikelihoodLst, hLst, xLabel=keys_test[x_idx], yLabel=keys_test[y_idx])
    print(bandwidthMatrix)

    # --- PLOT KDE ESTIMATES IN 1D --- #
    if plotting1D == True:
        for i in range(0, n):
            plot_histograms_with_pdf(data, bandwidthMatrix, dim=i)

    return bandwidthMatrix

# --- calculate the bandwidth matrix - 
def scv_objective(params, data, epsilon=1e-8, max_exp_arg=700):
    n, d = data.shape

    # Build lower-triangular matrix L from params
    lowerTriangularMatrix = np.zeros((d, d))
    tril_indices = np.tril_indices(d)
    lowerTriangularMatrix[tril_indices] = params

    # Bandwidth matrix H = L L^T, regularize
    choleskyMatrix = lowerTriangularMatrix @ lowerTriangularMatrix.T
    choleskyMatrix += epsilon * np.eye(d)

    # Inverse and determinant of H
    try:
        H_inv = np.linalg.inv(choleskyMatrix)
        det_H = np.linalg.det(choleskyMatrix)
    except np.linalg.LinAlgError:
        return 1e10

    if det_H <= 0:
        return 1e10

    norm_const = 1.0 / ((2 * np.pi) ** (d / 2) * np.sqrt(det_H))

    # Compute all pairwise differences: shape (n, n, d)
    diffs = data[:, np.newaxis, :] - data[np.newaxis, :, :]  # shape (n, n, d)

    # Compute all pairwise squared Mahalanobis distances using einsum:
    # Result shape (n, n)
    dists = np.einsum('ijk,kl,ijl->ij', diffs, H_inv, diffs)

    # Exclude diagonal (self-distances)
    np.fill_diagonal(dists, np.inf)  # large number so exp(-inf) = 0

    # Compute kernels matrix
    clipped_args = np.clip(-0.5 * dists, a_min=-np.inf, a_max=max_exp_arg)
    kernels = np.exp(clipped_args)

    # For each i, estimate is average of kernels over j != i
    estimate = norm_const * np.mean(kernels, axis=1)  # shape (n,)

    # Guard against invalid values
    if np.any(estimate <= 0) or not np.all(np.isfinite(estimate)):
        return 1e10

    # Compute SCV score
    score = -np.mean(np.log(estimate))

    return score

def estimate_bandwidth_matrix_scv(data, initial_scale=1.0):
    """
    Estimate bandwidth matrix H via SCV with numerical stability.

    Parameters:
        data: (n_samples, d) input data
        initial_scale: float, initial scale for diagonal of L

    Returns:
        H_opt: (d, d) estimated bandwidth matrix
    """
    n, d = data.shape

    # Initialize L as scaled diagonal matrix
    initialCholeskyMatrix = np.zeros((d, d))
    np.fill_diagonal(initialCholeskyMatrix, initial_scale * np.std(data, axis=0))

    initial_params = initialCholeskyMatrix[np.tril_indices(d)]

    result = minimize(
        scv_objective,
        initial_params,
        args=(data,),
        method='L-BFGS-B',
        options={'maxiter': 500}
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    L_opt = np.zeros((d, d))
    L_opt[np.tril_indices(d)] = result.x
    H_opt = L_opt @ L_opt.T

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

# --- Plot the results of CV for 2D
def plot_KDE_CV(mean_logLikelihoodLst, hLst):

    # Plot cross-validation scores
    plt.figure(figsize=(10, 6))
    plt.plot(hLst, mean_logLikelihoodLst, '-o')
    plt.xlabel('Bandwidth scaling factor (h_p multiplier)', fontsize=14)
    plt.ylabel('Mean Log Likelihood (CV score)', fontsize=14)
    plt.title('Cross-Validation Score', fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

# --- Plot KDE estimate and PDF estimate in 2D
def plot_kde_vs_pdf_2d(data, kde_vals, pdf_vals, grid_points):
    x_unique = np.unique(grid_points[:, 0])
    y_unique = np.unique(grid_points[:, 1])
    X, Y = np.meshgrid(x_unique, y_unique)

    # Downsample data for plotting if really large
    plot_data = data if data.shape[0] <= 10000 else data[::10]

    plt.scatter(plot_data[:, 0], plot_data[:, 1], c='dimgrey', s=10, alpha=0.3, label='Samples')
    plt.contour(X, Y, kde_vals, colors='navy', linewidths=1.5)
    plt.contour(X, Y, pdf_vals, colors='firebrick', linestyles='dashed', linewidths=1.5)

    legend_elements = [
        Line2D([0], [0], color='navy', lw=1.5, label='KDE Estimate pdf'),
        Line2D([0], [0], color='firebrick', lw=1.5, linestyle='dashed', label='Empirical pdf'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=6, label='Samples', alpha=0.5)
    ]

    plt.legend(handles=legend_elements, fontsize=12)
    plt.title(f'KDE vs Sample PDF', fontsize=16)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Plot 1D histograms - included for checking the underlying distributions
def plot_histograms_with_pdf(data, bandwidthMatrix, dim=0, bins=50):
    """
    Plot histogram, empirical PDF, and KDE estimate for one dimension of data,
    using bandwidth from the multidimensional bandwidth matrix.

    Args:
        data: (n, d) ndarray, dataset
        bandwidthMatrix: (d, d) ndarray, bandwidth matrix from multidim KDE
        dim: int, dimension index to plot (default 0)
        bins: int, number of histogram bins
    """
    
    # Extract 1D data for the selected dimension
    data_1d = data[:, dim]

    # Extract scalar bandwidth h = sqrt of diagonal element from bandwidthMatrix
    h = np.sqrt(bandwidthMatrix[dim, dim])

    # Compute mean and std for that dimension
    mean_x = np.mean(data_1d)
    std_x = np.std(data_1d, ddof=0)

    print(f"Using bandwidth h extracted from bandwidthMatrix diagonal: h = {h:.5f}")
    print(f"Mean = {mean_x:.5f}, Std = {std_x:.5f}")

    # Generate x values for plotting KDE and PDF
    x_vals = np.linspace(np.min(data_1d), np.max(data_1d), 500)

    # Empirical PDF assuming Gaussian distribution
    pdf_x = norm.pdf(x_vals, loc=mean_x, scale=std_x)

    # calculate KDE estimates
    kde_vals = np.zeros_like(x_vals)
    for i, x in enumerate(x_vals):
        diff = (x - data_1d) / h
        kde_vals[i] = np.mean(np.exp(-0.5 * diff**2)) / (np.sqrt(2 * np.pi) * h)

    # Plot histogram, empirical PDF, and KDE estimate
    plt.figure(figsize=(8, 5))
    plt.hist(data_1d, bins=bins, density=True, color='skyblue', edgecolor='black', alpha=0.6)
    plt.plot(x_vals, pdf_x, lw=2, label="Empirical PDF")
    plt.plot(x_vals, kde_vals, '--', lw=2, label="KDE Estimate PDF")
    plt.xlabel(f"Dimension {dim}", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title("1D Histogram with Empirical PDF and KDE Estimate", fontsize=14)
    plt.legend(fontsize=12)
    plt.show()

#############################################################################
### CALCULATING MOMENTS
#############################################################################

# --- generate grid for integration - not currently used
def setup_nd_grid(data, k=3, grid_res=150):
    """
    Create a multi-dimensional grid that spans k standard deviations around the mean.

    Parameters:
        data: (n, d) array of input data
        k: scalar, how many standard deviations to extend in each direction
        grid_res: number of grid points per dimension

    Returns:
        grid: list of 1D arrays (length d), one per dimension
        dV: differential volume element
    """
    d = data.shape[1]
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=1)

    grid = []
    for i in range(d):
        low = mean[i] - k * std[i]
        high = mean[i] + k * std[i]
        grid_i = np.linspace(low, high, grid_res)
        grid.append(grid_i)

    # Compute grid spacing (assumes uniform spacing in each dim)
    dxs = [(g[1] - g[0]) for g in grid]
    dV = np.prod(dxs)  # differential volume

    return grid, dV



# ---------------------------------------------
# --- KDE PDF Evaluation Function ---
# ---------------------------------------------

# --- calculate the pdf estimate for a single point
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
# --- Monte Carlo moment integration (Importance Sampling) ---
# ---------------------------------------------

def calc_moments_grid_KDE(data, bandwidth, grid, differentialElement):
    """
    Calculate zeroth, mean, variance, and covariance via KDE grid integration.
    """
    d = data.shape[1]
    mesh = np.meshgrid(*grid, indexing='ij')
    shape = mesh[0].shape
    f_hats = np.empty(shape)

    grid_points = np.stack([g.ravel() for g in mesh], axis=1)

    # KDE density estimation at grid points
    for i, pt in enumerate(grid_points):
        f_hats.ravel()[i] = calc_pdf_pointwise(pt, data, bandwidth)

    # Zeroth moment
    zerothMoment = np.sum(f_hats) * differentialElement

    # First moment (mean)
    meanVec = np.zeros(d)
    for dim in range(d):
        coords = mesh[dim]
        meanVec[dim] = np.sum(coords * f_hats) * differentialElement 

    # Second moment
    secondMomentMatrix = np.zeros((d, d))
    for i in range(grid_points.shape[0]):
        pt = grid_points[i]
        f_val = f_hats.ravel()[i]
        secondMomentMatrix += f_val * np.outer(pt, pt)
    secondMomentMatrix *= differentialElement

    # Covariance matrix
    covarianceMatrix = secondMomentMatrix - np.outer(meanVec, meanVec)
    varianceVec = np.diag(covarianceMatrix)

    return zerothMoment, meanVec, varianceVec, covarianceMatrix


# --- calculate the moments via importance sampling method
def calc_moments_importanceSampling(data, bandwidthMatrix, n_samplesMC):
    """
    Calculate the moment vector via Monte Carlo importance sampling of KDE.
    Supports n dimensions.

    integral of the form I = integral f_x * p_x dx can be approximated as 1/N * sum (p_x / q_x) * f_x
    where 
        p_x = actual distribution (usually hard to sample from)
        q_x = approximate / guess distribution (in theory a version of p_x that you can sample from)
        f_x = function of interest - in our case for zeorth moment = 1, first moment = x, second moment = x^2 etc
    """

    # Proposal distribution q(x) ~ N(mu, cov) - calculated from data, the sample mean and co-variance
    mu = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)

    print(mu, cov)
    print('FAFSDGFSDFGS')

    mu = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    epsilon = 1e-8
    cov = cov + epsilon * np.eye(cov.shape[0])

    samples = np.random.multivariate_normal(mu, cov, size=n_samplesMC)
    q_dist = multivariate_normal(mean=mu, cov=cov) # create multivariate normal object for q(x)
    q_vals = q_dist.pdf(samples)

    # p_vals = [calc_pdf_pointwise(point, data, bandwidthMatrix) for point in samples]
    p_vals = calc_pdf_batch(samples, data, bandwidthMatrix)
    weightsLst = p_vals / q_vals # shape (n_samples,)

    # zeroth Moment - normalisation check
    zerothMoment = np.mean(weightsLst) # f(x) = 1 thus I ~ 1/N sum of p(x) / q(x) - the mean of the weightsLst

    # Mean - first moment
    f_x_first = samples  # shape (n_samples, d)
    weighted_f_x = weightsLst[:, None] * f_x_first  # shape (n_samples, d) - broadcast along dimensions - basically duplicate the weightsLst element wise to mutliply by f_x_first
    weighted_sum = np.sum(weighted_f_x, axis=0)
    weight_sum = np.sum(weightsLst)
    firstMomentVec = weighted_sum / weight_sum # divide for normalisation 

    # Second Moment
    f_x_second = samples**2  # shape (n_samples, d)
    weighted_f_x = weightsLst[:, None] * f_x_second  # shape (n_samples, d)
    weighted_sum = np.sum(weighted_f_x, axis=0)
    weight_sum = np.sum(weightsLst)
    secondMomentVec = weighted_sum / weight_sum

    varianceVec = secondMomentVec - firstMomentVec**2

    return zerothMoment, firstMomentVec, varianceVec

# --- calculate the moments (0th, 1st and 2nd) and covariance via importance sampling method
def calc_moments_importanceSampling_ALL(data, bandwidthMatrix, n_samplesMC):
    """
    Calculate moments via Monte Carlo importance sampling of KDE.
    Supports n dimensions.

    Returns:
    - zerothMoment: scalar
    - firstMomentVec: vector of means (shape d,)
    - varianceVec: vector of variances (shape d,)
    - covarianceMatrix: full covariance matrix (shape d x d)
    """

    # Proposal distribution q(x) ~ N(mu, cov) estimated from data
    mu = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)

    # Draw samples from q(x)
    samples = np.random.multivariate_normal(mu, cov, size=n_samplesMC)
    q_dist = multivariate_normal(mean=mu, cov=cov)
    q_vals = q_dist.pdf(samples)

    # Evaluate p(x) via KDE at sample points
    p_vals = calc_pdf_batch(samples, data, bandwidthMatrix)  # shape (n_samplesMC,)

    # Importance weights
    weightsLst = p_vals / q_vals  # shape (n_samplesMC,)

    # Zeroth moment (normalization)
    zerothMoment = np.mean(weightsLst)

    # First moment (mean vector)
    weighted_sum = np.sum(weightsLst[:, None] * samples, axis=0)  # shape (d,)
    weight_sum = np.sum(weightsLst)
    firstMomentVec = weighted_sum / weight_sum

    # Second moment (needed for covariance)
    weighted_outer_sum = np.zeros((samples.shape[1], samples.shape[1]))
    for i in range(n_samplesMC):
        x = samples[i]
        w = weightsLst[i]
        weighted_outer_sum += w * np.outer(x, x)
    secondMomentMatrix = weighted_outer_sum / weight_sum  # shape (d, d)

    # Covariance matrix: E[xx^T] - E[x] E[x]^T - effectively E[XY] - E[X][Y] in vector form
    covarianceMatrix = secondMomentMatrix - np.outer(firstMomentVec, firstMomentVec)

    # Variance vector (diagonal of covariance)
    varianceVec = np.diag(covarianceMatrix)

    return zerothMoment, firstMomentVec, varianceVec, covarianceMatrix


import numpy as np
from scipy.stats import multivariate_normal

def calc_moments_mcmc_ALL(data, bandwidthMatrix, n_samplesMC, burn_in=1000, proposal_scale=1):
    """
    Calculate moments via MCMC sampling of KDE.
    Supports n dimensions.

    Args:
    - data: array of shape (n_points, d), original data
    - bandwidthMatrix: bandwidth matrix for KDE (d x d)
    - n_samplesMC: number of MCMC samples to collect after burn-in
    - burn_in: number of initial samples to discard (default 1000)
    - proposal_scale: scalar to scale proposal covariance (default 0.5)

    Returns:
    - zerothMoment: scalar (should be close to 1 for normalized KDE)
    - firstMomentVec: vector of means (shape d,)
    - varianceVec: vector of variances (shape d,)
    - covarianceMatrix: full covariance matrix (shape d x d)
    """

    d = data.shape[1]
    total_samples = n_samplesMC + burn_in

    # Estimate KDE mode or mean as initial state
    current_state = np.mean(data, axis=0)

    # Proposal distribution covariance (scaled from data covariance)
    proposal_cov = proposal_scale * np.cov(data, rowvar=False)

    # Precompute KDE normalization constant if needed (optional)
    # Here we assume calc_pdf_batch returns unnormalized KDE values or properly normalized ones.

    samples = np.zeros((n_samplesMC, d))
    current_pdf = calc_pdf_batch(np.array([current_state]), data, bandwidthMatrix)[0]

    for i in range(total_samples):
        # Propose new point
        proposal = np.random.multivariate_normal(current_state, proposal_cov)
        proposal_pdf = calc_pdf_batch(np.array([proposal]), data, bandwidthMatrix)[0]

        # Acceptance probability (MH ratio)
        alpha = proposal_pdf / current_pdf

        if alpha >= 1 or np.random.rand() < alpha:
            # Accept proposal
            current_state = proposal
            current_pdf = proposal_pdf

        if i >= burn_in:
            samples[i - burn_in] = current_state

    # Calculate moments from MCMC samples
    zerothMoment = 1.0  # KDE is a PDF, so integral approx = 1 by construction

    firstMomentVec = np.mean(samples, axis=0)
    covarianceMatrix = np.cov(samples, rowvar=False)
    varianceVec = np.diag(covarianceMatrix)

    return zerothMoment, firstMomentVec, varianceVec, covarianceMatrix


# ---------------------------------------------
# --- Errors ---
# ---------------------------------------------

# --- calculate errors via bootstrap for importance method 
def calc_bootstrap_error_mc_importance(data, bandwidthMatrix, n_bootstrap, n_samplesBootStrap):
    """
    Bootstrap standard error for importance sampling KDE moments.
    Extends to calculate covariance matrix error too.

    Returns:
        std_zeroth: scalar standard error of zeroth moment
        std_first: (d,) vector standard error of first moments
        std_variance: (d,) vector standard error of variance
        std_covariance: (d,d) matrix of standard errors for covariance matrix elements
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

# --- function passed into calc_bootstrapErrorEmpirical
def mean_stat(data_sample):
    return np.mean(data_sample, axis=0)

# --- function passed into calc_bootstrapErrorEmpirical
def variance_stat(data_sample):
    return np.var(data_sample, axis=0, ddof=1)

# --- function passed into calc_bootstrapErrorEmpirical
def covariance_stat(data_sample):
    return np.cov(data_sample, rowvar=False)

# --- calculate the errors for the empirical method, via bootstrap
def calc_bootstrapErrorEmpirical(data, stat_func, n_bootstrap):
    n = data.shape[0]
    indices = np.random.choice(n, size=(n_bootstrap, n), replace=True)
    samples = data[indices]
    stats = np.array([stat_func(sample) for sample in samples])
    error = np.std(stats, ddof=1)
    
    return error


def bootstrap_moment_errors_grid_KDE(data, bandwidth, grid, differentialElement, n_bootstrap=100):
    """
    Returns bootstrap errors (standard deviations) for:
    - Zeroth moment (scalar)
    - Mean (vector)
    - Variance (vector)
    - Covariance matrix (matrix)
    """
    d = data.shape[1]

    zeroth_vals = np.empty(n_bootstrap)
    mean_vals = np.empty((n_bootstrap, d))
    var_vals = np.empty((n_bootstrap, d))
    cov_vals = np.empty((n_bootstrap, d, d))

    for i in range(n_bootstrap):
        idxs = np.random.choice(len(data), size=len(data), replace=True)
        sample = data[idxs]
        Z, mu, var, cov = calc_moments_grid_KDE(sample, bandwidth, grid, differentialElement)

        zeroth_vals[i] = Z
        mean_vals[i] = mu
        var_vals[i] = var
        cov_vals[i] = cov

    zeroth_error = np.std(zeroth_vals, ddof=1)
    mean_error = np.std(mean_vals, axis=0, ddof=1)
    var_error = np.std(var_vals, axis=0, ddof=1)
    cov_error = np.std(cov_vals, axis=0, ddof=1)

    return zeroth_error, mean_error, var_error, cov_error



# ---------------------------------------------
# --- Run Moments Code ---
# ---------------------------------------------

def run_2D_momentCalculations(data, bandwidthMatrix, n_bootstrap=100, n_samplesMC=10000):
    
    # Compute moments by KDE MC integration
    # M0, M1, var = calc_moments_importanceSampling(data, bandwidthMatrix, n_samplesMC)
    # error_zeorth_is, bootstrapError_mean_is, bootstrapError_variance_is = calc_bootstrap_error_mc_importance(data, bandwidthMatrix, n_bootstrap, n_bootstrap)

    M0, M1, var, cov = calc_moments_importanceSampling_ALL(data, bandwidthMatrix, n_samplesMC)
    err_zeroth, err_mean, err_var, err_cov = calc_bootstrap_error_mc_importance(data, bandwidthMatrix, n_bootstrap, n_samplesMC)


    M0, M1, var, cov = calc_moments_mcmc_ALL(data, bandwidthMatrix, n_samplesMC)

    # Set up grid
    # grid, dV = setup_nd_grid(data, k=3, grid_res=550)
    # M0, M1, var, cov = calc_moments_grid_KDE(data, bandwidthMatrix, grid, dV)

    # Bootstrap errors only
    # err_zeroth, err_mean, err_var, err_cov = 1,1,1,1 # bootstrap_moment_errors_grid_KDE(data, bandwidthMatrix, grid, dV, n_bootstrap=100)

    print("Zeroth error:", err_zeroth)
    print("Mean error:", err_mean)
    print("Variance error:", err_var)
    print("Covariance error:\n", err_cov)

    # Empirical moments + bootstrap errors
    empirical_mean = np.mean(data, axis=0)
    empirical_variance = np.var(data, axis=0, ddof=1)
    empirical_covariance = np.cov(data, rowvar=False)

    bootstrapError_mean = calc_bootstrapErrorEmpirical(data, mean_stat, n_bootstrap)
    bootstrapError_variance = calc_bootstrapErrorEmpirical(data, variance_stat, n_bootstrap)
    bootstrap_covariance = calc_bootstrapErrorEmpirical(data, covariance_stat, n_bootstrap)

    print("\n--- Moments ---")
    print(f"Zeroth (KDE): {M0} with error {err_zeroth}\n")
    
    print(f"Mean (KDE): {M1} with error {err_mean}")
    print(f"Mean (Empirical): {empirical_mean} with error {bootstrapError_mean}\n")

    print(f"Variance (KDE): {var} with error {err_var}")
    print(f"Variance (Empirical): {empirical_variance} with error {bootstrapError_variance}\n")

    print(f"Covariance Matrix (KDE): {cov}\n with error {err_cov}\n")
    print(f"Covariance Matrix (Empirical): {empirical_covariance}\n with error {bootstrap_covariance}\n")

#############################################################################
#############################################################################

# --- Create grid and call plotting functions
def run_2D_KDE_estimates_plot(data, bandwidthMatrix, x_idx, y_idx, kdeGridRes=150):
    
    print(f'Plotting Dimensions {x_idx} and {y_idx}')

    # Extract data for only the two selected dimensions
    data_2d = data[:, [x_idx, y_idx]]

    # Create 2D grid only for selected dims
    grid_axes_2d = [np.linspace(np.min(data_2d[:, dim]) - 1, np.max(data_2d[:, dim]) + 1, kdeGridRes) for dim in range(2)]

    # Create 2D meshgrid and flatten
    X, Y = np.meshgrid(grid_axes_2d[0], grid_axes_2d[1], indexing='xy')
    grid_points_2d = np.vstack([X.ravel(), Y.ravel()]).T.astype(np.float32)  # shape (num_points, 2)

    # Compute 2D bandwidth matrix for selected dims from the full bandwidth
    optimalBandwidthMatrix_2d = bandwidthMatrix[np.ix_([x_idx, y_idx], [x_idx, y_idx])]

    # Compute KDE only on selected dims
    kde_vals_2d = calc_kdeGaussianEstimate_nD(grid_points_2d, data_2d, optimalBandwidthMatrix_2d).reshape(kdeGridRes, kdeGridRes)

    # Calculate sample PDF for selected dims
    sampleMean_2d = np.mean(data_2d, axis=0)
    sampleCovariance_2d = np.cov(data_2d, rowvar=False)
    sampleGaussian_2d = multivariate_normal(mean=sampleMean_2d, cov=sampleCovariance_2d)
    pdf_vals_2d = sampleGaussian_2d.pdf(grid_points_2d).reshape(kdeGridRes, kdeGridRes)

    # Plot with the 2D inputs    
    plot_kde_vs_pdf_2d(data_2d, kde_vals_2d, pdf_vals_2d, grid_points_2d)

# --- Main function 
def main():
    res_flav, res_ev = read_in_data()

    keys_ev = ['Sigma', 'V', 'V3', 'V8', 'T3', 'T8', 'c+', 'g', 'V15']
    keys_flav = ['d', 'u', 's', 'c', 'dbar', 'ubar', 'sbar', 'cbar', 'g']
    keys_test = ['u', 'g']
    index = 47

    # for plotting the 2D KDE - this index corresponds to the keys list
    x_idx, y_idx = 0,1
    
    data1 = prepare_data(res_flav, keys_test, index)
    # bandwidthMatrix = calc_bandwidthMatrix(data1, keys_test, x_idx, y_idx)
    bandwidthMatrix = estimate_bandwidth_matrix_scv(data1)
    print(bandwidthMatrix)

    if bandwidthMatrix[0][0] >= 1e-8:
        run_2D_KDE_estimates_plot(data1, bandwidthMatrix, x_idx, y_idx)
        run_2D_momentCalculations(data1, bandwidthMatrix)
    else:
        print('Bandwidth Matrix too small to sucessfully calculate moments')

if __name__ == "__main__":
    main()



