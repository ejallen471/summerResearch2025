import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from scipy.stats import norm
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

# --- Get 2D data at fixed index
def prepare_2d_data(res, key_x, key_y, index=25):
    return np.array([[r[key_x][index], r[key_y][index]] for r in res])

# --- Get the data and bring into lists etc
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
### BUILDING KDE STUFF
#############################################################################

# --- calculate the bandwidth matrix - diagonal, ignore covariance, much quicker
def calc_bandwidthMatrix(data, n=100000):
    
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
    print(bandwidthMatrix)

    return bandwidthMatrix

# --- Helper function for estimate_bandwidth_matrix_scv
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

# --- calculate the bandwidth matrix with the covariance terms - much slower
def estimate_bandwidth_matrix_scv(data, initial_scale=1.0):
    """
    Estimate bandwidth matrix H via Smooth Cross Validation (SCV) 

    Parameters:
        data: (n_samples, d) input data
        initial_scale: float, initial scale for diagonal of L

    Returns:
        H_opt: (d, d) estimated bandwidth matrix
    """
    n, d = data.shape

    # Initialise L as scaled diagonal matrix
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
    for i, x in enumerate(x_vals):
        diff = (x - data_1d) / h
        kde_vals[i] = np.mean(np.exp(-0.5 * diff**2)) / (np.sqrt(2 * np.pi) * h)

    return x_vals, pdf_vals, kde_vals

# --- Calculate KL divergence in 1D - between two choosen flavours
def calc_KLDivergence(data, kde_vals_x, kde_vals_y, pdf_x, pdf_y):

    # Assume uniform spacing
    x_vals = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), len(kde_vals_x))
    y_vals = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), len(kde_vals_y))
    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]

    # KL divergence: KDE vs True
    kl_x = np.sum(kde_vals_x * np.log(kde_vals_x / pdf_x)) * dx
    kl_y = np.sum(kde_vals_y * np.log(kde_vals_y / pdf_y)) * dy

    # print stuff
    print(f"\nKL divergence (X marginal): {kl_x:.6f}")
    print(f"KL divergence (Y marginal): {kl_y:.6f}\n")

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
    ax_main.set_ylabel(f"f(x)", fontsize=12)
    ax_main.set_xlabel(f"x", fontsize=12)
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
# --- KDE PDF Evaluation Function 1D ---
# ---------------------------------------------

def calc_pdf_pointwise_1D(x, data, h):

    n = data.shape[0]
    diffs = x - data[:, 0]  # shape (n,)

    # Gaussian kernel evaluations
    normConst = 1 / np.sqrt(2 * np.pi * h)
    kernel_vals = normConst * np.exp(-0.5 * ((diffs)** 2) / h)

    # KDE estimate is average kernel values
    pdf_val = np.sum(kernel_vals) / n

    return pdf_val

# ---------------------------------------------
# --- Grid Evaulation Moment Integration ---
# ---------------------------------------------

def calc_moment_integral_GridEvaluation_1d(data, bandwidth, order, grid, differentialElement):
    """
    Calculate moment integrals of KDE in 1D by grid evaluation.

    data: (n,) 1D data array
    bandwidth: scalar bandwidth variance (h)
    order: integer moment order (0 for zeroth moment, 1 for first, etc.)
    grid: 1D array of points at which to evaluate KDE
    differentialElement: scalar representing the grid spacing (dx)
    """

    # Evaluate KDE at grid points
    f_hats = np.empty_like(grid)
    for i, pt in enumerate(grid):
        f_hats[i] = calc_pdf_pointwise_1D(pt, data[:, None], bandwidth)
    
    if order == 0:
        # Zeroth moment: integral over KDE (scalar)
        return np.sum(f_hats) * differentialElement
    else:
        # Higher moments: integral of x^order * KDE(x)
        moment = np.sum((grid ** order) * f_hats) * differentialElement
        return moment

# ---------------------------------------------
# --- Empirical Moment Stat Functions ---
# ---------------------------------------------

# --- pass through function for calc_bootstrap_error
def mean_stat(data_sample):
    return np.mean(data_sample, axis=0)

# --- pass through function for calc_bootstrap_error
def variance_stat(data_sample):
    return np.var(data_sample, axis=0, ddof=1)

# ---------------------------------------------
# --- Bootstrap Error ---
# ---------------------------------------------

# --- calculate the error from empirical moments
def calc_bootstrap_error(data, stat_func, n_bootstrap):
    n = data.shape[0]
    indices = np.random.choice(n, size=(n_bootstrap, n), replace=True)
    samples = data[indices]
    stats = np.array([stat_func(sample) for sample in samples])
    
    # stats shape: (n_bootstrap,) if stat_func returns scalar
    # std with ddof=1 to get unbiased estimator
    error = np.std(stats, ddof=1)
    
    return error

# --- calculate the error from integral based moments
def calc_bootstrap_error_integral(data, bandwidth, grid, differentialElement, n_bootstrap=100):

    n = len(data)
    zeorth_estimates = np.empty(n_bootstrap)
    mean_estimates = np.empty(n_bootstrap)
    var_estimates = np.empty(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Bootstrap resample with replacement
        resample = np.random.choice(data, size=n, replace=True)
        
        M0 = calc_moment_integral_GridEvaluation_1d(resample, bandwidth, 1, grid, differentialElement)
        M1 = calc_moment_integral_GridEvaluation_1d(resample, bandwidth, 1, grid, differentialElement)
        M2 = calc_moment_integral_GridEvaluation_1d(resample, bandwidth, 2, grid, differentialElement)

        var = M2 - M1**2
        var_estimates[i] = var
        mean_estimates[i] = M1
        zeorth_estimates[i] = M0
    
    # Bootstrap standard error is std deviation of bootstrap moment estimates
    zeorth_std = np.std(zeorth_estimates, ddof=1)
    mean_std = np.std(mean_estimates, ddof=1)
    var_std = np.std(var_estimates, ddof=1)

    return zeorth_std, mean_std, var_std

# ---------------------------------------------
# --- Main Moments Function
# ---------------------------------------------

# --- calling functions and bringing everything together
def run_1D_momentCalculations(data, bandwidthMatrix, n_bootstrap=50, n_samplesBootStrap=10000):

    n_dims = data.shape[1]
    for dim in range(n_dims):
        print(f"\n--- Dimension {dim + 1} ---")

        data_min = np.min(data[:,dim])
        data_max = np.max(data[:,dim])
        data_std = np.std(data[:,dim])

        grid_min = data_min - 3 * data_std
        grid_max = data_max + 3 * data_std

        grid = np.linspace(grid_min, grid_max, 1000)
        dx = grid[1] - grid[0]
        bandwidth = bandwidthMatrix[dim][dim]

        # Zeroth moment
        M0 = calc_moment_integral_GridEvaluation_1d(data[:,dim], bandwidth, 0, grid, dx)
        M1 = calc_moment_integral_GridEvaluation_1d(data[:,dim], bandwidth, 1, grid, dx)
        M2 = calc_moment_integral_GridEvaluation_1d(data[:,dim], bandwidth, 2, grid, dx)
       
        # Calculate variance
        variance = M2 - M1**2
        
        # Empirical moments
        empirical_mean = np.mean(data[:,dim])
        empirical_variance = np.var(data[:,dim], ddof=1)

        # Empirical bootstrap errors 
        bootstrapError_mean = calc_bootstrap_error(data, mean_stat, n_bootstrap)
        bootstrapError_variance = calc_bootstrap_error(data, variance_stat, n_bootstrap)

        # Integral bootstrap errors 
        bootstrapError_zeroth_is, bootstrapError_mean_is, bootstrapError_variance_is = calc_bootstrap_error_integral(data[:,dim], bandwidth, grid, differentialElement=dx)

        print("\n--- Moments ---")
        print(f"Zeroth (KDE): {M0} with error {bootstrapError_zeroth_is}\n")

        print(f"Mean (KDE): {M1} with error {bootstrapError_mean_is}")
        print(f"Mean (Empirical): {empirical_mean} with error {bootstrapError_mean}\n")

        print(f"Variance (KDE): {variance} with error {bootstrapError_variance_is}")
        print(f"Variance (Empirical): {empirical_variance} with error {bootstrapError_variance}\n")
    
#############################################################################
### MAIN FUNCTION
#############################################################################

def main(plotting1D=True, KL_divergence=True):
    res_flav, res_ev = read_in_data()
    keys_ev = ['Sigma', 'V', 'V3', 'V8', 'T3', 'T8', 'c+', 'g', 'V15']
    keys_flav = ['d', 'u', 's', 'c', 'dbar', 'ubar', 'sbar', 'cbar', 'g']
    keys_flav = ['d', 'g']

    index = 28

    data = prepare_data(res_flav, keys_flav, index)
    bandwidthMatrix = calc_bandwidthMatrix(data)
    # bandwidthMatrix = estimate_bandwidth_matrix_scv(data)


    # --- Plot in 1D
    d = data.shape[1]
    if plotting1D == True:
        for dim in range(0, d):
            x_vals, pdf_vals, kde_vals = calc_pdf_and_kde_values(data, bandwidthMatrix, dim)
            plot_1D_histogram(data, x_vals, pdf_vals, kde_vals, dim)
            # plot_1D_histogram_withScatter(data, x_vals, pdf_vals, kde_vals, dim)

    # --- Calculate KL divergence 
    KL_idx = (0,1) # which distributions is the KL divergence calculated between 
    if KL_divergence == True:
        _, pdf_vals_x, kde_vals_x = calc_pdf_and_kde_values(data, bandwidthMatrix, dim=KL_idx[0])
        _, pdf_vals_y, kde_vals_y = calc_pdf_and_kde_values(data, bandwidthMatrix, dim=KL_idx[0])
        calc_KLDivergence(data, kde_vals_x, kde_vals_y, pdf_vals_x, pdf_vals_y)

    run_1D_momentCalculations(data, bandwidthMatrix)


if __name__ == "__main__":
    main()



