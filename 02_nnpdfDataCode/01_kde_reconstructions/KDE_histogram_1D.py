#############################################################################

# Program will plot a histogram with the empirical PDF and KDE PDF superimposed 
# for chosen flavour grid-index pair

#############################################################################

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

# --- calculate the bandwidth matrix - diagonal, ignore covariance, much quicker - NOT USED
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

    # Empirical PDF - assuming Gaussian distribution with sample mean and std
    pdf_vals = norm.pdf(x_vals, loc=mean_x, scale=std_x)

    # calculate KDE values
    kde_vals = np.zeros_like(x_vals)
    for i, x in enumerate(x_vals):
        diff = (x - data_1d) / h
        kde_vals[i] = np.mean(np.exp(-0.5 * diff**2)) / (np.sqrt(2 * np.pi) * h)

    return x_vals, pdf_vals, kde_vals

def calc_KLDivergence(data, kde_vals_x, kde_vals_y, pdf_x, pdf_y):
    # Assume uniform spacing in x and y
    x_vals = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), len(kde_vals_x))
    y_vals = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), len(kde_vals_y))
    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]

    # Clip to avoid log(0) or division by zero
    kde_vals_x = np.clip(kde_vals_x, 1e-300, None)
    pdf_x = np.clip(pdf_x, 1e-300, None)
    kde_vals_y = np.clip(kde_vals_y, 1e-300, None)
    pdf_y = np.clip(pdf_y, 1e-300, None)

    # Normalise
    kde_vals_x /= np.sum(kde_vals_x) * dx
    pdf_x /= np.sum(pdf_x) * dx
    kde_vals_y /= np.sum(kde_vals_y) * dy
    pdf_y /= np.sum(pdf_y) * dy

    # Compute KL divergence: D_KL(true || kde)
    kl_x = np.sum(pdf_x * np.log(pdf_x / kde_vals_x)) * dx
    kl_y = np.sum(pdf_y * np.log(pdf_y / kde_vals_y)) * dy

    print(f"\nKL divergence (X marginal): {kl_x:.6f}")
    print(f"KL divergence (Y marginal): {kl_y:.6f}\n")


# --- Plot histogram
def plot_1D_histogram(data, x_vals, pdf_vals, kde_vals, dim, bins=50):

    data_1d = data[:, dim]

    # Plot histogram, empirical PDF, and KDE estimate
    plt.figure(figsize=(8, 5))
    plt.hist(data_1d, bins=bins, density=True, color='#68A5A1', edgecolor='black', alpha=0.6)
    plt.plot(x_vals, kde_vals, '--', lw=2, label="KDE Estimate PDF")
    plt.plot(x_vals, pdf_vals, lw=2, label="Empirical PDF")
    # plt.xlabel(f"Gluon PDF")
    plt.ylabel("Probability Density")
    # plt.title("1D Histogram with Empirical PDF and KDE Estimate", fontsize=14)
    plt.legend()
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
### MAIN FUNCTION
#############################################################################

def main(plotting1D=True, KL_divergence=True):
    res_flav, res_ev = read_in_data()
    
    # keys_ev = ['Sigma', 'V', 'V3', 'V8', 'T3', 'T8', 'c+', 'g', 'V15']
    # keys_flav = ['d', 'u', 's', 'c', 'dbar', 'ubar', 'sbar', 'cbar', 'g']
    
    # choose flavours to loop through 
    keys_flav = ['d', 'g']

    # choose singe index between 1 and 50
    index = 28

    data = prepare_data(res_flav, keys_flav, index)
    bandwidthMatrix = calc_bandwidthMatrix(data)
    # bandwidthMatrix = estimate_bandwidth_matrix_scv(data)


    # --- Plot in 1D
    d = data.shape[1]
    print(d)
    if plotting1D == True:
        for dim in range(0, d):
            x_vals, pdf_vals, kde_vals = calc_pdf_and_kde_values(data, bandwidthMatrix, dim)
            plot_1D_histogram(data, x_vals, pdf_vals, kde_vals, dim)
            # plot_1D_histogram_withScatter(data, x_vals, pdf_vals, kde_vals, dim)

    # --- Calculate KL divergence 
    KL_idx = (0,1) # which distributions is the KL divergence calculated between 
    if KL_divergence == True:
        _, pdf_vals_x, kde_vals_x = calc_pdf_and_kde_values(data, bandwidthMatrix, dim=KL_idx[0])
        _, pdf_vals_y, kde_vals_y = calc_pdf_and_kde_values(data, bandwidthMatrix, dim=KL_idx[1])
        calc_KLDivergence(data, kde_vals_x, kde_vals_y, pdf_vals_x, pdf_vals_y)

if __name__ == "__main__":
    main()



