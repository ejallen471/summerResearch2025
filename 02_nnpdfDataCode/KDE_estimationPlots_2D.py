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

# --- helper function for estimate_bandwidth_matrix_scv
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

# --- calculate the bandwidth matrix - the whole matrix 
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

# --- Plot KDE estimate and PDF estimate in 2D
def plot_kde_vs_pdf_2d(data, kde_vals, pdf_vals, grid_points):
    x_unique = np.unique(grid_points[:, 0])
    y_unique = np.unique(grid_points[:, 1])
    X, Y = np.meshgrid(x_unique, y_unique)

    # Downsample data for plotting if really large
    plot_data = data if data.shape[0] <= 10000 else data[::10]

    plt.scatter(plot_data[:, 0], plot_data[:, 1], c='dimgrey', s=10, alpha=0.3, label='Samples')
    plt.contour(X, Y, pdf_vals, colors='navy', linewidths=1.5)
    plt.contour(X, Y, kde_vals, colors='firebrick', linestyles='dashed', linewidths=1.5)

    legend_elements = [
        Line2D([0], [0], color='navy', lw=1.5, label='Empirical PDF'),
        Line2D([0], [0], color='firebrick', lw=1.5, linestyle='dashed', label='KDE Estimate PDF'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=6, label='Samples', alpha=0.5)
    ]

    plt.legend(handles=legend_elements)
    # plt.title(f'KDE vs Sample PDF', fontsize=16)
    # plt.xlabel('X', fontsize=14)
    # plt.ylabel('Y', fontsize=14)
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
    plt.xlabel(f"Dimension {dim}")
    plt.ylabel("Density")
    plt.title("1D Histogram with Empirical PDF and KDE Estimate", fontsize=14)
    plt.legend(fontsize=12)
    plt.show()

#############################################################################
### KL DIVERGENCE
#############################################################################

# --- Calculate the KL divergence between the empirical distribution and kde distribution
def calc_KLDivergence_2D(grid_points, kde_vals_2d, pdf_vals_2d):
    """
    Compute KL divergence D_KL(P || Q) over a 2D grid:
    P = KDE estimate
    Q = reference PDF (true Gaussian)
    """

    # --- Flatten 2D grid values to 1D arrays
    p = kde_vals_2d.flatten()
    q = pdf_vals_2d.flatten()

    # --- Grid spacing
    x_unique = np.unique(grid_points[:, 0])
    y_unique = np.unique(grid_points[:, 1])
    dx = x_unique[1] - x_unique[0]
    dy = y_unique[1] - y_unique[0]
    dA = dx * dy

    # --- Clip to avoid log(0) and division by zero
    p = np.clip(p, 1e-300, None)
    q = np.clip(q, 1e-300, None)

    # --- Normalise both distributions over the grid
    p /= np.sum(p) * dA
    q /= np.sum(q) * dA

    # --- Compute KL divergence
    kl_terms = p * np.log(p / q)
    kl_terms = np.nan_to_num(kl_terms, nan=0.0, posinf=0.0, neginf=0.0)

    kl_2d = np.sum(kl_terms) * dA
    print(f"\nKL divergence (2D): {kl_2d:.6f}\n")
    return kl_2d


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
    kde_vals_2d = calc_kdeGaussianEstimate_nD(grid_points_2d, data_2d, optimalBandwidthMatrix_2d).reshape(X.shape)

    # Calculate sample PDF for selected dims
    sampleMean_2d = np.mean(data_2d, axis=0)
    sampleCovariance_2d = np.cov(data_2d, rowvar=False)
    sampleGaussian_2d = multivariate_normal(mean=sampleMean_2d, cov=sampleCovariance_2d)
    pdf_vals_2d = sampleGaussian_2d.pdf(grid_points_2d).reshape(X.shape)

    # Plot with the 2D inputs    
    plot_kde_vs_pdf_2d(data_2d, kde_vals_2d, pdf_vals_2d, grid_points_2d)

# --- Main function 
def main(kdeGridRes=150):
    res_flav, res_ev = read_in_data()

    keys_ev = ['Sigma', 'V', 'V3', 'V8', 'T3', 'T8', 'c+', 'g', 'V15']
    keys_flav = ['d', 'u', 's', 'c', 'dbar', 'ubar', 'sbar', 'cbar', 'g']
    keys_test = ['u', 'g']
    index = 28
    
    data = prepare_data(res_flav, keys_test, index)
    bandwidthMatrix = estimate_bandwidth_matrix_scv(data)
    # print(bandwidthMatrix)

    if bandwidthMatrix[0][0] >= 1e-8:
        run_2D_KDE_estimates_plot(data, bandwidthMatrix, x_idx=0, y_idx=1)

        # Create 2D grid only for selected dims
        grid_axes_2d = [np.linspace(np.min(data[:, dim]) - 1, np.max(data[:, dim]) + 1, kdeGridRes) for dim in range(2)]

        # Create 2D meshgrid and flatten
        X, Y = np.meshgrid(grid_axes_2d[0], grid_axes_2d[1], indexing='xy')
        grid_points_2d = np.vstack([X.ravel(), Y.ravel()]).T.astype(np.float32)  # shape (num_points, 2)
                
        # Compute KDE only on selected dims
        kde_vals_2d = calc_kdeGaussianEstimate_nD(grid_points_2d, data, bandwidthMatrix).reshape(kdeGridRes, kdeGridRes)

        # Calculate sample PDF for selected dims
        sampleMean_2d = np.mean(data, axis=0)
        sampleCovariance_2d = np.cov(data, rowvar=False)
        sampleGaussian_2d = multivariate_normal(mean=sampleMean_2d, cov=sampleCovariance_2d)
        pdf_vals_2d = sampleGaussian_2d.pdf(grid_points_2d).reshape(kdeGridRes, kdeGridRes)

        # Plot with the 2D inputs    
        plot_kde_vs_pdf_2d(data, kde_vals_2d, pdf_vals_2d, grid_points_2d)
        calc_KLDivergence_2D(grid_points_2d, kde_vals_2d, pdf_vals_2d)

    else:
        print('Bandwidth Matrix too small to sucessfully calculate moments')

if __name__ == "__main__":
    main()



