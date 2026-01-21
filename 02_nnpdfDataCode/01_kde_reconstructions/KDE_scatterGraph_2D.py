"""
Two-dimensional KDE and empirical PDF comparison for NNPDF replicas.

This module loads flavour-basis replica data, extracts two chosen parton
flavours at a fixed grid index, and compares a Gaussian kernel density
estimate (KDE) to the empirical Gaussian PDF fit to the same samples.

The main workflow is:

- load serialised replica data from the 00_data directory,
- prepare a two-dimensional sample array for the selected flavours,
- estimate a full bandwidth matrix via smoothed cross-validation (SCV),
- evaluate the KDE and empirical Gaussian on a 2D grid,
- visualise samples with overlaid PDF and KDE contours,
- optionally compute the 2D KL divergence between KDE and empirical PDF.

Utility helpers are also provided for 1D diagnostic histograms and for
finding the custom Matplotlib style file used across the project.
"""

#############################################################################

import os
import pickle
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize

#############################################################################

def _find_python_style():
    """
    Locate the ``pythonStyle.mplstyle`` file, if available.

    The search starts from the directory containing this file and walks up
    the directory tree until a file named ``pythonStyle.mplstyle`` is found.

    Returns
    -------
    str or None
        Absolute path to ``pythonStyle.mplstyle`` if found, otherwise
        ``None``.
    """

    base = os.path.abspath(__file__)
    while True:
        directory = os.path.dirname(base)
        if not directory or directory == os.path.sep:
            return None
        candidate = os.path.join(directory, "pythonStyle.mplstyle")
        if os.path.exists(candidate):
            return candidate
        base = directory


_style_path = _find_python_style()
if _style_path is not None:
    plt.style.use(_style_path)

np.random.seed(4)

#############################################################################
#############################################################################

# --- Load serialised data
def read_in_data():
    """
    Load serialised flavour- and evolution-basis replica data.

    The data are loaded from the 00_data directory located one level above
    this script. Two pickled objects are read: one in the flavour basis and
    one in the evolution basis.

    Returns
    -------
    generator of list of dict
        Generator yielding the flavour-basis replica list and the
        evolution-basis replica list.
    """

    data_dir = Path(__file__).resolve().parent.parent / "00_data"
    paths = [data_dir / "flavour_basis.pkl", data_dir / "evolution_basis.pkl"]
    return (pickle.load(open(p, "rb")) for p in paths)

# --- Extract 2D data at fixed index
def prepare_2d_data(res, key_x, key_y, index=25):
    """
    Extract two flavours at a fixed grid index into a 2D array.

    Parameters
    ----------
    res : sequence of dict
        Replica list where each element is a dictionary mapping flavour
        keys to arrays over the x-grid.
    key_x : str
        Flavour key to use for the x-dimension.
    key_y : str
        Flavour key to use for the y-dimension.
    index : int, optional
        Grid index to select from each replica, by default 25.

    Returns
    -------
    numpy.ndarray
        Array of shape (n_replicas, 2) containing the selected flavour
        values for each replica.
    """

    return np.array([[r[key_x][index], r[key_y][index]] for r in res])

# --- Get data and transform into one single array
def prepare_data(res, keys, indices=None):
    """
    Prepare replica data for one or more flavours and grid indices.

    Parameters
    ----------
    res : sequence of dict
        List of replicas, each a dictionary mapping flavour keys to
        one-dimensional arrays over the x-grid.
    keys : sequence of str
        Flavour keys to extract from each replica.
    indices : None, int or array-like of int, optional
        Grid index or indices to extract. If ``None``, all indices
        (0..49) are used. If an integer is provided, a single index is
        selected. If an array-like is provided, multiple indices are
        extracted.

    Returns
    -------
    numpy.ndarray
        If ``indices`` is an int or ``None``, the shape is
        ``(n_replicas, n_keys, n_indices)`` where ``n_indices`` is 50
        when ``indices`` is ``None`` and 1 when it is an int.
        If ``indices`` is array-like, the shape is
        ``(n_replicas, n_keys, len(indices))``.
    """

    num_replicas = len(res)
    num_keys = len(keys)

    if indices is None:
        # Use all indices (0..49)
        indices = np.arange(50)
    
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
    """
    Smoothed cross-validation (SCV) objective for multivariate Gaussian KDE.

    The bandwidth matrix is parameterised via its lower-triangular
    Cholesky factor. This function evaluates the negative log
    leave-one-out likelihood for a given set of Cholesky parameters.

    Parameters
    ----------
    params : numpy.ndarray
        One-dimensional array containing the lower-triangular entries
        of the Cholesky factor ``L`` of shape ``(d, d)``.
    data : numpy.ndarray
        Sample matrix of shape ``(n_samples, d)``.
    epsilon : float, optional
        Small positive value added to the diagonal of the bandwidth
        matrix for numerical stability, by default 1e-8.
    max_exp_arg : float, optional
        Upper bound for the exponent argument used when evaluating the
        Gaussian kernel to avoid overflow in ``exp``, by default 700.

    Returns
    -------
    float
        Scalar SCV score (negative log leave-one-out likelihood) to be
        minimised.
    """

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
    Estimate the full bandwidth matrix via SCV.

    The bandwidth matrix ``H`` is parameterised as ``H = L L^T`` where
    ``L`` is lower-triangular. The parameters of ``L`` are optimised by
    minimising the SCV objective.

    Parameters
    ----------
    data : numpy.ndarray
        Input sample matrix of shape ``(n_samples, d)``.
    initial_scale : float, optional
        Scaling factor applied to the initial diagonal of ``L`` based
        on the empirical standard deviation of each dimension,
        by default 1.0.

    Returns
    -------
    numpy.ndarray
        Estimated bandwidth matrix ``H`` of shape ``(d, d)``.

    Raises
    ------
    RuntimeError
        If the SCV optimisation does not converge successfully.
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
        raise RuntimeError(f"Optimisation failed: {result.message}")

    L_opt = np.zeros((d, d))
    L_opt[np.tril_indices(d)] = result.x
    H_opt = L_opt @ L_opt.T

    return H_opt
    
#############################################################################

# --- Estimate KDE at given points using batching 
def calc_kdeGaussianEstimate_nD(points, data, bandwidth, batch_size=50):
    """
    Evaluate a multivariate Gaussian KDE on a set of points.

    Parameters
    ----------
    points : numpy.ndarray
        Evaluation points of shape ``(m, d)``.
    data : numpy.ndarray
        Sample matrix of shape ``(n_samples, d)``.
    bandwidth : numpy.ndarray
        Bandwidth matrix ``H`` of shape ``(d, d)``.
    batch_size : int, optional
        Number of evaluation points to process per batch in order to
        reduce memory usage, by default 50.

    Returns
    -------
    numpy.ndarray
        Estimated densities at each evaluation point, shape ``(m,)``.
    """

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
def plot_kde_vs_pdf_2d(data, kde_vals, pdf_vals, grid_points, key_x, key_y, index):
    """
    Plot samples with empirical and KDE contour lines in 2D.

    Parameters
    ----------
    data : numpy.ndarray
        Sample matrix of shape ``(n_samples, 2)`` for the two selected
        flavours.
    kde_vals : numpy.ndarray
        KDE density values evaluated on the 2D grid, with shape
        matching the reshaped grid, e.g. ``(n_y, n_x)``.
    pdf_vals : numpy.ndarray
        Empirical Gaussian PDF values evaluated on the same 2D grid and
        with the same shape as ``kde_vals``.
    grid_points : numpy.ndarray
        Flattened grid coordinates of shape ``(n_points, 2)`` used to
        build the 2D mesh.
    key_x : str
        Flavour label for the x-axis.
    key_y : str
        Flavour label for the y-axis.
    index : int
        Grid index in the underlying x-grid associated with the plot.
    """

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
    plt.xlabel(f"{key_x}(x, Q)", fontsize=14)
    plt.ylabel(f"{key_y}(x, Q)", fontsize=14)
    plt.title(f"KDE vs Empirical PDF ({key_x}, {key_y}; grid index {index})", fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Plot 1D histograms - included for checking the underlying distributions
def plot_histograms_with_pdf(data, bandwidthMatrix, dim=0, bins=50):
    """
    Plot a 1D histogram with empirical Gaussian and KDE overlays.

    This is intended as a diagnostic to inspect the marginal
    distribution implied by the multidimensional bandwidth matrix.

    Parameters
    ----------
    data : numpy.ndarray
        Sample matrix of shape ``(n_samples, d)``.
    bandwidthMatrix : numpy.ndarray
        Full bandwidth matrix of shape ``(d, d)`` estimated from the
        multidimensional KDE.
    dim : int, optional
        Index of the dimension to plot, by default 0.
    bins : int, optional
        Number of histogram bins, by default 50.
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
    Compute the 2D KL divergence between KDE and empirical PDF.

    The divergence is approximated by numerical integration over the
    supplied 2D grid.

    Parameters
    ----------
    grid_points : numpy.ndarray
        Flattened grid coordinates of shape ``(n_points, 2)``.
    kde_vals_2d : numpy.ndarray
        KDE density values on the 2D grid, shape ``(n_y, n_x)``.
    pdf_vals_2d : numpy.ndarray
        Reference empirical Gaussian PDF values on the same grid and
        with the same shape as ``kde_vals_2d``.

    Returns
    -------
    float
        Approximation of the KL divergence :math:`D_{KL}(P \| Q)`.
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
def run_2D_KDE_estimates_plot(data, bandwidthMatrix, x_idx, y_idx, kdeGridRes=150, key_x=None, key_y=None, index=None):
    """
    Build a 2D grid, evaluate KDE and empirical PDF, and plot results.

    Parameters
    ----------
    data : numpy.ndarray
        Sample matrix of shape ``(n_samples, d)``.
    bandwidthMatrix : numpy.ndarray
        Full bandwidth matrix of shape ``(d, d)`` estimated from SCV.
    x_idx : int
        Index of the dimension in ``data`` to use for the x-axis.
    y_idx : int
        Index of the dimension in ``data`` to use for the y-axis.
    kdeGridRes : int, optional
        Number of grid points per axis when constructing the 2D grid,
        by default 150.
    key_x : str or None, optional
        Flavour label for the x-axis. If ``None``, a generic label based
        on ``x_idx`` is used.
    key_y : str or None, optional
        Flavour label for the y-axis. If ``None``, a generic label based
        on ``y_idx`` is used.
    index : int or None, optional
        Grid index associated with the selected slice. If ``None``, a
        placeholder is used in the plot title.
    """
    
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
    plot_kde_vs_pdf_2d(
        data_2d,
        kde_vals_2d,
        pdf_vals_2d,
        grid_points_2d,
        key_x if key_x is not None else f"dim{x_idx}",
        key_y if key_y is not None else f"dim{y_idx}",
        index if index is not None else "?",
    )

# --- Main function 
def main(kdeGridRes=150):
    """
    Run the full 2D KDE vs empirical PDF workflow for test flavours.

    The function loads replica data, selects a pair of test flavours at
    a fixed grid index, estimates the SCV bandwidth matrix, and
    generates 2D visualisations along with a KL divergence diagnostic.

    Parameters
    ----------
    kdeGridRes : int, optional
        Number of grid points per axis when constructing the 2D grid,
        by default 150.
    """

    res_flav, res_ev = read_in_data()

    keys_ev = ['Sigma', 'V', 'V3', 'V8', 'T3', 'T8', 'c+', 'g', 'V15']
    keys_flav = ['d', 'u', 's', 'c', 'dbar', 'ubar', 'sbar', 'cbar', 'g']
    
    # choose flavours and index here
    keys_test = ['u', 'g']
    index = 28
    
    data = prepare_data(res_flav, keys_test, index)
    bandwidthMatrix = estimate_bandwidth_matrix_scv(data)
    # print(bandwidthMatrix)

    if bandwidthMatrix[0][0] >= 1e-8:
        run_2D_KDE_estimates_plot(
            data,
            bandwidthMatrix,
            x_idx=0,
            y_idx=1,
            kdeGridRes=kdeGridRes,
            key_x=keys_test[0],
            key_y=keys_test[1],
            index=index,
        )

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
        plot_kde_vs_pdf_2d(data, kde_vals_2d, pdf_vals_2d, grid_points_2d, keys_test[0], keys_test[1], index)
        calc_KLDivergence_2D(grid_points_2d, kde_vals_2d, pdf_vals_2d)

    else:
        print('Bandwidth Matrix too small to sucessfully calculate moments')

if __name__ == "__main__":
    main()



