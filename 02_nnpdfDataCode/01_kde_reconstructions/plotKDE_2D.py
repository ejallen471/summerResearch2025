"""
Plot 2D KDE probability density functions for selected flavours.

This script implements the same 2D KDE construction (SCV bandwidth
selection and Gaussian kernel) as used in the 2D moments code, but it is
entirely self-contained and focuses purely on visualisation: it plots
contour lines of the KDE probability density function, optionally
alongside a Gaussian fit to the samples.
"""

import pickle
import warnings
from pathlib import Path

import numba
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import minimize
from scipy.stats import multivariate_normal


def read_in_data():
    """
    Load flavour- and evolution-basis replica data from the shared 00_data.

    Expects ``flavour_basis.pkl`` and ``evolution_basis.pkl`` in the
    ``00_data`` directory one level above this folder.

    Returns
    -------
    generator of dict
        A generator yielding first the flavour-basis replicas and then
        the evolution-basis replicas, as loaded from pickle files.
    """

    data_dir = Path(__file__).resolve().parent.parent / "00_data"
    paths = [data_dir / "flavour_basis.pkl", data_dir / "evolution_basis.pkl"]
    return (pickle.load(open(p, "rb")) for p in paths)


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

    indices = np.array(indices)
    data_array = np.empty((num_replicas, num_keys, len(indices)), dtype=float)
    for i, replica in enumerate(res):
        for j, key in enumerate(keys):
            data_array[i, j, :] = replica[key][indices]

    return data_array


@numba.njit(fastmath=True)
def _mahalanobis2_2x2(dx, dy, invH00, invH01, invH11):
    """
    Squared Mahalanobis distance for 2x2 bandwidth matrices.

    Parameters
    ----------
    dx, dy : float
        Coordinate differences along the two dimensions.
    invH00, invH01, invH11 : float
        Independent entries of the inverse bandwidth matrix ``H^{-1}``
        for a 2x2 symmetric matrix.

    Returns
    -------
    float
        The squared Mahalanobis distance corresponding to ``[dx, dy]``.
    """

    return invH00 * dx * dx + 2 * invH01 * dx * dy + invH11 * dy * dy


@numba.njit(fastmath=True)
def _scv_objective_numba(L_flat, data):
    """
    Numba-accelerated SCV objective for a 2x2 bandwidth matrix.

    Parameters
    ----------
    L_flat : numpy.ndarray
        One-dimensional array of length 3 containing the lower-triangular
        entries of the Cholesky factor ``L`` for a 2x2 bandwidth matrix.
    data : numpy.ndarray
        Sample matrix of shape ``(n_samples, 2)``.

    Returns
    -------
    float
        The negative log leave-one-out likelihood to be minimised.
    """

    n = data.shape[0]
    L00, L10, L11 = L_flat[0], L_flat[1], L_flat[2]

    # Regularised H
    H00 = max(L00 * L00, 1e-10)
    H01 = L00 * L10
    H11 = max(L10 * L10 + L11 * L11, 1e-10)

    detH = H00 * H11 - H01 * H01
    if detH <= 0.0:
        return np.inf

    invH00 = H11 / detH
    invH01 = -H01 / detH
    invH11 = H00 / detH

    log_norm_const = -np.log(2 * np.pi) - 0.5 * np.log(detH)
    total_log = 0.0

    for i in range(n):
        xi0, xi1 = data[i, 0], data[i, 1]

        # Leave-one-out Mahalanobis distances in log-space
        m2_max = -1e20
        m2_vals = np.empty(n - 1, dtype=np.float64)
        idx = 0
        for j in range(n):
            if i == j:
                continue
            dx = xi0 - data[j, 0]
            dy = xi1 - data[j, 1]
            m2 = -0.5 * _mahalanobis2_2x2(dx, dy, invH00, invH01, invH11)
            m2_vals[idx] = m2
            if m2 > m2_max:
                m2_max = m2
            idx += 1

        # log-sum-exp trick
        s = 0.0
        for k in range(n - 1):
            s += np.exp(m2_vals[k] - m2_max)
        p_i = (s / (n - 1)) * np.exp(log_norm_const + m2_max)

        if p_i <= 0.0 or not np.isfinite(p_i):
            return np.inf
        total_log += np.log(p_i)

    return -total_log / n


def scv_objective(params, data):
    """
    Wrapper around the numba SCV objective handling non-finite values.

    Parameters
    ----------
    params : numpy.ndarray
        Parameters describing the Cholesky factor ``L``.
    data : numpy.ndarray
        Sample matrix of shape ``(n_samples, 2)``.

    Returns
    -------
    float
        Finite SCV objective value or ``np.inf`` on failure.
    """

    try:
        val = _scv_objective_numba(params, data)
        return val if np.isfinite(val) else np.inf
    except Exception:
        return np.inf


def _make_positive_definite_2x2(M, eps=1e-5):
    """
    Symmetrise and lightly regularise a 2x2 matrix to make it SPD.

    Parameters
    ----------
    M : numpy.ndarray
        Input matrix of shape ``(2, 2)``.
    eps : float, optional
        Small positive number added to the diagonal when the determinant
        is too small or negative, by default ``1e-5``.

    Returns
    -------
    numpy.ndarray
        Symmetric positive-definite 2x2 matrix.
    """

    M = 0.5 * (M + M.T)
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if det <= eps:
        M[0, 0] += eps
        M[1, 1] += eps
    return M


def estimate_bandwidth_matrix_scv(data, maxiter=200):
    """
    Estimate a 2x2 bandwidth matrix using smoothed cross-validation.

    Parameters
    ----------
    data : numpy.ndarray
        Sample matrix of shape ``(n_samples, 2)``.
    maxiter : int, optional
        Maximum number of iterations for the L-BFGS-B optimiser,
        by default ``200``.

    Returns
    -------
    numpy.ndarray
        A symmetric positive-definite bandwidth matrix of shape ``(2, 2)``.
    """

    n, d = data.shape
    assert d == 2, "Only 2x2 supported"

    Sigma = np.cov(data, rowvar=False)
    Sigma = _make_positive_definite_2x2(Sigma)

    c = (4 / (d + 2)) ** (1 / (d + 4)) * n ** (-1 / (d + 4))
    L0 = np.linalg.cholesky(Sigma) * c
    initial_params = L0[np.tril_indices(d)]

    # Reasonable bounds on L
    bounds = [(1e-6, 1e2), (-1e2, 1e2), (1e-6, 1e2)]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = minimize(
                scv_objective,
                initial_params,
                args=(data,),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": maxiter, "ftol": 1e-6},
            )
    except Exception:
        result = None

    if result is None or not result.success or not np.isfinite(result.fun):
        # Fallback: scaled empirical covariance
        return Sigma * c ** 2

    L_opt = np.zeros((2, 2))
    L_opt[np.tril_indices(2)] = result.x
    H = L_opt @ L_opt.T

    return _make_positive_definite_2x2(H)


def calc_kdeGaussianEstimate_nD(points, data, bandwidth, batch_size=50):
    """
    Estimate a multivariate Gaussian KDE at given points using batching.

    Parameters
    ----------
    points : numpy.ndarray
        Evaluation points of shape ``(m, d)``.
    data : numpy.ndarray
        Sample matrix of shape ``(n_samples, d)``.
    bandwidth : numpy.ndarray
        Bandwidth matrix of shape ``(d, d)``.
    batch_size : int, optional
        Number of evaluation points to process per batch, by default ``50``.

    Returns
    -------
    numpy.ndarray
        Array of length ``m`` containing the KDE values at each point.
    """

    n, d = data.shape
    m = points.shape[0]

    bandwidth_inv = np.linalg.inv(bandwidth)
    det_bandwidth = np.linalg.det(bandwidth)
    norm_const = 1.0 / ((2 * np.pi) ** (d / 2) * np.sqrt(det_bandwidth))

    densities = np.empty(m, dtype=np.float32)

    # Calculate in batches to reduce memory footprint
    for start in range(0, m, batch_size):
        end = min(start + batch_size, m)
        batch = points[start:end].astype(np.float32)
        diffs = batch[:, np.newaxis, :] - data  # (b, n, d)
        D2 = np.einsum("bnd,dd,bnd->bn", diffs, bandwidth_inv, diffs)
        kernel_vals = norm_const * np.exp(-0.5 * D2)
        densities[start:end] = np.mean(kernel_vals, axis=1)

    return densities


def plot_kde_vs_pdf_2d(data, grid_points, kde_vals, pdf_vals):
    """
    Plot 2D KDE and Gaussian reference PDF as contour lines.

    Parameters
    ----------
    data : numpy.ndarray
        Sample matrix of shape ``(n_samples, 2)`` used to generate the KDE.
    grid_points : numpy.ndarray
        Grid points of shape ``(m, 2)`` on which densities were evaluated.
    kde_vals : numpy.ndarray
        KDE values on the grid, reshaped to match the meshgrid shape.
    pdf_vals : numpy.ndarray
        Gaussian reference PDF values on the same grid.

    Returns
    -------
    None
        The function creates and shows a matplotlib figure.
    """

    x_unique = np.unique(grid_points[:, 0])
    y_unique = np.unique(grid_points[:, 1])
    X, Y = np.meshgrid(x_unique, y_unique)

    # Downsample data for plotting if really large
    plot_data = data if data.shape[0] <= 10000 else data[::10]

    plt.figure(figsize=(7, 6))
    plt.scatter(plot_data[:, 0], plot_data[:, 1], c="dimgrey", s=8, alpha=0.3, label="Samples")
    plt.contour(X, Y, kde_vals, colors="navy", linewidths=1.5)
    plt.contour(X, Y, pdf_vals, colors="firebrick", linestyles="dashed", linewidths=1.5)

    legend_elements = [
        Line2D([0], [0], color="navy", lw=1.5, label="KDE pdf"),
        Line2D([0], [0], color="firebrick", lw=1.5, linestyle="dashed", label="Gaussian fit"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="k", markersize=5, label="Replicas"),
    ]

    plt.legend(handles=legend_elements)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    """
    Build a 2D KDE and plot its probability density function.

    This function loads replica data, selects two flavours at a fixed
    grid index, constructs a 2D Gaussian KDE with SCV-selected bandwidth,
    evaluates it and a Gaussian fit on a grid, and produces a contour plot.
    """

    res_flav, _ = read_in_data()

    # Choose flavours and grid index here
    keys_flav = ["u", "g"]
    index = 42

    # Extract 2D data (n_samples, 2)
    data_3d = prepare_data(res_flav, keys_flav, [index])
    data = data_3d[:, :, 0]

    # Bandwidth via SCV (same machinery as in momentEstimation_2D)
    bandwidthMatrix = estimate_bandwidth_matrix_scv(data)

    # Build a 2D grid over the data range
    kdeGridRes = 150
    grid_axes = [
        np.linspace(np.min(data[:, dim]) - 1.0, np.max(data[:, dim]) + 1.0, kdeGridRes)
        for dim in range(2)
    ]
    X, Y = np.meshgrid(grid_axes[0], grid_axes[1], indexing="xy")
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T.astype(np.float32)

    # KDE values on the grid
    kde_vals = calc_kdeGaussianEstimate_nD(grid_points, data, bandwidthMatrix).reshape(X.shape)

    # Gaussian reference PDF fitted to the same data
    mu = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    gauss = multivariate_normal(mean=mu, cov=cov)
    pdf_vals = gauss.pdf(grid_points).reshape(X.shape)

    plot_kde_vs_pdf_2d(data, grid_points, kde_vals, pdf_vals)


if __name__ == "__main__":
    main()
