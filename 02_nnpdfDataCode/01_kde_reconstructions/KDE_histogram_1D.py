"""
Plot 1D histograms with empirical and KDE PDFs for NNPDF replicas.

This module loads replica data in the flavour basis, constructs one-dimensional
kernel density estimates (KDEs) for chosen parton flavours at a fixed grid
index, and visualises the results as histograms with the empirical Gaussian
PDF and KDE estimate superimposed. It also provides utilities for bandwidth
selection and simple KL-divergence diagnostics.
"""

#############################################################################

import os
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from scipy.stats import norm

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


#############################################################################
#############################################################################

def read_in_data():
    """
    Load serialised flavour- and evolution-basis replica data.

    The function expects the files ``flavour_basis.pkl`` and
    ``evolution_basis.pkl`` to be located in the ``00_data`` directory that
    sits alongside this script's parent directory.

    Returns
    -------
    generator
        A generator yielding two objects ``(res_flav, res_ev)``, each being
        the unpickled contents of the corresponding file.
    """

    data_dir = Path(__file__).resolve().parent.parent / "00_data"
    paths = [data_dir / "flavour_basis.pkl", data_dir / "evolution_basis.pkl"]
    return (pickle.load(open(p, "rb")) for p in paths)


def prepare_data(res, keys, index):
    """
    Extract replica values for selected flavours at a single grid index.

    Parameters
    ----------
    res : list of dict
        List of NNPDF replicas. Each replica is a dictionary mapping
        flavour keys (e.g. 'u', 'd', 'g') to one-dimensional NumPy arrays
        defined on a fixed grid.
    keys : list of str
        Flavour keys to extract from each replica.
    index : int
        Grid index at which the replica values are extracted.

    Returns
    -------
    numpy.ndarray
        Array of shape (n_replicas, n_flavours) containing the replica
        values for the selected flavours at the specified grid index.
    """

    num_replicas = len(res)
    num_keys = len(keys)
    
    data_array = np.empty((num_replicas, num_keys), dtype=float)
    for i, replica in enumerate(res):
        for j, key in enumerate(keys):
            data_array[i, j] = replica[key][index]

    return data_array

#############################################################################
### BUILDING KDE STUFF
#############################################################################

def calc_bandwidthMatrix(data, n=100000):
    """
    Estimate a diagonal bandwidth matrix via cross-validation.

    This helper constructs a family of diagonal bandwidth matrices based on
    Silverman's rule-of-thumb scaling and selects the one that maximises the
    cross-validated log-likelihood. Off-diagonal covariance terms are
    ignored. Currently not used in the 2D code paths but kept for reference.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape ``(n_samples, d)`` containing the input samples.
    n : int, optional
        Unused argument retained for backwards compatibility.

    Returns
    -------
    numpy.ndarray
        Diagonal bandwidth matrix of shape ``(d, d)``.
    """
    
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
    bandwidthMatrix, _ = calc_kdeCrossValidation(data, H_Matrix_candidateLst, k=5, subsample_size=10000)
    print(bandwidthMatrix)

    return bandwidthMatrix

def calc_kdeCrossValidation(data, H_Matrix_candidateLst, k=5, subsample_size=10000):
    """
    Select a bandwidth matrix by k-fold cross-validation.

    Parameters
    ----------
    data : numpy.ndarray
        Input data of shape ``(n_samples, d)``.
    H_Matrix_candidateLst : sequence of numpy.ndarray
        List of candidate bandwidth matrices of shape ``(d, d)``.
    k : int, optional
        Number of folds for cross-validation.
    subsample_size : int, optional
        Maximum number of samples used for CV (for speed). If the dataset
        is smaller, all samples are used.

    Returns
    -------
    tuple
        ``(optimalBandwidthMatrix, mean_logLikelihoodLst)`` where
        ``optimalBandwidthMatrix`` is the best-performing candidate and
        ``mean_logLikelihoodLst`` contains the mean log-likelihood per
        candidate.
    """
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

#############################################################################

def calc_pdf_and_kde_values(data, bandwidthMatrix, dim):
    """
    Compute 1D empirical Gaussian PDF and KDE estimate for one dimension.

    Parameters
    ----------
    data : numpy.ndarray
        Sample array of shape ``(n_samples, d)``.
    bandwidthMatrix : numpy.ndarray
        Bandwidth matrix of shape ``(d, d)`` from which the scalar
        bandwidth for ``dim`` is taken from the diagonal.
    dim : int
        Column index of ``data`` to analyse.

    Returns
    -------
    tuple of numpy.ndarray
        ``(x_vals, pdf_vals, kde_vals)`` where ``x_vals`` is the grid of
        evaluation points, ``pdf_vals`` the empirical Gaussian PDF and
        ``kde_vals`` the kernel density estimate at those points.
    """

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
    """
    Compute 1D KL divergences between empirical and KDE marginals.

    The function assumes that the x and y evaluation grids are uniformly
    spaced and re-constructs them from the data range and the length of
    the supplied arrays.

    Parameters
    ----------
    data : numpy.ndarray
        Sample data of shape ``(n_samples, 2)``.
    kde_vals_x, kde_vals_y : numpy.ndarray
        KDE marginal values along the x and y directions.
    pdf_x, pdf_y : numpy.ndarray
        Reference empirical PDF marginal values along the x and y
        directions.

    Returns
    -------
    None
        The function prints the KL divergence for each marginal to stdout.
    """

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


def plot_1D_histogram(data, x_vals, pdf_vals, kde_vals, dim, flavour, index, bins=50):
    """
    Plot a 1D histogram with empirical PDF and KDE overlayed.

    Parameters
    ----------
    data : numpy.ndarray
        Replica samples of shape ``(n_replicas, n_flavours)``.
    x_vals : numpy.ndarray
        Grid of evaluation points used for the PDFs.
    pdf_vals : numpy.ndarray
        Empirical Gaussian PDF values at ``x_vals``.
    kde_vals : numpy.ndarray
        KDE estimate values at ``x_vals``.
    dim : int
        Column index selecting which flavour to show.
    flavour : str
        Name of the parton flavour corresponding to ``dim``.
    index : int
        Grid index at which the PDFs are evaluated.
    bins : int, optional
        Number of histogram bins.
    """

    data_1d = data[:, dim]

    # Plot histogram, empirical PDF, and KDE estimate
    plt.figure(figsize=(8, 5))
    plt.hist(
        data_1d,
        bins=bins,
        density=True,
        color="#68A5A1",
        edgecolor="black",
        alpha=0.6,
        label="Replica histogram",
    )
    plt.plot(x_vals, kde_vals, "--", lw=2, color="#861B61", label="KDE Estimate PDF")
    plt.plot(x_vals, pdf_vals, lw=2, color="#0B2C3D", label="Empirical PDF")

    plt.xlabel("f(x, Q)")
    plt.ylabel("Probability Density")
    plt.title(
        f"Replica Distribution (flavour: {flavour}, grid index: {index})",
        fontsize=14,
    )
    plt.legend()
    plt.show()

def plot_1D_histogram_withScatter(data, x_vals, pdf_vals, kde_vals, dim, flavour, index, bins=50):
    """
    Plot scatter of replica values plus rotated histogram and PDFs.

    Parameters
    ----------
    data : numpy.ndarray
        Replica samples of shape ``(n_replicas, n_flavours)``.
    x_vals : numpy.ndarray
        Grid of evaluation points used for the PDFs.
    pdf_vals : numpy.ndarray
        Empirical Gaussian PDF values at ``x_vals``.
    kde_vals : numpy.ndarray
        KDE estimate values at ``x_vals``.
    dim : int
        Column index selecting which flavour to show.
    flavour : str
        Name of the parton flavour corresponding to ``dim``.
    index : int
        Grid index at which the PDFs are evaluated.
    bins : int, optional
        Number of histogram bins.
    """
    data_1d = data[:, dim]

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=(4, 2), wspace=0.05)

    # Left: Scatter plot
    ax_main = fig.add_subplot(gs[0])
    ax_main.scatter(np.arange(len(data_1d)), data_1d, color='#68A5A1', s=3)
    ax_main.set_title(f"Replica Distribution (flavour: {flavour}, grid index: {index})",fontsize=14)
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

    # Overlay: empirical PDF (solid line, navy) and KDE estimate (dashed, plum)
    ax_hist.plot(kde_vals, x_vals, "--", lw=2, color="#861B61", label="KDE Estimate PDF")
    ax_hist.plot(pdf_vals, x_vals, lw=2, color="#0B2C3D", label="Empirical PDF")

    ax_hist.set_xlabel("f(x, Q)", fontsize=12, labelpad=12)
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
    """
    Run the 1D KDE analysis and produce plots/diagnostics.

    Parameters
    ----------
    plotting1D : bool, optional
        If ``True`` (default), produce 1D histogram plots for each selected
        flavour.
    KL_divergence : bool, optional
        If ``True`` (default), compute and print KL divergences for a
        chosen pair of marginals.
    """
    res_flav, res_ev = read_in_data()
    
    # keys_ev = ['Sigma', 'V', 'V3', 'V8', 'T3', 'T8', 'c+', 'g', 'V15']
    # keys_flav = ['d', 'u', 's', 'c', 'dbar', 'ubar', 'sbar', 'cbar', 'g']
    
    # choose flavours to loop through 
    keys_flav = ['c', 'cbar']

    # choose singe index between 1 and 50
    index = 12

    data = prepare_data(res_flav, keys_flav, index)
    bandwidthMatrix = calc_bandwidthMatrix(data)

    # --- Plot in 1D
    d = data.shape[1]
    print(d)
    if plotting1D == True:
        for dim in range(0, d):
            x_vals, pdf_vals, kde_vals = calc_pdf_and_kde_values(data, bandwidthMatrix, dim)
            plot_1D_histogram(data, x_vals, pdf_vals, kde_vals, dim, keys_flav[dim], index)
            # plot_1D_histogram_withScatter(data, x_vals, pdf_vals, kde_vals, dim, keys_flav[dim], index)

    # --- Calculate KL divergence 
    KL_idx = (0,1) # which distributions is the KL divergence calculated between 
    if KL_divergence == True:
        _, pdf_vals_x, kde_vals_x = calc_pdf_and_kde_values(data, bandwidthMatrix, dim=KL_idx[0])
        _, pdf_vals_y, kde_vals_y = calc_pdf_and_kde_values(data, bandwidthMatrix, dim=KL_idx[1])
        calc_KLDivergence(data, kde_vals_x, kde_vals_y, pdf_vals_x, pdf_vals_y)

if __name__ == "__main__":
    main()



