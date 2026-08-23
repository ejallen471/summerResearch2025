"""
We have a KDE mean vector and covariance matrix for every Q and want to draw
fixed-Q Gaussian PDF replicas for reconstruction diagnostics.

Run with the following command:

python pdfGenerator.py

This file does the following:

1. Read every flavour_basis_<Q> mean, covariance and correlation file.
2. Make each covariance positive semidefinite and draw multivariate-normal
   replicas at that fixed Q.
3. Calculate and save per-flavour KDE means and standard deviations and create
   comparison plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('../../pythonStyle.mplstyle')

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# ==========================================================
# ----------------------- READ FILES -----------------------
# ==========================================================

def readInFiles(base_dir, expected_size=405):
    """
    Read KDE outputs from flavour_basis folders.

    Parameters
    ----------
    base_dir : str
        Path containing `flavour_basis_<Q>` subfolders.
    expected_size : int, optional
        Expected dimensionality of covariance/correlation matrices
        and length of mean-vector arrays. Default is 405.

    Returns
    -------
    dict
        Mapping folder name -> dict with keys ``'Q'``, ``'covariance_kde'``,
        ``'correlation_kde'`` and ``'mean_vector_kde.csv'``.
    """
    data_store = {}

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if not (os.path.isdir(folder_path) and folder_name.startswith("flavour_basis_")):
            continue

        # Extract Q value from folder name
        try:
            Q_value = float(folder_name.split("_")[-1])
        except ValueError:
            print(f"Warning: Failed to parse Q from {folder_name}, skipping.")
            continue

        flavour_data = {"Q": Q_value}
        skip_folder = False

        # Read covariance + correlation
        for file_name in ["covariance_kde.csv", "correlation_kde.csv"]:
            file_path = os.path.join(folder_path, file_name)
            key_name = file_name.replace(".csv", "")

            if not os.path.exists(file_path):
                print(f"Missing {file_name} in {folder_name}, skipping.")
                skip_folder = True
                break

            matrix = pd.read_csv(file_path, header=None).values
            if matrix.shape != (expected_size, expected_size):
                print(f"Shape mismatch for {file_name} in {folder_name}, skipping.")
                skip_folder = True
                break

            flavour_data[key_name] = matrix

        if skip_folder:
            continue

        # Read mean vector
        mean_file = os.path.join(folder_path, "mean_vector_kde.csv")
        if not os.path.exists(mean_file):
            print(f"Missing mean_vector in {folder_name}, skipping.")
            continue

        mean_vec = pd.read_csv(mean_file, header=None).values.flatten()
        if mean_vec.shape[0] != expected_size:
            print(f"Bad mean vector length in {folder_name}, skipping.")
            continue

        flavour_data["mean_vector_kde.csv"] = mean_vec
        data_store[folder_name] = flavour_data

    return data_store


# ==========================================================
# ------------------- PDF REPLICA UTILS --------------------
# ==========================================================

def make_psd(matrix, eps=1e-8):
    """
    Return a symmetric positive-definite approximation of `matrix`.

    Parameters
    ----------
    matrix : ndarray
        Square matrix to project to the PSD cone.
    eps : float, optional
        Minimum eigenvalue to enforce. Default is 1e-8.

    Returns
    -------
    ndarray
        Symmetric positive-definite matrix.

    Raises
    ------
    np.linalg.LinAlgError
        If eigen-decomposition fails.
    """
    matrix = 0.5 * (matrix + matrix.T)
    try:
        eigvals, eigvecs = np.linalg.eigh(matrix)
    except np.linalg.LinAlgError:
        print("SVD did not converge while making PSD matrix. Skipping this dataset.")
        raise

    eigvals_clipped = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T


def generate_pdf_replica(mu, sigma, N=100):
    """
    Generate multivariate normal replicas from mean and covariance.

    Parameters
    ----------
    mu : ndarray
        1-D array of mean values.
    sigma : ndarray
        Covariance matrix.
    N : int, optional
        Number of replicas to generate. Default is 100.

    Returns
    -------
    ndarray
        Array of shape ``(N, len(mu))`` containing replicas.

    Raises
    ------
    np.linalg.LinAlgError
        If the covariance matrix cannot be made positive-definite.
    """
    try:
        sigma = make_psd(sigma)
        return np.random.multivariate_normal(mean=mu, cov=sigma, size=N)
    except np.linalg.LinAlgError:
        print("*** SVD did not converge during replica generation. Skipping this dataset.*** ")
        raise


def compute_mean_std_from_replicas(replicas, n_flavours=9, n_points=45):
    """
    Reshape replicas and compute per-flavour mean and standard deviation.

    Parameters
    ----------
    replicas : ndarray
        Array of shape ``(N, n_flavours*n_points)`` produced by
        :func:`generate_pdf_replica`.
    n_flavours : int, optional
        Number of flavours (default 9).
    n_points : int, optional
        Number of x-grid points per flavour (default 45).

    Returns
    -------
    tuple
        ``(mean, std)`` where each is an array of shape ``(n_flavours, n_points)``.
    """
    replicas_reshaped = replicas.reshape(replicas.shape[0], n_flavours, n_points)
    mean = replicas_reshaped.mean(axis=0)
    std  = replicas_reshaped.std(axis=0, ddof=1)
    return mean, std

def saveKDEResults(mean_vals, std_vals, Q, flavours):
    """
    Save KDE mean and std arrays to CSV files.

    Parameters
    ----------
    mean_vals : ndarray
        Array of shape ``(n_flavours, n_points)`` containing mean values.
    std_vals : ndarray
        Array of shape ``(n_flavours, n_points)`` containing standard
        deviations.
    Q : float
        The Q value used in the folder naming; used in output filenames.
    flavours : sequence
        Sequence of flavour name strings.
    """
    os.makedirs("KDE_mean", exist_ok=True)
    os.makedirs("KDE_std", exist_ok=True)

    for i, flav in enumerate(flavours):
        mean_filename = f"KDE_mean/mean_{flav}_Q={Q:.6e}.csv"
        std_filename = f"KDE_std/std_{flav}_Q={Q:.6e}.csv"

        np.savetxt(mean_filename, mean_vals[i], delimiter=",", fmt="%.6e")
        np.savetxt(std_filename, std_vals[i], delimiter=",", fmt="%.6e")

    print(f"Saved KDE results for Q = {Q:.6e}")


def plot_mean_std(mean, std, xgrid, folder_name, flavours=None):
    """Plot per-flavour mean with one-sigma band on a 3x3 grid.

    The layout, labels and axis limits are chosen to match the
    reference "PDFs in Flavour Basis" plots used elsewhere in this
    project.

    Parameters
    ----------
    mean : ndarray
        Mean values with shape ``(9, n_points)``.
    std : ndarray
        Standard deviations with shape ``(9, n_points)``.
    xgrid : ndarray
        1-D array of x-grid points.
    folder_name : str
        Folder name used to compose the output filename.
    flavours : sequence, optional
        Labels for the nine flavours; default ordering is
        ``['d','u','s','c','dbar','ubar','sbar','cbar','g']``.
    """

    if flavours is None:
        flavours = ['d', 'u', 's', 'c', 'dbar', 'ubar', 'sbar', 'cbar', 'g']

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('PDFs in Flavour Basis', fontsize=16, fontweight='bold')
    axes_flat = axes.flatten()

    # Order and labelling of flavours in the 3x3 grid
    plot_order = ['d', 'u', 's', 'c', 'dbar', 'ubar', 'sbar', 'cbar', 'g']
    index_map = [flavours.index(f) for f in plot_order]

    y_labels = [
        r'$xd(x)$',
        r'$xu(x)$',
        r'$xs(x)$',
        r'$xc$',
        r'$x \bar{d}(x)$',
        r'$x \bar{u}(x)$',
        r'$x \bar{s}(x)$',
        r'$x \bar{c}(x)$',
        r'$xg$',
    ]
    y_lims = [
        (0.30, 0.6),
        (0.35, 0.80),
        (0.0, 0.55),
        (-0.06, 0.15),
        (0.0, 0.55),
        (0.0, 0.55),
        (0.0, 0.55),
        (-0.06, 0.15),
        (0.5, 3.5),
    ]

    for ax, idx, y_label, y_lim in zip(axes_flat, index_map, y_labels, y_lims):
        # Use custom colours consistent with other PDF figures:
        # line in #692859 and band in #0f5248.
        ax.plot(
            xgrid,
            mean[idx],
            linewidth=2,
            color="#244161",
            label=r'$\textrm{mean} \pm \sigma$',
        )
        ax.fill_between(
            xgrid,
            mean[idx] - std[idx],
            mean[idx] + std[idx],
            color="#244161",
            alpha=0.25,
        )

        ax.set_ylabel(y_label, fontsize=20)
        ax.set_xlabel(r'$x$')
        ax.set_xscale('log')
        ax.set_xlim([1e-5, 1])
        ax.set_ylim(y_lim)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs("kde_pdf_figures", exist_ok=True)
    safe_name = folder_name.replace(" ", "_")
    out_file = f"kde_pdf_figures/PDF_KDE_{safe_name}.png"
    plt.savefig(out_file, dpi=150)
    print(f"Saved plot: {out_file}")
    # plt.close()


# ==========================================================
# ------------------------ MAIN ----------------------------
# ==========================================================

def main():

    XGRID_45 = np.array([
        2.0e-07, 3.0343e-07, 4.6035e-07, 6.9842e-07, 1.0596e-06, 1.6076e-06,
        2.4389e-06, 3.7002e-06, 5.6137e-06, 8.5168e-06, 1.2921e-05, 1.9603e-05,
        2.9738e-05, 4.5114e-05, 6.8437e-05, 1.0381e-04, 1.5746e-04, 2.3879e-04,
        3.6205e-04, 5.4878e-04, 8.3141e-04, 1.2587e-03, 1.9035e-03, 2.8739e-03,
        4.3285e-03, 6.4962e-03, 9.6992e-03, 1.4375e-02, 2.1089e-02, 3.0522e-02,
        4.3415e-02, 6.0480e-02, 8.2281e-02, 1.0914e-01, 1.4112e-01, 1.7803e-01,
        2.1950e-01, 2.6511e-01, 3.1439e-01, 3.6688e-01, 4.2217e-01, 4.7989e-01,
        5.3976e-01, 6.0147e-01, 6.6481e-01
    ])

    flavours = ['d', 'u', 's', 'c', 'dbar', 'ubar', 'sbar', 'cbar', 'g']

    base_dir = os.path.join("..", "02_covarianceGeneration")
    data_store = readInFiles(base_dir)
    if not data_store:
        print("No valid flavour_basis folders found.")
        return

    Nrep = 1000
    skipCount = 0
    totalCount = 0

    for folder_name, flavour_data in data_store.items():
        totalCount += 1

        Q = flavour_data["Q"]
        print(f"\n=== Processing {folder_name} (Q = {Q:.6e}) ===")

        cov_matrix = flavour_data['covariance_kde']
        mean_vec   = flavour_data['mean_vector_kde.csv']

        # Diagnostic checks
        cov_diag = np.diag(cov_matrix)
        print(f"Covariance diagonal range: {cov_diag.min():.3e} to {cov_diag.max():.3e}")
        print(f"Mean vector range: {mean_vec.min():.3e} to {mean_vec.max():.3e}")
        print(f"Relative uncertainty (std/mean): {np.sqrt(cov_diag.mean()) / np.abs(mean_vec).mean():.3f}")
        
        eigvals = np.linalg.eigvalsh(cov_matrix)
        print(f"Eigenvalue range: {eigvals.min():.3e} to {eigvals.max():.3e}")
        print(f"Negative eigenvalues: {(eigvals < 0).sum()}")

        try:
            replicas_flat = generate_pdf_replica(mean_vec, cov_matrix, Nrep)
            mean_vals, std_vals = compute_mean_std_from_replicas(replicas_flat)

            plot_mean_std(mean_vals, std_vals, XGRID_45, folder_name, flavours)
            saveKDEResults(mean_vals, std_vals, Q, flavours)

        except np.linalg.LinAlgError:
            print(f"Skipping {folder_name}: SVD convergence failure.")
            skipCount += 1

    print(f"\nFinished. Skipped {skipCount} / {totalCount} datasets.")


if __name__ == "__main__":
    main()
