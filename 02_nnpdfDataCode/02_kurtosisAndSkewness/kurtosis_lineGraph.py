"""
We have NNPDF replica distributions and want to study how empirical and
KDE-based excess kurtosis vary across x-grid indices.

Run with the following command:

python kurtosis_lineGraph.py

This file does the following:

1. Read replica data and calculate KDE and empirical kurtosis at each index.
2. Estimate uncertainty on both kurtosis calculations using bootstrap samples.
3. Plot excess kurtosis against grid index with error bars for every flavour.
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import norm

#############################################################################
#############################################################################

def read_in_data():
    """Load serialised flavour- and evolution-basis replica data.

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
    """Extract replica values for selected flavours at a single grid index.

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


def calc_bandwidthMatrix(data, n=100000):
    """Estimate a diagonal bandwidth matrix via cross-validation.

    This helper constructs a family of diagonal bandwidth matrices based on
    Silverman's rule-of-thumb scaling and selects the one that maximises the
    cross-validated log-likelihood. Off-diagonal covariance terms are
    ignored.

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
    bandwidthMatrix, _ = calc_kdeCrossValidation(
        data, H_Matrix_candidateLst, k=5, subsample_size=10000
    )

    return bandwidthMatrix


def calc_kdeCrossValidation(data, H_Matrix_candidateLst, k=5, subsample_size=10000):
    """Select a bandwidth matrix by k-fold cross-validation.

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
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
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


def calc_pdf_pointwise(point, data, bandwidth):
    """
    Evaluate a 1D Gaussian KDE probability density function at a point.

    Parameters
    ----------
    point : float
        Evaluation point.
    data : numpy.ndarray
        One-dimensional sample array of shape ``(n_samples,)`` or
        ``(n_samples, 1)``.
    bandwidth : float
        Scalar bandwidth parameter ``h``.

    Returns
    -------
    float
        KDE value at the evaluation point.
    """

    data = np.asarray(data).ravel()
    n = data.size
    d = 1

    diffs = data - point
    D2 = (diffs / bandwidth) ** 2
    norm_const = 1.0 / ((2 * np.pi) ** (d / 2) * bandwidth)
    kernel_vals = np.exp(-0.5 * D2)

    return (1.0 / n) * np.sum(norm_const * kernel_vals)


def calc_kurtosis(data, bandwidth, n_samplesMC):
    """
    Estimate excess kurtosis of the 1D KDE via importance sampling.

    A Gaussian proposal is fitted to the data and used to draw Monte
    Carlo samples. Importance weights are constructed from the ratio of
    KDE to proposal densities. Internally this computes the zeroth,
    first and second moments, but only the excess kurtosis is returned.

    Parameters
    ----------
    data : numpy.ndarray
        One-dimensional sample array of shape ``(n_samples,)``.
    bandwidth : float
        Scalar bandwidth parameter for the KDE.
    n_samplesMC : int
        Number of Monte Carlo samples drawn from the proposal.

    Returns
    -------
    float
        Excess kurtosis of the KDE, defined so that a Gaussian has
        value zero.
    """

    data = np.asarray(data).ravel()

    # Proposal distribution: Gaussian fit to the data
    mu = np.mean(data)
    var = np.var(data, ddof=1)

    samples = np.random.normal(mu, np.sqrt(var), size=n_samplesMC)
    q_vals = norm.pdf(samples, loc=mu, scale=np.sqrt(var))

    # Target distribution: KDE
    p_vals = np.array([calc_pdf_pointwise(point, data, bandwidth) for point in samples])

    weights = p_vals / q_vals

    # Zeroth moment (integral of p/q ~ 1 if proposal is good)
    zeroth_moment = np.mean(weights)

    # First moment (mean)
    weighted_mean = np.sum(weights * samples) / np.sum(weights)

    # Second moment (E[x^2])
    weighted_x2 = np.sum(weights * samples**2) / np.sum(weights)
    variance = weighted_x2 - weighted_mean**2

    # Fourth central moment
    centered_samples = samples - weighted_mean
    weighted_x4 = np.sum(weights * centered_samples**4) / np.sum(weights)
    excessKurtosis = (weighted_x4 / (variance**2)) - 3

    return excessKurtosis


def calc_empirical_kurtosis(data):
    """
    Compute empirical excess kurtosis for 1D data.

    Parameters
    ----------
    data : numpy.ndarray
        One-dimensional sample array.

    Returns
    -------
    float
        Empirical excess kurtosis.
    """

    data = np.asarray(data).ravel()
    mean = np.mean(data)
    variance = np.var(data, ddof=1)
    centered = data - mean
    fourth_moment = np.mean(centered**4)
    excess_kurtosis = (fourth_moment / (variance**2)) - 3

    return float(excess_kurtosis)


#############################################################################
### Bootstrap helpers
#############################################################################


def bootstrap_kde_kurtosis(data, bandwidth, n_bootstrap=250, n_samplesMC=20000):
    """
    Bootstrap standard error for KDE-based excess kurtosis.

    Parameters
    ----------
    data : numpy.ndarray
        One-dimensional sample array of shape ``(n_samples,)``.
    bandwidth : float
        Fixed KDE bandwidth ``h`` used for all bootstrap resamples.
    n_bootstrap : int, optional
        Number of bootstrap resamples, by default 250.
    n_samplesMC : int, optional
        Number of Monte Carlo samples in the importance-sampling
        estimator, by default 20000.

    Returns
    -------
    float
        Bootstrap standard error of the KDE-based excess kurtosis.
    """

    data = np.asarray(data).ravel()
    n = data.size
    kurt_samples = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        resampled = data[idx]
        kurt_samples[i] = calc_kurtosis(resampled, bandwidth, n_samplesMC=n_samplesMC)

    return float(np.std(kurt_samples, ddof=1))


def bootstrap_empirical_kurtosis(data, n_bootstrap=250):
    """
    Bootstrap standard error for empirical excess kurtosis.

    Parameters
    ----------
    data : numpy.ndarray
        One-dimensional sample array of shape ``(n_samples,)``.
    n_bootstrap : int, optional
        Number of bootstrap resamples, by default 250.

    Returns
    -------
    float
        Bootstrap standard error of the empirical excess kurtosis.
    """

    data = np.asarray(data).ravel()
    n = data.size
    kurt_samples = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        resampled = data[idx]
        kurt_samples[i] = calc_empirical_kurtosis(resampled)

    return float(np.std(kurt_samples, ddof=1))


#############################################################################
### Main plotting logic
#############################################################################


def main():
    """
    Plot KDE-based and empirical excess kurtosis vs grid index.

    For each chosen flavour and each x-grid index, this function:

    1. Extracts the replica values for that (flavour, index) pair.
    2. calculate the bandwidth via cross validation
    3. calculate the kurtosis empircally and via KDE
    4. Estimates bootstrap errors for both estimators.

    The results are collected into two line graphs (KDE and empirical),
    with one coloured line per flavour and error bars on each grid
    index.

    """

    res_flav, _ = read_in_data()

    # Flavours to include
    keys_flav = ['d', 'u', 's', 'c', 'dbar', 'ubar', 'sbar', 'cbar', 'g']

    latex_labels = {
        'd': r'$d$',
        'u': r'$u$',
        's': r'$s$',
        'c': r'$c$',
        'g': r'$g$',
        'dbar': r'$\bar{d}$',
        'ubar': r'$\bar{u}$',
        'sbar': r'$\bar{s}$',
        'cbar': r'$\bar{c}$',
    }

    # Determine the number of grid points from the first replica and flavour
    first_replica = res_flav[0]
    n_indices = len(first_replica[keys_flav[0]])
    indices = np.arange(n_indices)

    # Storage for results
    kurtosis_kde = {key: np.zeros(n_indices, dtype=float) for key in keys_flav}
    err_kde = {key: np.zeros(n_indices, dtype=float) for key in keys_flav}
    kurtosis_emp = {key: np.zeros(n_indices, dtype=float) for key in keys_flav}
    err_emp = {key: np.zeros(n_indices, dtype=float) for key in keys_flav}

    total = len(keys_flav) * n_indices
    counter = 0

    for key in keys_flav:
        print(f"*** Flavour {key} ***")
        for i, idx in enumerate(indices):
            counter += 1
            print(f"  {counter} / {total}  (index={idx})", end="\r")

            # Fetch data for this flavour and grid index
            data = prepare_data(res_flav, [key], idx)  # (n_replicas, 1)
            data_1d = data[:, 0]

            # Bandwidth selection and KDE-based kurtosis
            bandwidth_matrix = calc_bandwidthMatrix(data)
            bandwidth = float(np.sqrt(bandwidth_matrix[0, 0]))

            k_kde = calc_kurtosis(data_1d, bandwidth, n_samplesMC=20000)
            k_kde_err = bootstrap_kde_kurtosis(
                data_1d,
                bandwidth,
                n_bootstrap=250,
                n_samplesMC=20000,
            )

            # Empirical kurtosis and bootstrap error
            k_emp = calc_empirical_kurtosis(data_1d)
            k_emp_err = bootstrap_empirical_kurtosis(data_1d, n_bootstrap=250)

            kurtosis_kde[key][i] = k_kde
            err_kde[key][i] = k_kde_err
            kurtosis_emp[key][i] = k_emp
            err_emp[key][i] = k_emp_err

        print()  # newline after each flavour

    # Plot: two line graphs (KDE vs empirical)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    colormap = plt.cm.get_cmap('tab10', len(keys_flav))
    color_map = {key: colormap(i) for i, key in enumerate(keys_flav)}

    # Left: KDE-based excess kurtosis
    for key in keys_flav:
        axs[0].errorbar(
            indices,
            kurtosis_kde[key],
            yerr=err_kde[key],
            fmt='o-',
            capthick=1.5,
            capsize=2.5,
            markersize=2,
            color=color_map[key],
        )

    axs[0].set_xlabel("Grid point index")
    axs[0].set_ylabel("Excess kurtosis")
    axs[0].set_title("KDE-based excess kurtosis")
    axs[0].grid(True)

    # Right: Empirical excess kurtosis
    for key in keys_flav:
        axs[1].errorbar(
            indices,
            kurtosis_emp[key],
            yerr=err_emp[key],
            fmt='o-',
            capthick=1.5,
            capsize=2.5,
            markersize=2,
            color=color_map[key],
        )

    axs[1].set_xlabel("Grid point index")
    axs[1].set_title("Empirical excess kurtosis")
    axs[1].grid(True)

    # Shared legend
    handles = [Patch(facecolor=color_map[k], label=latex_labels[k]) for k in keys_flav]
    axs[1].legend(handles=handles, title="Flavour", loc='upper left', frameon=True)

    plt.tight_layout()
    plt.savefig("excess_kurtosis_vs_index_all_flavours.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
