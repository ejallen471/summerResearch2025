"""
We have NNPDF replica distributions and want to compare their empirical and
KDE-based one-dimensional excess kurtosis values.

Run with the following command:

python kurtosis_calcHistogram.py

This file does the following:

1. Read replica data for all selected flavours and x-grid indices.
2. Calculate empirical and KDE excess kurtosis values.
3. Plot histograms collecting the two kurtosis estimates.
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
    # print(f'Initial h_p: {h_p}')

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
    # print(bandwidthMatrix)

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

# --------------------------------------------------------
# --- Calc PDF
# --------------------------------------------------------

# --- calculate the pdf estimate for a single point
def calc_pdf_pointwise(point, data, bandwidth):
    """
    Evaluate a 1D Gaussian KDE probability density function at a single point.

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

# --------------------------------------------------------
# --- Monte Carlo moment integration (Importance Sampling)
# --------------------------------------------------------

def calc_kurtosis(data, bandwidth, n_samplesMC):
    """Estimate excess kurtosis of the 1D KDE via importance sampling.

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
    q_vals = norm.pdf(samples, loc=mu, scale=np.sqrt(var))  # shape (n_samples,)

    # Target distribution: KDE
    p_vals = np.array([calc_pdf_pointwise(point, data, bandwidth) for point in samples])  # shape (n_samples,)

    weights = p_vals / q_vals  # Importance weights

    # Zeroth moment (integral of p/q ~ 1 if proposal is good)
    zeroth_moment = np.mean(weights)
    # print(zeroth_moment)

    # First moment (mean)
    weighted_mean = np.sum(weights * samples) / np.sum(weights)

    # Second moment (E[x^2])
    weighted_x2 = np.sum(weights * samples**2) / np.sum(weights)
    variance = weighted_x2 - weighted_mean**2

    # Fourth central moment
    centered_samples = samples - weighted_mean
    weighted_x4 = np.sum(weights * centered_samples**4) / np.sum(weights)
    excessKurtosis = (weighted_x4 / (variance**2)) - 3 # minus three makes it excess kurtosis (without is just kurtosis)

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
#############################################################################

# --- Plot histograms of kurtosis
def plot_kurtosis_histograms(KDE_data, empirical_data):
    """
    Plot histograms of KDE-based and empirical excess kurtosis.

    Parameters
    ----------
    KDE_data : sequence of float
        List of KDE-based excess kurtosis values.
    empirical_data : sequence of float
        List of empirically estimated excess kurtosis values.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].hist(KDE_data, bins=100, color="#68A5A1", edgecolor='black')
    axs[0].set_xlabel('Excess Kurtosis')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('KDE-based excess kurtosis', fontsize=12)
    axs[0].tick_params(axis='both')  

    axs[1].hist(empirical_data, bins=100, color="#68A5A1", edgecolor='black')
    axs[1].set_xlabel('Excess Kurtosis')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Empirical excess kurtosis', fontsize=12)
    axs[1].tick_params(axis='both')  

    plt.tight_layout()
    plt.savefig("histogram_kurtosis.png", dpi=300)
    plt.show()

#############################################################################
#############################################################################

# --- Main Function
def main():
    """
    Compute kurtosis (KDE and empirical) and plot two histograms.

    For each chosen flavour and grid index, this function extracts the
    corresponding replica values, calculates the 1D bandwidth via cross validation 
    then constructs the 1D gaussian KDE probability density function. Using this the excess kurtosis is
    calculated via importance sampling. 
    
    In addition, the excess kurtosis is estimated directly from the samples
     
    finally we plot two histograms collecting these values across all (flavour, index) pairs.
    """

    res_flav, _ = read_in_data()

    # --- Change here for number of indices and flavours to include
    keys_flav = ['d', 'u', 's', 'c', 'dbar', 'ubar', 'sbar', 'cbar', 'g']
    indices = np.arange(5)

    kurtosis_KDE_Lst = []
    kurtosis_empirical_Lst = []

    total = len(keys_flav) * len(indices)
    count = 0

    for key in keys_flav:
        for index in indices:
            count += 1
            print(f'Processing {count} / {total}  (key={key}, index={index})')

            # Fetch 1D data for this flavour and index
            data = prepare_data(res_flav, [key], index)  # shape (n_replicas, 1)
            data_1d = data[:, 0]

            # Bandwidth calculation: use 2D (n_samples, 1) for calc_bandwidthMatrix
            bandwidthMatrix = calc_bandwidthMatrix(data)
            bandwidth = float(np.sqrt(bandwidthMatrix[0, 0]))

            # KDE-based kurtosis using that bandwidth
            kurtosis_KDE = calc_kurtosis(data_1d, bandwidth, n_samplesMC=20000)

            # Empirical kurtosis from the same samples
            kurtosis_empirical = calc_empirical_kurtosis(data_1d)

            kurtosis_KDE_Lst.append(float(kurtosis_KDE))
            kurtosis_empirical_Lst.append(float(kurtosis_empirical))

    plot_kurtosis_histograms(kurtosis_KDE_Lst, kurtosis_empirical_Lst)

if __name__ == "__main__":
    main()


