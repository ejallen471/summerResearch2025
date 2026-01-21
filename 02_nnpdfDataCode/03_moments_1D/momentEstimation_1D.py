
#############################################################################

# Calculate the first four standardised moments (zeroth, mean, variance,
# excess kurtosis) using 1D KDE and empirical methods, with bootstrap
# errors.

#############################################################################

import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from scipy.stats import norm

np.random.seed(4)

#############################################################################
### DATA LOADING, PREPARATION AND BANDWIDTH SELECTION
#############################################################################


def read_in_data():
    """
    Load flavour- and evolution-basis replica data from disk.

    The function expects the files ``flavour_basis.pkl`` and
    ``evolution_basis.pkl`` to be located in the ``00_data`` directory
    that sits one level above this script's directory.

    Returns
    -------
    generator
        Generator yielding two Python objects ``(res_flav, res_ev)``
        corresponding to the unpickled contents of the flavour- and
        evolution-basis files, respectively.
    """

    data_dir = Path(__file__).resolve().parent.parent / "00_data"
    paths = [data_dir / "flavour_basis.pkl", data_dir / "evolution_basis.pkl"]
    return (pickle.load(open(p, "rb")) for p in paths)


def prepare_data(res, keys, index):
    """
    Extract replica values for selected flavours at a single grid index.

    Parameters
    ----------
    res : sequence of dict
        List or other sequence of replicas. Each replica is a mapping
        from flavour key (e.g. ``'u'``, ``'d'``, ``'g'``) to a 1D NumPy
        array defined on a fixed grid.
    keys : sequence of str
        Flavour keys to extract from each replica.
    index : int
        Grid index at which to sample the arrays stored in ``res``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_replicas, n_flavours)`` containing the
        replica values for the requested flavours at the chosen index.
    """

    num_replicas = len(res)
    num_keys = len(keys)

    data_array = np.empty((num_replicas, num_keys), dtype=float)
    for i, replica in enumerate(res):
        for j, key in enumerate(keys):
            data_array[i, j] = replica[key][index]

    return data_array


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
        Number of folds to use in the cross-validation split.
    subsample_size : int, optional
        Maximum number of samples to use during CV (for speed). If
        ``n_samples <= subsample_size`` all points are used.

    Returns
    -------
    optimalBandwidthMatrix : numpy.ndarray
        The candidate bandwidth matrix that maximises the average
        log-likelihood over validation folds.
    mean_logLikelihoodLst : numpy.ndarray
        Array of mean log-likelihood values, one entry per candidate in
        ``H_Matrix_candidateLst``.
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
            dists = np.einsum("mnd,dd,mnd->mn", diffs, H_inv, diffs)
            K = norm_const * np.exp(-0.5 * dists)

            f_vals = np.mean(K, axis=1)
            f_vals = np.clip(f_vals, 1e-300, None)
            fold_log_likelihoods.append(np.mean(np.log(f_vals)))

        mean_logLikelihoodLst.append(np.mean(fold_log_likelihoods))

    mean_logLikelihoodLst = np.array(mean_logLikelihoodLst)
    optimal_idx = np.argmax(mean_logLikelihoodLst)
    optimalBandwidthMatrix = H_Matrix_candidateLst[optimal_idx]

    return optimalBandwidthMatrix, mean_logLikelihoodLst


def calc_bandwidthMatrix(data, n=100000):
    """
    Estimate a diagonal bandwidth matrix via cross-validation.

    This helper constructs a Silverman rule-of-thumb bandwidth vector
    and scales it by a set of factors in ``[0.5, 2.0]`` to form a list
    of diagonal bandwidth matrices. The best candidate is selected via
    :func:`calc_kdeCrossValidation`.

    Parameters
    ----------
    data : numpy.ndarray
        Input samples of shape ``(n_samples, d)``.
    n : int, optional
        Unused argument retained for backwards compatibility.

    Returns
    -------
    numpy.ndarray
        Diagonal bandwidth matrix of shape ``(d, d)``.
    """

    n, d = data.shape
    sigma = np.std(data, axis=0, ddof=1)
    h_p = (4 / (d + 2)) ** (1 / (d + 4)) * n ** (-1 / (d + 4)) * sigma

    scaling_factors = np.linspace(0.5, 2.0, 10)

    H_Matrix_candidateLst = []
    for s in scaling_factors:
        H_diag = (s * h_p) ** 2
        H_matrix = np.diag(H_diag)
        H_Matrix_candidateLst.append(H_matrix)

    bandwidthMatrix, _ = calc_kdeCrossValidation(
        data, H_Matrix_candidateLst, k=5, subsample_size=10000
    )

    return bandwidthMatrix

#############################################################################
### 1D KDE MOMENTS VIA IMPORTANCE SAMPLING
#############################################################################


def calc_pdf_pointwise(point, data, bandwidth):
    """
    Evaluate a 1D Gaussian KDE at a single point.

    Parameters
    ----------
    point : float
        Location ``x`` at which the KDE is evaluated.
    data : array_like
        One-dimensional sample array of shape ``(n_samples,)`` or
        ``(n_samples, 1)``.
    bandwidth : float
        Kernel bandwidth (standard deviation of the Gaussian kernel).

    Returns
    -------
    float
        Estimated probability density ``f(x)`` at ``point``.
    """

    data = np.asarray(data).ravel()
    n = data.size

    diffs = data - point
    D2 = (diffs / bandwidth) ** 2
    norm_const = 1.0 / (np.sqrt(2 * np.pi) * bandwidth)
    kernel_vals = np.exp(-0.5 * D2)

    return (1.0 / n) * np.sum(norm_const * kernel_vals)


def calc_kde_moments_1d(data, bandwidth, n_samplesMC):
    """
    Estimate 1D KDE moments (0th, mean, variance, excess kurtosis).

    Moments are computed via self-normalised importance sampling using a
    Gaussian proposal distribution fitted to the data. The KDE is the
    target density and the Gaussian fit provides the proposal.

    Parameters
    ----------
    data : array_like
        One-dimensional sample array.
    bandwidth : float
        Kernel bandwidth (standard deviation of the Gaussian kernel).
    n_samplesMC : int
        Number of Monte Carlo samples drawn from the proposal.

    Returns
    -------
    zeroth : float
        Zeroth moment of the KDE (should be close to ``1``).
    mean : float
        First moment (mean) of the KDE.
    variance : float
        Second central moment (variance) of the KDE.
    excess_kurtosis : float
        Excess kurtosis of the KDE, defined such that a Gaussian has
        value zero.
    """

    data = np.asarray(data).ravel()

    if data.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    # Proposal distribution: Gaussian fit to the data
    mu = np.mean(data)
    var = np.var(data, ddof=1)
    if not np.isfinite(var) or var <= 0.0:
        return np.nan, np.nan, np.nan, np.nan

    proposal_std = np.sqrt(var)
    samples = np.random.normal(mu, proposal_std, size=n_samplesMC)
    q_vals = norm.pdf(samples, loc=mu, scale=proposal_std)

    # Target distribution: KDE
    p_vals = np.array([calc_pdf_pointwise(point, data, bandwidth) for point in samples])

    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.divide(p_vals, q_vals, out=np.zeros_like(p_vals), where=(q_vals > 0))

    if not np.all(np.isfinite(weights)):
        return np.nan, np.nan, np.nan, np.nan

    # Zeroth moment (integral of KDE, should be ~1)
    zeroth_moment = float(np.mean(weights))

    weight_sum = np.sum(weights)
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        return zeroth_moment, np.nan, np.nan, np.nan

    # First and second raw moments
    weighted_mean = np.sum(weights * samples) / weight_sum
    weighted_x2 = np.sum(weights * samples**2) / weight_sum
    variance = weighted_x2 - weighted_mean**2

    if not np.isfinite(variance) or variance <= 0.0:
        return zeroth_moment, weighted_mean, np.nan, np.nan

    # Fourth central moment and excess kurtosis
    centered_samples = samples - weighted_mean
    weighted_x4 = np.sum(weights * centered_samples**4) / weight_sum
    excess_kurtosis = (weighted_x4 / (variance**2)) - 3.0

    return float(zeroth_moment), float(weighted_mean), float(variance), float(excess_kurtosis)

#############################################################################
### EMPIRICAL MOMENTS AND BOOTSTRAP ERRORS
#############################################################################

def empirical_moments_1d(data):
    """
    Compute empirical 1D zeroth, mean, variance and excess kurtosis.

    Parameters
    ----------
    data : array_like
        One-dimensional sample array.

    Returns
    -------
    zeroth : float
        Zeroth moment, always equal to ``1.0`` for empirical data.
    mean : float
        Sample mean.
    variance : float
        Unbiased sample variance (``ddof=1``).
    excess_kurtosis : float
        Empirical excess kurtosis, computed from the fourth central
        moment and the variance (Gaussian gives zero).
    """

    data = np.asarray(data).ravel()
    if data.size == 0:
        return np.nan, np.nan, np.nan, np.nan

    mean = np.mean(data)
    variance = np.var(data, ddof=1)
    if not np.isfinite(variance) or variance <= 0.0:
        return 1.0, float(mean), np.nan, np.nan

    centered = data - mean
    fourth_moment = np.mean(centered**4)
    excess_kurtosis = (fourth_moment / (variance**2)) - 3.0

    return 1.0, float(mean), float(variance), float(excess_kurtosis)


def calc_bootstrap_empirical_moments_1d(data, n_bootstrap):
    """
    Estimate bootstrap errors for empirical 1D moments.

    Parameters
    ----------
    data : array_like
        One-dimensional sample array.
    n_bootstrap : int
        Number of bootstrap resamples.

    Returns
    -------
    err_zeroth : float
        Standard deviation of the bootstrap zeroth-moment estimates.
    err_mean : float
        Standard deviation of the bootstrap mean estimates.
    err_var : float
        Standard deviation of the bootstrap variance estimates.
    err_kurt : float
        Standard deviation of the bootstrap excess-kurtosis estimates.
    """

    data = np.asarray(data).ravel()
    n = data.size
    if n == 0 or n_bootstrap <= 1:
        return np.nan, np.nan, np.nan, np.nan

    zeroth_vals = np.empty(n_bootstrap)
    mean_vals = np.empty(n_bootstrap)
    var_vals = np.empty(n_bootstrap)
    kurt_vals = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idxs = np.random.choice(n, size=n, replace=True)
        resample = data[idxs]
        m0, m1, v, k = empirical_moments_1d(resample)
        zeroth_vals[i] = m0
        mean_vals[i] = m1
        var_vals[i] = v
        kurt_vals[i] = k

    err_zeroth = float(np.std(zeroth_vals, ddof=1))
    err_mean = float(np.std(mean_vals, ddof=1))
    err_var = float(np.std(var_vals, ddof=1))
    err_kurt = float(np.std(kurt_vals, ddof=1))

    return err_zeroth, err_mean, err_var, err_kurt


def calc_bootstrap_kde_moments_1d(data, bandwidth, n_bootstrap, n_samplesMC):
    """
    Estimate bootstrap errors for KDE-based 1D moments.

    Parameters
    ----------
    data : array_like
        One-dimensional sample array used to build the KDE.
    bandwidth : float
        Kernel bandwidth (standard deviation of the Gaussian kernel).
    n_bootstrap : int
        Number of bootstrap resamples.
    n_samplesMC : int
        Number of Monte Carlo samples per bootstrap draw used in
        :func:`calc_kde_moments_1d`.

    Returns
    -------
    err_zeroth : float
        Standard deviation of the bootstrap zeroth-moment estimates.
    err_mean : float
        Standard deviation of the bootstrap mean estimates.
    err_var : float
        Standard deviation of the bootstrap variance estimates.
    err_kurt : float
        Standard deviation of the bootstrap excess-kurtosis estimates.
    """

    data = np.asarray(data).ravel()
    n = data.size
    if n == 0 or n_bootstrap <= 1:
        return np.nan, np.nan, np.nan, np.nan

    zeroth_vals = np.empty(n_bootstrap)
    mean_vals = np.empty(n_bootstrap)
    var_vals = np.empty(n_bootstrap)
    kurt_vals = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idxs = np.random.choice(n, size=n, replace=True)
        resample = data[idxs]
        m0, m1, v, k = calc_kde_moments_1d(resample, bandwidth, n_samplesMC)
        zeroth_vals[i] = m0
        mean_vals[i] = m1
        var_vals[i] = v
        kurt_vals[i] = k

    err_zeroth = float(np.std(zeroth_vals, ddof=1))
    err_mean = float(np.std(mean_vals, ddof=1))
    err_var = float(np.std(var_vals, ddof=1))
    err_kurt = float(np.std(kurt_vals, ddof=1))

    return err_zeroth, err_mean, err_var, err_kurt


#############################################################################
### COMPUTE AND PRINT 1D MOMENTS
#############################################################################


def run_1D_momentCalculations(data, bandwidthMatrix, keys_flav, index, n_bootstrap=100, n_samplesMC=20000):
    """
    Compute and print 1D KDE and empirical moments for each flavour.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape ``(n_replicas, n_flavours)`` containing replica
        values for each selected flavour at a fixed grid index.
    bandwidthMatrix : numpy.ndarray
        Diagonal bandwidth matrix returned by :func:`calc_bandwidthMatrix`.
    keys_flav : sequence of str
        Flavour labels corresponding to the columns of ``data``.
    index : int
        Grid index at which the replicas were extracted (used for
        labelling the printed output).
    n_bootstrap : int, optional
        Number of bootstrap resamples used to estimate uncertainties.
    n_samplesMC : int, optional
        Number of Monte Carlo samples used for KDE moment estimation.
    """

    n_dims = data.shape[1]

    for dim in range(n_dims):
        key = keys_flav[dim] if dim < len(keys_flav) else f"dim{dim}"
        data_1d = data[:, dim]
        bandwidth = float(np.sqrt(bandwidthMatrix[dim, dim]))

        print(f"\n=== Flavour {key}, index {index} (dimension {dim}) ===")

        # KDE moments and bootstrap errors
        m0_kde, mean_kde, var_kde, kurt_kde = calc_kde_moments_1d(data_1d, bandwidth, n_samplesMC)
        err0_kde, err_mean_kde, err_var_kde, err_kurt_kde = calc_bootstrap_kde_moments_1d(
            data_1d, bandwidth, n_bootstrap, n_samplesMC
        )

        # Empirical moments and bootstrap errors
        m0_emp, mean_emp, var_emp, kurt_emp = empirical_moments_1d(data_1d)
        err0_emp, err_mean_emp, err_var_emp, err_kurt_emp = calc_bootstrap_empirical_moments_1d(
            data_1d, n_bootstrap
        )

        print("KDE moments:")
        print(f"  Zeroth: {m0_kde} +/- {err0_kde}")
        print(f"  Mean: {mean_kde} +/- {err_mean_kde}")
        print(f"  Variance: {var_kde} +/- {err_var_kde}")
        print(f"  Excess kurtosis: {kurt_kde} +/- {err_kurt_kde}")

        print("Empirical moments:")
        print(f"  Zeroth: {m0_emp} +/- {err0_emp}")
        print(f"  Mean: {mean_emp} +/- {err_mean_emp}")
        print(f"  Variance: {var_emp} +/- {err_var_emp}")
        print(f"  Excess kurtosis: {kurt_emp} +/- {err_kurt_emp}")


#############################################################################
### MAIN FUNCTION
#############################################################################


def main():
    """
    Entry point: compute and print 1D moments only (no plotting).

    This function wires together data loading, flavour/index selection,
    bandwidth determination and the moment/uncertainty calculations in
    :func:`run_1D_momentCalculations`. Edit ``keys_flav`` and ``index``
    below to change which 1D marginals are analysed.
    """

    res_flav, _ = read_in_data()

    # Choose flavours and grid index here
    keys_flav = ['d', 'u', 's', 'c', 'dbar', 'ubar', 'sbar', 'cbar', 'g']
    index = 28

    data = prepare_data(res_flav, keys_flav, index)
    bandwidthMatrix = calc_bandwidthMatrix(data)

    run_1D_momentCalculations(data, bandwidthMatrix, keys_flav, index)


if __name__ == "__main__":
    main()



