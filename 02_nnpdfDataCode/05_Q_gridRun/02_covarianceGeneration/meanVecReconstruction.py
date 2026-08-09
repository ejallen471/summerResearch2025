"""
meanVecReconstruction.py

Build 1D mean vectors per flavour and grid point for KDE-based and
empirical reconstructions.

This module provides:
- utilities to read flavour replica data;
- a simple 1D bandwidth estimator (`estimate_bandwidth_matrix_scv_1d`);
- a numba-accelerated batch KDE PDF evaluator used by the 1D
    importance-sampling estimator;
- `construct_mean_vectors_1d` which returns both KDE and empirical mean
    vectors and writes them to CSV when run as a script.

Outputs
-------
- `mean_vector_kde.csv` — mean vector estimated by 1D KDE + importance
    sampling.
- `mean_vector_empirical.csv` — empirical mean vector computed directly
    from replicas.

"""

import pickle
import numpy as np
from pathlib import Path
import numba

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

#############################################################################
# --- Data preparation
#############################################################################

def read_flavour_data():
    """
    Find the first pickle file starting with 'flavour_basis' in the current directory
    and load the replica data.

    Returns
    -------
    object
        Deserialised flavour replica object loaded from pickle.
    """
    path = next(Path(".").glob("flavour_basis*.pkl"))
    return pickle.load(open(path, 'rb'))

def prepare_data_fixedIndex(res, key, idx):
    """
    Extract data for a single flavour at a single grid index across replicas.

    Parameters
    ----------
    res : list
        List of replica dicts.
    key : str
        Flavour key to extract.
    idx : int
        Grid index to extract.

    Returns
    -------
    ndarray
        Array of shape (n_replicas, 1).
    """
    data = np.array([r[key][idx] for r in res]).reshape(-1,1)
    return data

#############################################################################
# --- 1D KDE bandwidth estimation
#############################################################################

def estimate_bandwidth_matrix_scv_1d(data):
    """
    1D KDE bandwidth estimation.

    Returns
    -------
    ndarray
        1x1 bandwidth matrix (variance of the kernel).
    """
    n = data.shape[0]
    sigma = np.var(data, ddof=1)
    h = (4/3)**(1/5) * n**(-1/5) * sigma
    return np.array([[h**2]])

#############################################################################
# --- KDE mean computation via Monte Carlo
#############################################################################

@numba.njit
def calc_pdf_batch_numba(points, data, H_inv, norm_const):
    """
    Evaluate KDE PDF values for a batch of `points` given data and an
    inverse bandwidth matrix.

    Parameters
    ----------
    points : ndarray
        Array of shape (m, d) containing query points.
    data : ndarray
        Array of shape (n, d) with KDE data points.
    H_inv : ndarray
        Inverse of the bandwidth matrix (d x d).
    norm_const : float
        Normalisation constant for the Gaussian kernel.

    Returns
    -------
    pdf_vals : ndarray
        Array of length m with estimated PDF values at each query point.
    """
    m, d = points.shape
    n = data.shape[0]
    pdf_vals = np.zeros(m)
    for i in range(m):
        diff = points[i] - data
        maha = np.sum(diff @ H_inv * diff, axis=1)
        pdf_vals[i] = norm_const * np.mean(np.exp(-0.5 * maha))
    return pdf_vals

def calc_mean_KDE_1d(data, bandwidthMatrix, n_samplesMC=10000):
    """
    Compute KDE mean for 1D data via importance sampling.

    Returns
    -------
    ndarray
        Estimated mean as a (1,) array or NaN on failure.
    """
    mu = np.mean(data, axis=0)
    cov_proposal = np.var(data, ddof=1).reshape(1,1)
    samples = np.random.normal(loc=mu, scale=np.sqrt(cov_proposal[0,0]), size=(n_samplesMC, 1))

    H_inv = np.linalg.inv(bandwidthMatrix)
    det_H = np.linalg.det(bandwidthMatrix)
    norm_const = 1.0 / np.sqrt(2*np.pi * det_H)

    p_vals = calc_pdf_batch_numba(samples, data, H_inv, norm_const)

    # Proposal PDF (normal)
    q_vals = (1/np.sqrt(2*np.pi*cov_proposal[0,0])) * np.exp(-0.5 * ((samples - mu)/np.sqrt(cov_proposal[0,0]))**2).flatten()
    weights = np.where(q_vals>0, p_vals / q_vals, 0.0)
    weight_sum = np.sum(weights)

    if weight_sum <= 1e-10:
        return np.array([np.nan])

    return np.sum(weights[:,None]*samples, axis=0)/weight_sum

#############################################################################
# --- Construct mean vectors for PDF (1D only)
#############################################################################

def construct_mean_vectors_1d(res_flav, keys_flav, numberOfGridPoints=45, n_samplesMC=10000):
    """
    Build the mean vectors using 1D KDE and empirical for comparison.

    Parameters
    ----------
    res_flav : list
        Replica flavour data.
    keys_flav : list
        Flavour keys.
    numberOfGridPoints : int, optional
        Number of grid points per flavour.
    n_samplesMC : int, optional
        Number of Monte Carlo samples for KDE mean estimation.

    Returns
    -------
    mean_vector_kde : ndarray
        Shape (n_flavours * numberOfGridPoints,)
    mean_vector_emp : ndarray
        Shape (n_flavours * numberOfGridPoints,)
    """
    n_flav = len(keys_flav)
    mean_vector_kde = np.zeros(n_flav * numberOfGridPoints)
    mean_vector_emp = np.zeros(n_flav * numberOfGridPoints)

    for f_idx, flav in enumerate(keys_flav):
        for g_idx in range(numberOfGridPoints):
            data = prepare_data_fixedIndex(res_flav, flav, g_idx)
            bandwidth_matrix = estimate_bandwidth_matrix_scv_1d(data)

            # --- KDE mean
            mean_vec_kde = calc_mean_KDE_1d(data, bandwidth_matrix, n_samplesMC)
            mean_vector_kde[f_idx*numberOfGridPoints + g_idx] = mean_vec_kde[0]

            # --- Empirical mean
            mean_vector_emp[f_idx*numberOfGridPoints + g_idx] = np.mean(data)

    return mean_vector_kde, mean_vector_emp

#############################################################################
# --- Main
#############################################################################

def main():
    """
    Driver routine: build mean vectors and save them to CSV files.

    Behaviour
    ---------
    - Loads the first `flavour_basis*.pkl` file from the current
        directory;
    - Constructs KDE and empirical mean vectors using
        `construct_mean_vectors_1d`;
    - Saves `mean_vector_kde.csv` and `mean_vector_empirical.csv`.
    """
    res_flav = read_flavour_data()

    keys_flav = ['d','u','s','c','dbar','ubar','sbar','cbar','g']
    numberOfGridPoints = 45

    mean_vector_kde, mean_vector_emp = construct_mean_vectors_1d(res_flav, keys_flav, numberOfGridPoints)

    try:
        np.savetxt("mean_vector_kde.csv", mean_vector_kde, delimiter=",", fmt="%.6e")
        np.savetxt("mean_vector_empirical.csv", mean_vector_emp, delimiter=",", fmt="%.6e")
    except Exception as e:
        print(f"Error saving files: {e}")
    else:
        print("Mean vector KDE and Mean vector Empirical successfully saved to CSV files")


if __name__ == "__main__":
    main()
