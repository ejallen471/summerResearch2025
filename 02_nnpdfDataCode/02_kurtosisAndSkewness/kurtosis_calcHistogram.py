#############################################################################

# File creates histogram from calculating the kurtosis for each 1D distribution
# and plot as a histogram

#############################################################################

import pickle
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from scipy.stats import norm, multivariate_normal
from scipy import stats

# --- If running on separate laptop/computer, this will need commenting out 
plt.style.use('pythonStyle')
import pythonStyle as ed

#############################################################################
#############################################################################

# --- Load serialised data
def read_in_data():
    paths = [Path("./flavour_basis.pkl"), Path("./evolution_basis.pkl")]
    return (pickle.load(open(p, 'rb')) for p in paths)

# --- prepare each dimension and bring together
def prepare_data(res, keys, indices):

    num_replicas = len(res)
    num_keys = len(keys)
    
    # Single index case, output 2D (num_replicas, num_keys)
    data_array = np.empty((num_replicas, num_keys), dtype=float)
    for i, replica in enumerate(res):
        for j, key in enumerate(keys):
            data_array[i, j] = replica[key][indices]

    return data_array

#############################################################################
### BUILDING KDE STUFF
#############################################################################

### --- helper function for calc_bandwidth_1d
def calc_kdeCrossValidation_1d(data, hLst, k=5, subsample_size=10000):
    """
    Cross-validation of 1D KDE bandwidths.
    """
    n = data.shape[0]
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mean_logLikelihoodLst = []

    # Subsampling for speed
    if n > subsample_size:
        indices = np.random.choice(n, subsample_size, replace=False)
        data_sub = data[indices]
    else:
        data_sub = data

    for h in hLst:
        norm_const = 1.0 / np.sqrt(2 * np.pi * h ** 2)
        fold_log_likelihoods = []

        for train_idx, val_idx in kf.split(data_sub):
            X_train = data_sub[train_idx]
            X_val = data_sub[val_idx]

            diffs = X_val[:, None] - X_train[None, :]  # shape (m, n)
            dists = diffs ** 2
            K = norm_const * np.exp(-0.5 * dists / h ** 2)

            f_vals = np.mean(K, axis=1)
            f_vals = np.clip(f_vals, 1e-300, None)
            fold_log_likelihoods.append(np.mean(np.log(f_vals)))

        mean_logLikelihoodLst.append(np.mean(fold_log_likelihoods))

    mean_logLikelihoodLst = np.array(mean_logLikelihoodLst)
    optimal_idx = np.argmax(mean_logLikelihoodLst)
    optimal_bandwidth = hLst[optimal_idx]

    return optimal_bandwidth

# --- calculate the bandwidth parameter
def calc_bandwidth_1d(data, n=100000):
    """
    Bandwidth selection using Silverman's rule and cross-validation for 1D data.
    """
    
    n = data.shape[0]
    sigma = np.std(data, ddof=1)
    h_p = (4 / (1 + 2)) ** (1 / (1 + 4)) * n ** (-1 / (1 + 4)) * sigma
    # print(f'Initial h_p: {h_p}')

    scaling_factors = np.linspace(0.5, 2.0, 10)
    hLst = scaling_factors * h_p

    bandwidth = calc_kdeCrossValidation_1d(data, hLst, k=5, subsample_size=10000)
    # print(f'Optimal bandwidth: {bandwidth}')

    return bandwidth

#############################################################################
### CALCULATE MOMENTS
#############################################################################

# ---------------------------------------------
# --- KDE PDF Evaluation Function
# ---------------------------------------------

# --- calculate the pdf estimate for a single point
def calc_pdf_pointwise(point, data, bandwidthParameter):
    d = data.shape[1] if data.ndim > 1 else 1
    n = data.shape[0]

    diffs = data - point
    D2 = (diffs / bandwidthParameter) ** 2
    norm_const = 1.0 / ((2 * np.pi) ** (d / 2) * bandwidthParameter)
    kernel_vals = np.exp(-0.5 * D2)
    return (1.0 / n) * np.sum(norm_const * kernel_vals)

# ---------------------------------------------
# --- Monte Carlo moment integration (Importance Sampling)
# ---------------------------------------------

# --- calculate first four normalised cumulants from KDE 
def calc_moments_importanceSampling(data, bandwidth, n_samplesMC):

    # Proposal distribution: Gaussian fit to the data
    mu = np.mean(data)
    var = np.var(data, ddof=1)

    samples = np.random.normal(mu, np.sqrt(var), size=n_samplesMC)
    q_vals = stats.norm.pdf(samples, loc=mu, scale=np.sqrt(var))  # shape (n_samples,)

    # Target distribution: KDE
    p_vals = np.array([calc_pdf_pointwise(point, data, bandwidth) for point in samples])  # shape (n_samples,)

    weights = p_vals / q_vals  # Importance weights

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
    excessKurtosis = (weighted_x4 / (variance**2)) - 3 # minus three makes it excess kurtosis (without is just kurtosis)

    return zeroth_moment, weighted_mean, variance, excessKurtosis

# --- calculate first four normalised cumulants direct from data
def calc_empiricalMoments(data):
    """
    Compute empirical mean, variance, and kurtosis (per dimension).
    Kurtosis = E[(x - mu)^4] / sigma^4

    Parameters:
        data : array of shape (n_samples, d)

    Returns:
        mean : shape (d,)
        variance : shape (d,)
        kurtosis : shape (d,)
    """
    mean = np.mean(data, axis=0)
    variance = np.var(data, axis=0, ddof=1)  
    centered = data - mean
    fourth_moment = np.mean(centered**4, axis=0)
    excessKurtosis = (fourth_moment / (variance**2)) - 3
    return mean, variance, excessKurtosis 

# ---------------------------------------------
# --- Errors ---
# ---------------------------------------------

# --- calculate errors via bootstrap for importance method 
def calc_bootstrap_error_mc_importance(data, bandwidth, n_bootstrap, n_samplesBootStrap):
    """
    Bootstrap standard error for importance sampling KDE moments.
    Calculates standard errors of zeroth moment, mean, variance, and kurtosis.
    """

    n = data.shape[0] 
    zeroth_moments = np.zeros(n_bootstrap)
    means = np.zeros(n_bootstrap)
    variances = np.zeros(n_bootstrap)
    kurtoses = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        idxs = np.random.choice(n, size=n, replace=True)
        resampled_data = data[idxs]

        zerothMoment, mean, variance, kurtosis = calc_moments_importanceSampling(resampled_data, bandwidth, n_samplesMC=n_samplesBootStrap)

        zeroth_moments[i] = zerothMoment
        means[i] = mean
        variances[i] = variance
        kurtoses[i] = kurtosis

    std_zeroth = np.std(zeroth_moments, ddof=1)
    std_mean = np.std(means, ddof=1)
    std_variance = np.std(variances, ddof=1)
    std_kurtosis = np.std(kurtoses, ddof=1)

    return std_zeroth, std_mean, std_variance, std_kurtosis

def bootstrap_moment_errors(data, n_bootstrap=1000, seed=None):
    """
    Compute bootstrap standard errors for mean, variance, and kurtosis.

    Parameters:
        data : array of shape (n_samples, d)
        n_bootstrap : number of bootstrap resamples
        seed : optional, for reproducibility

    Returns:
        std_err_mean : shape (d,)
        std_err_var : shape (d,)
        std_err_kurt : shape (d,)
    """
    rng = np.random.default_rng(seed)
    n_samples, d = data.shape

    means = np.zeros((n_bootstrap, d))
    vars_ = np.zeros((n_bootstrap, d))
    kurts = np.zeros((n_bootstrap, d))

    for i in range(n_bootstrap):
        sample_indices = rng.integers(0, n_samples, size=n_samples)
        sample = data[sample_indices]
        mean, var, kurt = calc_empiricalMoments(sample)
        means[i] = mean
        vars_[i] = var
        kurts[i] = kurt

    std_err_mean = np.std(means, axis=0, ddof=1)
    std_err_var = np.std(vars_, axis=0, ddof=1)
    std_err_kurt = np.std(kurts, axis=0, ddof=1)

    return std_err_mean, std_err_var, std_err_kurt 

# --- Function to run the moments calculations
def run_1D_momentCalculations(data, bandwidthMatrix, n_bootstrap=250, n_samplesMC=10000):
    
    # --- KDE Integration to calculate moments
    zerothMoment, firstMomentVec, varianceVec, kurtosisVec = calc_moments_importanceSampling(data, bandwidthMatrix, n_samplesMC=n_samplesMC)
    error_zeroth_is, bootstrapError_mean_is, bootstrapError_variance_is, bootstrap_kurtosis_is = calc_bootstrap_error_mc_importance(data, bandwidthMatrix, n_bootstrap, n_samplesMC)

    # --- Empirical moments to calculate moments
    empirical_mean, empirical_variance, empirical_kurtosis = calc_empiricalMoments(data)
    bootstrapError_mean, bootstrapError_variance, bootstrap_kurtosis =  bootstrap_moment_errors(data)

    # print("\n--- Moments ---")
    # print(f"Zeroth (KDE): {zerothMoment} with error {error_zeroth_is}\n")
    
    # print(f"Mean (KDE): {firstMomentVec} with error {bootstrapError_mean_is}")
    # print(f"Mean (Empirical): {empirical_mean} with error {bootstrapError_mean}\n")

    # print(f"Variance (KDE): {varianceVec} with error {bootstrapError_variance_is}")
    # print(f"Variance (Empirical): {empirical_variance} with error {bootstrapError_variance}\n")

    # print(f"Excess kurtosis (KDE): {kurtosisVec} with error {bootstrap_kurtosis_is}\n")
    # print(f"Excess kurtosis  (Empirical): {empirical_kurtosis} with error {bootstrap_kurtosis}\n")

    return kurtosisVec, bootstrap_kurtosis_is, empirical_kurtosis, bootstrap_kurtosis

#############################################################################
#############################################################################

# ---------------------------------------------
# --- Graphs etc of 450 dimensions stuff
# ---------------------------------------------

# --- Filter to set cut-off point
def filter_kurtosis_data(kurtosis_list, error_list, max_kurtosis=5):
    """Filters kurtosis values and corresponding errors based on a threshold (defined by max_kurtosis)."""
    
    filtered_kurtosis = []
    filtered_errors = []
    
    for k, err in zip(kurtosis_list, error_list):
        if k <= max_kurtosis:
            filtered_kurtosis.append(k)
            filtered_errors.append(err)
    
    return filtered_kurtosis, filtered_errors

# --- plot big histogram of kurtosis
def plot_kurtosis_histograms(KDE_data, empirical_data):
    """Plots histograms of filtered kurtosis data."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].hist(KDE_data, bins=100, color="#68A5A1", edgecolor='black')
    axs[0].set_xlabel('Excess Kurtosis')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Integral Method', fontsize=12)
    axs[0].tick_params(axis='both')  

    axs[1].hist(empirical_data, bins=100, color="#68A5A1", edgecolor='black')
    axs[1].set_xlabel('Excess Kurtosis')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Empirical Method', fontsize=12)
    axs[1].tick_params(axis='both')  

    plt.tight_layout()
    plt.savefig("histogram_kurtosis.png", dpi=300)
    plt.show()

#############################################################################
#############################################################################

# --- Main Function
def main():
    res_flav, res_ev = read_in_data()

    # keys_ev = ['Sigma', 'V', 'V3', 'V8', 'T3', 'T8', 'c+', 'g', 'V15']
    
    ## --- Change here for number of indices and flavours to include
    keys_flav = ['d'] # 'u', 's', 'c', 'dbar', 'ubar', 'sbar', 'cbar', 'g']
    indices = np.arange(5)
    
    kurtosis_KDE_Lst = []
    kurtosis_KDE_errorLst = []
    kurtosis_empirical_Lst = []
    kurtosis_empirical_errorLst = []

    total = len(keys_flav) * len(indices)
    count = 0

    # Loop and data collection
    for key in keys_flav:
        for index in indices:
            count += 1
            print(f'Processing {count} / {total}  (key={key}, index={index})')        
            
            # fetch data
            data = prepare_data(res_flav, [key], index)

            # Run analysis
            bandwidthParameter = calc_bandwidth_1d(data)
            kurtosis_KDE, kurtosisError_KDE, kurtosis_empirical, kurtosisError_empirical = run_1D_momentCalculations(data, bandwidthParameter)

            # Collect results
            kurtosis_KDE_Lst.append(kurtosis_KDE)  
            kurtosis_KDE_errorLst.append(kurtosisError_KDE)
            kurtosis_empirical_Lst.append(float(kurtosis_empirical))
            kurtosis_empirical_errorLst.append(float(kurtosisError_empirical))

    # Filter results
    filtered_KDE, _ = filter_kurtosis_data(kurtosis_KDE_Lst, kurtosis_KDE_errorLst)
    filtered_empirical, _ = filter_kurtosis_data(kurtosis_empirical_Lst, kurtosis_empirical_errorLst)
    plot_kurtosis_histograms(filtered_KDE, filtered_empirical)

if __name__ == "__main__":
    main()



