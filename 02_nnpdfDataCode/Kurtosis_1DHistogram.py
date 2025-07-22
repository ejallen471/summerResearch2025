import pickle
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from scipy.stats import norm, multivariate_normal

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
def prepare_2d_data(res, key_x, key_y, index):
    return np.array([[r[key_x][index], r[key_y][index]] for r in res])

# --- prepare each dimension and bring together
def prepare_data(res, keys, indices):

    num_replicas = len(res)
    num_keys = len(keys)
    
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
#############################################################################

# ---------------------------------------------
# --- Create KDE Model
# ---------------------------------------------

# --- Cross-validation of KDE bandwidth matrix- using subsampling for speed
def calc_kdeCrossValidation_nD(data, H_Matrix_candidateLst, k=5, subsample_size=10000):
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

# --- Plot the 1D histograms - plotting1D must == True for this to occur
def plot_histograms_with_pdf(data, bandwidthMatrix, dim, bins=50):
    # Extract 1D data for the selected dimension
    data_1d = data[:, dim]
    # print(dim)

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

    # Calculate the KDE estimate
    kde_vals = np.zeros_like(x_vals)
    for i, x in enumerate(x_vals):
        diff = (x - data_1d) / h
        kde_vals[i] = np.mean(np.exp(-0.5 * diff**2)) / (np.sqrt(2 * np.pi) * h)

    # Plot histogram with empirical PDF and KDE estimate
    plt.figure(figsize=(8, 5))
    plt.hist(data_1d, bins=bins, density=True, color='skyblue', edgecolor='black', alpha=0.6)
    plt.plot(x_vals, pdf_x, lw=2, label="Empirical PDF")
    plt.plot(x_vals, kde_vals, '--', lw=2, label="KDE Estimate PDF")
    plt.xlabel(f"Dimension {dim}", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title("1D Histogram with Empirical PDF and KDE Estimate", fontsize=14)
    plt.legend(fontsize=12)
    plt.show()

# --- Calculate KL divergence in 1D
def oneDimensionalKLDivergence(data, kde_vals_x, kde_vals_y, pdf_x, pdf_y):

    # Assume uniform spacing
    x_vals = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), len(kde_vals_x))
    y_vals = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), len(kde_vals_y))
    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]

    # KL divergence: KDE vs True
    kl_x = np.sum(kde_vals_x * np.log(kde_vals_x / pdf_x)) * dx
    kl_y = np.sum(kde_vals_y * np.log(kde_vals_y / pdf_y)) * dy

    # print stuff
    print(f"KL divergence (X marginal): {kl_x:.6f}")
    print(f"KL divergence (Y marginal): {kl_y:.6f}")

# --- Plot the 2D KDE and empirical PDF estimate - choose dimensions to plot 
def plot_kde_vs_pdf_2d(data, kde_vals, pdf_vals, grid_points, xLabel, yLabel):
    x_unique = np.unique(grid_points[:, 0])
    y_unique = np.unique(grid_points[:, 1])
    X, Y = np.meshgrid(x_unique, y_unique)

    # Downsample data for scatter plot if large
    plot_data = data if data.shape[0] <= 10000 else data[::10]

    plt.scatter(plot_data[:, 0], plot_data[:, 1], c='dimgrey', s=10, alpha=0.3, label='Samples')
    plt.contour(X, Y, kde_vals, colors='navy', linewidths=1.5)
    plt.contour(X, Y, pdf_vals, colors='firebrick', linestyles='dashed', linewidths=1.5)

    legend_elements = [
        Line2D([0], [0], color='navy', lw=1.5, label='KDE pdf'),
        Line2D([0], [0], color='firebrick', lw=1.5, linestyle='dashed', label='True pdf'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=6, label='Samples', alpha=0.5)
    ]

    plt.legend(handles=legend_elements, fontsize=12)
    plt.title(f'KDE vs Analytical PDF', fontsize=16)
    plt.xlabel(f'{xLabel}', fontsize=14)
    plt.ylabel(f'{yLabel}', fontsize=14)
    plt.grid(True)
    plt.tight_layout()    
    plt.show()

    mae = np.mean(np.abs(kde_vals - pdf_vals))
    print(f"MAE between KDE and Analytical PDF: {mae:.6f}")

#############################################################################
#############################################################################

# ---------------------------------------------
# --- KDE PDF Evaluation
# ---------------------------------------------

# --- Evaulate the KDE PDF for a single point (x,y) in 1D
def calc_pdf_pointwise_1D(point, data, bandwidthMatrix):
    """
    Evaluate KDE PDF at a single 1D point.

    point: (1,) or scalar - evaluation point
    data: (n, 1) array of data points
    bandwidthMatrix: (1,1) bandwidth matrix (scalar bandwidth squared)

    Returns:
        pdf_val: scalar KDE value at point
    """
    x = point

    # Extract bandwidth scalar
    h = bandwidthMatrix[0, 0]

    # Number of data points
    n = data.shape[0]

    # Compute differences (x - x_i)
    diffs = x - data[:, 0]  # shape (n,)

    # Gaussian kernel evaluations
    normConst = 1 / np.sqrt(2 * np.pi * h)
    kernel_vals = normConst * np.exp(-0.5 * ((diffs)** 2) / h)

    # KDE estimate is average kernel values
    pdf_val = np.sum(kernel_vals) / n

    return pdf_val

# ---------------------------------------------
# --- Integration
# ---------------------------------------------

# --- Evaulate the integral via grid evaluation
def calc_moment_integral_GridEvaluation_1d(data, bandwidth, order, grid, differentialElement):
    """
    Calculate moment integrals of KDE in 1D by grid evaluation.

    data: (n,) 1D data array
    bandwidth: scalar bandwidth variance (h)
    order: integer moment order (0 for zeroth moment, 1 for first, etc.)
    grid: 1D array of points at which to evaluate KDE
    differentialElement: scalar representing the grid spacing (dx)
    """
    # Evaluate KDE at grid points
    f_hats = np.empty_like(grid)
    for i, pt in enumerate(grid):
        f_hats[i] = calc_pdf_pointwise_1D(pt, data[:, None], bandwidth)
    
    if order == 0:
        # Zeroth moment: integral over KDE (scalar)
        return np.sum(f_hats) * differentialElement
    else:
        # Higher moments: integral of x^order * KDE(x)
        moment = np.sum((grid ** order) * f_hats) * differentialElement
        return moment

# ---------------------------------------------
# --- Bootstrap errors
# ---------------------------------------------

# --- Evaulate the error on the integral values via bootstrap
def calc_bootstrap_error_variance_kurtosis_1d(data, bandwidth, grid, differentialElement, n_bootstrap=100):

    n = len(data)
    zeorth_estimates = np.empty(n_bootstrap)
    mean_estimates = np.empty(n_bootstrap)
    var_estimates = np.empty(n_bootstrap)
    kurt_estimates = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        resample = np.random.choice(data, size=n, replace=True)

        # Moments from KDE on resample
        M0 = calc_moment_integral_GridEvaluation_1d(resample, bandwidth, 0, grid, differentialElement)
        M1 = calc_moment_integral_GridEvaluation_1d(resample, bandwidth, 1, grid, differentialElement)
        M2 = calc_moment_integral_GridEvaluation_1d(resample, bandwidth, 2, grid, differentialElement)
        M3 = calc_moment_integral_GridEvaluation_1d(resample, bandwidth, 3, grid, differentialElement)
        M4 = calc_moment_integral_GridEvaluation_1d(resample, bandwidth, 4, grid, differentialElement)

        var = M2 - M1**2
        kurt = (M4 - 4*M3*M1 + 6*M2*(M1**2) - 3*(M1**4)) / var**2

        zeorth_estimates[i] = M0
        mean_estimates[i] = M1
        var_estimates[i] = var
        kurt_estimates[i] = kurt

    # Compute the error
    zeorth_std = np.std(zeorth_estimates, ddof=1)
    mean_std = np.std(mean_estimates, ddof=1)
    var_std = np.std(var_estimates, ddof=1)
    kurt_std = np.std(kurt_estimates, ddof=1)

    return zeorth_std, mean_std, var_std, kurt_std

# --- Evaulate the error on the empricial values via bootstrap
def calc_bootstrap_error_empirical(data, stat_func, n_bootstrap):
    n = data.shape[0]
    indices = np.random.choice(n, size=(n_bootstrap, n), replace=True)
    samples = data[indices]
    stats = np.array([stat_func(sample) for sample in samples])
    
    # stats shape: (n_bootstrap,) if stat_func returns scalar
    # std with ddof=1 to get unbiased estimator
    error = np.std(stats, ddof=1)
    
    return error

# --- Function to pass into calc_bootstrap_error_empirical that calculates the mean
def mean_stat(data_sample):
    return np.mean(data_sample, axis=0)

# --- Function to pass into calc_bootstrap_error_empirical that calculates the variance
def variance_stat(data_sample):
    return np.var(data_sample, axis=0, ddof=1)

# --- Function to pass into calc_bootstrap_error_empirical that calculates the kurtosis
def kurtosis_stat(data_sample):
    empirical_mean = np.mean(data_sample)
    return np.mean((data_sample - empirical_mean) ** 4) / (np.std(data_sample, ddof=1) ** 4) - 3

# --- Function to run the moments calculations
def run_1D_momentCalculations(data, bandwidthMatrix, n_bootstrap=50, n_gridIntegralResolution=1000):

    n_dims = data.shape[1]

    # intialise for returning
    kurtosisLst_integral = []
    kurtosisErrorLst_integral = []
    kurtosisLst_empirical = []
    kurtosisErrorLst_empirical = []

    for dim in range(n_dims-1):
        print(f"\n--- Dimension {dim + 1} ---")

        data_min = np.min(data[:,dim])
        data_max = np.max(data[:,dim])
        data_std = np.std(data[:,dim])

        grid_min = data_min - 3 * data_std
        grid_max = data_max + 3 * data_std

        grid = np.linspace(grid_min, grid_max, n_gridIntegralResolution)
        dx = grid[1] - grid[0]

        # Zeroth moment
        M0 = calc_moment_integral_GridEvaluation_1d(data[:,dim], bandwidthMatrix, 0, grid, dx)
        M1 = calc_moment_integral_GridEvaluation_1d(data[:,dim], bandwidthMatrix, 1, grid, dx)
        M2 = calc_moment_integral_GridEvaluation_1d(data[:,dim], bandwidthMatrix, 2, grid, dx)
        M3 = calc_moment_integral_GridEvaluation_1d(data[:,dim], bandwidthMatrix, 3, grid, dx)
        M4 = calc_moment_integral_GridEvaluation_1d(data[:,dim], bandwidthMatrix, 4, grid, dx)
        
        # Calculate variance
        variance = M2 - M1**2
        
        # Calculate fourth cumulant
        fourth_cumulant = M4 - 4*M3*M1 + 6*M2*(M1**2) - 3*(M1**4)

        # Calculate kurtosis (excess kurtosis)
        kurtosis = fourth_cumulant / (variance**2) - 3
        kurtosisLst_integral.append(kurtosis)

        # Empirical moments
        empirical_mean = np.mean(data[:,dim])
        empirical_variance = np.var(data[:,dim], ddof=1)
        empirical_kurtosis = np.mean((data[:,dim] - empirical_mean) ** 4) / (np.std(data[:,dim], ddof=1) ** 4) - 3
        kurtosisLst_empirical.append(empirical_kurtosis)

        # Empirical bootstrap errors 
        bootstrapError_mean = calc_bootstrap_error_empirical(data, mean_stat, n_bootstrap)
        bootstrapError_variance = calc_bootstrap_error_empirical(data, variance_stat, n_bootstrap)
        bootstrapError_kurtosis = calc_bootstrap_error_empirical(data, kurtosis_stat, n_bootstrap)
        kurtosisErrorLst_empirical.append(bootstrapError_kurtosis)

        # Integral bootstrap errors 
        bootstrapError_zeroth_is, bootstrapError_mean_is, bootstrapError_variance_is, bootstrapError_kurtosis_is = calc_bootstrap_error_variance_kurtosis_1d(data[:,dim], bandwidthMatrix, grid=grid, differentialElement=dx, n_bootstrap=100)
        kurtosisErrorLst_integral.append(bootstrapError_kurtosis_is)

        print("\n--- Moments ---")
        print(f"Zeroth (KDE): {M0} with error {bootstrapError_zeroth_is}\n")

        print(f"Mean (KDE): {M1} with error {bootstrapError_mean_is}")
        print(f"Mean (Empirical): {empirical_mean} with error {bootstrapError_mean}\n")

        print(f"Variance (KDE): {variance} with error {bootstrapError_variance_is}")
        print(f"Variance (Empirical): {empirical_variance} with error {bootstrapError_variance}\n")

        print(f"Kurtosis (KDE): {kurtosis} with error {bootstrapError_kurtosis_is}")
        print(f"Kurtosis (Empirical): {empirical_kurtosis} with error {bootstrapError_kurtosis}\n")
    
    return kurtosisLst_integral, kurtosisErrorLst_integral, kurtosisLst_empirical, kurtosisErrorLst_empirical

#############################################################################
#############################################################################

# ---------------------------------------------
# --- Graphs etc of 450 dimensions stuff
# ---------------------------------------------

def filter_kurtosis_data(kurtosis_list, error_list, max_kurtosis=3):
    """Filters kurtosis values and corresponding errors based on a threshold."""
    filtered_kurtosis = []
    filtered_errors = []
    for k, err in zip(kurtosis_list, error_list):
        if k <= max_kurtosis:
            filtered_kurtosis.append(k)
            filtered_errors.append(err)
    return filtered_kurtosis, filtered_errors

def plot_kurtosis_histograms(integral_data, empirical_data):
    """Plots histograms of filtered kurtosis data."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].hist(integral_data, bins=100, color="#68A5A1", edgecolor='black')
    axs[0].set_xlabel('Excess Kurtosis', fontsize=12)
    axs[0].set_ylabel('Frequency', fontsize=12)
    # axs[0].set_title('Histogram of Kurtosis Values (Integral Method)', fontsize=12)
    axs[0].tick_params(axis='both', labelsize=12)  


    axs[1].hist(empirical_data, bins=100, color="#68A5A1", edgecolor='black')
    axs[1].set_xlabel('Excess Kurtosis', fontsize=12)
    axs[1].set_ylabel('Frequency', fontsize=12)
    # axs[1].set_title('Histogram of Kurtosis Values (Empirical Method)', fontsize=12)
    axs[1].tick_params(axis='both', labelsize=12)  

    plt.tight_layout()
    plt.savefig("histogram_kurtosis.png", dpi=300)
    plt.show()

def save_kurtosis_results(integral_kurtosis, integral_errors, empirical_kurtosis, empirical_errors, filename="kurtosis_results.txt"):
    """Saves kurtosis data and errors to a text file."""
    with open(filename, 'w') as f:
        f.write(f'kurtosisLst1_integral: {integral_kurtosis}\n')
        f.write(f'kurtosisErrorLst1_integral: {integral_errors}\n')
        f.write(f'kurtosisLst1_empirical: {empirical_kurtosis}\n')
        f.write(f'kurtosisErrorLst1_empirical: {empirical_errors}\n')

def plot_kurtosis_with_errors(integral_data, integral_errors, empirical_data, empirical_errors):
    """Plots kurtosis values with error bars."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    ax1.errorbar(range(len(integral_data)), integral_data, yerr=integral_errors, fmt='o', capsize=5)
    ax1.axhline(0, linestyle='--', linewidth=1)
    ax1.set_title('Integral Kurtosis')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Integral Kurtosis')
    ax1.grid(True)

    ax2.errorbar(range(len(empirical_data)), empirical_data, yerr=empirical_errors, fmt='s', capsize=5)
    ax2.axhline(0, linestyle='--', linewidth=1)
    ax2.set_title('Empirical Kurtosis')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Empirical Kurtosis')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

#############################################################################
#############################################################################

# --- Main analysis pipeline
def run_gaussian_kde_analysis(data, keys_test, n=100000, kdeGridRes=150, plotting1D=False, plotting2D=False):
    
    # --- BUILD BANDWIDTH MATRIX - we assume no co-variance
    n, d = data.shape
    
    # Calculate Silverman bandwidth vector
    sigma = np.std(data, axis=0, ddof=1)
    h_p = (4 / (d + 2)) ** (1 / (d + 4)) * n ** (-1 / (d + 4)) * sigma
    print(f'Initial h_p: {h_p}')

    # Create candidate bandwidth matrices through scaling factors
    scaling_factors = np.linspace(0.5, 2.0, 10)

    H_Matrix_candidateLst = []
    hLst = []
    for s in scaling_factors:
        H_diag = (s * h_p) ** 2 
        H_matrix = np.diag(H_diag)
        H_Matrix_candidateLst.append(H_matrix)
        hLst.append(H_diag)

    # Cross-validation
    bandwidthMatrix, mean_logLikelihoodLst = calc_kdeCrossValidation_nD(data, H_Matrix_candidateLst, k=5, subsample_size=10000)

    # --- PLOT KDE ESTIMATES IN 1D 
    if plotting1D == True:
        for i in range(0, d):
            plot_histograms_with_pdf(data, bandwidthMatrix, dim=i)

    # --- EVAULATE AND PLOT KDE AND SAMPLE PDF IN 2D
    if plotting2D == True:
        
        # state dimensions to plot
        x_dimension_idx = 0
        y_dimension_idx = 1 
        print(f'Plotting Dimensions {x_dimension_idx} and {y_dimension_idx}')

        # Extract data for only the two selected dimensions
        data_2d = data[:, [x_dimension_idx, y_dimension_idx]]
        n_dims_2d = 2

        # Create 2D grid only for selected dims
        grid_axes_2d = [np.linspace(np.min(data_2d[:, dim]) - 1, np.max(data_2d[:, dim]) + 1, kdeGridRes) for dim in range(n_dims_2d)]

        # Create 2D meshgrid and flatten
        X, Y = np.meshgrid(grid_axes_2d[0], grid_axes_2d[1], indexing='xy')
        grid_points_2d = np.vstack([X.ravel(), Y.ravel()]).T.astype(np.float32)  # shape (num_points, 2)

        # Compute 2D bandwidth matrix for selected dims from the full bandwidth
        optimalBandwidthMatrix_2d = bandwidthMatrix[np.ix_([x_dimension_idx, y_dimension_idx], [x_dimension_idx, y_dimension_idx])]

        # Compute KDE only on selected dims
        kde_vals_2d = calc_kdeGaussianEstimate_nD(grid_points_2d, data_2d, optimalBandwidthMatrix_2d).reshape(kdeGridRes, kdeGridRes)

        # Calculate sample PDF for selected dims
        sampleMean_2d = np.mean(data_2d, axis=0)
        sampleCovariance_2d = np.cov(data_2d, rowvar=False)
        sampleGaussian_2d = multivariate_normal(mean=sampleMean_2d, cov=sampleCovariance_2d)
        pdf_vals_2d = sampleGaussian_2d.pdf(grid_points_2d).reshape(kdeGridRes, kdeGridRes)

        # Plot with the 2D inputs    
        plot_kde_vs_pdf_2d(data_2d, kde_vals_2d, pdf_vals_2d, grid_points_2d, xLabel=keys_test[x_dimension_idx], yLabel=keys_test[y_dimension_idx])

    return data, bandwidthMatrix


# --- Main Function
def main():
    res_flav, res_ev = read_in_data()

    keys_ev = ['Sigma', 'V', 'V3', 'V8', 'T3', 'T8', 'c+', 'g', 'V15']
    keys_flav = ['d', 'u', 's', 'c', 'dbar', 'ubar', 'sbar', 'cbar', 'g']
    keys_test = ['u', 'g']
    indices = np.arange(50)
    # index = 21
    
    # data1 = prepare_data(res_flav, keys_test, index)
    # data1, bandwidthMatrix = run_gaussian_kde_analysis(data1, keys_test )
    # kurtosisLst1_integral, kurtosisErrorLst1_integral,  kurtosisLst1_empirical, kurtosisErrorLst1_empirical = run_1D_momentCalculations(data1, bandwidthMatrix)

    kurtosisLst1_integralLst = []
    kurtosisErrorLst1_integralLst = []
    kurtosisLst1_empiricalLst = []
    kurtosisErrorLst1_empiricalLst = []

    for selected_index in indices:
        print(f'*** INDEX {selected_index} ***')
        data1 = prepare_data(res_flav, keys_flav, int(selected_index))

        # Run analysis
        data1, bandwidthMatrix = run_gaussian_kde_analysis(data1, keys_flav)
        kurtosisLst1_integral, kurtosisErrorLst1_integral, kurtosisLst1_empirical, kurtosisErrorLst1_empirical = run_1D_momentCalculations(data1, bandwidthMatrix)

        # Collect results
        kurtosisLst1_integralLst.extend(kurtosisLst1_integral)  
        kurtosisErrorLst1_integralLst.extend(kurtosisErrorLst1_integral)
        kurtosisLst1_empiricalLst.extend(kurtosisLst1_empirical)
        kurtosisErrorLst1_empiricalLst.extend(kurtosisErrorLst1_empirical)

    # --- Run plots etc
    filtered_integral, filtered_integral_errors = filter_kurtosis_data(kurtosisLst1_integralLst, kurtosisErrorLst1_integralLst)
    filtered_empirical, filtered_empirical_errors = filter_kurtosis_data(kurtosisLst1_empiricalLst, kurtosisErrorLst1_empiricalLst)

    plot_kurtosis_histograms(filtered_integral, filtered_empirical)
    save_kurtosis_results(kurtosisLst1_integralLst, kurtosisErrorLst1_integralLst, kurtosisLst1_empiricalLst, kurtosisErrorLst1_empiricalLst)
    plot_kurtosis_with_errors(filtered_integral, filtered_integral_errors,filtered_empirical, filtered_empirical_errors)


if __name__ == "__main__":
    main()



