"""
Helpers to compare KDE-derived and LHAPDF-derived PDF uncertainties.

This module loads per-flavour mean and standard-deviation CSVs produced
by the KDE pipeline and by LHAPDF evaluations, and provides plotting
utilities that compare the two sets of uncertainties. The primary
comparisons are the normalised difference in standard deviation and the
difference expressed in units of expected statistical fluctuation.

The functions in this file are intended for interactive inspection and
are executed when the module is run as a script.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('../../pythonStyle.mplstyle')

#############################################################################
### READ IN DATA
#############################################################################

def read_in_kde(mean_dir="KDE_mean", std_dir="KDE_std"):
    """
    Load KDE mean and standard-deviation CSVs into a nested mapping.

    Parameters
    ----------
    mean_dir : str, optional
        Directory containing per-flavour mean CSV files. Default is
        ``'KDE_mean'``.
    std_dir : str, optional
        Directory containing per-flavour std CSV files. Default is
        ``'KDE_std'``.

    Returns
    -------
    dict
        Mapping ``Q_string`` -> ``{'mean': {flavour: array}, 'std': {flavour: array}}``.

    Notes
    -----
    Filenames are expected in the format ``mean_<flavour>_Q=<Q>.csv``
    and ``std_<flavour>_Q=<Q>.csv`` where ``<Q>`` is a numeric value in
    scientific notation. Keys are normalised to the LHAPDF-style
    scientific format (``'%.6e'``).
    """
    pattern = r"^(mean|std)_([a-zA-Z]+)_Q=([0-9.eE+-]+)\.csv$"

    data_dict = {}

    # Map std filenames for quick lookup
    std_files = {f: os.path.join(std_dir, f) for f in os.listdir(std_dir)}

    for file_name in os.listdir(mean_dir):
        match = re.match(pattern, file_name)
        if not match:
            continue

        _, flavour, Q_str = match.groups()
        key = f"{float(Q_str):.6e}"

        if key not in data_dict:
            data_dict[key] = {"mean": {}, "std": {}}

        mean_file = os.path.join(mean_dir, file_name)

        std_file_name = f"std_{flavour}_Q={key}.csv"
        if std_file_name not in std_files:
            print(f"[WARN] Missing std file for {file_name}")
            continue

        std_file = std_files[std_file_name]

        data_dict[key]["mean"][flavour] = np.loadtxt(mean_file, delimiter=",")
        data_dict[key]["std"][flavour] = np.loadtxt(std_file, delimiter=",")

    print(f"KDE loaded: {len(data_dict)} Q-points")
    return data_dict

def read_in_lhapdf(mean_dir="LHAPDF_mean", std_dir="LHAPDF_std"):
    """
    Load LHAPDF mean and standard-deviation CSVs into a nested mapping.

    Parameters
    ----------
    mean_dir : str, optional
        Directory containing LHAPDF mean CSV files. Default is
        ``'LHAPDF_mean'``.
    std_dir : str, optional
        Directory containing LHAPDF std CSV files. Default is
        ``'LHAPDF_std'``.

    Returns
    -------
    dict
        Mapping ``Q_string`` -> ``{'mean': {flavour: array}, 'std': {flavour: array}, 'Q': float}``.

    Notes
    -----
    Filenames are expected to follow the pattern
    ``mean_<flavour>_Q=<Q>.csv`` and ``std_<flavour>_Q=<Q>.csv``. Keys are
    normalised to the LHAPDF-style scientific format (``'%.6e'``).
    """

    pattern = r"(mean|std)_(\w+)_Q=([0-9.eE+-]+)\.csv"
    data_dict = {}

    # Load means
    for filename in os.listdir(mean_dir):
        match = re.match(pattern, filename)
        if not match:
            continue

        _, flav, Q_str = match.groups()
        key = f"{float(Q_str):.6e}"

        if key not in data_dict:
            data_dict[key] = {"mean": {}, "std": {}, "Q": float(Q_str)}

        data_dict[key]["mean"][flav] = np.loadtxt(os.path.join(mean_dir, filename), delimiter=",")

    # Load std
    for filename in os.listdir(std_dir):
        match = re.match(pattern, filename)
        if not match:
            continue

        _, flav, Q_str = match.groups()
        key = f"{float(Q_str):.6e}"

        if key not in data_dict:
            data_dict[key] = {"mean": {}, "std": {}, "Q": float(Q_str)}

        data_dict[key]["std"][flav] = np.loadtxt(os.path.join(std_dir, filename), delimiter=",")

    print(f"LHAPDF loaded: {len(data_dict)} Q-points")
    return data_dict

def read_in_xgrid(path=None):
    """
    Load the standard 45-point x-grid from a text file.

    Parameters
    ----------
    path : str, optional
        Path to the x-grid file. If omitted, uses the repository-standard
        file under ``00_data/XGRID_45.txt`` relative to this module.

    Returns
    -------
    ndarray
        1-D array of x-grid points.
    """
    if path is None:
        here = os.path.dirname(__file__)
        path = os.path.normpath(os.path.join(here, "..", "..", "00_data", "XGRID_45.txt"))

    return np.loadtxt(path)

#############################################################################
### Calculate statistics (for diagnostics)
#############################################################################

def run_diagnostics(data, flavour):
    """
    Prints statistical diagnostics for the data and calculates robust plotting limits.
    Returns: (robust_min, robust_max)
    """
    print(f"\n" + "="*40)
    print(f" DIAGNOSTICS FOR FLAVOUR: {flavour}")
    print(f"="*40)
    
    n_total = len(data)
    if n_total == 0:
        return -1, 1

    # 1. Count Zeros and Near Zeros
    n_exact_zeros = np.sum(data == 0.0)
    threshold_tiny = 1e-6
    n_tiny = np.sum(np.abs(data) < threshold_tiny)
    n_neg_outliers = np.sum(data < -threshold_tiny)
    
    print(f"Total Data Points:    {n_total}")
    print(f"Exact Zeros (0.0):    {n_exact_zeros}")
    print(f"Near Zeros (< 1e-6):  {n_tiny}  ({n_tiny/n_total*100:.1f}%)")
    print(f"Negative Outliers:    {n_neg_outliers}  ({n_neg_outliers/n_total*100:.1f}%)")
    
    # 2. Distribution Spread
    percs = np.percentile(data, [0, 1, 25, 50, 75, 99, 100])
    
    print("-" * 20)
    print("DISTRIBUTION SPREAD:")
    print(f"Min (0%):   {percs[0]:.4e}")
    print(f"Median:     {percs[3]:.4e}")
    print(f"Max (100%): {percs[6]:.4e}")
    print("-" * 20)

    # 3. Calculate Robust Limits (IQR method)
    iqr_lower = percs[2] # 25th
    iqr_upper = percs[4] # 75th
    
    # Expand view to 3x the IQR
    view_width = (iqr_upper - iqr_lower) * 3.0
    if view_width == 0: view_width = 1e-9
    
    robust_min = percs[3] - view_width
    robust_max = percs[3] + view_width
    
    print(f"Suggested Linear Plot Range: {robust_min:.4e} to {robust_max:.4e}")
    return robust_min, robust_max

#############################################################################
### Normalised differences (and plotting functions)
#############################################################################

def plot_histogram(values, title, xlabel, bins=40, range=None, density=True,
                   output_dir=None, filename=None, showPlots=False):
    """
    Plot a histogram for pooled PDF comparison metrics.

    Parameters
    ----------
    values : array-like
        1D array of values to histogram (e.g. Δ_norm over x and Q).
    title : str
        Plot title.
    xlabel : str
        Label for x-axis.
    bins : int, optional
        Number of histogram bins.
    range : tuple, optional
        Histogram range.
    density : bool, optional
        Normalise histogram to unit area.
    """
    values = np.asarray(values)
    values = values[np.isfinite(values)]  

    plt.figure(figsize=(7,5))
    plt.hist(values, bins=bins, range=range, density=density,
             histtype="stepfilled", alpha=0.6)

    plt.xlabel(xlabel)
    plt.ylabel("Density" if density else "Counts")
    plt.title(title)
    plt.axvline(0.0, linestyle="--", color="black", linewidth=1.5) # dotted reference line at zero
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            safe_title = re.sub(r"[^A-Za-z0-9_]+", "_", title).strip("_")
            filename = f"{safe_title}.png"
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=150)
        print(f"Saved histogram: {out_path}")

    if showPlots:
        plt.show()
    else:
        plt.close()

def normalised_variance_difference(KDE, LHAPDF, XGRID, FLAVS, showPlots=False):
    """
    Compute and plot normalised mean differences between LHAPDF and KDE.

    The quantity plotted is::

        (mean_LHAPDF - mean_KDE) / sqrt(std_LHAPDF^2 + std_KDE^2)

    for each flavour and Q-value. Points are scatter-plotted on the
    standard 45-point x-grid.

    Parameters
    ----------
    KDE : dict
        Mapping produced by :func:`KDE_read_mean_std_data`.
    LHAPDF : dict
        Mapping produced by :func:`readInLhapdf`.
    """

    for flav in FLAVS:
        print(f"\n=== Plotting flavour: {flav} ===")

        all_diffs = []

        for key in sorted(LHAPDF.keys()):

            # Q must exist in KDE and LHAPRF
            if key not in KDE:
                print(f"Missing Q={key} in KDE -> Skipping.")
                continue
            
            # Flavour must exist in KDE and LHAPRF
            if flav not in LHAPDF[key]["std"]:
                print(f"Missing {flav} in LHAPDF for Q={key} -> Skipping.")
                continue
            if flav not in KDE[key]["std"]:
                print(f"Missing {flav} in KDE for Q={key} -> Skipping.")
                continue

            # Normalised error difference
            num = LHAPDF[key]["std"][flav]**2 - KDE[key]["std"][flav]**2
            denom = np.sqrt(LHAPDF[key]["std"][flav]**4 + KDE[key]["std"][flav]**4)
            diff = num / denom

            all_diffs.append(diff)

        if all_diffs:
            all_diffs_flat = np.concatenate(all_diffs)
            plot_histogram(
                values=all_diffs_flat,
                title=f"Normalised Variance Difference – {flav}",
                xlabel=r"$(\sigma_{LHAPDF}^2-\sigma_{KDE}^2) / \sqrt{\sigma_{LHAPDF}^4+\sigma_{KDE}^4}$",
                # range=(-3, 3),
                output_dir="normalisedVarianceDifference_plots",
                filename=f"normalisedVarianceDifference_{flav}.png",
                showPlots=showPlots
            )

def variance_of_variance(KDE, LHAPDF, FLAVS, N_replicas=1000, showPlots=True):
    """
    Plot the normalised difference of the variance-of-variance (VoV)
    between KDE and LHAPDF.

    The quantity plotted is:

        (VoV_KDE - VoV_LHAPDF) / sqrt(VoV_KDE^2 + VoV_LHAPDF^2)

    which is bounded in [-1, 1].
    """

    output_dir = "varianceOfVarianceDifference_plots"
    os.makedirs(output_dir, exist_ok=True)

    for flav in FLAVS:
        print(f"\n=== Processing flavour: {flav} ===")
        all_diffs = []

        # --- 1. Compute VoV differences ---
        for key in sorted(LHAPDF.keys()):
            if key not in KDE:
                continue
            if flav not in LHAPDF[key]["std"]:
                continue
            if flav not in KDE[key]["std"]:
                continue

            std_L = LHAPDF[key]["std"][flav]
            std_K = KDE[key]["std"][flav]

            # Variance of the variance
            vov_L = 2.0 * std_L**4 / (N_replicas - 1)
            vov_K = 2.0 * std_K**4 / (N_replicas - 1)

            denom = np.sqrt(vov_L**2 + vov_K**2)

            with np.errstate(divide="ignore", invalid="ignore"):
                diff = (vov_K - vov_L) / denom

            diff = diff[np.isfinite(diff)]
            all_diffs.append(diff)

        if not all_diffs:
            continue

        all_diffs_flat = np.concatenate(all_diffs)

        # --- 2. Plot ---
        plt.figure(figsize=(8, 5))

        plt.hist(
            all_diffs_flat,
            bins=80,
            range=(-1.0, 1.0),
            histtype="stepfilled",
            alpha=0.8,
            linewidth=0.6
        )

        plt.xlabel(
            r"$(\mathrm{VoV}_{\mathrm{KDE}} - \mathrm{VoV}_{\mathrm{LHAPDF}})"
            r"/\sqrt{\mathrm{VoV}_{\mathrm{KDE}}^2 + \mathrm{VoV}_{\mathrm{LHAPDF}}^2}$"
        )
        plt.ylabel("Counts")

        plt.title(f"Normalised variance-of-variance difference – {flav}")
        plt.grid(True, alpha=0.3)

        filename = f"VoV_Difference_{flav}.png"
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot: {out_path}")

        if showPlots:
            plt.show()
        else:
            plt.close()

def plot_vov_side_by_side(KDE, LHAPDF, FLAVS, N_replicas=1000, output_dir="VoV_SideBySide_plots", showPlots=False):

    os.makedirs(output_dir, exist_ok=True)

    for flav in FLAVS:
        print(f"\n=== Generating VoV comparison for: {flav} ===")

        vov_kde_list = []
        vov_lhapdf_list = []

        # --- 1. Collect data ---
        for key in sorted(LHAPDF.keys()):
            if key not in KDE:
                continue
            if flav not in LHAPDF[key]["std"]:
                continue
            if flav not in KDE[key]["std"]:
                continue

            std_K = KDE[key]["std"][flav]
            std_L = LHAPDF[key]["std"][flav]

            vov_K = 2.0 * std_K**4 / (N_replicas - 1)
            vov_L = 2.0 * std_L**4 / (N_replicas - 1)

            vov_kde_list.append(vov_K)
            vov_lhapdf_list.append(vov_L)

        if not vov_kde_list:
            continue

        vov_kde_flat = np.concatenate(vov_kde_list)
        vov_lhapdf_flat = np.concatenate(vov_lhapdf_list)

        # --- 2. Common linear binning ---
        all_data = np.concatenate([vov_kde_flat, vov_lhapdf_flat])

        finite = np.isfinite(all_data)
        if not np.any(finite):
            continue

        xmin = np.min(all_data[finite])
        xmax = np.max(all_data[finite])

        # Safety: avoid zero-width range
        if xmin == xmax:
            xmin = 0.0
            xmax = xmax * 1.1 if xmax > 0 else 1.0

        bins = np.linspace(-1, 1, 100)

        # --- 3. Plotting ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        axes[0].hist(
            vov_kde_flat,
            bins=bins,
            density=True,
            color= "#287085",
            alpha=0.7,
            linewidth=0.5
        )

        axes[0].set_title(f"KDE: VoV ({flav})")
        axes[0].set_xlabel(r"$2\sigma_{\mathrm{KDE}}^4 / (N_{\mathrm{rep}} - 1)$")
        axes[0].set_ylabel("Density")
        axes[0].grid(alpha=0.3)
        axes[0].set_xlim(-1,1)

        axes[1].hist(
            vov_lhapdf_flat,
            bins=bins,
            density=True,
            color= "#722b5b",
            alpha=0.7,
            linewidth=0.5
        )
        axes[1].set_title(f"LHAPDF: VoV ({flav})")
        axes[1].set_xlabel(r"$2\sigma_{\mathrm{LHAPDF}}^4 / (N_{\mathrm{rep}} - 1)$")
        axes[1].grid(alpha=0.3)
        axes[1].set_xlim(-1,1)

        plt.tight_layout()

        filename = f"VoV_{flav}.png"
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")

        if showPlots:
            plt.show()
        else:
            plt.close()


#############################################################################
### Pull distributions
#############################################################################

def pull_distribution(KDE, LHAPDF, FLAVS, showPlots=False):
    """
    Build true PDF pull distributions per flavour.

    The pull is defined as::

        pull = (mean_KDE - mean_LHAPDF)
               / sqrt(std_KDE^2 + std_LHAPDF^2)

    This tests PDF central-value closure, not uncertainty closure.
    """

    for flav in FLAVS:
        pulls = []

        for key in sorted(LHAPDF.keys()):
            if key not in KDE:
                continue
            if flav not in LHAPDF[key]["mean"] or flav not in KDE[key]["mean"]:
                continue
            if flav not in LHAPDF[key]["std"] or flav not in KDE[key]["std"]:
                continue

            mean_lhapdf = LHAPDF[key]["mean"][flav]
            mean_kde = KDE[key]["mean"][flav]

            std_lhapdf = LHAPDF[key]["std"][flav]
            std_kde = KDE[key]["std"][flav]

            denom = np.sqrt(std_lhapdf**2 + std_kde**2)

            with np.errstate(divide='ignore', invalid='ignore'):
                pull_vals = (mean_kde - mean_lhapdf) / denom

            pull_vals = pull_vals[np.isfinite(pull_vals)]
            if pull_vals.size:
                pulls.append(pull_vals)

        if not pulls:
            print(f"No valid pulls for flavour {flav}")
            continue

        pulls = np.concatenate(pulls)

        plot_histogram(
            values=pulls,
            title=f"PDF Pull Distribution ({flav} flavour)",
            xlabel=r"$(\mu_{KDE}-\mu_{LHAPDF}) / \sqrt{\sigma_{KDE}^2+\sigma_{LHAPDF}^2}$",
            output_dir="pullDistributions_plots",
            filename=f"pullDistribution_{flav}.png",
            showPlots=showPlots
        )



#############################################################################
### Main Function
#############################################################################

def main(showPlots=False):
    KDE = read_in_kde()
    LHAPDF = read_in_lhapdf()
    XGRID = read_in_xgrid()
    FLAVS = ['u','d','s','ubar','dbar','sbar','c','cbar','g']

    # --- Run difference in error analytics 
    # normalised_variance_difference(KDE, LHAPDF, XGRID, FLAVS, showPlots)
    # pull_distribution(KDE, LHAPDF, FLAVS, showPlots)
    variance_of_variance(KDE, LHAPDF, FLAVS, N_replicas=1000, showPlots=showPlots)
    plot_vov_side_by_side(KDE, LHAPDF, FLAVS, N_replicas=1000, output_dir="VoV_SideBySide_plots", showPlots=showPlots)



if __name__ == "__main__":
    main()
