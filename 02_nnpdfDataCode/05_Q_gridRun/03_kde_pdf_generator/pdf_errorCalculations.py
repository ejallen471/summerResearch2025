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
    # Reference line at zero
    plt.axvline(0.0, linestyle="--", color="black", linewidth=1.5)
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

def normalised_mean_difference(KDE, LHAPDF, XGRID, FLAVS, showPlots=False):
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
            num = LHAPDF[key]["std"][flav] - KDE[key]["std"][flav]
            denom = np.sqrt(LHAPDF[key]["std"][flav]**2 + KDE[key]["std"][flav]**2)
            diff = num / denom

            all_diffs.append(diff)

        if all_diffs:
            all_diffs_flat = np.concatenate(all_diffs)
            plot_histogram(
                values=all_diffs_flat,
                title=f"Normalised Mean Error Difference – {flav}",
                xlabel=r"$(\sigma_{LHAPDF}-\sigma_{KDE}) / \sqrt{\sigma_{LHAPDF}^2+\sigma_{KDE}^2}$",
                range=(-3, 3),
                output_dir="normalisedMeanDifference_plots",
                filename=f"normalisedMeanDifference_{flav}.png",
                showPlots=showPlots
            )




def normalised_difference_in_fluctuations(KDE, LHAPDF, XGRID, FLAVS,N_replicas=1000, showPlots=False):
    """
    Compute and plot differences in standard deviations in units of the
    expected statistical fluctuation (error on the error).

    The quantity plotted is::

        (std_LHAPDF - std_KDE) /
        sqrt(delta_std_LHAPDF^2 + delta_std_KDE^2)

    where::

        delta_std = std / sqrt(2 * (N_replicas - 1))
    """

    for flav in FLAVS:
        print(f"\n=== Plotting flavour: {flav} ===")

        all_diffs = []

        for key in sorted(LHAPDF.keys()):

            if key not in KDE:
                continue
            if flav not in LHAPDF[key]["std"]:
                continue
            if flav not in KDE[key]["std"]:
                continue

            std_L = LHAPDF[key]["std"][flav]
            std_K = KDE[key]["std"][flav]

            # Error on the error (Gaussian MC)
            delta_L = std_L / np.sqrt(2 * (N_replicas - 1))
            delta_K = std_K / np.sqrt(2 * (N_replicas - 1))

            denom = np.sqrt(delta_L**2 + delta_K**2)

            # Safety guard
            if not np.all(np.isfinite(denom)):
                continue

            diff = (std_L - std_K) / denom
            all_diffs.append(diff)

        if all_diffs:
            all_diffs_flat = np.concatenate(all_diffs)

            plot_histogram(
                values=all_diffs_flat,
                title=f"Error Difference in Units of Fluctuation – {flav}",
                xlabel=r"$(\sigma_{LHAPDF}-\sigma_{KDE}) / \sqrt{\delta\sigma_{LHAPDF}^2+\delta\sigma_{KDE}^2}$",
                # range=(-5, 5),
                output_dir="normalisedDifferenceInFluctuations_plots",
                filename=f"normalisedDifferenceInFluctuations_{flav}.png",
                showPlots=showPlots
            )



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
    normalised_mean_difference(KDE, LHAPDF, XGRID, FLAVS, showPlots)
    normalised_difference_in_fluctuations(KDE, LHAPDF, XGRID, FLAVS, N_replicas=1000, showPlots=showPlots)
    pull_distribution(KDE, LHAPDF, FLAVS, showPlots)


if __name__ == "__main__":
    main()
