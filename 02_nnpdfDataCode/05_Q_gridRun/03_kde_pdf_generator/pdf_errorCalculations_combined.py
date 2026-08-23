"""
We have KDE-versus-original uncertainty diagnostics for many flavours and Q
values and want combined plots showing all flavours together.

Run with the following command:

python pdf_errorCalculations_combined.py

This file does the following:

1. Read the KDE and original-LHAPDF mean and standard-deviation CSV files.
2. Calculate combined pull, variance-difference and variance-of-variance data.
3. Plot and save aggregated shaded histograms for all flavours.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Optional: Set a style if available, otherwise default
try:
    plt.style.use('../../pythonStyle.mplstyle')
except OSError:
    plt.style.use('ggplot')
    pass

#############################################################################
### DATA LOADING
#############################################################################

def read_in_kde(mean_dir="KDE_mean", std_dir="KDE_std"):
    """Load KDE mean and std CSVs."""
    pattern = r"^(mean|std)_([a-zA-Z]+)_Q=([0-9.eE+-]+)\.csv$"
    data_dict = {}
    
    if not os.path.exists(mean_dir) or not os.path.exists(std_dir):
        print(f"[WARN] Directories {mean_dir} or {std_dir} not found.")
        return {}

    std_files = {f: os.path.join(std_dir, f) for f in os.listdir(std_dir)}

    for file_name in os.listdir(mean_dir):
        match = re.match(pattern, file_name)
        if not match: continue

        _, flavour, Q_str = match.groups()
        key = f"{float(Q_str):.6e}"

        if key not in data_dict:
            data_dict[key] = {"mean": {}, "std": {}}

        mean_path = os.path.join(mean_dir, file_name)
        std_name = f"std_{flavour}_Q={key}.csv"
        
        if std_name in std_files:
            data_dict[key]["mean"][flavour] = np.loadtxt(mean_path, delimiter=",")
            data_dict[key]["std"][flavour] = np.loadtxt(std_files[std_name], delimiter=",")

    print(f"KDE loaded: {len(data_dict)} Q-points")
    return data_dict

def read_in_lhapdf(mean_dir="LHAPDF_mean", std_dir="LHAPDF_std"):
    """Load LHAPDF mean and std CSVs."""
    pattern = r"(mean|std)_(\w+)_Q=([0-9.eE+-]+)\.csv"
    data_dict = {}

    if not os.path.exists(mean_dir) or not os.path.exists(std_dir):
        print(f"[WARN] Directories {mean_dir} or {std_dir} not found.")
        return {}

    # Load Means
    for filename in os.listdir(mean_dir):
        match = re.match(pattern, filename)
        if match:
            _, flav, Q_str = match.groups()
            key = f"{float(Q_str):.6e}"
            if key not in data_dict: data_dict[key] = {"mean": {}, "std": {}}
            data_dict[key]["mean"][flav] = np.loadtxt(os.path.join(mean_dir, filename), delimiter=",")

    # Load Stds
    for filename in os.listdir(std_dir):
        match = re.match(pattern, filename)
        if match:
            _, flav, Q_str = match.groups()
            key = f"{float(Q_str):.6e}"
            if key not in data_dict: data_dict[key] = {"mean": {}, "std": {}}
            data_dict[key]["std"][flav] = np.loadtxt(os.path.join(std_dir, filename), delimiter=",")

    print(f"LHAPDF loaded: {len(data_dict)} Q-points")
    return data_dict

#############################################################################
### METRIC CALCULATION
#############################################################################

def calculate_metrics(KDE, LHAPDF, FLAVS, N_replicas=1000):
    """
    Iterates through all Q points and flavours to collect metric arrays.
    Returns a dictionary of dictionaries.
    """
    
    collections = {
        "pulls": {f: [] for f in FLAVS},
        "norm_var_diff": {f: [] for f in FLAVS},
        "vov_kde": {f: [] for f in FLAVS},   # Raw KDE VoV
        "vov_lha": {f: [] for f in FLAVS},   # Raw LHAPDF VoV
        "vov_diff": {f: [] for f in FLAVS}   # Normalised Diff
    }

    for key in sorted(LHAPDF.keys()):
        if key not in KDE: continue

        for flav in FLAVS:
            if (flav not in KDE[key]["std"] or flav not in LHAPDF[key]["std"] or
                flav not in KDE[key]["mean"] or flav not in LHAPDF[key]["mean"]):
                continue

            # --- Extract Data ---
            m_K = KDE[key]["mean"][flav]
            m_L = LHAPDF[key]["mean"][flav]
            s_K = KDE[key]["std"][flav]
            s_L = LHAPDF[key]["std"][flav]

            # --- 1. Pulls ---
            denom_pull = np.sqrt(s_K**2 + s_L**2)
            with np.errstate(divide='ignore', invalid='ignore'):
                pull = (m_K - m_L) / denom_pull
            collections["pulls"][flav].append(pull)

            # --- 2. Normalised Variance Difference ---
            var_K = s_K**2
            var_L = s_L**2
            denom_var = np.sqrt(var_L**2 + var_K**2)
            with np.errstate(divide='ignore', invalid='ignore'):
                nvd = (var_L - var_K) / denom_var
            collections["norm_var_diff"][flav].append(nvd)

            # --- 3. Variance of Variance (VoV) ---
            vov_K = 2.0 * s_K**4 / (N_replicas - 1)
            vov_L = 2.0 * s_L**4 / (N_replicas - 1)
            
            collections["vov_kde"][flav].append(vov_K)
            collections["vov_lha"][flav].append(vov_L)

            # --- 4. Normalised VoV Difference ---
            denom_vov = np.sqrt(vov_L**2 + vov_K**2)
            with np.errstate(divide='ignore', invalid='ignore'):
                vovd = (vov_K - vov_L) / denom_vov
            collections["vov_diff"][flav].append(vovd)

    # Flatten lists into arrays
    final_metrics = {m: {} for m in collections}
    for metric in collections:
        for flav in FLAVS:
            if collections[metric][flav]:
                arr = np.concatenate(collections[metric][flav])
                final_metrics[metric][flav] = arr[np.isfinite(arr)]
            else:
                final_metrics[metric][flav] = np.array([])
                
    return final_metrics

#############################################################################
### PLOTTING UTILS
#############################################################################

def plot_single_histogram(data_map, title, xlabel, output_name, xlim=None, log_y=False):
    """Plots histograms for all flavours on a single figure with shading."""
    plt.figure(figsize=(9, 6))
    
    flavours = [f for f in data_map.keys() if len(data_map[f]) > 0]
    
    for flav in flavours:
        data = data_map[flav]
        if len(data) == 0: continue
        
        # Binning logic
        if xlim:
            data_zoomed = data[(data >= xlim[0]) & (data <= xlim[1])]
            if len(data_zoomed) == 0: continue
            bins = np.linspace(xlim[0], xlim[1], 50)
        else:
            if len(data) < 2: continue
            low, high = np.percentile(data, [1, 99])
            if low == high: low, high = min(data), max(data)
            span = high - low
            if span == 0: span = 1e-9
            bins = np.linspace(low - 0.1*span, high + 0.1*span, 50)

        # Plot Outline
        n, bins_out, patches = plt.hist(data, bins=bins, density=True, histtype='step',
                                        linewidth=1.5, label=flav, alpha=1.0, zorder=3)
        # Plot Fill
        col = patches[0].get_edgecolor()
        plt.hist(data, bins=bins, density=True, histtype='stepfilled',
                 color=col, alpha=0.2, zorder=2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    # plt.grid(alpha=0.3, linestyle=':', zorder=0)
    plt.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)

    if log_y: plt.yscale('log')
    if xlim: plt.xlim(xlim)

    plt.tight_layout()
    
    out_dir = "combined_plots_shaded"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_name)
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")
    plt.close()

def plot_vov_side_by_side(kde_map, lha_map, title, xlabel, output_name, log_y=True):
    """
    Plots two subplots side-by-side: KDE VoV and LHAPDF VoV.
    Shares Y-axis and X-axis range for direct comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)
    
    flavours = [f for f in kde_map.keys() if len(kde_map[f]) > 0]
    
    # 1. Determine common range for x-axis across BOTH datasets
    all_data = []
    for f in flavours:
        all_data.append(kde_map[f])
        all_data.append(lha_map[f])
    all_data = np.concatenate(all_data)
    
    # Robust range (1st to 99th percentile)
    low, high = np.percentile(all_data, [1, 99])
    if low == high: low, high = min(all_data), max(all_data)
    
    # Create fixed bins for both plots
    bins = np.linspace(low, high, 50)

    # --- Helper to plot on axis ---
    def plot_on_axis(ax, data_map, sub_title):
        for flav in flavours:
            data = data_map[flav]
            if len(data) == 0: continue
            
            # Clip data to range for clean histograms
            data_clipped = data[(data >= low) & (data <= high)]
            
            # Outline
            n, _, patches = ax.hist(data_clipped, bins=bins, density=True, histtype='step',
                                    linewidth=1.5, label=flav, alpha=1.0, zorder=3)
            # Fill
            col = patches[0].get_edgecolor()
            ax.hist(data_clipped, bins=bins, density=True, histtype='stepfilled',
                    color=col, alpha=0.2, zorder=2)
        
        ax.set_title(sub_title)
        ax.set_xlabel(xlabel)
        # ax.grid(alpha=0.3, linestyle=':', zorder=0)
        if log_y: ax.set_yscale('log')

    # 2. Plot KDE (Left)
    plot_on_axis(axes[0], kde_map, "KDE: Un-normalised VoV")
    axes[0].set_ylabel("Density")

    # 3. Plot LHAPDF (Right)
    plot_on_axis(axes[1], lha_map, "LHAPDF: Un-normalised VoV")
    
    # Single Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5))
    
    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0, 0.90, 1])

    out_dir = "combined_plots_shaded"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_name)
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")
    plt.close()

#############################################################################
### MAIN
#############################################################################

def main():
    FLAVS = ['d', 'u', 's', 'c', 'dbar', 'ubar', 'sbar', 'cbar', 'g']
    
    kde_data = read_in_kde()
    lhapdf_data = read_in_lhapdf()
    
    if not kde_data or not lhapdf_data:
        print("Data loading failed.")
        return

    print("Calculating metrics...")
    metrics = calculate_metrics(kde_data, lhapdf_data, FLAVS)

    # 1. Pull Distributions
    plot_single_histogram(
        metrics["pulls"],
        title=" ", # "Pull Distribution",
        xlabel=r"Pull",
        output_name="Combined_Pulls.png",
        xlim=(-0.2, 0.2)
    )

    # 2. Normalised Variance Difference
    plot_single_histogram(
        metrics["norm_var_diff"],
        title=" ", # "Normalised Variance Difference",
        xlabel=r"Norm. Var Diff",
        output_name="Combined_Norm_Var_Diff.png",
        xlim=(-1.1, 1.1)
    )

    # 3. Normalised VoV Difference
    plot_single_histogram(
        metrics["vov_diff"],
        title=" ", # "Normalised VoV Difference",
        xlabel=r"Norm. VoV Diff",
        output_name="Combined_VoV_Diff.png",
        xlim=(-1.1, 1.1)
    )
    
    # 4. Side-by-Side Un-normalised VoV
    plot_vov_side_by_side(
        metrics["vov_kde"],
        metrics["vov_lha"],
        title=" ", # "Un-normalised Variance of Variance",
        xlabel=r"$2\sigma^4 / (N-1)$",
        output_name="SideBySide_VoV_Raw.png",
        log_y=True
    )

if __name__ == "__main__":
    main()
