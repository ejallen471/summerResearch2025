"""
Plot raw 1D histograms for selected NNPDF replica flavours.

This module loads replica data in the flavour and evolution bases,
extracts chosen parton flavours at a fixed grid index, and visualises the
resulting replica distributions as normalised histograms. For each
histogram an empirical Gaussian PDF, fitted from the sample mean and
standard deviation, is overlaid for comparison. No KDE is used here; this
script is intended as a quick diagnostic view of the underlying replica
spread.
"""

#############################################################################

import os
import pickle
import numpy as np
from pathlib import Path
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
    Load the serialised flavour and evolution basis data.

    The function expects the files ``flavour_basis.pkl`` and
    ``evolution_basis.pkl`` to be present in the ``00_data`` directory
    located alongside this script's parent directory.

    Returns
    -------
    generator
        Generator yielding the flavour-basis replica list and the
        evolution-basis replica list.
    """

    data_dir = Path(__file__).resolve().parent.parent / "00_data"
    paths = [data_dir / "flavour_basis.pkl", data_dir / "evolution_basis.pkl"]
    return (pickle.load(open(p, "rb")) for p in paths)


def prepare_data(res, keys, index):
    """
    Prepare an array of replica values for selected flavours.

    Parameters
    ----------
    res : list[dict]
        List of replica dictionaries, each mapping flavour keys to arrays.
    keys : list[str]
        Flavour keys to extract from each replica.
    indices : int or array-like, optional
        Grid index or indices to extract. If ``None``, all indices in the
        range ``0..49`` are used.

    Returns
    -------
    numpy.ndarray
        If ``indices`` is an int, an array of shape ``(n_replicas, n_keys)``.
        Otherwise, an array of shape ``(n_replicas, n_keys, n_indices)``.
    """
    num_replicas = len(res)
    num_keys = len(keys)
    
    # Single index case, output 2D (num_replicas, num_keys)
    data_array = np.empty((num_replicas, num_keys), dtype=float)
    for i, replica in enumerate(res):
        for j, key in enumerate(keys):
            data_array[i, j] = replica[key][index]


    return data_array

#############################################################################
### PLOTTING 
#############################################################################

def plot_1D_histogram(data, dim, flavour, index, bins=50):
    """
    Plot a 1D normalised histogram with empirical Gaussian overlay.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape ``(n_replicas, n_keys)`` as returned by
        :func:`prepare_data` for a single grid index.
    dim : int
        Column index selecting which flavour to plot.
    flavour : str
        Name of the parton flavour (from ``keys_flav``).
    index : int
        Grid index used for this histogram.
    bins : int, optional
        Number of histogram bins (default is 50).
    """

    data_1d = data[:, dim]

    # Empirical Gaussian PDF from sample mean and std
    mean = np.mean(data_1d)
    std = np.std(data_1d, ddof=0)
    x_vals = np.linspace(data_1d.min(), data_1d.max(), 500)
    pdf_vals = norm.pdf(x_vals, loc=mean, scale=std)

    plt.figure(figsize=(8, 5))
    plt.hist(
        data_1d,
        bins=bins,
        density=True,
        color="#68A5A1",
        edgecolor="black",
        alpha=0.6,
        label="Replica histogram",
    )
    plt.plot(x_vals, pdf_vals, color="black", lw=2, label="Empirical Gaussian PDF")
    plt.xlabel("f(x, Q)")
    plt.title(f"Replica Distribution (flavour: {flavour}, grid index: {index})")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_1D_histogram_withScatter(data, dim, bins=50):
    """
    Plot a scatter of replica values and the corresponding histogram.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape ``(n_replicas, n_keys)`` as returned by
        :func:`prepare_data` for a single grid index.
    dim : int
        Column index selecting which flavour to plot.
    bins : int, optional
        Number of histogram bins (default is 50).
    """

    data_1d = data[:, dim]

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=(4, 2), wspace=0.05)

    # Left: Scatter plot
    ax_main = fig.add_subplot(gs[0])
    ax_main.scatter(np.arange(len(data_1d)), data_1d, color='#68A5A1', s=3)
    ax_main.set_ylabel("f(x)", fontsize=12)
    ax_main.set_xlabel("x", fontsize=12)
    ax_main.set_title("1D Scatter Plot and Histogram", fontsize=16)
    ax_main.tick_params(axis='both', labelsize=12)
    ax_main.grid(True)

    # Right: Rotated histogram (density on x, value on y)
    ax_hist = fig.add_subplot(gs[1], sharey=ax_main)
    hist, bin_edges = np.histogram(data_1d, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bar_width = bin_edges[1] - bin_edges[0]
    ax_hist.barh(
        bin_centers,
        hist,
        height=bar_width,
        color="#68A5A1",
        edgecolor='black',
    )

    ax_hist.set_xlabel("Probability Density", fontsize=12, labelpad=12)
    ax_hist.tick_params(axis='x', labelsize=12)
    ax_hist.tick_params(axis='y', left=False, labelleft=False)
    ax_hist.set_xlim(left=0)

    plt.tight_layout()
    plt.show()

#############################################################################
### MAIN FUNCTION
#############################################################################

def main(plotting1D=True):
    """
    Entry point to generate 1D histograms for selected flavours.

    Parameters
    ----------
    plotting1D : bool, optional
        If ``True`` (default), plot 1D histograms for each selected flavour.
    """

    res_flav, res_ev = read_in_data()
    
    # keys_ev = ['Sigma', 'V', 'V3', 'V8', 'T3', 'T8', 'c+', 'g', 'V15']
    # keys_flav = ['d', 'u', 's', 'c', 'dbar', 'ubar', 'sbar', 'cbar', 'g']
    
    # choose flavours to loop through 
    keys_flav = ['d', 'g']

    # choose singe index between 1 and 50
    index = 28

    data = prepare_data(res_flav, keys_flav, index)

    # --- Plot in 1D (histograms only)
    d = data.shape[1]
    print(d)
    if plotting1D is True:
        for dim in range(0, d):
            plot_1D_histogram(data, dim, keys_flav[dim], index)
            # plot_1D_histogram_withScatter(data, dim)

if __name__ == "__main__":
    main()



