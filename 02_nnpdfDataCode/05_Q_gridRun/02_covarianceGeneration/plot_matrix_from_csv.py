"""
plot_matrix_from_csv.py

Small helper script to visualise covariance and correlation matrices
that are saved as CSV files by the reconstruction pipeline.

Behaviour
---------
- Loads    
        `correlation_kde.csv`, 
        `correlation_empirical.csv`,
        `covariance_kde.csv`  
        `covariance_empirical.csv` 
    from the current working directory, cleans/clamps invalid values 
    and writes comparison PNG images 
        `correlation_comparison.png` 
        `covariance_comparison.png`

"""

import numpy as np
from matplotlib import pyplot as plt
plt.style.use('../../pythonStyle.mplstyle')

##############################################################################
##############################################################################

# Helper to clamp correlation values
def clamp_correlation(matrix):
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=-1.0)  # Remove NaN/inf
    return np.clip(matrix, -1.0, 1.0)  # Clamp to [-1, 1]

# Helper to clean covariance
def clean_matrix(matrix):
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=np.finfo(np.float64).max, neginf=-np.finfo(np.float64).max)
    return matrix

##############################################################################
##############################################################################

def plot_matrix_correlation(matrix1, matrix2, title1, title2, cbar_label, save_filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # First matrix
    im0 = axes[0].imshow(matrix1, aspect='equal', interpolation='none')
    axes[0].invert_yaxis()
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].grid(False)
    axes[0].set_title(title1)

    # Second matrix
    im1 = axes[1].imshow(matrix2, aspect='equal', interpolation='none')
    axes[1].invert_yaxis()
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].grid(False)
    axes[1].set_title(title2)

    # Add colourbar only for the second matrix
    cbar = fig.colorbar(im1, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    plt.subplots_adjust(wspace=0.3)
    plt.savefig(save_filename, bbox_inches='tight')
    plt.show()

def plot_matrix_covariance(matrix1, matrix2, title1, title2, cbar_label, save_filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    vmin = -1
    vmax = 50

    # First matrix
    im0 = axes[0].imshow(matrix1, aspect='equal', interpolation='none', vmin=vmin, vmax=vmax)
    axes[0].invert_yaxis()
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].grid(False)
    axes[0].set_title(title1)

    # Second matrix
    im1 = axes[1].imshow(matrix2, aspect='equal', interpolation='none' , vmin=vmin, vmax=vmax)
    axes[1].invert_yaxis()
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].grid(False)
    axes[1].set_title(title2)

    # Add colourbar only for the second matrix
    cbar = fig.colorbar(im1, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    plt.subplots_adjust(wspace=0.3)
    plt.savefig(save_filename, bbox_inches='tight')
    plt.show()

##############################################################################
##############################################################################

def main():
    """
    Load matrices from CSV, clean them and produce comparison PNGs.

    Parameters
    ----------
    save_prefix : str, optional
        Filename prefix for saved PNGs (default: 'comparison'). The
        produced files are 
        `<save_prefix>_correlation.png` and
        `<save_prefix>_covariance.png`.
    """

    kde_correlation = clamp_correlation(np.loadtxt('correlation_kde.csv', delimiter=','))
    empirical_correlation = clamp_correlation(np.loadtxt('correlation_empirical.csv', delimiter=','))

    kde_covariance = clean_matrix(np.loadtxt('covariance_kde.csv', delimiter=','))
    empirical_covariance = clean_matrix(np.loadtxt('covariance_empirical.csv', delimiter=','))

    # Plot correlation comparison
    plot_matrix_correlation(
        kde_correlation,
        empirical_correlation,
        'KDE Correlation',
        'Empirical Correlation',
        'Correlation',
        'matrix_correlation.png'
    )

    # Plot covariance comparison
    plot_matrix_covariance(
        kde_covariance,
        empirical_covariance,
        'KDE Covariance',
        'Empirical Covariance',
        'Covariance',
        'matrix_correlation.png'
    )


if __name__ == '__main__':
    main()

