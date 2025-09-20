import numpy as np
from matplotlib import pyplot as plt

# If running on separate laptop/computer, this will need commenting out 
plt.style.use('pythonStyle')
import pythonStyle as ed

# Load matrices
kde_correlation = np.loadtxt('correlation_kde.csv', delimiter=',')
empirical_correlation = np.loadtxt('correlation_empirical.csv', delimiter=',')
kde_covariance = np.genfromtxt('covariance_kde.csv', delimiter=',')
empirical_covariance = np.loadtxt('empirical_covariance.csv', delimiter=',')

def plot_matrix_comparison(matrix1, matrix2, title1, title2, cbar_label, save_filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot matrix 1
    im0 = axes[0].imshow(matrix1, aspect='equal')
    # axes[0].set_title(title1, fontsize=12)
    axes[0].invert_yaxis()
    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])
    axes[0].grid(False)

    # Plot matrix two
    im1 = axes[1].imshow(matrix2, aspect='equal')
    # axes[1].set_title(title2, fontsize=12)
    axes[1].invert_yaxis()
    axes[1].set_xticklabels([])
    axes[1].set_yticklabels([])
    axes[1].grid(False)

    # Add colorbar to the second plot (linked to im1)
    cbar = fig.colorbar(im1, ax=axes[1])
    cbar.set_label(cbar_label)

    plt.subplots_adjust(wspace=0.3)
    plt.savefig(save_filename, bbox_inches='tight')
    plt.show()



plot_matrix_comparison(kde_correlation, empirical_correlation, 'title1', 'title2', 'Correlation', 'correlation_15Grid')
plot_matrix_comparison(kde_covariance, empirical_covariance, 'title1', 'title2', 'Covariance', 'covariance_15Grid')