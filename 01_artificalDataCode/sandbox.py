import numpy as np
from scipy.stats import gaussian_kde, norm
import warnings

def process_2d_blocks(data, bandwidth='scott', num_points=500):
    """
    Split high-dimensional data into 2D blocks, compute empirical and KDE moments,
    and KL divergence (1D marginals). No np.trapz used.

    Parameters:
        data: ndarray, shape (n_samples, d_total)
        bandwidth: KDE bandwidth method ('scott', 'silverman', or float)
        num_points: number of grid points per axis for numerical integration

    Returns:
        results: list of dicts with keys:
            'block', 'mean_empirical', 'cov_empirical',
            'mean_kde', 'cov_kde', 'kl_div_x', 'kl_div_y'
    """
    n_samples, d_total = data.shape
    assert d_total % 2 == 0, "Number of dimensions must be even"

    results = []

    for i in range(0, d_total, 2):
        print(f'{i}')
        dims = (i, i + 1)
        block = data[:, dims]  # shape: (n_samples, 2)

        # --- Empirical statistics
        mean_emp = np.mean(block, axis=0)
        cov_emp = np.cov(block, rowvar=False)

        # --- KDE estimation
        kde = gaussian_kde(block.T, bw_method=bandwidth)

        try:
            # Grid setup for numerical integration
            x_grid = np.linspace(block[:, 0].min(), block[:, 0].max(), num_points)
            y_grid = np.linspace(block[:, 1].min(), block[:, 1].max(), num_points)
            dx = x_grid[1] - x_grid[0]
            dy = y_grid[1] - y_grid[0]

            X, Y = np.meshgrid(x_grid, y_grid)
            grid_points = np.stack([X.ravel(), Y.ravel()])  # shape (2, num_points^2)

            # Evaluate KDE over grid
            kde_vals = kde(grid_points).reshape(num_points, num_points)
            kde_vals /= np.sum(kde_vals) * dx * dy  # normalize manually

            # Marginal densities
            px = np.sum(kde_vals, axis=0) * dy  # ∫f(x, y) dy
            py = np.sum(kde_vals, axis=1) * dx  # ∫f(x, y) dx

            # Means
            mean_kde_x = np.sum(px * x_grid) * dx
            mean_kde_y = np.sum(py * y_grid) * dy
            mean_kde = np.array([mean_kde_x, mean_kde_y])

            # Variances
            var_kde_x = np.sum(px * (x_grid - mean_kde_x) ** 2) * dx
            var_kde_y = np.sum(py * (y_grid - mean_kde_y) ** 2) * dy
            cov_kde = np.array([[var_kde_x, 0], [0, var_kde_y]])

        except Exception as e:
            warnings.warn(f"KDE or integration failed for block {dims}: {e}")
            mean_kde = np.full(2, np.nan)
            cov_kde = np.full((2, 2), np.nan)

        results.append({
            'block': dims,
            'mean_empirical': mean_emp,
            'cov_empirical': cov_emp,
            'mean_kde': mean_kde,
            'cov_kde': cov_kde,
        })

    return results



def reconstruct_covariance_matrix(results, d_total):
    """
    Reconstruct full covariance matrix from overlapping 2x2 blocks.

    Parameters:
        results: output of process_2d_blocks
        d_total: total number of original dimensions (e.g. 450)

    Returns:
        cov_full: (d_total, d_total) averaged covariance matrix
    """
    cov_full = np.zeros((d_total, d_total))
    counts = np.zeros((d_total, d_total))

    for result in results:
        i, j = result['block']
        cov_block = result['cov_kde']

        if np.any(np.isnan(cov_block)):
            continue  # skip blocks that failed

        cov_full[i:i+2, i:i+2] += cov_block
        counts[i:i+2, i:i+2] += 1

    # Avoid divide-by-zero
    mask = counts > 0
    cov_full[mask] /= counts[mask]

    return cov_full


data = np.random.randn(1000, 450)  # Replace with real data
results = process_2d_blocks(data)

# Print summary for first few
for r in results[:3]:
    print(f"Block {r['block']}")
    print(f"  Mean empirical: {r['mean_empirical']}")
    print(f"  Mean KDE:       {r['mean_kde']}")


cov_kde_full = reconstruct_covariance_matrix(results, d_total=450)

print(cov_kde_full.shape)  # should be (450, 450)

import matplotlib.pyplot as plt

# --- Plot full KDE covariance matrix
plt.figure(figsize=(10, 8))
im = plt.imshow(cov_kde_full, cmap='viridis', interpolation='nearest')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.title('Full 450×450 KDE Covariance Matrix')
plt.xlabel('Dimension')
plt.ylabel('Dimension')
plt.tight_layout()
plt.show()
