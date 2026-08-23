### 01_kde_reconstructions — README

Overview
--------

This directory contains the code used to reconstruct 1D and 2D parton probability density functions (PDFs) from NNPDF replica samples using non‑parametric kernel density estimation (KDE). The main aims are to:

- inspect the empirical distribution of parton densities at fixed kinematic grid points (in \(x\) and \(Q\)) across replicas,
- fit smooth KDE approximations to these empirical distributions using data–driven bandwidth selection, and
- compare KDE reconstructions with simple Gaussian approximations ("empirical PDFs") and quantify their agreement via KL divergence.

The KDE machinery is Gaussian and supports both diagonal and full bandwidth matrices, estimated by cross‑validation or Smooth Cross‑Validation (SCV).


Files and Directories
---------------------

- `KDE_histogram_1D.py`
	- 1D KDE reconstruction for selected flavours at a single grid index.
	- Produces histograms with both the empirical Gaussian PDF and KDE
		estimate superimposed, and can compute 1D KL divergences for
		selected marginals.

- `KDE_scatterGraph_2D.py`
	- 2D KDE reconstruction for a pair of flavours at a fixed grid index.
	- Plots sample scatter, the empirical bivariate Gaussian PDF, and the
		KDE estimate as contour plots, and reports the 2D KL divergence.

- `plot_histograms.py`
	- Convenience script to visualise replica histograms, for chosen flavours at a given grid index.
	No KDE, only empirical gaussian is plotted as well.

All plotting scripts automatically locate the shared `pythonStyle.mplstyle` file one layer higher up the tree 


Key implementation details
--------------------------

#### Data Format

The dataset consists of 1000 NNPDF replicas, each evaluated at 50 grid points in the momentum fraction \(x\) space. Each replica is stored as a Python dictionary mapping parton flavour keys to 1D NumPy arrays of length 50.  

The canonical flavour order is:

- `d` (down), `u` (up), `s` (strange), `c` (charm)
- `dbar` (anti-down), `ubar` (anti-up), `sbar` (anti-strange), `cbar` (anti-charm)
- `g` (gluon)

For KDE analysis:  
- `prepare_data()` extracts the replica values for selected flavours at a given grid index (1D) or multiple indices (higher-dimensional arrays).  
- `prepare_2d_data()` selects two flavours at a single grid index, producing a `(1000, 2)` array for 2D KDE and joint distribution analysis.  


#### 1D Implementation

The workflow uses the following functions: `prepare_data()`, `calc_bandwidthMatrix()`, `calc_kdeCrossValidation()`, `calc_pdf_and_kde_values()`, `plot_1D_histogram()`, and `calc_KLDivergence()`.

We use `calc_bandwidthMatrix(data)` to construct a diagonal bandwidth matrix. Conceptually, this is equivalent to performing separate 1D bandwidth calculations for each flavour, but using a matrix formulation improves efficiency and allows future extension to multiple dimensions.

The initial bandwidths are calculated using a theoretical rule-of-thumb (Scott / Silverman formula):

`h_rule = σ * n^(-1/5)`

where `σ` is the sample standard deviation and `n` is the number of replicas, note we exclude the 1.06 factor that is usually included because it was found not to make a difference

Then the candidate bandwidth matrices are created by scaling the initial values with a range of factors for cross validation. This cross validation is done using `calc_kdeCrossValidation()`, which performs k-fold cross-validation. For each candidate bandwidth, the log-likelihood of validation folds under the KDE fitted on training folds is computed. The bandwidth that maximises the mean log-likelihood is selected as the optimal diagonal bandwidth matrix.

Then with the selected bandwidths, `calc_pdf_and_kde_values()` computes:

- **empirical Gaussian 1D PDF** using the sample mean and standard deviation.
- **1D KDE PDF** using the bandwidth corresponding to each flavour.

and this is visulised through

- `plot_1D_histogram()` displays a histogram of replica values overlaid with the empirical PDF and KDE.
- `plot_1D_histogram_withScatter()` provides a scatter plot of replica values alongside a rotated histogram.

Then to evauluate results, we use `calc_KLDivergence()` to compute the Kullback-Leibler divergence between the empirical PDF and the KDE estimate for each selected flavour.

#### 2D Implementation 

We use `estimate_bandwidth_matrix_scv()` here to calculate the full 2×2 bandwidth matrix for two selected flavours at a given grid index. This is the multivariate equivalent of the 1D bandwidth calculation, allowing correlations between flavours to be captured via the off-diagonal elements.  

The theoretical bandwidth is initially estimated from the standard deviations of the two dimensions, and then optimised using Smoothed Cross-Validation (SCV), implemented in `scv_objective()`. In SCV we construct a positive-definite bandwidth matrix from parameters, compute the leave-one-out KDE for all points, and calculates the negative log-likelihood. Minimising this function selects the bandwidth that best balances smoothness and fidelity to the data. This is a similar method to jackknife

Next, we create a 2D evaluation grid covering the ranges of the two selected flavours and compute the KDE values with `calc_kdeGaussianEstimate_nD()`.  We also compute the empirical 2D Gaussian PDF using the sample mean and covariance of the two dimensions via `multivariate_normal.pdf()`.  

Finally, the results are visualised using `plot_kde_vs_pdf_2d()`, which overlays:

- the scatter of the replica points,
- the empirical 2D Gaussian PDF contours (solid lines),
- the KDE estimate contours (dashed lines).

Optionally, the 2D KL divergence between the KDE and the empirical PDF is calculated with `calc_KLDivergence_2D()`, providing a quantitative measure of how closely the KDE matches the empirical distribution.

Outputs
-------

The scripts in this directory produce Matplotlib figures rather than
writing files by default. Typical plots include:

- 1D histograms of replica values with
- optional scatter‑plus‑histogram layouts for more detailed inspection;
- 2D scatter plots overlaid with contours of the empirical Gaussian and KDE density estimates.

KL divergence values are printed for quick quantitative checks

Assumptions and limitations
---------------------------

- For numerical stability, small regularisation terms (`epsilon`) are added to bandwidth matrices; extremely ill‑conditioned data may still cause optimisation failures.
- The SCV‑based bandwidth optimisation can be computationally expensive for large numbers of replicas or higher dimensions; the 1D helper uses subsampling and diagonal bandwidths for speed.
- The current scripts focus on fixed grid indices; evolution in \(Q\) (or across multiple indices) is not modelled explicitly here.


How to run
----------

From this directory:

```bash
python KDE_histogram_1D.py
python KDE_scatterGraph_2D.py
python plot_histograms.py
```

Each script has a `main` function where you can adjust:

- the list of flavours (e.g. `keys_flav = ['d', 'g']`),
- the chosen grid index (e.g. `index = 28`),
- plotting flags (e.g. enabling KL divergence diagnostics).


Dependencies
------------

- Python packages: `numpy`, `matplotlib`, `scipy`, `scikit-learn`.
- The repository contains a `requirements.txt` one level above this
	directory; a convenient way to install required packages is:

```bash
pip install -r ../requirements.txt
```

Further notes
-------------

- All plotting modules attempt to locate `pythonStyle.mplstyle` by walking one level upward from the script location; keep that file in the top‑level `02_nnpdfDataCode` directory so styles resolve correctly.
- The data files `flavour_basis.pkl` and `evolution_basis.pkl` are expected in `../00_data/` relative to this directory.
