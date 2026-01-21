### 04_moments_2D — README

Overview
--------

This directory contains tools for studying 2D moments of NNPDF replica
distributions using kernel density estimation (KDE) and empirical
estimators. The focus is on the 2D mean vectors and covariance
matrices for pairs of flavours at a fixed x-grid index.

The only script is `momentEstimation_2D.py`, which computes KDE and
empirical estimates of the mean and covariance, together with bootstrap
uncertainties. Plotting has been separated out into dedicated scripts
to keep the numerical moment calculations clean.


Files
-----

- `momentEstimation_2D.py` — main entry point for computing 2D moments.
	Loads replica data, selects two flavours and a grid index, constructs
	a 2D Gaussian KDE with bandwidth chosen via smoothed cross-validation
	(SCV), and prints KDE-based and empirical mean vectors and covariance
	matrices with bootstrap errors.


Data and inputs
---------------

- Replica data are read from the shared `../00_data/` directory, in
	particular from `flavour_basis.pkl` (and `evolution_basis.pkl` for
	consistency with other code).
- Within the scripts you can choose:
	- the pair of flavours to analyse (e.g. `u` vs `g`), and
	- the x-grid index at which to evaluate the 2D distributions.


Key implementation details
--------------------------

- The 2D KDE uses multivariate Gaussian kernels with a full 2×2 bandwidth matrix.
- The bandwidth matrix is determined by an SCV (smoothed
	cross-validation) procedure implemented in `estimate_bandwidth_matrix_scv`,
	using a Cholesky parameterisation and optimisation via L-BFGS-B.
- KDE evaluation is performed in batches for efficiency using
	`calc_kdeGaussianEstimate_nD`.
- KDE moments (mean vector and covariance matrix) are computed by
	importance sampling, drawing from a Gaussian proposal matched to the
	empirical mean and covariance of the data.
- Empirical moments use standard sample mean and covariance.
- Bootstrap errors are obtained by resampling replicas and recomputing
	both empirical and KDE-based moments.


Assumptions and limitations
---------------------------

- The code assumes that the replica pickles in `../00_data/` have the
	same structure as in the rest of the project (lists of dictionaries
	keyed by flavour name, each mapping to 1D arrays over the x-grid).
- Calculations are performed for one pair of flavours and a single
	x-grid index at a time; correlations across more dimensions or across
	multiple x-points are not treated here.
- The KDE uses Gaussian kernels only; no alternative kernels are
	implemented.
- The plotting script is intended for qualitative inspection of the 2D
	distributions and is not required to obtain numerical moment values.


How to run
----------

From this directory, you can run the numerical moment code and the plotting script directly, for example:

```bash
python momentEstimation_2D.py
python plotKDE_2D.py
```

You can edit the flavour choices and grid index inside the scripts to target different 2D flavour combinations.


Dependencies
------------

- Python packages: `numpy`, `scipy`, `numba`.
- The repository contains a `requirements.txt` one level above this
	directory; a convenient way to install required packages is:

```bash
pip install -r ../requirements.txt
```

Further notes
-------------

- All files use the shared `../00_data/` directory for `flavour_basis.pkl` and `evolution_basis.pkl`.
