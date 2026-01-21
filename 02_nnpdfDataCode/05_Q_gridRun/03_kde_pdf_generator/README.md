### 03_kde_pdf_generator — README

Overview
--------

- This folder contains the code to generate PDF replicas from KDE-derived mean vectors and
	covariance matrices, compute per-flavour mean and standard deviation
	on the standard 45-point x-grid, and produce diagnostic plots.

Files and Directories
---------------------

- `pdfGenerator.py` — Generate multivariate normal replicas using the
	KDE mean vector and covariance from each `flavour_basis_<Q>` folder
	under `../02_covarianceGeneration`. Saves per-flavour mean and std
	CSVs under `KDE_mean/` and `KDE_std/`. Plotting helpers are present
	but plotting is only performed when the script is run interactively
	(images are saved when enabled).
- `pdf_errorCalculations.py` — Comparison utilities that load
	`KDE_mean`/`KDE_std` and `LHAPDF_mean`/`LHAPDF_std` inputs and
	produce diagnostic scatter plots comparing uncertainties.
- `KDE_mean/`, `KDE_std/` — Output directories created by
	`pdfGenerator.py` containing CSVs named like
	`mean_<flavour>_Q=<Q>.csv` and `std_<flavour>_Q=<Q>.csv`.
- `LHAPDF_mean/`, `LHAPDF_std/` — input directories for the
	LHAPDF comparison pipeline.


Key implementation details
--------------------------

- **X-grid:** both scripts use a fixed 45-point grid (hard-coded) that
	corresponds to the standard evaluation points used across the
	pipeline.
- **Replica generation:** `pdfGenerator.py` generates `Nrep` samples of
	the full PDF vector by drawing multivariate normal replicas with the
	KDE mean vector and covariance. The covariance matrix is forced to
	be positive-definite by symmetric projection followed by eigenvalue
	clipping (minimum eigenvalue ~ 1e-8) to ensure stability.
- **Reshaping & statistics:** replicas are reshaped to
	`(Nrep, 9, 45)` (9 flavours × 45 x-points) and per-flavour means and
	standard deviations are computed and written to CSV.
- **Flavour ordering:** `['u','d','s','c','ubar','dbar','sbar','cbar','g']`.

Outputs
-------

- `KDE_mean/mean_<flavour>_Q=<Q>.csv` — per-flavour mean on the 45-point
	x-grid.
- `KDE_std/std_<flavour>_Q=<Q>.csv` — per-flavour standard deviation.
- Optional plot files (PNG) when plotting is enabled by the scripts.

Assumptions and limitations
---------------------------

- The scripts expect `flavour_basis_<Q>` directories under
	`../02_covarianceGeneration` containing `covariance_kde.csv` and
	`mean_vector_kde.csv`. Covariance matrices should be shape `(405,405)`
	and mean vectors length `405` (9×45).
- Generating large numbers of replicas for full 405-dimensional
	vectors may be memory-intensive. If memory is constrained, reduce
	`Nrep` or process Q-values individually.
- If eigen-decomposition fails for a covariance matrix, that dataset
	is skipped; the scripts print warnings in such cases.


How to run
----------

Run the generator from the `03_pdfGenerator` directory so the style
file path resolves correctly:

```bash
cd 05_Q_gridRun/03_pdfGenerator
python pdfGenerator.py

```

This script reads `../02_covarianceGeneration` for `flavour_basis_<Q>`
folders. To produce comparison plots with LHAPDF outputs:

```bash
python pdf_errorCalculations.py
```

Dependencies
------------

- Python packages: `numpy`, `pandas`, `matplotlib`.
- The repository contains a `requirements.txt` at the project root (for all files); a
	convenient way to install required packages is:

```bash
pip install -r ../../requirements.txt
```

Further notes
-------------

- See `05_Q_gridRun/02_covarianceGeneration/README.md` for details about
	how the KDE mean vectors and covariance matrices are produced and
	any memmap/temporary-file behaviour used by that pipeline.
- The plotting modules use the repository `pythonStyle.mplstyle` file
	for consistent aesthetics; ensure it remains at the project root.