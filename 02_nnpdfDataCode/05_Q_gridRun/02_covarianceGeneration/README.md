### 02_covariance_generation — README

Overview
--------

This directory contains the code and helpers used to reconstruct covariance
and correlation matrices from NNDPF replica data for different values of
`Q` (see `../../00_data/01_Q_ValuesExplained.txt` for how these were determined).
The implementation supports two reconstruction routes:

- a KDE-based reconstruction
- an empirical reconstruction

Files
-----

- `covarianceMatrixReconstruction.py` — main module that builds full
  covariance and correlation matrices. Produces memmap temporary files
  during accumulation and final CSV outputs for KDE and empirical
  reconstructions.
- `meanVecReconstruction.py` — constructs mean vectors per flavour/grid
  point using a 1D KDE (importance sampling) and an empirical mean for
  comparison; saves CSV outputs.
- `plot_matrix_from_csv.py` — plotting helper to visualise
  CSV outputs
- per-`Q` directories — results and intermediate files (memmaps and
  CSVs) are stored under the relevant `Q` folder when the pipeline is
  executed per-`Q`.

Key implementation details
--------------------------

KDE-based reconstruction
- Pairwise 2D KDE is performed for each pair of flavour/grid indices.
- Bandwidth estimation: a small, robust smooth cross-validation (SCV)
  procedure is used to estimate a 2×2 symmetric positive-definite
  bandwidth matrix. The implementation parameterises the bandwidth via
  a Cholesky factor and optimises the SCV objective with L-BFGS-B.
- Numerical stabilisation / regularisation:
  - The Cholesky-to-bandwidth mapping ensures positive-definiteness;
  - A helper `_make_positive_definite_2x2` enforces a minimum determinant
    and adds a small diagonal `eps` when required;
  - SCV optimisation returns a reasonable fallback (`Sigma * c**2`)
    when optimisation fails.
- KDE moment estimation is performed via importance-sampling Monte
  Carlo: samples are drawn from a Gaussian proposal (empirical mean
  and covariance) and reweighted by the KDE density values to estimate
  the KDE mean and covariance.
- For efficiency the KDE evaluation loop is numba-compiled (`@numba.njit`)
  for batch PDF evaluation and the SCV objective.

Empirical reconstruction
- The empirical approach uses standard sample covariance (`np.cov`) on
  flattened replica vectors (multiple flavours and indices). The code
  computes covariance and correlation matrices directly from the
  replicas for comparison with KDE results.

Parallelism and memory-mapped accumulation
- Large reconstructions are parallelised with `joblib.Parallel`.
- To avoid excessive memory use the code accumulates partial results into
  `numpy.memmap` files. Default temporary memmap filenames follow the
  pattern `<tmp_prefix>_cov_kde.dat`, `<tmp_prefix>_cov_emp.dat`,
  `<tmp_prefix>_count_kde.dat`, `<tmp_prefix>_count_emp.dat`.
- After parallel batches are processed these memmaps are normalised by
  their count matrices to produce the final covariance matrices that are saved as .csv files

Outputs
-------

Typical outputs written by the modules are:

- `covariance_kde.csv` — normalised covariance matrix from KDE
- `correlation_kde.csv` — correlation matrix converted from KDE
- `covariance_empirical.csv` — empirical covariance matrix
- `correlation_empirical.csv` — empirical correlation matrix
-- (Bootstrap error files have been removed from the pipeline; empirical
  reconstruction now writes only covariance and correlation CSVs.)
- memmap files (`*.dat`) used as temporary accumulators during the
  parallel build 


Assumptions and limitations
---------------------------

- Flavour ordering: the code expects the flavours to appear in the
  canonical order used in the project. Changing ordering requires
  consistent mapping via `flav_to_index`.
- Grid points: when unspecified the code uses 45 grid points 
- Numerical stability:
  - Extremely small sample sizes or degenerate input covariances may
    produce NaNs; the code attempts to return NaNs on failure and uses
    fallbacks for bandwidth estimation.
  - Importance sampling depends on the overlap between the proposal and
    KDE target; if the effective sample size (sum of weights) is near
    zero the estimator returns NaN for that entry.
- Reproducibility: for Monte Carlo results, set the RNG explicitly
  (the code uses `np.random` by default). Consider seeding before runs
  for repeatable results.

Performance notes
-----------------

- Numba-compiled sections significantly speed up KDE evaluations and the
  SCV objective but require a first-call compilation overhead.
- The SCV optimiser (L-BFGS-B) can be expensive per 2×2 pair. 
- Disk space: memmap accumulators are `dim × dim × 8 bytes` for each
  covariance or count matrix. `

How to run
----------

Run the main module from the `02_covarianceGeneration/flavour_basis_<Q_value>` directory
where the correct `flavour_basis_<Q_value>.pkl` is stored. Then the commands
to run are:

```bash
cd /path/to/02_covarianceGeneration/<Q_value>
python covarianceMatrixReconstruction.py
python meanVecReconstruction.py
python plot_matrix_from_csv.py

```

Notes
- The script reads the flavour pickle from the current directory, so make
  sure the correct `flavour_basis_<Q_value>.pkl` is present there before
  running.
- Toggle behaviour (e.g. whether to run the KDE reconstruction or to
  remove temporary memmaps) by editing the boolean defaults in the
  `main()` function definition at the top of `covarianceMatrixReconstruction.py`.

Dependencies
------------

See the project-level `requirements.txt`. Key packages used here include
 `numpy`, `scipy`, `numba`, `joblib`, `tqdm` and `matplotlib`.

the plotting helper uses the repository style
file `pythonStyle.mplstyle` located at the project root. 

Ensure this file is present when generating plots (the plotting script references
it with a relative path: `../../pythonStyle.mplstyle`).

