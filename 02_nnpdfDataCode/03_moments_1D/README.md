### 03_moments_1D — README

Overview
--------

This directory contains tools for studying 1D moments of NNPDF replica distributions using kernel density estimation (KDE) and empirical estimators. The focus is on the first four moments in one dimension:

- zeroth moment (normalisation check)
- mean
- variance
- excess kurtosis.

The only script is `momentEstimation_1D.py`, which operates on selected flavours at a fixed x-grid index and computes these moments both from the empirical sample and from a 1D Gaussian KDE, together with bootstrap uncertainties. The script is purely numerical.


Files
-----

- `momentEstimation_1D.py` — main entry point for computing 1D moments. Loads replica data, selects a flavour and grid index, constructs a 1D Gaussian KDE with bandwidth chosen by Silverman's rule and cross-validation, and prints empirical and KDE-based estimates of the zeroth, first, second, and fourth moments, together with bootstrap errors.


Data and inputs
---------------

- Replica data are read from the shared `../00_data/` directory, in particular from `flavour_basis.pkl` (and `evolution_basis.pkl`).
- Within the script you can choose:
	- the flavour(s) to analyse (e.g. `u`, `d`, `g`), and
	- the x-grid index at which to evaluate the 1D distributions.


Key implementation details
--------------------------

- KDE construction mirrors the 1D reconstruction code in `01_kde_reconstructions`, using Gaussian kernels.
- Bandwidth selection follows Silverman's rule-of-thumb, optionally refined via cross-validation (as implemented in the script).
- Empirical moments are computed directly from the sample of replicas at the chosen grid point.
- KDE moments are obtained by numerical integration/importance sampling of the KDE.
- Bootstrap errors are provided for both empirical and KDE-based estimates by resampling replicas.


Assumptions and limitations
---------------------------

- The code assumes that the replica pickles in `../00_data/` have the same structure as in the rest of the project (lists of dictionaries keyed by flavour name, each mapping to 1D arrays over the x-grid).
- The moment calculations are performed at a single x-grid index at a time; correlations between different x points are not considered here.
- The KDE uses Gaussian kernels only; no alternative kernels are implemented.


How to run
----------

From this directory, run the main script directly, for example:

```bash
python momentEstimation_1D.py
```

You can edit the flavour list and grid index inside the script to target different 1D distributions.


Dependencies
------------

- Python packages: `numpy`, `scipy`.
- The repository contains a `requirements.txt` one level above this
	directory; a convenient way to install required packages is:

```bash
pip install -r ../requirements.txt
```

Further notes
-------------

- All files use the shared `../00_data/` directory for `flavour_basis.pkl` and `evolution_basis.pkl`.
