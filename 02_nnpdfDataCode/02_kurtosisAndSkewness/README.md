### 02_kurtosisAndSkewness — README

Overview
--------

This directory contains scripts for quantifying and visualising higher
moments of NNPDF replica distributions, focusing on excess kurtosis and
skewness. The goal is to understand how non-Gaussian each parton density
is across the x-grid by comparing:

- **empirical moments**, computed directly from the replica samples, and
- **KDE-based moments**, where a Gaussian kernel density estimate is
	constructed and its moments are evaluated either by importance sampling
	or by numerical integration over a grid.

Most plots are produced as either:

- histograms of the moment values (kurtosis or skewness) accumulated over
	many flavour/grid-index combinations, or
- line graphs of a given moment as a function of grid index, for each
	flavour.


Files and scripts
-----------------

- `kurtosis_calcHistogram.py`
	- Computes excess kurtosis for selected flavours and grid indices.
	- Uses a 1D Gaussian KDE with bandwidth chosen by a diagonal
		bandwidth matrix and cross‑validation (matching the
		`01_kde_reconstructions` implementation) plus importance sampling to
		estimate excess kurtosis.
	- Also computes empirical excess kurtosis directly from the replica
		samples.
	- Aggregates these values over the chosen (flavour, index) pairs and
		produces two histograms: one for KDE-based excess kurtosis and
		one for empirical excess kurtosis.

- `kurtosis_lineGraph.py`
	- Computes excess kurtosis as a function of grid index for all
		flavours at once.
	- For each flavour and grid index, extracts the 1D replica samples,
		selects a 1D KDE bandwidth via the same cross‑validated diagonal
		bandwidth machinery, and estimates excess kurtosis from the KDE by
		importance sampling.
	- In parallel, computes empirical excess kurtosis and bootstrap
		standard errors for both the KDE-based and empirical estimators.
	- Produces a single figure with two line graphs (subplots):
		left = KDE-based excess kurtosis vs grid index with error bars, and
		right = empirical excess kurtosis vs grid index with error bars;
		one coloured line per flavour.

- `skewness_calcHistogram.py`
	- Skewness analogue of `kurtosis_calcHistogram.py`.
	- Computes skewness for selected flavours and grid indices using
		the same 1D KDE + importance‑sampling machinery, but with the
		standardised third central moment instead of the fourth.
	- Also computes empirical skewness directly from the samples.
	- Aggregates skewness values and produces two histograms: one for
		KDE-based skewness and one for empirical skewness.

- `Skewness_lineGraph.py`
	- Skewness analogue of `kurtosis_lineGraph.py`.
	- For a chosen set of flavours, and for every grid index, computes
		KDE-based skewness (via the same bandwidth and importance‑sampling
		setup) and empirical skewness, together with bootstrap errors for
		both.
	- Produces a single figure with two line graphs: left = KDE-based
		skewness vs grid index with error bars; right = empirical skewness
		vs grid index with error bars.


Data and inputs
---------------

All scripts expect the NNPDF replica data (in flavour and evolution
bases) as pickle files stored in the shared `00_data` directory one
level above this folder:

- `../00_data/flavour_basis.pkl`
- `../00_data/evolution_basis.pkl`

Each script controls which flavours and grid indices are used
via small lists or `np.arange(...)` declarations near the top of
`main()`. For example, in the kurtosis histogram script you will find
something like:

- `keys_flav = ['d']`
- `indices = np.arange(5)`

which can be adjusted depending on how many points and which flavours
you want to include in the study.


Key implementation details
--------------------------

KDE implementation now follows the approach used in the
`01_kde_reconstructions` folder: a diagonal bandwidth matrix is
constructed via Silverman’s rule‑of‑thumb and refined by
cross‑validation, then a Gaussian kernel is used to build a 1D KDE for
each (flavour, index) sample.

**Moment estimation from KDE**

For both kurtosis and skewness, the main scripts use the same strategy:

- **Importance sampling from a Gaussian proposal**: draw samples from a
	Gaussian fitted to the data, evaluate both the KDE and the proposal on
	those samples, and form importance weights \(w = p/q\). Moments are
	then estimated as weighted expectations over the proposal samples.

In this framework, excess kurtosis is defined as

\[
	\kappa = \frac{E[(X - \mu)^4]}{\sigma^4} - 3,
\]

and skewness is defined as

\[
	\gamma_1 = \frac{E[(X - \mu)^3]}{\sigma^3}.
\]

**Empirical moments and bootstrap errors**

Each script also computes empirical moments directly from the
replica values, using standard formulas for the mean, variance,
skewness, and kurtosis. To estimate uncertainties, bootstrap resampling
is employed:

- draw many resampled datasets (with replacement),
- recompute the moment of interest for each resample,
- take the standard deviation over bootstrap realisations as the
	standard error.

These errors are then used to draw error bars in the line plots or to
characterise the spread of histogrammed values.

How to run
----------

From this directory, you can run any of the scripts directly, for
example:

```bash
python kurtosis_calcHistogram.py
python kurtosis_lineGraph.py
python kurtosis_lineGraph_PerFlavour.py
python "skewness_calcHistogram.py"   
python Skewness_lineGraph.py
```

Before running, you may want to edit the `keys_flav`, `indices`, or
related variables in each script’s `main()` function to match the
flavours and grid points you care about. The scripts will write PNG
files (e.g. `histogram_kurtosis.png`,
`excess_kurtosis_vs_index_all_flavours.png`,
`histogram_skewness.png`, `skewness_vs_index_all_flavours.png`) into the
current directory.


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

- All plotting modules use `plt.style.use('../pythonStyle.mplstyle')`
	directly; keep `pythonStyle.mplstyle` in the parent `02_nnpdfDataCode` directory
- All files use the shared `../00_data/` directory for `flavour_basis.pkl` and `evolution_basis.pkl`.

