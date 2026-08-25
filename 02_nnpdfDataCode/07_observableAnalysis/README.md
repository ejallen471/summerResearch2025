# 07_observableAnalysis

This folder contains a calculation of a physical observable
using the original and KDE-reconstructed PDF ensembles. 

The two LHAPDF sets are:

- `NNPDF_original`: the local concise name for the official
  `NNPDF40_nnlo_as_01180_1000` set;
- `KDE_reconstruction`: the 1,000 KDE-generated replicas plus their central
  member.

## Files

- `lhapdf_calc_observables.py` contains the classes that load the PineAPPL
  grid, select the supported partonic channels, evaluate individual LHAPDF
  members, calculate complete replica ensembles, calculate summary statistics
  and apply the shared plotting style.
- `analyse_observables.ipynb` runs the calculation, compares the two PDF
  ensembles and plots their propagated PDF uncertainties.
- `LHCB_DY_8TEV.pineappl.lz4` is the external PineAPPL interpolation grid. It
  is not committed to Git.
- `prediction_uncertainties.png` is created when the final plotting cell is
  run.
- `analyse_predictions_nnpdf40.ipynb` is older exploratory work based on
  precomputed predictions and is separate from the current PineAPPL example.

## Observable

The PineAPPL grid describes forward Drell--Yan production at LHCb:

```text
proton + proton -> Z/gamma* -> lepton + antilepton
```

The metadata identifies the observable as `d sigma / d y_ll`, where
`y_ll` is the rapidity of the dilepton pair. This can be understood as the
direction of the reconstructed Z/gamma* system. The grid has 18 bins from 2.0
to 4.5.

The PineAPPL convolution divides each bin by its stored bin width. The plotted
quantity is a differential cross section in pb

```text
d sigma / d y_ll [pb]
```

## Common-domain restrictions

`KDE_reconstruction` contains only:

```text
d, u, s, c, dbar, ubar, sbar, cbar, g
```

The PineAPPL grid also contains bottom and photon channels. Those channels are
excluded from both PDF ensembles, leaving the common light-flavour/gluon
channels `0, 1, 3, 5, 6, 8`.

The reconstructed x-grid ends at `x_max = 0.664813948`. The PineAPPL grid can
request values up to x = 1. Contributions above the reconstructed limit are
therefore set to zero for both ensembles. Tiny floating-point differences at
an actual grid boundary are clipped to the boundary, but actual out-of-range
values are not extrapolated.

These restrictions make the result a common-domain proof of concept. It is not
the complete published LHCb cross section.

## Replica calculation and PDF uncertainty

The observable is evaluated using members 1 to 1,000 of each LHAPDF set.
the uncertainty plot uses the statistics of the 1,000 replicas.

For every observable bin, the code calculates:

- the replica mean;
- the sample standard deviation, using `ddof=1`.

The shaded band is the replica mean plus or minus one sample standard
deviation.

## Setup and running

Activate the environment from the repository root:

```bash
source .venv/bin/activate
```

Install the recorded Python packages if needed:

```bash
python -m pip install -r 02_nnpdfDataCode/06_convertToLhapdfFormat/requirements.txt
```

The LHAPDF Python bindings must also be available in this environment. Before
running the notebook, generate and validate `KDE_reconstruction` in
`06_convertToLhapdfFormat`. Its `output` directory must contain both
`NNPDF_original` and `KDE_reconstruction`.

Download the PineAPPL grid into this folder with:

```bash
curl -L \
  https://data.nnpdf.science/pineappl/test-data/LHCB_DY_8TEV.pineappl.lz4 \
  -o 02_nnpdfDataCode/07_observableAnalysis/LHCB_DY_8TEV.pineappl.lz4
```

Then open `analyse_observables.ipynb`, select the repository `.venv` kernel and
run the cells in order. A 10-replica calculation is performed first as a small
check before the two full 1,000-replica calculations.
