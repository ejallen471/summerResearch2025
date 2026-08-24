# 04_nnpdf_reference

This stage reads `NNPDF_original` through LHAPDF and provides reference results
for comparison with `KDE_reconstruction`. It does not create a new
LHAPDF-format PDF set.

`NNPDF_original` is the concise local name for the official source set
`NNPDF40_nnlo_as_01180_1000`.

## Contents

- `generate_reference_statistics.py` evaluates the 1,000 `NNPDF_original`
  replicas on the project's 45-point x-grid and Q-grid, then calculates
  per-flavour means and standard deviations and produces reference plots.
- `lhapdf_pdf_set_figures/` contains reference plots generated from the
  `NNPDF_original` set.

The separate `06_convertToLhapdfFormat` stage is responsible for developing
the exporter for the reconstructed PDF set. Observable analysis is kept in
the separate `07_observableAnalysis` stage.
