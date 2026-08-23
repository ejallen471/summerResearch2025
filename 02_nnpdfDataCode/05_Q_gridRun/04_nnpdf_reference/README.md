# 04_nnpdf_reference

This stage reads the original `NNPDF40_nnlo_as_01180_1000` set through
LHAPDF and provides reference results for comparison with the KDE
reconstruction. It does not create a new LHAPDF-format PDF set.

## Contents

- `generate_reference_statistics.py` evaluates the 1,000 original NNPDF4.0
  replicas on the project's 45-point x-grid and Q-grid, then calculates
  per-flavour means and standard deviations and produces reference plots.
- `lhapdf_pdf_set_figures/` contains reference plots generated from the
  original NNPDF4.0 set.

The separate `06_convertToLhapdfFormat` stage is responsible for developing
the exporter for the reconstructed PDF set. Observable analysis is kept in
the separate `07_observableAnalysis` stage.
