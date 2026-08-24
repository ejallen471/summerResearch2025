# 07_observableAnalysis

This folder contains work related to physical-observable predictions and their
comparison across PDF ensembles. It is separate from Q-grid reconstruction and
LHAPDF export.

## Contents

- `analyse_predictions_nnpdf40.ipynb` explores precomputed observable
  predictions from the `NNPDF_original` replicas. It checks reconstruction of
  the central theory prediction from replica predictions and distinguishes
  observables that depend linearly or quadratically on PDFs.

## Required external input

The notebook expects `predictions_with_NNPDF40_by_replica.pkl`. That file is not
currently present in this folder.

## Future role

This stage contains comparisons between observables calculated with
`NNPDF_original` and `KDE_reconstruction`, including work using PineAPPL.
