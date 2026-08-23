# 07_observableAnalysis

This folder contains work related to physical-observable predictions and their
comparison across PDF ensembles. It is separate from Q-grid reconstruction and
LHAPDF export.

## Contents

- `analyse_predictions_nnpdf40.ipynb` explores precomputed observable
  predictions from the original NNPDF4.0 replicas. It checks reconstruction of
  the central theory prediction from replica predictions and distinguishes
  observables that depend linearly or quadratically on PDFs.

## Required external input

The notebook expects `predictions_with_NNPDF40_by_replica.pkl`. That file is not
currently present in this folder.

## Future role

This stage will eventually contain comparisons between observables calculated
with the original NNPDF4.0 set and the reconstructed LHAPDF set, including work
using PineAPPL.
