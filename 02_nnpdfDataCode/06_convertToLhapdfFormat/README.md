# 06_convertToLhapdfFormat

This folder contains the original rough LHAPDF prototype and a separate
implementation for converting the fixed-Q KDE reconstruction into LHAPDF's
`lhagrid1` format.

## Contents

- `generate_lhapdf.py` is a rough prototype for writing LHAPDF `.info` and
  `.dat` files. AI generated
- `kde_lhapdf_generator.py` is the reconstruction script. It reads the
  49 fixed-Q mean vectors and covariance matrices, validates their dimensions
  and symmetry, generates replicas with a shared latent vector across Q, and
  writes LHAPDF `.info` and `.dat` files. Its
  reconstruction metadata is read from separate files in `00_data`. The files are
    - `02_Q_values.txt`,
    - `04_flavours.txt`,
    - `05_flavour_to_pid.csv`,
    - `06_lhapdf_pids.txt`,
    - `07_alpha_s_values.txt`.
  Another alpha-s Q-grid file can be passed with `--alpha-s-q-grid`.
- `validate_lhapdf_set.py` checks the generated folder and
  metadata, every member's `lhagrid1` layout, the central-value/flavour
  mapping and seeded replica reproduction. It can also require the real LHAPDF
  library to load and evaluate the set.

## Replica construction

Each replica uses one seeded standard-normal latent vector at every Q. At each
Q, the vector is transformed using the symmetric positive-semidefinite square
root of that Q's covariance matrix. This preserves the reconstructed fixed-Q
mean and covariance in expectation while defining an explicit approximate
relationship between Q points.

The exported domain contains the nine reconstructed flavours, 45 x-points up
to approximately 0.665, and 49 unique Q-points. Bottom and anti-bottom values
are not invented.

## Usage

Activate the enviroment from the repository root with:

```bash
source .venv/bin/activate
```

The Python dependency is recorded in `requirements.txt`.

Run a small export first:

```bash
python kde_lhapdf_generator.py --num-replicas 10
```

The intended full command is:

```bash
python kde_lhapdf_generator.py --num-replicas 1000
```

Validate an existing set with:

```bash
python validate_lhapdf_set.py
```
