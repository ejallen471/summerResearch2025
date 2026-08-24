"""
We have a generated KDE PDF set in LHAPDF format, containing one .info file,
one central .dat member and a collection of replica .dat members.

Run with

python validate_lhapdf_set.py

This file does the following:

1. Read the expected flavour order, flavour-to-PID mapping, LHAPDF PID order,
   Q values and alpha-s values from 00_data.

2. Find the generated set directory and check that it contains an .info file
   whose name matches the set directory.

3. Read NumMembers from the .info file. Check that the correct number of .dat
   files exists and that their names form a continuous sequence:

       <SET NAME>_0000.dat, <SET NAME>_0001.dat, ...

4. Parse every .dat member independently from the generator. Check the
   PdfType, lhagrid1 format declaration, --- delimiters, x-grid, Q-grid,
   flavour PIDs, number of data rows, number of values per row and finite PDF
   values. Check that member 0000 is central and all other members are
   replicas.

5. Check that every member uses the same strictly increasing x-grid and Q-grid
   and the expected LHAPDF flavour order. Confirm that the grid contains 45 x
   values and 49 Q values.

6. Check the .info metadata, including the format, member counts, flavour
   order, x and Q limits, uncertainty type, alpha-s Q values and alpha-s
   values.

7. Read the original KDE mean vectors and covariance matrices from
   02_covarianceGeneration. Check their dimensions, finite values and Q order.

8. Reorder the original KDE mean vectors into LHAPDF flavour order and confirm
   that member 0000 matches those means exactly.

9. Recreate the covariance square roots, random latent vectors and replicas
   using the seed stored in the .info file. Confirm that every generated
   replica can be reproduced numerically.

10. Load the complete set using the installed LHAPDF library. Confirm that
    LHAPDF finds the expected number of members and that xfxQ returns the
    expected central values at selected low, middle and high x and Q knots.

The script ends with PASS only when the structural, numerical and LHAPDF
library checks all succeed.
"""

import argparse
import ast
import csv
import re
from pathlib import Path

import numpy as np


#############################################################################
### Constants and Flavour Ordering
#############################################################################

DATA_DIR = Path(__file__).resolve().parents[1] / "00_data"
Q_FOLDER_RE = re.compile(r"^flavour_basis_(?P<q>[0-9.eE+-]+)$")


def load_string_lines(path):
    values = tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{path} must contain unique, non-empty values")
    return values


def load_integer_list(path):
    try:
        values = tuple(
            int(token) for token in path.read_text().replace(",", " ").split()
        )
    except ValueError as error:
        raise ValueError(f"{path} contains a non-integer PID") from error
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{path} must contain unique PIDs")
    return values


def load_float_list(path):
    try:
        values = tuple(
            float(token) for token in path.read_text().replace(",", " ").split()
        )
    except ValueError as error:
        raise ValueError(f"{path} contains a non-numeric value") from error
    if not values or not np.all(np.isfinite(values)):
        raise ValueError(f"{path} must contain finite numeric values")
    return values


def load_flavour_to_pid(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"flavour", "pid"}:
        raise ValueError(f"{path} must have flavour,pid columns")
    try:
        mapping = {row["flavour"].strip(): int(row["pid"]) for row in rows}
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} contains an invalid flavour or PID") from error
    if len(mapping) != len(rows) or len(set(mapping.values())) != len(mapping):
        raise ValueError(f"{path} contains duplicate flavours or PIDs")
    return mapping


FLAVOURS = load_string_lines(
    DATA_DIR / "04_flavours.txt"
)
FLAVOUR_TO_PID = load_flavour_to_pid(
    DATA_DIR / "05_flavour_to_pid.csv"
)
EXPECTED_PIDS = load_integer_list(DATA_DIR / "06_lhapdf_pids.txt")
EXPECTED_ALPHA_S_QS = load_float_list(DATA_DIR / "02_Q_values.txt")
EXPECTED_ALPHA_S_VALS = load_float_list(
    DATA_DIR / "07_alpha_s_values.txt"
)

if set(FLAVOUR_TO_PID) != set(FLAVOURS):
    raise ValueError("Flavours and flavour-to-PID mapping do not agree")
if set(EXPECTED_PIDS) != set(FLAVOUR_TO_PID.values()):
    raise ValueError("LHAPDF PID order and flavour-to-PID mapping do not agree")


#############################################################################
### Parsing Helpers
#############################################################################

def parse_info_file(path):
    """Read the simple YAML-style fields used by an LHAPDF info file."""

    metadata = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"{path}:{line_number} has no ':' separator")
        key, value = line.split(":", 1)
        if key in metadata:
            raise ValueError(f"{path} contains duplicate field {key!r}")
        metadata[key] = value.strip()
    return metadata


def metadata_integer(metadata, key):
    try:
        return int(metadata[key])
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid or missing integer metadata field {key}") from error


def metadata_float(metadata, key):
    try:
        return float(metadata[key])
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid or missing float metadata field {key}") from error


def metadata_list(metadata, key):
    try:
        values = ast.literal_eval(metadata[key])
    except (KeyError, SyntaxError, ValueError) as error:
        raise ValueError(f"Invalid or missing list metadata field {key}") from error
    if not isinstance(values, list):
        raise ValueError(f"Metadata field {key} is not a list")
    return values


def parse_float_row(text, description):
    """Parse a whitespace-separated row without accepting partial input."""

    try:
        values = np.array([float(value) for value in text.split()], dtype=float)
    except ValueError as error:
        raise ValueError(f"Invalid numeric value in {description}") from error
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"Empty or non-finite numeric row in {description}")
    return values


def parse_member(path):
    """Parse one single-subgrid lhagrid1 member without using the writer."""

    lines = path.read_text().splitlines()
    if len(lines) < 8:
        raise ValueError(f"{path} is too short to be an lhagrid1 member")
    if lines[0] not in {"PdfType: central", "PdfType: replica"}:
        raise ValueError(f"{path} has an invalid PdfType header")
    if lines[1] != "Format: lhagrid1":
        raise ValueError(f"{path} is not Format: lhagrid1")
    if lines[2] != "---" or lines[-1] != "---":
        raise ValueError(f"{path} has invalid subgrid delimiters")

    x_grid = parse_float_row(lines[3], f"{path} x-grid")
    q_grid = parse_float_row(lines[4], f"{path} Q-grid")
    try:
        pids = tuple(int(value) for value in lines[5].split())
    except ValueError as error:
        raise ValueError(f"{path} has an invalid flavour-ID row") from error
    if x_grid.size == 0 or q_grid.size == 0 or len(pids) == 0:
        raise ValueError(f"{path} contains an empty grid")

    data_lines = lines[6:-1]
    expected_rows = x_grid.size * q_grid.size
    if len(data_lines) != expected_rows:
        raise ValueError(
            f"{path} has {len(data_lines)} data rows; expected {expected_rows}"
        )
    rows = [
        parse_float_row(line, f"{path} data row {index + 1}")
        for index, line in enumerate(data_lines)
    ]
    if any(row.size != len(pids) for row in rows):
        raise ValueError(f"{path} has a data row with the wrong number of flavours")
    values = np.vstack(rows)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains a non-finite PDF value")

    # lhagrid1 stores x in the outer loop and Q in the inner loop.
    values = values.reshape(x_grid.size, q_grid.size, len(pids)).transpose(1, 2, 0)
    return lines[0].split(":", 1)[1].strip(), x_grid, q_grid, pids, values


#############################################################################
### Source Reconstruction Checks
#############################################################################

def discover_sources(covariance_dir, expected_dimension):
    """Load source means and covariances in increasing Q order."""

    sources = []
    for folder in covariance_dir.iterdir():
        match = Q_FOLDER_RE.match(folder.name) if folder.is_dir() else None
        if not match:
            continue
        mean = np.loadtxt(folder / "mean_vector_kde.csv", delimiter=",").reshape(-1)
        covariance = np.loadtxt(folder / "covariance_kde.csv", delimiter=",")
        if mean.shape != (expected_dimension,):
            raise ValueError(f"{folder}/mean_vector_kde.csv has shape {mean.shape}")
        if covariance.shape != (expected_dimension, expected_dimension):
            raise ValueError(f"{folder}/covariance_kde.csv has shape {covariance.shape}")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise ValueError(f"{folder} contains non-finite source values")
        sources.append((float(match.group("q")), mean, covariance))
    sources.sort(key=lambda item: item[0])
    return sources


def values_in_pid_order(flat_values, x_count):
    """Convert flavour blocks to the LHAPDF PID order."""

    flavours = flat_values.reshape(len(FLAVOURS), x_count)
    flavour_index = {name: index for index, name in enumerate(FLAVOURS)}
    pid_to_flavour = {pid: name for name, pid in FLAVOUR_TO_PID.items()}
    return np.stack(
        [flavours[flavour_index[pid_to_flavour[pid]]] for pid in EXPECTED_PIDS]
    )


def covariance_sqrt(covariance):
    """Reproduce the documented symmetric PSD projection."""

    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, 0.0)
    return (eigenvectors * np.sqrt(clipped)) @ eigenvectors.T


#############################################################################
### Validation Checks
#############################################################################

def validate_metadata(metadata, set_name, member_count, x_grid, q_grid):
    required = {
        "SetDesc", "Format", "DataVersion", "NumMembers", "NumErrorMembers",
        "Particle", "Flavors", "OrderQCD", "FlavorScheme", "NumFlavors",
        "ErrorType", "XMin", "XMax", "QMin", "QMax", "AlphaS_Qs",
        "AlphaS_Vals",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"{set_name}.info is missing fields: {', '.join(missing)}")
    if metadata["Format"] != "lhagrid1":
        raise ValueError("Info metadata does not specify Format: lhagrid1")
    if metadata["ErrorType"] != "replicas":
        raise ValueError("Info metadata does not specify ErrorType: replicas")
    if metadata_integer(metadata, "NumMembers") != member_count:
        raise ValueError("NumMembers does not match the number of member files")
    if metadata_integer(metadata, "NumErrorMembers") != member_count - 1:
        raise ValueError("NumErrorMembers must equal NumMembers - 1")
    if tuple(metadata_list(metadata, "Flavors")) != EXPECTED_PIDS:
        raise ValueError("Info Flavors do not match the reconstructed PID order")
    for key, actual in (
        ("XMin", x_grid[0]), ("XMax", x_grid[-1]),
        ("QMin", q_grid[0]), ("QMax", q_grid[-1]),
    ):
        if not np.isclose(metadata_float(metadata, key), actual, rtol=5e-9, atol=0.0):
            raise ValueError(f"Metadata {key} does not match the member grid")
    alpha_q = np.asarray(metadata_list(metadata, "AlphaS_Qs"), dtype=float)
    alpha_values = np.asarray(metadata_list(metadata, "AlphaS_Vals"), dtype=float)
    if alpha_q.size != alpha_values.size or alpha_q.size == 0:
        raise ValueError("AlphaS_Qs and AlphaS_Vals have incompatible lengths")
    if np.any(np.diff(alpha_q) < 0) or not np.all(np.isfinite(alpha_values)):
        raise ValueError("AlphaS metadata is not finite and non-decreasing in Q")
    if not np.array_equal(alpha_q, np.asarray(EXPECTED_ALPHA_S_QS)):
        raise ValueError("AlphaS_Qs metadata does not match 02_Q_values.txt")
    if not np.array_equal(alpha_values, np.asarray(EXPECTED_ALPHA_S_VALS)):
        raise ValueError("AlphaS_Vals metadata does not match 07_alpha_s_values.txt")


def validate_with_lhapdf(output_parent, set_name, members, central):
    """Ask the installed LHAPDF implementation to parse and evaluate the set."""

    try:
        import lhapdf
    except ImportError as error:
        raise RuntimeError("LHAPDF Python bindings are not installed") from error
    lhapdf.pathsPrepend(str(output_parent.resolve()))
    pdf_set = lhapdf.mkPDFs(set_name)
    if len(pdf_set) != members:
        raise ValueError(f"LHAPDF loaded {len(pdf_set)} members; expected {members}")
    pdf = pdf_set[0]
    x_grid, q_grid = central[0], central[1]
    values = central[2]
    for q_index in {0, q_grid.size // 2, q_grid.size - 1}:
        for x_index in {0, x_grid.size // 2, x_grid.size - 1}:
            for flavour_index, pid in enumerate(EXPECTED_PIDS):
                actual = pdf.xfxQ(pid, float(x_grid[x_index]), float(q_grid[q_index]))
                expected = values[q_index, flavour_index, x_index]
                if not np.isclose(actual, expected, rtol=1e-6, atol=1e-12):
                    raise ValueError(
                        f"LHAPDF knot mismatch for pid={pid}, x={x_grid[x_index]}, "
                        f"Q={q_grid[q_index]}: {actual} != {expected}"
                    )


#############################################################################
### Main Function
#############################################################################

def build_parser():
    script_dir = Path(__file__).resolve().parent
    data_root = script_dir.parent
    parser = argparse.ArgumentParser(description="Validate a generated KDE LHAPDF set")
    parser.add_argument(
        "--set-dir", type=Path,
        default=script_dir / "output" / "KDE_reconstruction",
    )
    parser.add_argument(
        "--covariance-dir", type=Path,
        default=data_root / "05_Q_gridRun" / "02_covarianceGeneration",
    )
    parser.add_argument(
        "--skip-replica-reproduction", action="store_true",
        help="Skip exact seeded reproduction of every replica",
    )
    return parser


def main():
    args = build_parser().parse_args()
    set_dir = args.set_dir.resolve()
    set_name = set_dir.name
    info_path = set_dir / f"{set_name}.info"
    if not set_dir.is_dir():
        raise FileNotFoundError(f"Set directory does not exist: {set_dir}")
    if not info_path.is_file():
        raise FileNotFoundError(f"Matching info file does not exist: {info_path}")

    metadata = parse_info_file(info_path)
    expected_members = metadata_integer(metadata, "NumMembers")
    member_paths = sorted(set_dir.glob(f"{set_name}_[0-9][0-9][0-9][0-9].dat"))
    if len(member_paths) != expected_members:
        raise ValueError(
            f"Found {len(member_paths)} member files; metadata expects {expected_members}"
        )
    expected_names = [f"{set_name}_{index:04d}.dat" for index in range(expected_members)]
    if [path.name for path in member_paths] != expected_names:
        raise ValueError("Member filenames are not a continuous zero-based sequence")

    parsed_members = []
    reference_x = reference_q = None
    for index, member_path in enumerate(member_paths):
        pdf_type, x_grid, q_grid, pids, values = parse_member(member_path)
        expected_type = "central" if index == 0 else "replica"
        if pdf_type != expected_type:
            raise ValueError(f"{member_path} is {pdf_type}; expected {expected_type}")
        if pids != EXPECTED_PIDS:
            raise ValueError(f"{member_path} has unexpected flavour IDs {pids}")
        if np.any(np.diff(x_grid) <= 0) or np.any(np.diff(q_grid) <= 0):
            raise ValueError(f"{member_path} grids are not strictly increasing")
        if index == 0:
            reference_x, reference_q = x_grid, q_grid
        elif not np.array_equal(x_grid, reference_x) or not np.array_equal(q_grid, reference_q):
            raise ValueError(f"{member_path} grids differ from the central member")
        parsed_members.append(values)

    validate_metadata(
        metadata, set_name, len(member_paths), reference_x, reference_q
    )
    if reference_x.size != 45 or reference_q.size != 49:
        raise ValueError(
            f"Expected a 45 x 49 grid; found {reference_x.size} x {reference_q.size}"
        )
    print(f"PASS FORMAT: {len(member_paths)} members on a 45 x 49 grid")

    dimension = len(FLAVOURS) * reference_x.size
    sources = discover_sources(args.covariance_dir.resolve(), dimension)
    source_q = np.array([item[0] for item in sources])
    if not np.allclose(source_q, reference_q, rtol=5e-9, atol=0.0):
        raise ValueError("Source Q folders do not match the generated Q-grid")
    expected_central = np.stack(
        [values_in_pid_order(item[1], reference_x.size) for item in sources]
    )
    if not np.array_equal(parsed_members[0], expected_central):
        error = float(np.max(np.abs(parsed_members[0] - expected_central)))
        raise ValueError(f"Central member does not match source means; max error={error}")
    print("PASS NUMERICAL: central values and flavour mapping match the sources exactly")

    if not args.skip_replica_reproduction:
        seed = metadata_integer(metadata, "ReconstructionSeed")
        square_roots = [covariance_sqrt(item[2]) for item in sources]
        rng = np.random.default_rng(seed)
        for member_index in range(1, len(parsed_members)):
            latent = rng.standard_normal(dimension)
            expected = np.stack([
                values_in_pid_order(source[1] + root @ latent, reference_x.size)
                for source, root in zip(sources, square_roots)
            ])
            if not np.allclose(parsed_members[member_index], expected, rtol=5e-12, atol=0.0):
                error = float(np.max(np.abs(parsed_members[member_index] - expected)))
                raise ValueError(
                    f"Member {member_index} is not reproducible from its seed; "
                    f"max error={error}"
                )
        print(f"PASS NUMERICAL: reproduced all {len(parsed_members) - 1} replicas from the seed")

    try:
        validate_with_lhapdf(
            set_dir.parent, set_name, len(member_paths),
            (reference_x, reference_q, parsed_members[0]),
        )
    except (RuntimeError, ValueError) as error:
        print(f"FAIL LHAPDF: {error}")
        return 1
    print("PASS LHAPDF: the installed LHAPDF library loaded and evaluated the set")

    print("PASS: validation completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
