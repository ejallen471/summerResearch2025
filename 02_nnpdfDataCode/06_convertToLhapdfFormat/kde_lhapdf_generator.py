"""
We have the PDFs from KDE reconstruction in the form of mean and covariance matrices for each Q value.

run with following cmd

python kde_lhapdf_generator.py --num-replicas 10 --overwrite

File does the following:

    1. read in the supporting data / information, found in 00_data

    2. collect and sort the Q values - looking through the folders in 02_covarianceGeneration,
    which are named flavour_basis_<Q> and extracting the Q value

    3. for each unique Q value, read in the mean and covariance and check the covariance is symmetric by

    relative asymmetry = ||covariance - covariance transpose||
                        --------------------------------------
                            max(||covariance||, 1)

    then if this is less than 10^-4, take the average of the two covariance elements and continue. if greater, script ends

    4. Checks the eigenvalues >=0 and calculate root covariance matrix - done through np.linalg.eigh.
    If the eigenvalues are negative, they are replaced with zero

    5. Stores the processed information for each Q value in a ReconstructionAtQ object
    (storing q, mean, covariance and covariance_sqrt)

    6. Take each mean vector and reshape into (flavours, x-points),
    then combine for all Q-values so we have (Q-values, flavours, x-points) - written as KDEReconstructed_NNPDF40_<MEMBER NUMBER>.dat

    7. generate random latent vector from standard normal distribution,
    405 dimensional with fixed seed (20250801) - done for each replica

    8. transfom this latent vector by multiplying by covariance sqrt and adding the mean of the PDF

    9. order to make sure the output order is cbar, sbar, ubar, dbar, g, d, u, s, c

    10. write .info file

    11. write one .dat file for each PDF member - KDEReconstructed_NNPDF40_<MEMBER NUMBER>.dat.

"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import shutil

import numpy as np


#############################################################################
### Constants and Flavour Ordering
#############################################################################

DATA_DIR = Path(__file__).resolve().parents[1] / "00_data"
Q_FOLDER_RE = re.compile(r"^flavour_basis_(?P<q>[0-9.eE+-]+)$")

def _load_string_lines(path):
    # Blank lines are ignored so the small configuration files remain readable.
    values = tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{path} must contain unique, non-empty values")
    return values


def _load_integer_list(path):
    # Accept comma-separated or whitespace-separated values.
    tokens = path.read_text().replace(",", " ").split()
    try:
        values = tuple(int(token) for token in tokens)
    except ValueError as error:
        raise ValueError(f"{path} contains a non-integer PID") from error
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{path} must contain unique PIDs")
    return values


def _load_float_list(path):
    # Alpha-s values are stored as plain text rather than Python source code.
    tokens = path.read_text().replace(",", " ").split()
    try:
        values = tuple(float(token) for token in tokens)
    except ValueError as error:
        raise ValueError(f"{path} contains a non-numeric value") from error
    if not values or not np.all(np.isfinite(values)):
        raise ValueError(f"{path} must contain finite numeric values")
    return values


def _load_flavour_to_pid(path):
    # The CSV connects the readable flavour names to the PDG identifiers used
    # by LHAPDF.
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


# These files define the input block order and the required LHAPDF output order.
FLAVOURS = _load_string_lines(DATA_DIR / "04_flavours.txt")
FLAVOUR_TO_PID = _load_flavour_to_pid(DATA_DIR / "05_flavour_to_pid.csv")
LHAPDF_PIDS = _load_integer_list(DATA_DIR / "06_lhapdf_pids.txt")

# Values copied from the AlphaS_Vals field of NNPDF40_nnlo_as_01180_1000.info.
ALPHAS_VALS = _load_float_list(DATA_DIR / "07_alpha_s_values.txt")

@dataclass(frozen=True)
class ReconstructionAtQ:
    """All information required to generate replicas at one fixed Q value."""

    q: float
    # The next three arrays have dimension 405 or 405 x 405.
    mean: np.ndarray
    covariance: np.ndarray
    covariance_sqrt: np.ndarray
    relative_asymmetry: float
    minimum_eigenvalue: float
    clipped_eigenvalues: int
    relative_psd_correction: float


#############################################################################
### Formatting and Input Helpers
#############################################################################

def _format_sequence(values, precision=8):
    # LHAPDF data rows contain space-separated scientific-notation values.
    return " ".join(f"{float(value):.{precision}e}" for value in values)


def _format_yaml_list(values):
    # The .info file uses YAML-style bracketed lists.
    return "[" + ", ".join(f"{float(value):.8e}" for value in values) + "]"


def _load_vector(path, expected_size):
    # Flatten the saved mean so its order stays flavour block, then x index.
    array = np.loadtxt(path, delimiter=",").reshape(-1)
    if array.shape != (expected_size,):
        raise ValueError(f"{path} has shape {array.shape}; expected ({expected_size},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{path} contains non-finite values")
    return array


def _load_matrix(path, expected_size):
    # A covariance must cover every pair of the 9 x 45 reconstructed values.
    array = np.loadtxt(path, delimiter=",")
    expected_shape = (expected_size, expected_size)
    if array.shape != expected_shape:
        raise ValueError(f"{path} has shape {array.shape}; expected {expected_shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{path} contains non-finite values")
    return array


def load_alpha_s_q_grid(path, reconstructed_q_grid):
    """Load and validate the alpha-s Q grid stored by the data-generation stage."""

    try:
        # The stored file uses commas and line breaks only for readability.
        values = np.fromstring(path.read_text().replace(",", " "), sep=" ")
    except OSError as error:
        raise FileNotFoundError(f"Could not read alpha-s Q grid: {path}") from error
    if values.shape != (len(ALPHAS_VALS),):
        raise ValueError(
            f"{path} contains {values.size} alpha-s Q values; "
            f"expected {len(ALPHAS_VALS)}"
        )
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) < 0):
        raise ValueError("Alpha-s Q values must be finite and non-decreasing")
    # Q=4.92 appears twice in the alpha-s metadata because it is a threshold
    # boundary, but it appears only once in the reconstructed PDF grid.
    unique_values = np.unique(values)
    if not np.allclose(unique_values, reconstructed_q_grid, rtol=5e-9, atol=0.0):
        raise ValueError(
            "Unique alpha-s Q values do not match the reconstructed PDF Q-grid"
        )
    return values


#############################################################################
### Covariance Validation and Replica Generation
#############################################################################

def symmetric_psd_sqrt(covariance):
    """Return a symmetric PSD square root and correction diagnostics."""

    # Remove small numerical disagreement between mirrored matrix entries.
    symmetric = 0.5 * (covariance + covariance.T)

    # A real symmetric eigendecomposition gives the unique symmetric square
    # root once all eigenvalues are non-negative.
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    minimum = float(eigenvalues[0])
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    clipped_count = int(np.count_nonzero(eigenvalues < 0.0))
    # Negative modes cannot describe real Gaussian fluctuations. Replace only
    # those modes with zero; genuine zero and positive modes are preserved.
    clipped = np.maximum(eigenvalues, 0.0)
    covariance_psd = (eigenvectors * clipped) @ eigenvectors.T
    covariance_sqrt = (eigenvectors * np.sqrt(clipped)) @ eigenvectors.T
    # Record how much the PSD projection changed the supplied covariance.
    denominator = max(float(np.linalg.norm(symmetric, ord="fro")), scale)
    correction = float(
        np.linalg.norm(covariance_psd - symmetric, ord="fro") / denominator
    )
    return covariance_sqrt, minimum, clipped_count, correction


def discover_reconstructions(covariance_dir, x_grid):
    """Load, validate, sort, and factor every fixed-Q reconstruction."""

    # There are 9 flavours x 45 x points = 405 variables at every Q.
    dimension = len(FLAVOURS) * x_grid.size
    candidates = []
    # Folder names are the source of the reconstructed PDF Q-grid.
    for folder in covariance_dir.iterdir():
        match = Q_FOLDER_RE.match(folder.name) if folder.is_dir() else None
        if match:
            candidates.append((float(match.group("q")), folder))
    if not candidates:
        raise FileNotFoundError(f"No flavour_basis_<Q> folders found in {covariance_dir}")
    if len(candidates) != len({q for q, _ in candidates}):
        raise ValueError("Reconstruction directory contains duplicate Q values")

    reconstructions = []
    for q, folder in sorted(candidates):
        # Each Q folder must contain one KDE mean and covariance in the same
        # flattened flavour/x order.
        mean = _load_vector(folder / "mean_vector_kde.csv", dimension)
        covariance = _load_matrix(folder / "covariance_kde.csv", dimension)
        # Cov(i,j) must equal Cov(j,i). Large disagreement indicates damaged
        # input rather than harmless floating-point noise.
        asymmetry = np.linalg.norm(covariance - covariance.T, ord="fro")
        covariance_norm = max(np.linalg.norm(covariance, ord="fro"), 1.0)
        relative_asymmetry = float(asymmetry / covariance_norm)
        if relative_asymmetry > 1e-4:
            raise ValueError(
                f"{folder / 'covariance_kde.csv'} is not symmetric: "
                f"relative asymmetry={relative_asymmetry:.3e}"
            )
        # The square root transforms a standard-normal vector into correlated
        # fluctuations with the corrected covariance.
        root, minimum, clipped_count, correction = symmetric_psd_sqrt(covariance)
        reconstructions.append(
            ReconstructionAtQ(
                q=q,
                mean=mean,
                covariance=0.5 * (covariance + covariance.T),
                covariance_sqrt=root,
                relative_asymmetry=relative_asymmetry,
                minimum_eigenvalue=minimum,
                clipped_eigenvalues=clipped_count,
                relative_psd_correction=correction,
            )
        )
    return reconstructions


def generate_member_values(reconstructions, latent_vector):
    """Return values shaped ``(n_q, n_flavours, n_x)``."""

    if not reconstructions:
        raise ValueError("At least one reconstruction is required")
    dimension = reconstructions[0].mean.size
    if latent_vector is not None and latent_vector.shape != (dimension,):
        raise ValueError(
            f"Latent vector has shape {latent_vector.shape}; expected ({dimension},)"
        )
    flat_values = []
    for reconstruction in reconstructions:
        # A missing latent vector denotes the central member, which is exactly
        # the reconstructed mean at every Q.
        values = reconstruction.mean.copy()
        if latent_vector is not None:
            # The caller supplies the same latent vector at every Q. This is
            # the explicit approximation used because cross-Q covariances were
            # not reconstructed.
            values += reconstruction.covariance_sqrt @ latent_vector
        flat_values.append(values)
    n_x = dimension // len(FLAVOURS)
    return np.stack(flat_values).reshape(
        len(reconstructions), len(FLAVOURS), n_x
    )


def values_in_lhapdf_order(values):
    """Reorder ``(Q, internal flavour, x)`` to LHAPDF flavour order."""

    # Translate from the stored flavour-block order to the order written in
    # every LHAPDF member file.
    index_by_pid = {
        FLAVOUR_TO_PID[name]: index for index, name in enumerate(FLAVOURS)
    }
    return values[:, [index_by_pid[pid] for pid in LHAPDF_PIDS], :]


#############################################################################
### LHAPDF File Writers
#############################################################################

def write_info_file(
    output_dir,
    set_name,
    num_replicas,
    x_grid,
    q_grid,
    alpha_s_q_grid,
    seed,
):
    """Write metadata derived from the source NNPDF4.0 set."""

    path = output_dir / f"{set_name}.info"
    # NumMembers includes member 0000 (central); NumErrorMembers counts only
    # the randomly generated replicas.
    lines = [
        'SetDesc: "Gaussian moment reconstruction of NNPDF4.0 using pairwise KDE fixed-Q covariances"',
        "Format: lhagrid1",
        "DataVersion: 1",
        f"NumMembers: {num_replicas + 1}",
        f"NumErrorMembers: {num_replicas}",
        "Particle: 2212",
        f"Flavors: [{', '.join(str(pid) for pid in LHAPDF_PIDS)}]",
        "OrderQCD: 2",
        "FlavorScheme: variable",
        "NumFlavors: 4",
        "ErrorType: replicas",
        f"XMin: {x_grid[0]:.8e}",
        f"XMax: {x_grid[-1]:.8e}",
        f"QMin: {q_grid[0]:.8e}",
        f"QMax: {q_grid[-1]:.8e}",
        "MZ: 9.1187600e+01",
        "MUp: 0",
        "MDown: 0",
        "MStrange: 0",
        "MCharm: 1.5100000e+00",
        "MBottom: 4.9200000e+00",
        "MTop: 1.7250000e+02",
        "AlphaS_MZ: 0.1180024",
        "AlphaS_OrderQCD: 2",
        # These alpha-s fields are from the source NNPDF4.0 set.
        "AlphaS_Type: ipol",
        f"AlphaS_Qs: {_format_yaml_list(alpha_s_q_grid)}",
        f"AlphaS_Vals: {_format_yaml_list(ALPHAS_VALS)}",
        "AlphaS_Lambda4: 0.342207",
        "AlphaS_Lambda5: 0.239",
        f"ReconstructionSeed: {seed}",
        "CrossQModel: shared_symmetric_psd_latent_vector",
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def write_member_file(
    output_dir,
    set_name,
    member_index,
    x_grid,
    q_grid,
    values,
):
    """Write one central or replica member in LHAPDF6 lhagrid1 format."""

    # In memory the axes are Q, flavour, x. The writer below changes only the
    # iteration order required on disk; it does not change the values.
    expected_shape = (q_grid.size, len(LHAPDF_PIDS), x_grid.size)
    if values.shape != expected_shape:
        raise ValueError(f"Member values have shape {values.shape}; expected {expected_shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Member values contain non-finite values")
    path = output_dir / f"{set_name}_{member_index:04d}.dat"
    pdf_type = "central" if member_index == 0 else "replica"
    with path.open("w") as handle:
        # Each .dat file contains one LHAPDF subgrid delimited by --- lines.
        handle.write(f"PdfType: {pdf_type}\nFormat: lhagrid1\n---\n")
        handle.write(_format_sequence(x_grid) + "\n")
        handle.write(_format_sequence(q_grid) + "\n")
        handle.write(" ".join(str(pid) for pid in LHAPDF_PIDS) + "\n")
        # lhagrid1 requires x as the outer loop and Q as the inner loop.
        for x_index in range(x_grid.size):
            for q_index in range(q_grid.size):
                handle.write(
                    _format_sequence(values[q_index, :, x_index], precision=12) + "\n"
                )
        handle.write("---\n")
    return path


#############################################################################
### Full Export Pipeline
#############################################################################

def export_set(
    covariance_dir,
    x_grid_path,
    alpha_s_q_grid_path,
    output_parent,
    set_name,
    num_replicas,
    seed,
    overwrite,
):
    """Build an LHAPDF set from all fixed-Q reconstructions."""

    # Validate user-controlled arguments before creating or replacing output.
    if num_replicas < 1:
        raise ValueError("num_replicas must be at least one")
    if not re.fullmatch(r"[A-Za-z0-9_]+", set_name):
        raise ValueError("set_name may contain only letters, digits, and underscores")
    # All covariance inputs were reconstructed on this same 45-point x-grid.
    x_grid = np.loadtxt(x_grid_path).reshape(-1)
    if x_grid.shape != (45,):
        raise ValueError(f"{x_grid_path} contains {x_grid.size} x points; expected 45")
    if not np.all(np.isfinite(x_grid)) or np.any(np.diff(x_grid) <= 0):
        raise ValueError("x-grid must be finite and strictly increasing")

    # Loading also performs symmetry checks and constructs every PSD square root.
    reconstructions = discover_reconstructions(covariance_dir, x_grid)
    if len(reconstructions) != 49:
        raise ValueError(f"Found {len(reconstructions)} Q points; expected 49")
    q_grid = np.array([item.q for item in reconstructions])
    if np.any(np.diff(q_grid) <= 0):
        raise ValueError("Q-grid must be strictly increasing")
    alpha_s_q_grid = load_alpha_s_q_grid(alpha_s_q_grid_path, q_grid)

    output_dir = output_parent / set_name
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    write_info_file(
        output_dir, set_name, num_replicas, x_grid, q_grid, alpha_s_q_grid, seed,
    )
    # LHAPDF convention uses member 0000 for the central prediction.
    central = values_in_lhapdf_order(generate_member_values(reconstructions, None))
    write_member_file(output_dir, set_name, 0, x_grid, q_grid, central)

    # A fixed seed makes the entire replica set exactly reproducible.
    rng = np.random.default_rng(seed)
    dimension = len(FLAVOURS) * x_grid.size
    for member_index in range(1, num_replicas + 1):
        # Reusing z at every Q supplies a smooth, explicit cross-Q identity for
        # a member even though only fixed-Q covariances are available.
        latent = rng.standard_normal(dimension)
        member = values_in_lhapdf_order(
            generate_member_values(reconstructions, latent)
        )
        write_member_file(output_dir, set_name, member_index, x_grid, q_grid, member)

    return output_dir


#############################################################################
### Main Function
#############################################################################

def build_parser():
    # Resolve defaults relative to this file so the command works regardless
    # of the directory from which it is launched.
    script_dir = Path(__file__).resolve().parent
    data_root = script_dir.parent
    parser = argparse.ArgumentParser(
        description="Export the KDE fixed-Q reconstruction to LHAPDF lhagrid1"
    )
    parser.add_argument(
        "--covariance-dir", type=Path,
        default=data_root / "05_Q_gridRun" / "02_covarianceGeneration",
    )
    parser.add_argument(
        "--x-grid", type=Path, default=data_root / "00_data" / "XGRID_45.txt"
    )
    parser.add_argument(
        "--alpha-s-q-grid", type=Path,
        default=data_root / "00_data" / "02_Q_values.txt",
    )
    parser.add_argument("--output-dir", type=Path, default=script_dir / "output")
    parser.add_argument("--set-name", default="KDEReconstructed_NNPDF40")
    parser.add_argument("--num-replicas", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20250801)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    # Parse the command line, run the complete export, then print a short record
    # of the generated ensemble and its cross-Q assumption.
    args = build_parser().parse_args(argv)
    output = export_set(
        covariance_dir=args.covariance_dir,
        x_grid_path=args.x_grid,
        alpha_s_q_grid_path=args.alpha_s_q_grid,
        output_parent=args.output_dir,
        set_name=args.set_name,
        num_replicas=args.num_replicas,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Exported {args.num_replicas + 1} members to {output}")
    print(f"Random seed: {args.seed}")
    print("Cross-Q model: shared symmetric-PSD latent vector")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
