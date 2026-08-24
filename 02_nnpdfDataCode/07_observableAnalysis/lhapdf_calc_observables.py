"""
Calculate a Drell-Yan observable using PineAPPL and LHAPDF.

This file contains the calculation classes to be imported and run from the notebook.

The selected PineAPPL grid contains bottom and photon contributions, but the
KDE-reconstructed PDF set contains only d, u, s, c, their antiquarks and the
gluon. Both PDF ensembles will therefore be evaluated using the same supported
light-flavour QCD channels: 0, 1, 3, 5, 6 and 8. Contributions outside the
reconstructed x-domain are also set to zero for both ensembles. The result is
a common-domain proof of concept, not the complete LHCb cross section.

In this file we do the following

1. Define the paths, PDF-set names and supported PineAPPL channels.
2. Load the PineAPPL grid and configure the local LHAPDF search path.
3. Construct the proton convolution and light-flavour channel mask.
4. Calculate the observable for one PDF member.
5. Calculate the observable for all requested members of a PDF ensemble.
6. Calculate the ensemble mean, standard deviation and percentile interval.
7. Return the bin information needed by the plotting notebook.
"""

from pathlib import Path

import lhapdf
import matplotlib.pyplot as plt
import numpy as np

from pineappl.convolutions import Conv, ConvType
from pineappl.grid import Grid


#############################################################################
### Paths and Constants
#############################################################################

SCRIPT_DIR = Path(__file__).resolve().parent

# Keeping the paths relative to this file means the notebook can be opened
# from a different working directory without losing the data or style files.
STYLE_PATH = SCRIPT_DIR.parent / "pythonStyle.mplstyle"
GRID_PATH = SCRIPT_DIR / "LHCB_DY_8TEV.pineappl.lz4"
RECONSTRUCTED_SET_DIR = SCRIPT_DIR.parent / "06_convertToLhapdfFormat" / "output"

# These are the short local names used in this project. NNPDF_original is a
# renamed local copy of the official NNPDF40_nnlo_as_01180_1000 set.
ORIGINAL_SET_NAME = "NNPDF_original"
RECONSTRUCTED_SET_NAME = "KDE_reconstruction"

# These channels contain only d, u, s, c, their antiquarks and the gluon.
# The other channels need bottom quarks or photons, which KDE_reconstruction
# does not contain, so they are left out of both sides of the comparison.
SUPPORTED_CHANNELS = (0, 1, 3, 5, 6, 8)


#############################################################################
### Plot Formatting
#############################################################################

class ObservablePlotStyle:
    """Apply the shared plot style and standard legend position."""

    def __init__(self, style_path=STYLE_PATH):
        self.style_path = Path(style_path)

    def apply(self):
        """Load the repository Matplotlib style file."""

        if not self.style_path.is_file():
            raise FileNotFoundError(f"Matplotlib style not found: {self.style_path}")
        plt.style.use(self.style_path)

    def add_legend(self, axes):
        """Place a two-column legend centrally below an axes."""

        # The legend sits outside the grey plotting area so that it cannot
        # cover either prediction or its uncertainty band.
        axes.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=2,
        )

    def save(self, figure, filename):
        """Save a notebook figure in the observable-analysis folder."""

        # bbox_inches="tight" is important here because the legend is below
        # the axes and would otherwise be cut off in the saved image.
        output_path = SCRIPT_DIR / filename
        figure.savefig(output_path, bbox_inches="tight")
        return output_path


#############################################################################
### PineAPPL Observable Calculation
#############################################################################

class PineAPPLObservable:
    """Load the grid and calculate its observable with LHAPDF members."""

    def __init__(
        self,
        grid_path=GRID_PATH,
        reconstructed_set_dir=RECONSTRUCTED_SET_DIR,
        supported_channels=SUPPORTED_CHANNELS,
    ):
        self.grid_path = Path(grid_path)
        self.reconstructed_set_dir = Path(reconstructed_set_dir)
        self.supported_channels = tuple(supported_channels)

        # These are filled by load(). Starting with None makes it clear if a
        # calculation is attempted before the PineAPPL grid has been loaded.
        self.grid = None
        self.proton = None
        self.channel_mask = None
        self.analysis_x_min = None
        self.analysis_x_max = None
        self.analysis_q2_min = None
        self.analysis_q2_max = None

    def load(self):
        """Load the PineAPPL grid and prepare its convolution inputs."""

        if not self.grid_path.is_file():
            raise FileNotFoundError(f"PineAPPL grid not found: {self.grid_path}")
        if not self.reconstructed_set_dir.is_dir():
            raise FileNotFoundError(
                f"Reconstructed LHAPDF directory not found: "
                f"{self.reconstructed_set_dir}"
            )

        # LHAPDF normally searches its system installation. Our two local PDF
        # folders live in output/, so that directory is added to the front of
        # the search path before either set is requested.
        lhapdf.setVerbosity(0)
        lhapdf.pathsPrepend(str(self.reconstructed_set_dir))

        self.grid = Grid.read(str(self.grid_path))

        # The grid describes two ordinary incoming protons. "polarized=False"
        # means that we are not keeping track of proton spin, and
        # "time_like=False" identifies an incoming rather than outgoing PDF.
        proton_type = ConvType(polarized=False, time_like=False)
        self.proton = Conv(convolution_types=proton_type, pid=2212)
        self.channel_mask = self.make_channel_mask()

        # The original PDF covers a wider x range than the reconstruction. A
        # fair comparison therefore uses the smaller reconstructed domain for
        # both sets, rather than giving the original set extra phase space.
        reconstructed_central = lhapdf.mkPDF(RECONSTRUCTED_SET_NAME, 0)
        self.analysis_x_min = reconstructed_central.xMin
        self.analysis_x_max = reconstructed_central.xMax
        self.analysis_q2_min = reconstructed_central.q2Min
        self.analysis_q2_max = reconstructed_central.q2Max
        return self

    def make_channel_mask(self):
        """Select only the channels supported by the reconstructed PDF set."""

        if self.grid is None:
            raise RuntimeError("Load the PineAPPL grid before making the channel mask")

        number_of_channels = len(self.grid.channels())
        invalid_channels = [
            channel for channel in self.supported_channels
            if channel < 0 or channel >= number_of_channels
        ]
        if invalid_channels:
            raise ValueError(
                f"Channel indices are outside the grid: {invalid_channels}"
            )

        # PineAPPL expects one True/False value for every channel in the grid.
        # We begin with everything switched off and enable only the six
        # light-flavour/gluon channels that both PDF sets can supply.
        channel_mask = np.zeros(number_of_channels, dtype=bool)
        channel_mask[list(self.supported_channels)] = True
        return channel_mask

    def load_pdf_member(self, set_name, member_index):
        """Load one member from an LHAPDF set."""

        if self.grid is None:
            raise RuntimeError("Call load() before loading PDF members")
        if member_index < 0:
            raise ValueError("member_index cannot be negative")
        if set_name not in lhapdf.availablePDFSets():
            raise FileNotFoundError(
                f"LHAPDF set {set_name} is not installed in any active "
                f"LHAPDF search path"
            )

        pdf_set = lhapdf.getPDFSet(set_name)
        if member_index >= pdf_set.size:
            raise ValueError(
                f"{set_name} contains {pdf_set.size} members; "
                f"member {member_index} does not exist"
            )
        return lhapdf.mkPDF(set_name, member_index)

    def make_pdf_function(self, pdf):
        """Return an xfxQ2 callback restricted to the common PDF domain."""

        # Each LHAPDF member advertises the range over which it can safely be
        # evaluated. PineAPPL will call the small function below many times,
        # passing it a parton ID, x and Q squared.
        x_min = pdf.xMin
        x_max = pdf.xMax
        q2_min = pdf.q2Min
        q2_max = pdf.q2Max

        def clip_boundary(value, lower, upper, name):
            # A grid boundary can differ by a final floating-point digit after
            # being written and read. That tiny difference is harmless, but a
            # genuinely out-of-range request should never be silently clipped.
            if value < lower:
                if np.isclose(value, lower, rtol=1e-8, atol=0.0):
                    return lower
                raise ValueError(f"PineAPPL requested {name}={value} below {lower}")
            if value > upper:
                if np.isclose(value, upper, rtol=1e-8, atol=0.0):
                    return upper
                raise ValueError(f"PineAPPL requested {name}={value} above {upper}")
            return value

        def xfx_q2(pid, x, q2):
            # PineAPPL and LHAPDF can represent the same boundary with slightly
            # different final digits. Clip only floating-point-level boundary
            # differences. Genuine out-of-domain x contributions are omitted
            # from both ensembles, rather than extrapolated from either PDF.
            if x < self.analysis_x_min:
                if np.isclose(x, self.analysis_x_min, rtol=1e-8, atol=0.0):
                    x = self.analysis_x_min
                else:
                    return 0.0
            if x > self.analysis_x_max:
                if np.isclose(x, self.analysis_x_max, rtol=1e-8, atol=0.0):
                    x = self.analysis_x_max
                else:
                    return 0.0

            if q2 < self.analysis_q2_min or q2 > self.analysis_q2_max:
                raise ValueError(
                    f"PineAPPL requested Q2={q2} outside the common PDF domain "
                    f"[{self.analysis_q2_min}, {self.analysis_q2_max}]"
                )

            safe_x = clip_boundary(x, x_min, x_max, "x")
            safe_q2 = clip_boundary(q2, q2_min, q2_max, "Q2")

            # LHAPDF returns x times the PDF, which is exactly the convention
            # expected by PineAPPL's convolution callback.
            return pdf.xfxQ2(pid, safe_x, safe_q2)

        return xfx_q2

    def calculate_member(self, set_name, member_index):
        """Return the observable prediction for one PDF member."""

        pdf = self.load_pdf_member(set_name, member_index)

        # PineAPPL combines its stored perturbative weights with this member's
        # PDFs and alpha_s. The result contains one cross-section value for
        # each muon-direction bin in the grid.
        prediction = self.grid.convolve(
            pdg_convs=[self.proton],
            xfxs=[self.make_pdf_function(pdf)],
            alphas=pdf.alphasQ2,
            channel_mask=self.channel_mask,
        )
        prediction = np.asarray(prediction, dtype=float)

        if prediction.shape != (self.grid.bins(),):
            raise ValueError(
                f"Unexpected prediction shape {prediction.shape}; "
                f"expected {(self.grid.bins(),)}"
            )
        if not np.all(np.isfinite(prediction)):
            raise ValueError(
                f"{set_name} member {member_index} produced non-finite values"
            )
        return prediction

    def calculate_ensemble(self, set_name, member_indices):
        """Return one observable prediction row for every requested member."""

        member_indices = list(member_indices)
        if not member_indices:
            raise ValueError("At least one PDF member must be requested")

        predictions = []
        for position, member_index in enumerate(member_indices, start=1):
            predictions.append(self.calculate_member(set_name, member_index))

            # A full run evaluates 1,000 members and can take a while. This
            # progress message reassures us that the calculation is advancing.
            if position % 100 == 0 or position == len(member_indices):
                print(
                    f"Calculated {position}/{len(member_indices)} members "
                    f"from {set_name}"
                )
        # Rows correspond to replicas and columns correspond to observable
        # bins. This is the shape used by ObservableStatistics below.
        return np.vstack(predictions)

    def bin_limits(self):
        """Return the lower and upper pseudorapidity limit of every bin."""

        if self.grid is None:
            raise RuntimeError("Call load() before requesting bin information")
        limits = self.grid.bin_limits()
        return np.asarray(
            [[bin_limits[0][0], bin_limits[0][1]] for bin_limits in limits],
            dtype=float,
        )

    def bin_centres(self):
        """Return the pseudorapidity value at the centre of every bin."""

        # For a bin [low, high], the plotted x position is simply their average.
        return self.bin_limits().mean(axis=1)


#############################################################################
### PDF Uncertainty Calculation
#############################################################################

class ObservableStatistics:
    """Calculate summary statistics from replica observable predictions."""

    def __init__(self, replica_predictions):
        self.replica_predictions = np.asarray(replica_predictions, dtype=float)
        self.validate()

    def validate(self):
        """Check that the prediction array is finite and two-dimensional."""

        if self.replica_predictions.ndim != 2:
            raise ValueError(
                "replica_predictions must have shape (replicas, observable bins)"
            )
        if self.replica_predictions.shape[0] < 2:
            raise ValueError("At least two replica predictions are required")
        if self.replica_predictions.shape[1] < 1:
            raise ValueError("The prediction array contains no observable bins")
        if not np.all(np.isfinite(self.replica_predictions)):
            raise ValueError("The prediction array contains non-finite values")
        return None

    def mean(self):
        """Return the replica mean in every observable bin."""

        return self.replica_predictions.mean(axis=0)

    def standard_deviation(self):
        """Return the sample standard deviation in every observable bin."""

        # ddof=1 uses the usual sample standard deviation because the replicas
        # are a finite sample of the underlying PDF probability distribution.
        return self.replica_predictions.std(axis=0, ddof=1)

    def percentile_interval(self, lower=16.0, upper=84.0):
        """Return a percentile-based PDF uncertainty interval."""

        if not 0.0 <= lower < upper <= 100.0:
            raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100")
        # The central 16th-to-84th percentile interval contains 68% of the
        # replica predictions and does not assume that they are symmetric.
        return np.percentile(
            self.replica_predictions, [lower, upper], axis=0
        )

    def summary(self):
        """Return all uncertainty statistics needed by the notebook."""

        lower, upper = self.percentile_interval()
        return {
            "mean": self.mean(),
            "standard_deviation": self.standard_deviation(),
            "lower_68": lower,
            "upper_68": upper,
        }
