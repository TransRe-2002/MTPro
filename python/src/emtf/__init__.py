"""
emtf — Python wrapper for the OSU EMTF magnetotelluric processing suite.

Wraps the Fortran ``dnff`` (cascade decimation + FFT) and ``tranmt``
(robust single-station / remote-reference transfer function estimation)
binaries in a Pythonic API.  A drop-in alternative to ``emtf_rs`` that
uses the original Fortran code path.

Typical pipeline::

    from emtf import dnff, tranmt, ZFile
    from emtf.types import DecimationLevel, FrequencyBand, ProcessingOptions

    # 1) Time series → Fourier coefficients
    fc_workspace = dnff(
        input_files=["data.bin"],
        decimation_levels=[...],
        num_channels=5,
    )

    # 2) FC files → impedance (Z file)
    z_workspace = tranmt(
        fc_files=sorted((fc_workspace.path / "FC").glob("*.f*")),
        bands=[FrequencyBand(1, 8, 9), ...],
        station_name="STA01",
        options=ProcessingOptions(robust=True),
    )

    # 3) Parse Z file
    zf = ZFile.read("path/to/Z.station.zss")
    print(zf.periods)   # → array of periods (s)
    print(zf.Zxy)       # → complex impedance tensor component

    # 4) Export to EDI
    zf.to_edi("output.edi")
"""

from emtf._binaries import resolve_binary, get_platform
from emtf._runner import Workspace
from emtf._config import (
    render_paths, render_decset, render_pwset,
    render_bands, render_options, render_tranmt_control,
)
from emtf.pipeline import dnff
from emtf.pipeline.dnff import dnff_from_arrays
from emtf.pipeline.tranmt import tranmt
from emtf import api
from emtf.api import (
    BandConfig,
    SpectrumResult,
    estimate_single_station_spectrum,
    estimate_remote_reference_spectrum,
)
from emtf.band_presets import (
    fine_bands,
    half_decade_bands,
    preset_bands,
    standard_bands,
)
from emtf.io import (
    ZFile, ZFileData,
    read_z_file, write_z_file,
    read_sp, SystemParams,
    BandedResult, EdiMetadata,
    write_edi, format_edi,
    from_result, to_banded_result, export_result_to_folder,
)
from emtf.io.binary import read_binary, write_binary, write_asc, write_sp
from emtf.types import (
    DecimationLevel, FrequencyBand,
    ProcessingOptions, Station, TransferFunction,
)

__all__ = [
    # binary resolution
    "resolve_binary", "get_platform",
    # workspace / config
    "Workspace",
    "render_paths", "render_decset", "render_pwset",
    "render_bands", "render_options", "render_tranmt_control",
    "api",
    # pipeline (low-level)
    "dnff", "dnff_from_arrays", "tranmt",
    # estimate (high-level, emtf_rs-compatible)
    "BandConfig", "SpectrumResult",
    "estimate_single_station_spectrum", "estimate_remote_reference_spectrum",
    "standard_bands", "fine_bands", "half_decade_bands", "preset_bands",
    # I/O
    "ZFile", "ZFileData", "read_z_file", "write_z_file",
    "BandedResult", "EdiMetadata", "write_edi", "format_edi",
    "from_result", "to_banded_result", "export_result_to_folder",
    "read_sp", "SystemParams",
    "read_binary", "write_binary", "write_asc", "write_sp",
    # types
    "DecimationLevel", "FrequencyBand",
    "ProcessingOptions", "Station", "TransferFunction",
]
__version__ = "0.1.0"
