"""IO: parsers for EMTF file formats.

- ``fc.py`` — Fourier coefficient (``.fN``) binary files from ``dnff``
- ``zfile.py`` — Impedance transfer function (``Z.*``) ASCII files from ``tranmt``
- ``sp.py`` — System parameter (``.sp``) files
- ``edi.py`` — EDI export (SEG standard MT data interchange format)
"""

from emtf.io.zfile import ZFile, ZFileData, read_z_file, write_z_file
from emtf.io.sp import read_sp, SystemParams
from emtf.io.binary import read_binary, write_binary
from emtf.io.edi import (
    BandedResult,
    EdiMetadata,
    export_result_to_folder,
    format_edi,
    from_result,
    to_banded_result,
    write_edi,
)

__all__ = [
    "ZFile", "ZFileData", "read_z_file", "write_z_file",
    "read_sp", "SystemParams",
    "read_binary", "write_binary",
    "BandedResult", "EdiMetadata", "write_edi", "format_edi",
    "from_result", "to_banded_result", "export_result_to_folder",
]
