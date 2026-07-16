"""Pipeline wrappers around EMTF Fortran binaries.

- ``dnff`` — cascade decimation + FFT (time series → FC files)
- ``tranmt`` — robust single-station / remote-reference TF estimation (FC files → Z files)
"""

from emtf.pipeline.dnff import dnff

__all__ = ["dnff"]
