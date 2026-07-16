#!/usr/bin/env python3
"""Basic EMTF pipeline example — alternative to emtf_rs using Fortran binaries.

This example demonstrates the two-stage workflow:

1. ``dnff`` — cascade decimation + FFT (time series → Fourier coefficients)
2. ``tranmt`` — robust single-station TF estimation (FC files → Z impedance file)
3. Parse the Z-file and export to EDI

Usage::

    python examples/basic_pipeline.py path/to/data.bin

Or use the EMTF example test data (run the EMTF test pipeline first)::

    cd EMTF/examples && bash INSTALL  # builds binaries and processes test data
    python ../../examples/basic_pipeline.py EMTF/examples/DATA/test1.bin

The script expects 5-channel binary time series files (Ex Ey Hx Hy Hz) at
the same sampling rate.  Adjust ``num_channels`` and ``decimation_levels``
for your specific survey setup.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from emtf import dnff, tranmt, ZFile
from emtf.types import DecimationLevel, FrequencyBand, ProcessingOptions


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <binary_datafile> [datafile2 ...]")
        sys.exit(1)

    input_files = [Path(f) for f in sys.argv[1:]]
    for f in input_files:
        if not f.exists():
            print(f"File not found: {f}")
            sys.exit(1)

    # ── Step 1: dnff → FC files ─────────────────────────────────────
    # Standard 1 Hz MT setup: 6 cascade levels, 128-sample windows,
    # decimation factor 4 between levels
    decimation_levels = [
        DecimationLevel(nwin=128, window_length=32.0, decimation_factor=1, nfilt=0),
        DecimationLevel(nwin=128, window_length=32.0, decimation_factor=4, nfilt=4,
                        fil_1=0.2154, fil_2=0.1911, fil_3=0.1307, fil_4=0.0705),
        DecimationLevel(nwin=128, window_length=32.0, decimation_factor=4, nfilt=4,
                        fil_1=0.2154, fil_2=0.1911, fil_3=0.1307, fil_4=0.0705),
        DecimationLevel(nwin=128, window_length=32.0, decimation_factor=4, nfilt=4,
                        fil_1=0.2154, fil_2=0.1911, fil_3=0.1307, fil_4=0.0705),
        DecimationLevel(nwin=128, window_length=32.0, decimation_factor=4, nfilt=4,
                        fil_1=0.2154, fil_2=0.1911, fil_3=0.1307, fil_4=0.0705),
        DecimationLevel(nwin=128, window_length=32.0, decimation_factor=4, nfilt=4,
                        fil_1=0.2154, fil_2=0.1911, fil_3=0.1307, fil_4=0.0705),
    ]

    print("Stage 1/2: dnff — time series → Fourier coefficients")
    ws_dnff = dnff(
        input_files=input_files,
        decimation_levels=decimation_levels,
        num_channels=5,
        cleanup=False,
    )
    fc_files = sorted((ws_dnff.path / "FC").glob("*.f*"))
    print(f"  → {len(fc_files)} FC file(s) in {ws_dnff.path}/FC/")
    for f in fc_files:
        print(f"      {f.name}")

    # ── Step 2: tranmt → Z impedance file ───────────────────────────
    # Frequency bands covering ~30 s to ~10,000 s period
    bands = [
        FrequencyBand(1, 8, 9),   # ~30 s    (decimation level 1)
        FrequencyBand(1, 6, 7),   # ~40 s
        FrequencyBand(1, 5, 5),   # ~50 s
        FrequencyBand(2, 14, 17),  # ~100 s   (decimation level 2)
        FrequencyBand(2, 11, 13),  # ~130 s
        FrequencyBand(2, 9, 10),   # ~160 s
        FrequencyBand(2, 7, 8),    # ~210 s
        FrequencyBand(2, 6, 6),    # ~250 s
        FrequencyBand(2, 5, 5),    # ~310 s
        FrequencyBand(3, 14, 17),  # ~390 s   (decimation level 3)
        FrequencyBand(3, 11, 13),  # ~500 s
        FrequencyBand(3, 9, 10),   # ~630 s
        FrequencyBand(3, 7, 8),    # ~850 s
        FrequencyBand(3, 6, 6),    # ~1000 s
        FrequencyBand(3, 5, 5),    # ~1280 s
        FrequencyBand(4, 14, 17),  # ~1500 s
        FrequencyBand(4, 11, 13),  # ~2000 s
        FrequencyBand(4, 9, 10),   # ~2500 s
        FrequencyBand(4, 7, 8),    # ~3400 s
        FrequencyBand(4, 6, 6),    # ~4000 s
        FrequencyBand(4, 5, 5),    # ~5000 s
        FrequencyBand(5, 14, 17),  # ~6000 s
        FrequencyBand(5, 11, 13),  # ~8000 s
        FrequencyBand(5, 9, 10),   # ~10000 s
    ]

    options = ProcessingOptions(
        header="Robust Single station",
        robust=True,
        remote_reference=False,
    )

    station_name = input_files[0].stem
    print(f"\nStage 2/2: tranmt — FC files → impedance")
    ws_tranmt = tranmt(
        fc_files=fc_files,
        bands=bands,
        station_name=station_name,
        options=options,
        num_channels=5,
        cleanup=False,
    )

    z_files = sorted((ws_tranmt.path / "MT").glob("Z.*"))
    print(f"  → {len(z_files)} Z-file(s) in {ws_tranmt.path}/MT/")

    # ── Step 3: Parse and export ────────────────────────────────────
    for zf_path in z_files:
        print(f"\n  Reading {zf_path.name} ...")
        zf = ZFile.read(zf_path)
        print(f"    Station: {zf.stname}")
        print(f"    Bands: {zf.nbt}")
        print(f"    Period range: {zf.periods[0]:.1f} – {zf.periods[-1]:.1f} s")
        print(f"    Zxy[0] = {zf.Zxy[0]:.4e}")
        print(f"    Zyx[0] = {zf.Zyx[0]:.4e}")
        print(f"    Rho_xy[0] = {zf.apparent_resistivity_xy()[0]:.1f} ohm·m")
        print(f"    Phi_xy[0] = {zf.phase_xy_deg()[0]:.1f}°")

        # Export to EDI
        edi_path = zf_path.with_suffix(".edi")
        zf.to_edi(edi_path)
        print(f"\n  EDI exported → {edi_path}")

    print(f"\nWorkspaces retained for inspection:")
    print(f"  dnff:    {ws_dnff.path}")
    print(f"  tranmt:  {ws_tranmt.path}")


if __name__ == "__main__":
    main()
