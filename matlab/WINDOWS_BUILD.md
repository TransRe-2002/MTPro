# Building the Windows TS Converter

`mtpro_ts_convert.exe` converts MATLAB `TS` objects to the plain `tsStruct`
representation used by MTPro, and converts `tsStruct` files back to `TS`
objects. The executable must be built on Windows; MATLAB Compiler does not
cross-compile between Linux and Windows.

## Requirements

- 64-bit Windows;
- MATLAB with the MATLAB Compiler product installed;
- PowerShell 5.1 or later.

The resulting executable requires the matching, freely redistributable
MATLAB Runtime on end-user machines. Use the Runtime release stated by the
compiler-generated `readme.txt`.

## Build and test

Open PowerShell in the repository (or extracted source bundle) and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\build_matlab.ps1
```

The script locates `mcc`, builds the converter, performs a TS-to-struct and
struct-to-TS smoke test, then copies the executable to:

```text
python\src\bin\win64\mtpro_ts_convert.exe
```

If MATLAB is not on `PATH`, provide its installation root:

```powershell
.\scripts\build_matlab.ps1 `
  -MatlabRoot "C:\Program Files\MATLAB\R2025b"
```

Useful options:

- `-SkipSmokeTest`: compile without running the bundled MAT fixture;
- `-NoInstall`: leave the result in `matlab\build\win64`;
- `-OutputDir <path>`: choose another compiler output directory.

After the executable has been copied into `python\src\bin\win64`, rerun the
Windows Nuitka build. Its existing `--include-data-dir=python/src/bin=bin`
option will include the converter in the application automatically.
