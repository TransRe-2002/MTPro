param(
    [string]$MatlabRoot = "",
    [string]$OutputDir = "",
    [switch]$SkipSmokeTest,
    [switch]$NoInstall
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Resolve-Path (Join-Path $ScriptDir "..")
$MatlabDir = Join-Path $RootDir "matlab"
$InstallDir = Join-Path $RootDir "python\src\bin\win64"

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $MatlabDir "build\win64"
}

$Sources = @(
    (Join-Path $MatlabDir "main.m"),
    (Join-Path $MatlabDir "TS.m"),
    (Join-Path $MatlabDir "struct_to_ts.m"),
    (Join-Path $MatlabDir "ts_to_struct.m")
)

foreach ($Source in $Sources) {
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Missing MATLAB source: $Source"
    }
}

function Find-MatlabCompiler {
    param([string]$RequestedRoot)

    $Candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($RequestedRoot)) {
        $Candidates += Join-Path $RequestedRoot "bin\mcc.bat"
        $Candidates += Join-Path $RequestedRoot "bin\mcc.exe"
    }

    foreach ($Name in @("mcc.bat", "mcc.exe", "mcc")) {
        $Command = Get-Command $Name -ErrorAction SilentlyContinue
        if ($null -ne $Command) {
            $Candidates += $Command.Source
        }
    }

    $ProgramFiles = [Environment]::GetFolderPath("ProgramFiles")
    if (-not [string]::IsNullOrWhiteSpace($ProgramFiles)) {
        $Installations = Get-ChildItem (Join-Path $ProgramFiles "MATLAB") `
            -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending
        foreach ($Installation in $Installations) {
            $Candidates += Join-Path $Installation.FullName "bin\mcc.bat"
            $Candidates += Join-Path $Installation.FullName "bin\mcc.exe"
        }
    }

    foreach ($Candidate in $Candidates) {
        if (-not [string]::IsNullOrWhiteSpace($Candidate) -and
            (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $Candidate).Path
        }
    }

    throw @"
Cannot find the Windows MATLAB Compiler (mcc).

Install MATLAB Compiler, add MATLAB\bin to PATH, or run:
  .\scripts\build_matlab.ps1 -MatlabRoot "C:\Program Files\MATLAB\R2025b"
"@
}

$Mcc = Find-MatlabCompiler $MatlabRoot
Write-Host "=== MATLAB Compiler ==="
Write-Host $Mcc

$ResolvedOutputDir = [System.IO.Path]::GetFullPath($OutputDir)
if (Test-Path -LiteralPath $ResolvedOutputDir) {
    Remove-Item -LiteralPath $ResolvedOutputDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $ResolvedOutputDir | Out-Null

$MccArgs = @(
    "-m", $Sources[0],
    "-o", "mtpro_ts_convert",
    "-d", $ResolvedOutputDir,
    "-a", $Sources[1],
    "-a", $Sources[2],
    "-a", $Sources[3]
)

Write-Host "=== Compiling mtpro_ts_convert.exe ==="
& $Mcc @MccArgs
if ($LASTEXITCODE -ne 0) {
    throw "MATLAB Compiler failed with exit code $LASTEXITCODE"
}

$Converter = Join-Path $ResolvedOutputDir "mtpro_ts_convert.exe"
if (-not (Test-Path -LiteralPath $Converter -PathType Leaf)) {
    throw "Build finished without expected executable: $Converter"
}

if (-not $SkipSmokeTest) {
    $Fixture = Join-Path $RootDir "python\tests\fixtures\ts_object.mat"
    if (-not (Test-Path -LiteralPath $Fixture -PathType Leaf)) {
        throw "Smoke-test fixture is missing: $Fixture (use -SkipSmokeTest to bypass)"
    }

    $SmokeDir = Join-Path ([System.IO.Path]::GetTempPath()) `
        ("mtpro_ts_convert_" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $SmokeDir | Out-Null
    try {
        $StructFile = Join-Path $SmokeDir "converted_struct.mat"
        $RoundTripFile = Join-Path $SmokeDir "roundtrip_ts.mat"

        Write-Host "=== Smoke test: TS object -> tsStruct ==="
        & $Converter "ts2struct" $Fixture $StructFile
        if ($LASTEXITCODE -ne 0 -or
            -not (Test-Path -LiteralPath $StructFile -PathType Leaf)) {
            throw "TS-to-struct smoke test failed"
        }

        Write-Host "=== Smoke test: tsStruct -> TS object ==="
        & $Converter "struct2ts" $StructFile $RoundTripFile
        if ($LASTEXITCODE -ne 0 -or
            -not (Test-Path -LiteralPath $RoundTripFile -PathType Leaf)) {
            throw "Struct-to-TS smoke test failed"
        }
    }
    finally {
        if (Test-Path -LiteralPath $SmokeDir) {
            Remove-Item -LiteralPath $SmokeDir -Recurse -Force
        }
    }
}

if (-not $NoInstall) {
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    $InstalledConverter = Join-Path $InstallDir "mtpro_ts_convert.exe"
    Copy-Item -LiteralPath $Converter -Destination $InstalledConverter -Force
    Write-Host "Installed: $InstalledConverter"
}

Write-Host "Build complete: $Converter"
Write-Host "Target machines need the matching MATLAB Runtime installed."
