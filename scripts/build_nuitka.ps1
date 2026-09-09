param(
    [switch]$InstallDeps,
    [switch]$BuildRust,
    [switch]$Zip,
    [switch]$KeepBuild,
    [switch]$Installer,
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Resolve-Path (Join-Path $ScriptDir "..")
$PythonDir = Join-Path $RootDir "python"
$VenvPython = Join-Path $PythonDir ".venv\Scripts\python.exe"
$SrcMain = Join-Path $PythonDir "src\main.py"
$SrcBin = Join-Path $PythonDir "src\bin"
$BuildDir = Join-Path $PythonDir "build\nuitka-windows"
$OutputDir = Join-Path $RootDir "outputs"
$DistName = "mtpro-windows-x86_64"
$DistDir = Join-Path $OutputDir $DistName

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = $VenvPython
}

if (-not (Test-Path $PythonExe)) {
    Write-Error @"
Missing project Python:
  $PythonExe

Create the Windows venv first, then install dependencies, for example:
  py -3.14 -m venv python\.venv
  .\python\.venv\Scripts\python.exe -m pip install -r python\requirements.txt
  .\python\.venv\Scripts\python.exe -m pip install -r python\requirements-build.txt
"@
}

if (-not (Test-Path $SrcMain)) {
    Write-Error "Missing application entry point: $SrcMain"
}

if ($InstallDeps) {
    & $PythonExe -m pip install --upgrade pip
    & $PythonExe -m pip install -r (Join-Path $PythonDir "requirements.txt")
    & $PythonExe -m pip install -r (Join-Path $PythonDir "requirements-build.txt")
}

if ($BuildRust) {
    $RustDir = Join-Path $RootDir "rust_lib"
    $RustDist = Join-Path $RustDir "dist"

    if (-not (Test-Path $RustDir)) {
        Write-Error "Missing Rust extension directory: $RustDir"
    }

    Push-Location $RustDir
    try {
        & $PythonExe -m maturin build --release --out $RustDist
        $Wheel = Get-ChildItem $RustDist -Filter "rust_lib-*.whl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($null -eq $Wheel) {
            Write-Error "maturin did not produce a rust_lib wheel in $RustDist"
        }
        & $PythonExe -m pip install --force-reinstall $Wheel.FullName
    }
    finally {
        Pop-Location
    }
}

& $PythonExe -c "import nuitka" | Out-Null
& $PythonExe -c "import PySide6" | Out-Null
& $PythonExe -c "import rust_lib" | Out-Null

$ExcludedUnusedPackages = @(
    "annotated_types",
    "astropy",
    "astropy_iers_data",
    "aurora",
    "certifi",
    "cftime",
    "charset_normalizer",
    "dns",
    "email_validator",
    "greenlet",
    "idna",
    "llvmlite",
    "loguru",
    "lxml",
    "mt_io",
    "mt_timeseries",
    "mt_metadata",
    "mth5",
    "numba",
    "obspy",
    "openpyxl",
    "pandas_stubs",
    "psutil",
    "pydantic",
    "pydantic_core",
    "pyerfa",
    "pyproj",
    "requests",
    "scipy_stubs",
    "sqlalchemy",
    "typing_inspection",
    "urllib3",
    "yaml"
)

$NuitkaArgs = @(
    "--standalone",
    "--follow-imports",
    "--enable-plugin=pyside6",
    "--static-libpython=no",
    "--include-package=ui",
    "--include-package=core",
    "--include-package=base",
    "--include-package=processor",
    "--include-package=io_utils",
    "--include-package=utils",
    "--include-package=emtf",
    "--include-data-dir=$SrcBin=bin",
    "--include-module=rust_lib",
    "--remove-output",
    "--lto=yes",
    "--noinclude-pytest-mode=nofollow",
    "--noinclude-unittest-mode=nofollow",
    "--noinclude-pydoc-mode=nofollow",
    "--noinclude-IPython-mode=nofollow",
    "--noinclude-setuptools-mode=nofollow",
    "--mingw64",
    "--output-dir=$BuildDir",
    "--assume-yes-for-downloads",
    "--show-progress"
)

foreach ($Package in $ExcludedUnusedPackages) {
    $NuitkaArgs += "--nofollow-import-to=$Package"
}

# 有应用图标时给 exe 嵌入图标(可选,缺省不影响构建)
$IconFile = Join-Path $PythonDir "src\mtpro.ico"
if (Test-Path $IconFile) {
    $NuitkaArgs += "--windows-icon-from-ico=$IconFile"
}

$NuitkaArgs += $SrcMain

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if (Test-Path $BuildDir) {
    Remove-Item -Recurse -Force $BuildDir
}
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

$PreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $PythonDir "src"

try {
    Write-Host "=== Building MTPro Windows standalone with Nuitka ==="
    & $PythonExe -m nuitka @NuitkaArgs
}
finally {
    $env:PYTHONPATH = $PreviousPythonPath
}

$MainDist = Join-Path $BuildDir "main.dist"
if (-not (Test-Path $MainDist)) {
    Write-Error "Nuitka build finished without expected dist directory: $MainDist"
}

if (Test-Path $DistDir) {
    Remove-Item -Recurse -Force $DistDir
}
Move-Item $MainDist $DistDir
Copy-Item (Join-Path $PythonDir "requirements.txt") $DistDir

# Nuitka 以入口脚本名命名产物(main.exe),统一改名为 MTPro.exe
$MainExe = Join-Path $DistDir "main.exe"
if (Test-Path $MainExe) {
    Rename-Item -Path $MainExe -NewName "MTPro.exe"
} else {
    Write-Warning "未找到 $MainExe,跳过改名(安装包将不可用)"
}

if (-not $KeepBuild) {
    Remove-Item -Recurse -Force $BuildDir
}

if ($Zip) {
    $ZipPath = Join-Path $OutputDir "$DistName.zip"
    if (Test-Path $ZipPath) {
        Remove-Item -Force $ZipPath
    }
    Compress-Archive -Path $DistDir -DestinationPath $ZipPath
    Write-Host "Zip: $ZipPath"
}

if ($Installer) {
    $IsccCommand = Get-Command iscc -ErrorAction SilentlyContinue
    if ($null -eq $IsccCommand) {
        Write-Error "未找到 Inno Setup (iscc)。请先安装: winget install JRSoftware.InnoSetup"
    }

    # 版本号优先级: MTPRO_VERSION 环境变量 → git describe --tags → 1.0.0
    $Version = $env:MTPRO_VERSION
    if ([string]::IsNullOrWhiteSpace($Version) -and (Get-Command git -ErrorAction SilentlyContinue)) {
        $Describe = & git describe --tags 2>$null
        if ($Describe -match '\d+\.\d+\.\d+') {
            $Version = $Matches[0]
        }
    }
    if ([string]::IsNullOrWhiteSpace($Version)) {
        $Version = "1.0.0"
    }

    $IssFile = Join-Path $ScriptDir "mtpro_windows.iss"
    Write-Host "=== Building installer (v$Version) with Inno Setup ==="
    & $IsccCommand.Source "/DMyAppVersion=$Version" $IssFile
    if ($LASTEXITCODE -ne 0) {
        Write-Error "iscc 失败,退出码 $LASTEXITCODE"
    }
}

$TargetExe = Join-Path $DistDir "MTPro.exe"
Write-Host "Output: $TargetExe"
