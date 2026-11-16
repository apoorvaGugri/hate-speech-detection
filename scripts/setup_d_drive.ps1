<#
Keeps this project's large downloads off the C: drive.

    .\scripts\setup_d_drive.ps1                 # caches under D:\Caches
    .\scripts\setup_d_drive.ps1 -CacheRoot E:\Caches

What it does
  * PIP_CACHE_DIR, HF_HOME, TORCH_HOME -> saved as USER environment variables
    (permanent, apply to every new terminal) and set in this terminal.
  * TEMP / TMP -> set for THIS terminal only, because pip unpacks large wheels
    (PyTorch) in the temp folder. Not saved permanently: changing TEMP for the
    whole account would affect every other program.

Safe to run again at any time. Run it in each new terminal before `pip install`.
The project folder and its .venv are wherever you put them; this script only
moves caches.
#>
param(
    [string]$CacheRoot = "D:\Caches"
)

$ErrorActionPreference = "Stop"

$dirs = @{
    PIP_CACHE_DIR = Join-Path $CacheRoot "pip"
    HF_HOME       = Join-Path $CacheRoot "huggingface"
    TORCH_HOME    = Join-Path $CacheRoot "torch"
}
$tempDir = Join-Path $CacheRoot "tmp"

foreach ($path in @($dirs.Values) + $tempDir) {
    New-Item -ItemType Directory -Force -Path $path | Out-Null
}

foreach ($name in $dirs.Keys) {
    [Environment]::SetEnvironmentVariable($name, $dirs[$name], "User")   # permanent
    Set-Item -Path "Env:$name" -Value $dirs[$name]                       # this terminal
}

$env:TEMP = $tempDir
$env:TMP = $tempDir

Write-Host "Cache locations now in effect:"
foreach ($name in @("PIP_CACHE_DIR", "HF_HOME", "TORCH_HOME", "TEMP")) {
    Write-Host ("  {0,-14} {1}" -f $name, (Get-Item "Env:$name").Value)
}
$drive = (Get-Item $CacheRoot).PSDrive
Write-Host ("Free space on {0}: {1:N1} GB" -f $drive.Name, ($drive.Free / 1GB))
