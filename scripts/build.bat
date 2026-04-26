@echo off
setlocal

REM Path to VS Developer Command Prompt bootstrap (contains parens, handle carefully).
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CARGO_BIN=%USERPROFILE%\.cargo\bin"

if not exist "%VCVARS%" goto missing_vcvars
if not exist "%CARGO_BIN%\cargo.exe" goto missing_cargo

call "%VCVARS%" >nul
if errorlevel 1 goto vcvars_failed

set "PATH=%CARGO_BIN%;%PATH%"
cargo %*
exit /b %errorlevel%

:missing_vcvars
echo ERROR: vcvars64.bat not found at:
echo   %VCVARS%
echo Install Visual Studio Build Tools with the "Desktop development with C++" workload.
exit /b 1

:missing_cargo
echo ERROR: cargo.exe not found at:
echo   %CARGO_BIN%\cargo.exe
echo Install Rust via: winget install Rustlang.Rustup
exit /b 1

:vcvars_failed
echo ERROR: vcvars64.bat call failed.
exit /b 1
