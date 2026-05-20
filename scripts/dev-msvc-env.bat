@echo off
REM Wrapper: prime MSVC environment, then run whatever was passed on the
REM command line. Used from bash on Windows hosts where VS env vars are
REM not part of the parent shell. Example:
REM   cmd.exe //c "scripts\\dev-msvc-env.bat cargo check --workspace"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul
if errorlevel 1 (
    echo Failed to prime MSVC environment
    exit /b 1
)
%*
