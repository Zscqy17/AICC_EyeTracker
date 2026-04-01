@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if defined PYTHON_BIN (
    "%PYTHON_BIN%" -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo [launcher] PYTHON_BIN is set but is not a supported Python 3.10-3.12 interpreter.
        exit /b 1
    )
    set "PYTHON_CMD="%PYTHON_BIN%""
    goto run_launcher
)

call :try_candidate "py -3.12"
if not errorlevel 1 goto run_launcher
call :try_candidate "py -3.11"
if not errorlevel 1 goto run_launcher
call :try_candidate "py -3.10"
if not errorlevel 1 goto run_launcher
call :try_candidate "python"
if not errorlevel 1 goto run_launcher

echo [launcher] No supported Python 3.10-3.12 interpreter was found.
exit /b 1

:try_candidate
cmd /c %~1 -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>nul
if errorlevel 1 exit /b 1
set "PYTHON_CMD=%~1"
exit /b 0

:run_launcher
call %PYTHON_CMD% "%SCRIPT_DIR%launch_tracker.py" %*
exit /b %ERRORLEVEL%