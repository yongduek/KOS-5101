@echo off
setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "TIMEOUT=120"
set "SPECIFIC_NB="

:parse_args
if "%~1"=="" goto done_args
if /i "%~1"=="--timeout" (
    set "TIMEOUT=%~2"
    shift
    shift
    goto parse_args
)
if /i "%~x1"==".ipynb" (
    set "SPECIFIC_NB=%~1"
    shift
    goto parse_args
)
echo Unknown argument: %~1
exit /b 1

:done_args
echo ============================================================
echo   IRT Notebook Executor
echo   Directory : %SCRIPT_DIR%
echo   Timeout   : %TIMEOUT% sec per cell
echo ============================================================
echo.

where jupyter >nul 2>&1
if %ERRORLEVEL% == 0 goto use_jupyter

if exist "run_notebooks.py" (
    where python >nul 2>&1
    if %ERRORLEVEL% == 0 goto use_python

    where python3 >nul 2>&1
    if %ERRORLEVEL% == 0 goto use_python3
)

echo [ERROR] Neither jupyter nor python was found.
echo.
echo   Install one of the following:
echo.
echo     conda install -c conda-forge jupyter cmdstanpy
echo     pip install jupyter nbconvert cmdstanpy
echo.
exit /b 1

:use_jupyter
echo [OK] Found jupyter -- using nbconvert.
echo.
if not "%SPECIFIC_NB%"=="" (
    call :run_one "%SPECIFIC_NB%"
    goto done
)
for %%f in (*.ipynb) do call :run_one "%%f"
goto done

:use_python
echo [OK] jupyter not found -- using run_notebooks.py with python.
echo.
if not "%SPECIFIC_NB%"=="" (
    python run_notebooks.py --timeout %TIMEOUT% "%SPECIFIC_NB%"
) else (
    python run_notebooks.py --timeout %TIMEOUT%
)
goto done

:use_python3
echo [OK] jupyter not found -- using run_notebooks.py with python3.
echo.
if not "%SPECIFIC_NB%"=="" (
    python3 run_notebooks.py --timeout %TIMEOUT% "%SPECIFIC_NB%"
) else (
    python3 run_notebooks.py --timeout %TIMEOUT%
)
goto done

:done
echo.
echo ============================================================
echo   Done. Open the .ipynb files to view results.
echo ============================================================
exit /b 0

:run_one
set "nb=%~1"
echo   Running: %nb% ...
jupyter nbconvert --to notebook --execute --allow-errors --inplace "--ExecutePreprocessor.timeout=%TIMEOUT%" "%nb%" >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo   [Done] %nb%
) else (
    echo   [Finished with errors - output saved] %nb%
)
exit /b 0
