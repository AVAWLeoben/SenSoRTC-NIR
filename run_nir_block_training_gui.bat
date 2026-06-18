@echo off
setlocal

REM ------------------------------------------------------------
REM Start nir_block_training_gui.py from the NIR conda environment.
REM Place this .bat file in the same folder as hsi_labeler.py.
REM ------------------------------------------------------------

cd /d "%~dp0"

set "ENV_NAME=NIR"
set "SCRIPT_PATH=%~dp0nir_block_training_gui.py"

if not exist "%SCRIPT_PATH%" (
    echo ERROR: Could not find SenSoRTC-NIR.py next to this batch file.
    echo Expected: "%SCRIPT_PATH%"
    echo.
    pause
    exit /b 1
)

set "CONDA_BAT="

REM Common Miniconda/Anaconda install locations.
if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%ProgramData%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%ProgramData%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%ProgramData%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%ProgramData%\anaconda3\condabin\conda.bat"

REM Fall back to conda from PATH.
if not defined CONDA_BAT (
    for /f "delims=" %%I in ('where conda.bat 2^>nul') do (
        if not defined CONDA_BAT set "CONDA_BAT=%%I"
    )
)
if not defined CONDA_BAT (
    for /f "delims=" %%I in ('where conda 2^>nul') do (
        if not defined CONDA_BAT set "CONDA_BAT=%%I"
    )
)

if not defined CONDA_BAT (
    echo ERROR: Could not find conda.bat.
    echo Install Miniconda/Anaconda or add conda to PATH.
    echo.
    pause
    exit /b 1
)

echo Using conda: "%CONDA_BAT%"
echo Activating environment: %ENV_NAME%
echo.

call "%CONDA_BAT%" activate "%ENV_NAME%"
if errorlevel 1 (
    echo.
    echo ERROR: Failed to activate conda environment "%ENV_NAME%".
    echo Check that the environment exists:
    echo     conda env list
    echo.
    pause
    exit /b 1
)

echo Running: "%SCRIPT_PATH%"
echo.
python "%SCRIPT_PATH%"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo Programm exited with code %EXIT_CODE%.
exit %EXIT_CODE%