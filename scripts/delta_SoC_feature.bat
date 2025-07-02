@echo off
REM Set PYTHON_PATH if Python is not in your system's PATH
REM For example: set PYTHON_PATH="C:\Python39\python.exe"
REM If python is in PATH, you can just use "python"
SET PYTHON_EXE=python
SET SCRIPT_DIR=%~dp0
REM SET PYTHON_SCRIPT=%SCRIPT_DIR%soc_prediction_pipeline.py
SET PYTHON_FE_SCRIPT_NAME=%SCRIPT_DIR%add_delta_soc.py
SET BASE_DATA_DIR=%SCRIPT_DIR%..\SOC_estimation_dataset_G10
SET OUTPUT_DIR=%SCRIPT_DIR%..\output_data

SET SCRIPT_DIR=%~dp0
ECHO Script directory (SCRIPT_DIR) is: %SCRIPT_DIR%

SET PYTHON_FE_SCRIPT=%PYTHON_FE_SCRIPT_NAME%

ECHO Python Feature Engineering script path (PYTHON_FE_SCRIPT) is: %PYTHON_FE_SCRIPT%
IF EXIST "%PYTHON_FE_SCRIPT%" (
    ECHO   OK: Python FE script found.
) ELSE (
    ECHO   ERROR: Python FE script NOT found at the path above! Check filename and location.
    GOTO EndDebug
)
ECHO.

SET INPUT_DATA_DIR=%SCRIPT_DIR%..\output_data
SET INPUT_CSV_FILE=%INPUT_DATA_DIR%\FINAL_MASTER_DATASET_with_SOC.csv
ECHO Input CSV file path is: %INPUT_CSV_FILE%

REM --- Check if input files exists ---
ECHO Checking for input file...
IF NOT EXIST "%INPUT_CSV_FILE%" (
    echo   ERROR: Input CSV file not found at %INPUT_CSV_FILE%
    echo   Please ensure 'FINAL_MASTER_DATASET_with_SOC.csv' was generated successfully.
    GOTO EndDebug
) ELSE (
    echo   OK: Input CSV file found.
)
ECHO Input file check complete.
ECHO.

REM --- Prepare the Command to Run ---
SET CMD_TO_RUN=%PYTHON_EXE% "%PYTHON_FE_SCRIPT%" "%INPUT_CSV_FILE%" "%OUTPUT_DIR%"

ECHO Starting Feature Engineering Python script...
ECHO The command that will be executed is:
ECHO %CMD_TO_RUN%
ECHO.

REM --- Run the Feature Engineering Script ---
%CMD_TO_RUN%

pause