@echo off
ECHO Starting Feature Engineering Batch Script...
ECHO.

REM Assumes this script is in a 'scripts' folder.
REM Assumes 'feature_engineer.py' (the feature engineering Python script) is in the same folder.
REM Assumes input file 'FINAL_MASTER_DATASET_with_SOC.csv' is in '..\output_data\'.
REM Output 'FINAL_MASTER_DATASET_with_Features.csv' will also be in '..\output_data\'.

REM --- Configuration ---
SET PYTHON_EXE=python
REM If 'python' is not in your PATH, or you need a specific version (e.g., python3),
REM provide the full path to python.exe, e.g.:
REM SET PYTHON_EXE=C:\Users\YourUser\AppData\Local\Programs\Python\Python39\python.exe
ECHO Current PYTHON_EXE is set to: %PYTHON_EXE%
ECHO.

SET SCRIPT_DIR=%~dp0
ECHO Script directory (SCRIPT_DIR) is: %SCRIPT_DIR%

SET PYTHON_FE_SCRIPT_NAME=feature_engineer_mod.py
SET PYTHON_FE_SCRIPT=%SCRIPT_DIR%%PYTHON_FE_SCRIPT_NAME%
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
SET OUTPUT_CSV_FILE=%INPUT_DATA_DIR%\FINAL_MASTER_DATASET_with_Features.csv

ECHO Input CSV file path is: %INPUT_CSV_FILE%
ECHO Output CSV file path will be: %OUTPUT_CSV_FILE%
ECHO.

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
SET CMD_TO_RUN=%PYTHON_EXE% "%PYTHON_FE_SCRIPT%" --input_csv_path "%INPUT_CSV_FILE%" --output_csv_path "%OUTPUT_CSV_FILE%"

ECHO Starting Feature Engineering Python script...
ECHO The command that will be executed is:
ECHO %CMD_TO_RUN%
ECHO.

REM --- Run the Feature Engineering Script ---
%CMD_TO_RUN%

ECHO.
ECHO Python script execution finished (or attempted).
ECHO Checking for output file: %OUTPUT_CSV_FILE%
IF EXIST "%OUTPUT_CSV_FILE%" (
    echo   OK: Feature engineering output file found!
    echo   Output saved to: %OUTPUT_CSV_FILE%
) ELSE (
    echo   ERROR: Feature engineering script may have failed or did not produce the output file.
    echo   Please check the console for any error messages from the Python script itself.
)
ECHO.

:EndDebug
echo Batch script finished.
pause
