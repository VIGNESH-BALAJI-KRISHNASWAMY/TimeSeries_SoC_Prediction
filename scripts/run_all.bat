@echo off
REM Batch script to automate SoC data processing,
REM now with options for SOC calculation and Source_File_Name retention.

REM --- Configuration ---
SET PYTHON_EXE=python
REM SET PYTHON_EXE=C:\Users\YourUser\AppData\Local\Programs\Python\Python39\python.exe

SET SCRIPT_DIR=%~dp0
SET PYTHON_SCRIPT=%SCRIPT_DIR%process_data.py
SET BASE_DATA_DIR=%SCRIPT_DIR%..\SOC_estimation_dataset_G10
SET OUTPUT_DIR=%SCRIPT_DIR%..\output_data

REM --- USER CHOICES FOR FINAL PROCESSING ---
REM Choose SOC calculation method: "none", "prof_global", "prof_per_file"
REM SET SOC_METHOD_CHOICE=prof_per_file
SET SOC_METHOD_CHOICE=prof_global
REM SET SOC_METHOD_CHOICE=none

REM Decide whether to drop Source_File_Name from the final output of process_data.py
REM To drop it, set DROP_SOURCE_NAME_FLAG=--drop_source_name_final
REM To keep it, set DROP_SOURCE_NAME_FLAG=
SET DROP_SOURCE_NAME_FLAG=--drop_source_name_final
REM SET DROP_SOURCE_NAME_FLAG=

REM Output filename will depend on whether SOC is calculated
REM Let's keep the final name consistent for now, but you can change it.
REM SET FINAL_OUTPUT_FILENAME=FINAL_MASTER_DATASET_Processed.csv
SET FINAL_OUTPUT_FILENAME=FINAL_MASTER_DATASET_With_SOC.csv
REM If SOC_METHOD_CHOICE is "none", you might name it FINAL_MASTER_DATASET_NoSOC.csv
REM If SOC_METHOD_CHOICE is "prof_global" or "prof_per_file", it could be FINAL_MASTER_DATASET_WithSOC.csv


REM Create output directory if it doesn't exist
IF NOT EXIST "%OUTPUT_DIR%" (
    echo Creating output directory: %OUTPUT_DIR%
    mkdir "%OUTPUT_DIR%"
)

REM --- Temperature folders and their values ---
SET TEMPERATURE_FOLDERS[0]=0degC,0
SET TEMPERATURE_FOLDERS[1]=10degC,10
SET TEMPERATURE_FOLDERS[2]=25degC,25
SET TEMPERATURE_FOLDERS[3]=40degC,40

REM --- Step 1: Process each temperature folder ---
echo Starting processing of individual temperature folders...
echo.

FOR /L %%N IN (0,1,3) DO (
    CALL SET FOLDER_TEMP_PAIR=%%TEMPERATURE_FOLDERS[%%N]%%
    CALL :ProcessPair "%%FOLDER_TEMP_PAIR%%"
)

GOTO :CombineStep

:ProcessPair
    FOR /F "tokens=1,2 delims=," %%A IN ("%~1") DO (
        SET TEMP_FOLDER_NAME=%%A
        SET TEMP_VALUE=%%B
    )
    echo Processing: %TEMP_FOLDER_NAME% (Temperature: %TEMP_VALUE% C)
    SET INPUT_TEMP_DIR=%BASE_DATA_DIR%\%TEMP_FOLDER_NAME%
    SET OUTPUT_TEMP_MASTER_FILE=%OUTPUT_DIR%\%TEMP_FOLDER_NAME%_master.csv
    IF EXIST "%INPUT_TEMP_DIR%" (
        %PYTHON_EXE% "%PYTHON_SCRIPT%" --mode process_temp --input_dir "%INPUT_TEMP_DIR%" --temp_value %TEMP_VALUE% --output_path "%OUTPUT_TEMP_MASTER_FILE%"
        echo   Finished processing %TEMP_FOLDER_NAME%.
        echo.
    ) ELSE (
        echo   WARNING: Input directory not found: %INPUT_TEMP_DIR%
        echo.
    )
    GOTO :EOF

:CombineStep
REM --- Step 2: Combine all master files, optionally calculate SOC, and optionally drop Source_File_Name ---
echo.
echo Starting final combination and processing step...
SET FINAL_OUTPUT_FILE_FULLPATH=%OUTPUT_DIR%\%FINAL_OUTPUT_FILENAME%
SET FINAL_METADATA_LOG_FILE=%OUTPUT_DIR%\experiment_metadata_log.csv 

echo   Input directory for master files (and temp metadata): %OUTPUT_DIR%
echo   SOC Calculation Method Chosen: %SOC_METHOD_CHOICE%
echo   Final Output File will be: %FINAL_OUTPUT_FILE_FULLPATH%
echo   Source_File_Name drop flag: %DROP_SOURCE_NAME_FLAG%
echo   Final metadata log will be: %FINAL_METADATA_LOG_FILE%

%PYTHON_EXE% "%PYTHON_SCRIPT%" --mode combine_all --input_dir "%OUTPUT_DIR%" --output_path "%FINAL_OUTPUT_FILE_FULLPATH%" --metadata_log_output_path "%FINAL_METADATA_LOG_FILE%" --soc_method %SOC_METHOD_CHOICE% %DROP_SOURCE_NAME_FLAG%

echo.
echo All processing complete!
IF EXIST "%FINAL_OUTPUT_FILE_FULLPATH%" (
    echo Final dataset should be available at: %FINAL_OUTPUT_FILE_FULLPATH%
) ELSE (
    echo ERROR: Final output dataset NOT found! Check Python script errors.
)
IF EXIST "%FINAL_METADATA_LOG_FILE%" (
    echo Metadata log should be available at: %FINAL_METADATA_LOG_FILE%
) ELSE (
    echo WARNING: Metadata log NOT found!
)
echo.
pause
