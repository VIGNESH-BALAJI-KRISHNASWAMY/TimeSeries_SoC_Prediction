import pandas as pd
import os
import glob
import argparse
import re
import csv 
from io import StringIO 
import numpy as np 

# --- Configuration for Header Detection ---
EXPECTED_HEADER_KEYWORDS = [
    "Time Stam", "Step", "Status", "Prog Time", "Step Time", "Cycle",
    "Voltage", "Current", "Temperatu", "Capacity", "WhAccu", "Cnt"
]
MIN_KEYWORDS_FOR_HEADER = 5 
MAX_LINES_TO_SCAN_FOR_HEADER = 50 

LINES_TO_INSPECT_AFTER_CANDIDATE = 3 
REQUIRED_DATA_LIKE_LINES_IN_WINDOW = 2 

MIN_DATA_COLUMNS_FOR_DATA_LINE = 3 
MIN_NUMERIC_PERCENT_FOR_DATA_LINE = 0.4 


# --- Configuration for Column Standardization ---
STANDARDIZED_COLUMN_PATTERNS = {
    "Timestamp_Full": [r"time\s*stam"],
    "Step_Number": [r"^Step$", r"step\s*(\[#\])?(\s*\(step\))?"], 
    "Status_Text": [r"^Status$"],
    "Program_Time_s_str": [r"prog\s*time"], 
    "Step_Time_s_str": [r"step\s*time"],    
    "Cycle_Number": [r"^Cycle$", r"cycle\s*(\[#\])?(\s*\(cycle\))?"], 
    "Cycle_Level_Text": [r"cycle\s*leve"], 
    "Procedure_Text": [r"^Procedure$"],   
    "Voltage_V": [r"^Voltage$", r"volt(age)?\s*(\[V\])?", r"u\s*(\[V\])?"],
    "Current_A": [r"^Current$", r"curr(ent)?\s*(\[A\])?", r"i\s*(\[A\])?"],
    "Cell_Temperature_C": [
        r"^Temperatu$", 
        r"cell\s*temp(eratu(re)?)?\s*(\[C\])?", 
        r"temp(eratu(re)?)?\s*(\[C\])?" 
    ],
    "Capacity_Ah": [r"^Capacity$", r"capacity\s*(\[Ah\])?", r"q\s*(\[Ah\])?"],
    "Energy_Wh": [r"^WhAccu$", r"whaccu\s*(\[Wh\])?"],
    "Count_#": [r"^Cnt$", r"cnt\s*(\[Cnt\])?"]
}

COLUMNS_TO_CONVERT_TYPES = {
    "Voltage_V": "float",
    "Current_A": "float",
    "Cell_Temperature_C": "float",
    "Capacity_Ah": "float", 
    "Cycle_Number": "Int64",
    "Step_Number": "Int64",   
    "Energy_Wh": "float",
    "Count_#": "Int64"        
}

METADATA_LABEL_PATTERNS = {
    "start_time": [re.compile(r"^\s*Start Time\s*[:;]?\s*$", re.IGNORECASE)], 
    "end_time": [re.compile(r"^\s*End Time\s*[:;]?\s*$", re.IGNORECASE)],
    "battery_type": [re.compile(r"^\s*Battery Name\s*[:;]?\s*$", re.IGNORECASE)],
    "nominal_voltage_V_meta": [re.compile(r"^\s*Nominal voltage\s*(\[V\])?\s*[:;]?\s*$", re.IGNORECASE)], 
    "max_voltage_V_meta": [re.compile(r"^\s*(?:Charge,\s*)?Max(?:imum)?\s*Voltage\s*(\[V\])?\s*[:;]?\s*$", re.IGNORECASE)],
    "gassing_voltage_V_meta": [re.compile(r"^\s*Gassing\s*Voltage\s*(\[V\])?\s*[:;]?\s*$", re.IGNORECASE)],
    "nominal_capacity_Ah_meta": [re.compile(r"^\s*Nominal capacity\s*(\[Ah\])?\s*[:;]?\s*$", re.IGNORECASE)]
}
MAX_METADATA_LINES_TO_SCAN = 30 


def reorder_csv_columns(input_file, output_file, new_order):
    # Read the input CSV file
    with open(input_file, 'r', newline='') as f_in:
        reader = csv.reader(f_in)
        header = next(reader)
        
        # Check if all columns in new_order exist in the header
        missing_columns = [col for col in new_order if col not in header]
        if missing_columns:
            raise ValueError(f"Columns not found in CSV: {missing_columns}")
        
        # Create mapping: new_order columns -> their positions in original header
        column_indices = [header.index(col) for col in new_order]
        
        # Prepare reordered rows
        reordered_rows = [new_order]  # New header
        
        # Process each data row
        for row in reader:
            reordered_row = [row[idx] for idx in column_indices]
            reordered_rows.append(reordered_row)
    
    # Write to output file
    with open(output_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerows(reordered_rows)






def extract_metadata_from_file(csv_file_path, label_patterns_dict, max_lines):
    metadata = {key: None for key in label_patterns_dict.keys()}
    found_keys = set() 
    try:
        with open(csv_file_path, 'r', encoding='latin1') as f:
            for i, line_str in enumerate(f):
                if i >= max_lines: break
                line_str = line_str.strip()
                if not line_str: continue
                line_buffer = StringIO(line_str)
                csv_parser = csv.reader(line_buffer, delimiter=',')
                cells = []
                try: cells = [cell.strip() for cell in next(csv_parser)]
                except StopIteration: continue
                if not cells: continue
                for cell_idx, cell_content in enumerate(cells):
                    for meta_key, patterns in label_patterns_dict.items():
                        if meta_key in found_keys: continue
                        for pattern in patterns:
                            if pattern.fullmatch(cell_content): 
                                if cell_idx + 1 < len(cells):
                                    value = cells[cell_idx + 1].strip()
                                    value = value.split(';')[0].strip() 
                                    if value: metadata[meta_key] = value; found_keys.add(meta_key); break 
                            if meta_key in found_keys: break 
                    if len(found_keys) == len(label_patterns_dict): break 
                if len(found_keys) == len(label_patterns_dict): break       
    except Exception as e:
        print(f"    Warning: Could not extract metadata from {os.path.basename(csv_file_path)}. Error: {e}")
    for key in label_patterns_dict.keys(): 
        if key not in metadata: metadata[key] = None
    return metadata

def basic_clean_column_name(col_name):
    name_str = str(col_name).strip()
    if not name_str: return "unnamed_original_col"
    name = re.sub(r'\s+|-', '_', name_str) 
    name = re.sub(r'[\[\]\(\)\:\.]+', '', name) 
    name = re.sub(r'[^0-9a-zA-Z_]', '', name) 
    name = re.sub(r'_+', '_', name)
    name = name.strip('_') 
    if not name: 
        name = re.sub(r'[^0-9a-zA-Z]+', '_', name_str) 
        name = re.sub(r'_+', '_', name).strip('_')
        if not name: return "unnamed_original_col"
    if name and name[0].isdigit(): name = '_' + name
    return name

def standardize_column_names(df, column_patterns, filename_for_logging=""):
    print(f"    Attempting to standardize columns for {filename_for_logging}:")
    print(f"      Original columns (as read by pandas): {list(df.columns)}")
    potential_renames = {} 
    for original_col_name in df.columns:
        potential_renames[original_col_name] = []
        for std_name, patterns in column_patterns.items():
            for pattern in patterns:
                try:
                    if re.search(pattern, str(original_col_name), re.IGNORECASE):
                        potential_renames[original_col_name].append(std_name)
                except Exception as e:
                    print(f"      Regex error for pattern '{pattern}' on column '{original_col_name}': {e}")
    final_rename_map = {}
    used_std_names = set()
    for original_col, mapped_std_names in potential_renames.items():
        if len(mapped_std_names) == 1:
            std_name_candidate = mapped_std_names[0]
            if std_name_candidate not in used_std_names:
                final_rename_map[original_col] = std_name_candidate
                used_std_names.add(std_name_candidate)
                print(f"      Exact Map: '{original_col}'  ==>  '{std_name_candidate}' (via patterns for {std_name_candidate})")
    for original_col, mapped_std_names in potential_renames.items():
        if original_col in final_rename_map: continue 
        if mapped_std_names:
            chosen_std_name = None
            for std_name_candidate in mapped_std_names:
                if std_name_candidate not in used_std_names:
                    chosen_std_name = std_name_candidate; break
            if chosen_std_name:
                final_rename_map[original_col] = chosen_std_name
                used_std_names.add(chosen_std_name)
                print(f"      Selected Map: '{original_col}'  ==>  '{chosen_std_name}' (from options: {mapped_std_names})")
            else: print(f"      Conflict/No Unused Map: '{original_col}' matched {mapped_std_names}, but target(s) already used. Will undergo basic cleaning.")
        else: print(f"      No Standard Map: '{original_col}' did not match any defined patterns. Will undergo basic cleaning.")
    df.rename(columns=final_rename_map, inplace=True)
    columns_after_std_rename = list(df.columns)
    basic_cleaned_map = {}
    current_df_columns_set = set(df.columns) 
    for col_name_original_case in columns_after_std_rename: 
        is_a_standardized_target_name = col_name_original_case in final_rename_map.values()
        if not is_a_standardized_target_name:
            cleaned_name = basic_clean_column_name(col_name_original_case)
            if cleaned_name != col_name_original_case: 
                final_unique_cleaned_name = cleaned_name; suffix_counter = 1
                while final_unique_cleaned_name in current_df_columns_set and final_unique_cleaned_name != col_name_original_case:
                    final_unique_cleaned_name = f"{cleaned_name}_{suffix_counter}"; suffix_counter += 1
                if final_unique_cleaned_name != col_name_original_case:
                    basic_cleaned_map[col_name_original_case] = final_unique_cleaned_name
                    current_df_columns_set.add(final_unique_cleaned_name) 
                    if col_name_original_case in current_df_columns_set: current_df_columns_set.remove(col_name_original_case)
                    print(f"      Basic Clean: '{col_name_original_case}'  ==>  '{final_unique_cleaned_name}'")
    if basic_cleaned_map: df.rename(columns=basic_cleaned_map, inplace=True)
    print(f"      Final columns after all standardization: {list(df.columns)}")
    return df

def is_potentially_numeric(value_str):
    if not isinstance(value_str, str): return False
    value_str = value_str.strip()
    if not value_str: return False
    try: float(value_str.replace(',', '.')); return True
    except ValueError:
        if re.search(r"[a-zA-Z%]", value_str): return False 
        return False

def find_header_row(csv_file_path, keywords, min_keyword_matches, max_lines_to_scan, 
                    lines_to_inspect_after_candidate, required_data_like_lines_in_window,
                    min_data_cols_for_data_line, min_numeric_percent_for_data_line):
    print(f"    Finding header for: {os.path.basename(csv_file_path)}")
    try:
        with open(csv_file_path, 'r', encoding='latin1') as f: lines = [line.strip() for line in f.readlines()] 
    except FileNotFoundError: print(f"      Error: File not found for header detection: {csv_file_path}"); return None
    except Exception as e: print(f"      Error reading file for header detection {csv_file_path}: {e}"); return None
    for i, candidate_header_str in enumerate(lines):
        if i >= max_lines_to_scan: print(f"      Scanned {max_lines_to_scan} lines, header not confidently identified."); break
        if not candidate_header_str: continue
        candidate_header_fields = [part.strip() for part in candidate_header_str.split(',')]
        num_candidate_header_fields = len(candidate_header_fields)
        if num_candidate_header_fields < min_data_cols_for_data_line: continue 
        normalized_candidate_parts = [part.lower() for part in candidate_header_fields] 
        keyword_matches_count = sum(1 for keyword in keywords if any(keyword.lower() in part for part in normalized_candidate_parts))
        if keyword_matches_count >= min_keyword_matches or num_candidate_header_fields > (min_data_cols_for_data_line + 2) : 
            print(f"      Line {i}: '{candidate_header_str[:100]}...' - Fields: {num_candidate_header_fields}, Keywords found: {keyword_matches_count}")
        if keyword_matches_count >= min_keyword_matches:
            print(f"        Candidate header on line {i} (keywords matched: {keyword_matches_count} >= {min_keyword_matches}, fields: {num_candidate_header_fields}). Inspecting subsequent lines...")
            data_like_lines_found_in_window = 0
            for j in range(1, lines_to_inspect_after_candidate + 1):
                line_index_to_check = i + j
                if line_index_to_check < len(lines):
                    line_to_check_str = lines[line_index_to_check]
                    if not line_to_check_str: print(f"          Line {line_index_to_check}: (empty) - Not data-like."); continue
                    line_to_check_fields = [part.strip() for part in line_to_check_str.split(',')]
                    num_fields_in_line_to_check = len(line_to_check_fields)
                    if num_fields_in_line_to_check < min_data_cols_for_data_line:
                        print(f"          Line {line_index_to_check}: '{line_to_check_str[:100]}...' - Too few columns ({num_fields_in_line_to_check} < {min_data_cols_for_data_line}). Not data-like."); continue
                    numeric_fields_count = sum(1 for field in line_to_check_fields if is_potentially_numeric(field))
                    percent_numeric = numeric_fields_count / num_fields_in_line_to_check if num_fields_in_line_to_check > 0 else 0
                    if percent_numeric >= min_numeric_percent_for_data_line:
                        print(f"          Line {line_index_to_check}: '{line_to_check_str[:100]}...' - Columns: {num_fields_in_line_to_check}, Numeric: {numeric_fields_count} ({percent_numeric:.2%}). LOOKS DATA-LIKE."); data_like_lines_found_in_window += 1
                    else: print(f"          Line {line_index_to_check}: '{line_to_check_str[:100]}...' - Columns: {num_fields_in_line_to_check}, Numeric: {numeric_fields_count} ({percent_numeric:.2%}). Not enough numeric data. NOT DATA-LIKE.")
                else: print(f"          Line {line_index_to_check}: (beyond end of file)."); break 
            if data_like_lines_found_in_window >= required_data_like_lines_in_window:
                print(f"        CONFIRMED: Line {i} is header. Found {data_like_lines_found_in_window} data-like lines in the next {lines_to_inspect_after_candidate} lines (required {required_data_like_lines_in_window})."); return i 
            else: print(f"        REJECTED: Candidate on line {i}. Found only {data_like_lines_found_in_window} data-like lines in window (required {required_data_like_lines_in_window}).")
    print(f"    Header not confidently identified in {os.path.basename(csv_file_path)} after scanning all candidates."); return None

# --- SOC Calculation Functions (Professor's Method) ---
def calculate_soc_prof_global(df, capacity_ah_col='Capacity_Ah', soc_col_name_prof='SoC_Percentage'): # MODIFIED soc_col_name
    print("    Calculating SOC using GLOBAL Max Discharge method...")
    df_soc = df.copy()
    max_discharge_col_name = 'Max_Discharge_Ah_Global' # MODIFIED column name

    if capacity_ah_col not in df_soc.columns:
        print(f"      Warning: '{capacity_ah_col}' not found. Skipping global SOC calculation.")
        df_soc[soc_col_name_prof] = np.nan
        df_soc[max_discharge_col_name] = np.nan
        return df_soc

    df_soc[f'{capacity_ah_col}_numeric'] = pd.to_numeric(df_soc[capacity_ah_col], errors='coerce')
    if df_soc[f'{capacity_ah_col}_numeric'].isnull().all():
        print(f"      Warning: All '{capacity_ah_col}_numeric' are NaN. Skipping global SOC calculation.")
        df_soc[soc_col_name_prof] = np.nan
        df_soc[max_discharge_col_name] = np.nan
        df_soc.drop(columns=[f'{capacity_ah_col}_numeric'], inplace=True, errors='ignore')
        return df_soc

    global_min_capacity = df_soc[f'{capacity_ah_col}_numeric'].min()
    global_max_discharge_val = np.nan
    if pd.notna(global_min_capacity) and global_min_capacity < 0: # Assumes discharge makes Capacity_Ah negative
        global_max_discharge_val = abs(global_min_capacity)
        print(f"      Global Max_Discharge calculated: {global_max_discharge_val:.4f} Ah")
    elif pd.notna(global_min_capacity) and global_min_capacity == 0 and df_soc[f'{capacity_ah_col}_numeric'].max() > 0: # Handles cases where Capacity_Ah might be remaining capacity (positive)
        global_max_discharge_val = df_soc[f'{capacity_ah_col}_numeric'].max()
        print(f"      Global Max_Discharge (from max positive Capacity_Ah): {global_max_discharge_val:.4f} Ah")

    df_soc[max_discharge_col_name] = global_max_discharge_val 

    if pd.isna(global_max_discharge_val) or global_max_discharge_val == 0:
        print(f"      Warning: Global Max_Discharge is 0 or NaN. SOC for all rows will be NaN.")
        df_soc[soc_col_name_prof] = np.nan
    else:
        # Using professor's literal formula
        df_soc[soc_col_name_prof] = abs (df_soc[f'{capacity_ah_col}_numeric']) / global_max_discharge_val
        print(f"      Calculated '{soc_col_name_prof}' using global method (literal formula). Sample: {df_soc[soc_col_name_prof].head().values}")
        # Note: This might produce SOC outside 0-1 or negative SOC depending on Capacity_Ah definition.
        # If Capacity_Ah is 0 when full and -max_discharge when empty, SOC will be 0 to -1.
        # If Capacity_Ah is max_discharge when full and 0 when empty, SOC will be 1 to 0.
    
    df_soc.drop(columns=[f'{capacity_ah_col}_numeric'], inplace=True, errors='ignore')
    return df_soc

def calculate_soc_prof_per_file(df, capacity_ah_col='Capacity_Ah', source_file_col='Source_File_Name', soc_col_name_prof='SoC_Percentage'): # MODIFIED soc_col_name
    print("    Calculating SOC using PER-FILE Max Discharge method...")
    df_soc = df.copy()
    max_discharge_col_name = 'Max_Discharge_Ah_Per_File' # MODIFIED column name

    if capacity_ah_col not in df_soc.columns:
        print(f"      Warning: '{capacity_ah_col}' not found. Skipping per-file SOC calculation.")
        df_soc[soc_col_name_prof] = np.nan
        df_soc[max_discharge_col_name] = np.nan
        return df_soc
    if source_file_col not in df_soc.columns:
        print(f"      Warning: '{source_file_col}' not found for grouping. Skipping per-file SOC calculation.")
        df_soc[soc_col_name_prof] = np.nan
        df_soc[max_discharge_col_name] = np.nan
        return df_soc

    df_soc[f'{capacity_ah_col}_numeric'] = pd.to_numeric(df_soc[capacity_ah_col], errors='coerce')
    df_soc[soc_col_name_prof] = np.nan 
    df_soc[max_discharge_col_name] = np.nan 

    for file_name, group_df in df_soc.groupby(source_file_col):
        print(f"      Processing SOC for file: {file_name}")
        if group_df[f'{capacity_ah_col}_numeric'].isnull().all():
            print(f"        Warning: All '{capacity_ah_col}_numeric' are NaN for {file_name}. SOC will be NaN.")
            continue

        min_cap_file = group_df[f'{capacity_ah_col}_numeric'].min()
        max_discharge_file_val = np.nan
        
        if pd.notna(min_cap_file) and min_cap_file < 0:
            max_discharge_file_val = abs(min_cap_file)
            print(f"        Max_Discharge for {file_name} (from min negative Capacity_Ah): {max_discharge_file_val:.4f} Ah")
        elif pd.notna(min_cap_file) and min_cap_file == 0 and group_df[f'{capacity_ah_col}_numeric'].max() > 0 : # Check if Capacity_Ah is logged as positive remaining/charged capacity
            max_discharge_file_val = group_df[f'{capacity_ah_col}_numeric'].max()
            print(f"        Max_Discharge for {file_name} (from max positive Capacity_Ah): {max_discharge_file_val:.4f} Ah")
        
        df_soc.loc[group_df.index, max_discharge_col_name] = max_discharge_file_val 

        if pd.isna(max_discharge_file_val) or max_discharge_file_val == 0:
            print(f"        Warning: Max_Discharge for {file_name} is 0 or NaN. SOC for this file will be NaN.")
            continue
            
        # Using professor's literal formula
        soc_values = group_df[f'{capacity_ah_col}_numeric'] / max_discharge_file_val
        df_soc.loc[group_df.index, soc_col_name_prof] = soc_values
        # NO CLIPPING APPLIED to strictly follow the literal formula. User must interpret the range.
        print(f"        Calculated '{soc_col_name_prof}' for {file_name} (literal formula). Sample: {df_soc.loc[group_df.index, soc_col_name_prof].head().values}")

    df_soc.drop(columns=[f'{capacity_ah_col}_numeric'], inplace=True, errors='ignore')
    return df_soc


def process_temperature_folder(input_dir, temp_value, output_csv_path):
    all_csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not all_csv_files: print(f"No CSV files found in {input_dir}"); return

    list_of_dataframes = []
    current_folder_metadata_list = [] 
    
    print(f"\nProcessing temperature: {temp_value}°C from directory: {input_dir}")

    for csv_file in all_csv_files:
        original_file_name = os.path.basename(csv_file)
        file_metadata = extract_metadata_from_file(csv_file, METADATA_LABEL_PATTERNS, MAX_METADATA_LINES_TO_SCAN)
        file_metadata['Source_File_Name'] = original_file_name 
        file_metadata['Ambient_Temperature_C_param'] = temp_value 
        current_folder_metadata_list.append(file_metadata) 
        print(f"    Extracted metadata for {original_file_name}: {file_metadata}")

        header_row_index = find_header_row(csv_file, EXPECTED_HEADER_KEYWORDS,
                                           MIN_KEYWORDS_FOR_HEADER, MAX_LINES_TO_SCAN_FOR_HEADER,
                                           LINES_TO_INSPECT_AFTER_CANDIDATE, 
                                           REQUIRED_DATA_LIKE_LINES_IN_WINDOW,
                                           MIN_DATA_COLUMNS_FOR_DATA_LINE, 
                                           MIN_NUMERIC_PERCENT_FOR_DATA_LINE)
        if header_row_index is None: print(f"    Warning: Could not identify header row in {original_file_name} for processing. Skipping."); continue
        
        try:
            header_line_content = None; parsed_column_names = []
            with open(csv_file, 'r', encoding='latin1') as f_hdr:
                all_lines_for_header = f_hdr.readlines() 
                if header_row_index < len(all_lines_for_header): header_line_content = all_lines_for_header[header_row_index].strip()
            if not header_line_content: print(f"    Error: Identified header line {header_row_index} is empty or could not be read from {original_file_name}. Skipping."); continue
            header_file_like = StringIO(header_line_content)
            csv_parser = csv.reader(header_file_like, delimiter=',') 
            try: parsed_column_names = [name.strip() for name in next(csv_parser)]
            except StopIteration: parsed_column_names = []
            if not parsed_column_names or (len(parsed_column_names) == 1 and not parsed_column_names[0]): 
                 print(f"    WARNING-POTENTIAL_HEADER_SPLIT_ISSUE: Manually parsed header for {original_file_name} resulted in empty or single empty column name: {parsed_column_names}. Skipping file."); continue
            elif len(parsed_column_names) == 1 and parsed_column_names[0] == header_line_content : 
                 print(f"    WARNING-POTENTIAL_HEADER_SPLIT_ISSUE: Manually parsed header for {original_file_name} did not split and resulted in: {parsed_column_names}. This might indicate a delimiter problem in the header line itself. Check file encoding and delimiters.")

            df = pd.read_csv(csv_file, skiprows=header_row_index + 1, names=parsed_column_names, encoding='latin1', on_bad_lines='warn', sep=',')
            if df.empty: print(f"    Warning: DataFrame is empty after reading {original_file_name} with manually parsed headers and skiprows. Skipping."); continue
            df = standardize_column_names(df, STANDARDIZED_COLUMN_PATTERNS, original_file_name)
            df['Source_File_Name'] = original_file_name 
            for std_col_name, dtype in COLUMNS_TO_CONVERT_TYPES.items(): 
                if std_col_name in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[std_col_name]):
                        try:
                            if dtype == "float": df[std_col_name] = pd.to_numeric(df[std_col_name], errors='coerce')
                            elif dtype == "Int64": df[std_col_name] = pd.to_numeric(df[std_col_name], errors='coerce').astype('Int64')
                        except Exception as e: print(f"    Warning: Could not convert standardized column '{std_col_name}' to {dtype} in {original_file_name}. Error: {e}")
            list_of_dataframes.append(df)
        except pd.errors.EmptyDataError: print(f"    Error: File {original_file_name} is empty or became empty after header processing.")
        except Exception as e: print(f"    Error reading/processing file {original_file_name}: {e}")
    
    if not list_of_dataframes: print(f"No dataframes were created for {input_dir}. Output file for main data will not be generated."); 
    else:
        try: combined_df = pd.concat(list_of_dataframes, ignore_index=True, sort=False)
        except Exception as e: print(f"Error during concatenation for {input_dir}: {e}"); return 
        combined_df["Ambient_Temperature_C"] = temp_value
        desired_order = ['Source_File_Name', 'Ambient_Temperature_C']
        for col in COLUMNS_TO_CONVERT_TYPES.keys():
            if col not in desired_order: desired_order.append(col)
        for col in STANDARDIZED_COLUMN_PATTERNS.keys():
            if col not in desired_order: desired_order.append(col)
        final_cols_order = [col for col in desired_order if col in combined_df.columns]
        remaining_cols = [col for col in combined_df.columns if col not in final_cols_order]
        final_cols_order.extend(remaining_cols)
        try: combined_df = combined_df[final_cols_order]
        except KeyError as e:
            print(f"    Warning: Key error during column reordering for {output_csv_path}. This can happen if expected columns are missing. Proceeding with available columns. Error: {e}")
            valid_order = [col for col in final_cols_order if col in combined_df.columns];
            if valid_order: combined_df = combined_df[valid_order]
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        try:
            combined_df.to_csv(output_csv_path, index=False)
            print(f"Successfully saved combined data for {temp_value}°C to {output_csv_path}. Shape: {combined_df.shape}")
        except Exception as e: print(f"Error saving combined data to {output_csv_path}: {e}")

    if current_folder_metadata_list:
        temp_metadata_df = pd.DataFrame(current_folder_metadata_list)
        output_dir_for_temp = os.path.dirname(output_csv_path) 
        if not output_dir_for_temp: output_dir_for_temp = "." 
        base_name_for_temp_files = os.path.basename(input_dir) 
        temp_metadata_file_path = os.path.join(output_dir_for_temp, f"{base_name_for_temp_files}_metadata_temp.csv")
        os.makedirs(output_dir_for_temp, exist_ok=True)
        try:
            temp_metadata_df.to_csv(temp_metadata_file_path, index=False)
            print(f"  Successfully saved temporary metadata for {temp_value}°C to {temp_metadata_file_path}")
        except Exception as e:
            print(f"  Error saving temporary metadata for {temp_value}°C to {temp_metadata_file_path}: {e}")
    else:
        print(f"  No metadata extracted for folder {input_dir}, temporary metadata file not created.")


def combine_master_files(master_files_input_dir, final_data_output_path, final_metadata_log_path, 
                         soc_method_choice, drop_source_name_final_flag): 
    all_master_files = glob.glob(os.path.join(master_files_input_dir, "*_master.csv"))
    final_df = None 

    if not all_master_files: print(f"No master data CSV files found in {master_files_input_dir} to combine."); 
    else:
        list_of_master_dataframes = []
        print(f"\nCombining master data files from: {master_files_input_dir}")
        for master_file in all_master_files:
            print(f"  Reading master data file: {os.path.basename(master_file)}")
            try: df = pd.read_csv(master_file, low_memory=False); list_of_master_dataframes.append(df)
            except Exception as e: print(f"    Error reading master data file {master_file}: {e}")
        if not list_of_master_dataframes: print("No master dataframes were loaded. Final dataset will not be generated.")
        else:
            final_df = pd.concat(list_of_master_dataframes, ignore_index=True, sort=False)
            print(f"  Successfully combined master data files. Shape before SOC calc & final drops: {final_df.shape}")
            
            # --- NEW: Remove rows that have a high number of empty (NaN) columns ---
            original_row_count_before_nan_drop = len(final_df)
            if final_df.shape[1] > 0 : # Ensure there are columns to evaluate
                # Keep rows that have at least (total_columns - 3) non-NaN values.
                # This means drop rows if they have 4 or more NaN values.
                # Example: If 10 columns, keep rows with at least 10-3 = 7 non-NaNs.
                # If a row has only 1 or 2 non-NaNs, it will be dropped if total_columns >= 5 or 6.
                min_non_nan_to_keep = final_df.shape[1] - 3 
                
                if min_non_nan_to_keep < 1 and final_df.shape[1] > 0 : 
                    # If total columns is 1, 2, or 3, this makes thresh=0, -1, or -2.
                    # We should ensure we keep rows with at least 1 actual value.
                    min_non_nan_to_keep = 1 
                    print(f"  INFO: Adjusted min_non_nan_to_keep to {min_non_nan_to_keep} for row cleaning due to low total column count ({final_df.shape[1]}).")
                
                if final_df.shape[1] == 0: # Should not happen if concat was successful
                     print("  INFO: DataFrame has no columns, skipping NaN row cleaning.")
                elif min_non_nan_to_keep <= 0 and final_df.shape[1] > 0 : 
                    # This case means total_columns <= 3, and we'd keep rows even if all are NaN based on the formula.
                    # The min_non_nan_to_keep = 1 adjustment above handles this better.
                    print(f"  INFO: Row cleaning threshold (min_non_nan_to_keep={min_non_nan_to_keep}) is low, implies keeping rows with very few non-NaNs. This specific condition might not drop rows as intended if total columns is very small.")
                else:
                    final_df.dropna(thresh=min_non_nan_to_keep, inplace=True)
                    rows_dropped_by_nan = original_row_count_before_nan_drop - len(final_df)
                    if rows_dropped_by_nan > 0:
                        print(f"  INFO: Removed {rows_dropped_by_nan} rows that had 4 or more NaN values (i.e., fewer than {min_non_nan_to_keep} non-NaN values).")
                    else:
                        print("  INFO: No rows were removed based on the 'at least 4 NaN values' criterion.")
            else:
                print("  INFO: DataFrame has no columns after concatenation, skipping NaN row cleaning.")
            print(f"  Shape after row cleaning based on NaN count: {final_df.shape}")
            if final_df.empty:
                print("WARNING: DataFrame is empty after NaN row cleaning. Subsequent processing might fail or produce an empty output file.")
                # Potentially save empty df and exit if that's desired
            # --- End of new row cleaning section ---
            
            print("\nProcessing Timestamps...")

            if "Timestamp_Full" in final_df.columns and not final_df.empty:
                try:
                    # Convert to datetime
                    final_df['DateTime_Object'] = pd.to_datetime(
                        final_df['Timestamp_Full'], 
                        format='%m/%d/%Y %I:%M:%S %p',
                        errors='coerce'
                    )
        
                    # Find minimum timestamp
                    min_timestamp = final_df['DateTime_Object'].min()
                    print(f"  Minimum Timestamp_Full in entire dataset: {min_timestamp}")
        
                    # Calculate seconds since min timestamp
                    final_df['Time_Seconds'] = (
                        final_df['DateTime_Object'] - min_timestamp
                    ).dt.total_seconds()
        
                    # Split into Date and Time columns
                    split_result = final_df['Timestamp_Full'].str.split(expand=True, n=2)
                    if len(split_result.columns) >= 3:
                        final_df['Date'] = split_result[0]
                        final_df['Time'] = split_result[1] + " " + split_result[2]
                    else:
                        print("  Warning: Couldn't split Timestamp_Full into Date/Time properly")
                        final_df['Date'] = np.nan
                        final_df['Time'] = np.nan
        
                    # Sort by timestamp
                    final_df.sort_values('DateTime_Object', inplace=True)
        
                    # Drop temporary datetime column
                    final_df.drop(columns=['DateTime_Object'], inplace=True)
        
                    print("  Successfully processed timestamps and added:")
                    print("    - Time_Seconds: Seconds since minimum timestamp")
                    print("    - Date: Extracted date string")
                    print("    - Time: Extracted time string with AM/PM")
                    print("  Rows sorted by Timestamp_Full")
        
                except Exception as e:
                    print(f"  Error processing Timestamp_Full: {e}")
            else:
                print("  'Timestamp_Full' column not found or empty dataset. Skipping timestamp processing.")
            # --- Conditionally Calculate SOC ---
            soc_col_name_to_use = "SoC_Percentage" # Professor's requested name
            if soc_method_choice == "prof_global":
                final_df = calculate_soc_prof_global(final_df, soc_col_name_prof=soc_col_name_to_use)
            elif soc_method_choice == "prof_per_file":
                if 'Source_File_Name' not in final_df.columns:
                    print("  ERROR: 'Source_File_Name' column is required for 'prof_per_file' SOC calculation method but it's not in the DataFrame. Skipping SOC calculation.")
                    final_df[soc_col_name_to_use] = np.nan 
                    final_df['Per_File_Max_Discharge_Ah'] = np.nan 
                else:
                    final_df = calculate_soc_prof_per_file(final_df, soc_col_name_prof=soc_col_name_to_use)
            elif soc_method_choice == "none":
                print("  SOC calculation skipped as per configuration.")
            else:
                print(f"  Warning: Unknown SOC method '{soc_method_choice}'. SOC calculation skipped.")

            cols_to_drop_prof_feedback = ['Procedure_Text', 'Cycle_Level_Text']
            existing_cols_to_drop = [col for col in cols_to_drop_prof_feedback if col in final_df.columns]
            if existing_cols_to_drop:
                final_df.drop(columns=existing_cols_to_drop, inplace=True)
                print(f"  Dropped columns based on professor feedback: {existing_cols_to_drop}")
            
            unnamed_cols_to_remove = []
            for col_name in final_df.columns:
                if col_name == "unnamed_original_col" or re.fullmatch(r"unnamed_original_col_\d+", col_name):
                    unnamed_cols_to_remove.append(col_name)
            if unnamed_cols_to_remove:
                final_df.drop(columns=unnamed_cols_to_remove, inplace=True)
                print(f"  INFO: Automatically removed the following 'unnamed_original_col' (and suffixed versions): {unnamed_cols_to_remove}")
                
                # --- Timestamp Processing ---
            print("\nProcessing Timestamps...")
            
           
            try:
                output_dir = os.path.dirname(args.output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                final_df.to_csv(args.output_path, index=False)
                print("Successfully saved refined dataset.")
                print(f"Final shape: {final_df.shape}")
            except Exception as e:
                print(f"Error saving output CSV: {e}")

            
            if drop_source_name_final_flag: 
                if 'Source_File_Name' in final_df.columns:
                    final_df.drop(columns=['Source_File_Name'], inplace=True)
                    print("  Dropped 'Source_File_Name' column from the final dataset as requested.")
                else:
                    print("  Info: --drop_source_name_final was set, but 'Source_File_Name' column was not found to drop.")
                if 'Program_Time_s_str' in final_df.columns:
                    final_df.drop(columns=['Program_Time_s_str'], inplace=True)
                    print("  Dropped 'Program_Time_s_str' column from the final dataset as requested.")
                else:
                    print("  Info: --drop_source_name_final was set, but 'Program_Time_s_str' column was not found to drop.")
                if 'Step_Time_s_str' in final_df.columns:
                    final_df.drop(columns=['Step_Time_s_str'], inplace=True)
                    print("  Dropped 'Step_Time_s_str' column from the final dataset as requested.")
                else:
                    print("  Info: --drop_source_name_final was set, but 'Step_Time_s_str' column was not found to drop.")
                if 'Max_Discharge_Ah_Global' in final_df.columns:
                    final_df.drop(columns=['Max_Discharge_Ah_Global'], inplace=True)
                    print("  Dropped 'Max_Discharge_Ah_Global' column from the final dataset as requested.")
                else:
                    print("  Info: --drop_source_name_final was set, but 'Max_Discharge_Ah_Global' column was not found to drop.")
                if 'Cycle_Number' in final_df.columns:
                    final_df.drop(columns=['Cycle_Number'], inplace=True)
                    print("  Dropped 'Cycle_Number' column from the final dataset as requested.")
                else:
                    print("  Info: --drop_source_name_final was set, but 'Cycle_Number' column was not found to drop.")
                if 'Status_Text' in final_df.columns:
                    final_df.drop(columns=['Status_Text'], inplace=True)
                    print("  Dropped 'Status_Text' column from the final dataset as requested.")
                else:
                    print("  Info: --drop_source_name_final was set, but 'Status_Text' column was not found to drop.")
                
            else:
                print("  Kept 'Source_File_Name' column in the final dataset.")

            # --- Column Reordering ---
            desired_order = []
            # Add key identifiers first if they exist
            if 'Source_File_Name' in final_df.columns: desired_order.append('Source_File_Name') 
            if 'Ambient_Temperature_C' in final_df.columns: desired_order.append('Ambient_Temperature_C')
            # Add new SOC and Max_Discharge columns early if they exist
            if soc_col_name_to_use in final_df.columns: desired_order.append(soc_col_name_to_use) 
            if 'Global_Max_Discharge_Ah' in final_df.columns: desired_order.append('Global_Max_Discharge_Ah')
            if 'Per_File_Max_Discharge_Ah' in final_df.columns: desired_order.append('Per_File_Max_Discharge_Ah')

            # Add other important columns
            for col in COLUMNS_TO_CONVERT_TYPES.keys(): 
                if col not in desired_order and col in final_df.columns : desired_order.append(col)
            for col in STANDARDIZED_COLUMN_PATTERNS.keys(): 
                # Only add if it exists in final_df and isn't one of the dropped ones or already added
                if col not in desired_order and col in final_df.columns and col not in existing_cols_to_drop and col not in unnamed_cols_to_remove: 
                    desired_order.append(col)
            
            current_cols_in_df = set(final_df.columns)
            final_cols_order = [col for col in desired_order if col in current_cols_in_df] # Ensure all in order exist
            remaining_cols = [col for col in current_cols_in_df if col not in final_cols_order] # Add any other columns
            final_cols_order.extend(remaining_cols)
            
            try: 
                final_df = final_df[final_cols_order]
            except KeyError as e:
                print(f"    Warning: Key error during final column reordering for {final_data_output_path}. Proceeding with available columns. Error: {e}")
                valid_order = [col for col in final_cols_order if col in final_df.columns];
                if valid_order: final_df = final_df[valid_order]

            os.makedirs(os.path.dirname(final_data_output_path), exist_ok=True)
            final_df.to_csv(final_data_output_path, index=False)
            print(f"Successfully saved final processed data to {final_data_output_path}. Final shape: {final_df.shape}")
            print(f"  Final columns in {os.path.basename(final_data_output_path)}: {list(final_df.columns)}")

    # --- Combine temporary metadata logs ---
    all_temp_metadata_files = glob.glob(os.path.join(master_files_input_dir, "*_metadata_temp.csv"))
    if not all_temp_metadata_files:
        print("No temporary metadata files found to combine for the final metadata log.")
        return 

    list_of_metadata_dataframes = []
    print(f"\nCombining temporary metadata files from: {master_files_input_dir}")
    for temp_meta_file in all_temp_metadata_files:
        print(f"  Reading temporary metadata file: {os.path.basename(temp_meta_file)}")
        try:
            df_meta = pd.read_csv(temp_meta_file)
            list_of_metadata_dataframes.append(df_meta)
        except Exception as e:
            print(f"    Error reading temporary metadata file {temp_meta_file}: {e}")

    if list_of_metadata_dataframes:
        final_metadata_df = pd.concat(list_of_metadata_dataframes, ignore_index=True, sort=False)
        metadata_cols = ['Source_File_Name', 'Ambient_Temperature_C_param'] 
        metadata_cols.extend(list(METADATA_LABEL_PATTERNS.keys()))
        ordered_metadata_cols = [col for col in metadata_cols if col in final_metadata_df.columns]
        ordered_metadata_cols.extend([col for col in final_metadata_df.columns if col not in ordered_metadata_cols])
        
        # Ensure all columns in ordered_metadata_cols actually exist in final_metadata_df before reindexing
        valid_metadata_cols_order = [col for col in ordered_metadata_cols if col in final_metadata_df.columns]
        final_metadata_df = final_metadata_df[valid_metadata_cols_order]

        os.makedirs(os.path.dirname(final_metadata_log_path), exist_ok=True)
        final_metadata_df.to_csv(final_metadata_log_path, index=False)
        print(f"Successfully saved final metadata log to {final_metadata_log_path}. Shape: {final_metadata_df.shape}")
        print("  Attempting to delete temporary metadata files...")
        for temp_meta_file in all_temp_metadata_files:
            try: os.remove(temp_meta_file); print(f"    Deleted temporary metadata file: {os.path.basename(temp_meta_file)}")
            except Exception as e: print(f"    Error deleting temporary metadata file {os.path.basename(temp_meta_file)}: {e}")
    else:
        print("No actual metadata was collected from temporary files to create the final metadata log.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and combine battery SoC CSV data. Optionally calculates SOC and drops Source_File_Name.")
    parser.add_argument("--mode", choices=['process_temp', 'combine_all'], required=True)
    parser.add_argument("--input_dir", help="Input directory.")
    parser.add_argument("--temp_value", type=float, help="Temperature value (for 'process_temp' mode).")
    parser.add_argument("--output_path", help="Output CSV file path for processed temp data or final combined data.")
    parser.add_argument("--metadata_log_output_path", default="experiment_metadata_log.csv", 
                        help="Filename for the collected metadata log (used in 'combine_all' mode). Default: experiment_metadata_log.csv, saved in the same directory as the final data output.")
    parser.add_argument("--soc_method", choices=['none', 'prof_global', 'prof_per_file'], default='none',
                        help="SOC calculation method to apply during 'combine_all' mode. 'prof_global' uses global max discharge, 'prof_per_file' uses per-file max discharge. Default: 'none'.")
    parser.add_argument("--drop_source_name_final", action='store_true', 
                        help="If set, 'Source_File_Name' will be dropped from the final output of 'combine_all' mode.")

    args = parser.parse_args()

    if args.mode == 'process_temp':
        if not all([args.input_dir, args.temp_value is not None, args.output_path]):
            parser.error("--input_dir, --temp_value, and --output_path are required for 'process_temp' mode.")
        process_temperature_folder(args.input_dir, args.temp_value, args.output_path)
    elif args.mode == 'combine_all':
        if not all([args.input_dir, args.output_path]): 
            parser.error("--input_dir (for master and temp metadata files) and --output_path (for final data file) are required.")
        final_output_dir_for_log = os.path.dirname(args.output_path) 
        if not final_output_dir_for_log: final_output_dir_for_log = args.input_dir 
        final_metadata_log_full_path = os.path.join(final_output_dir_for_log, os.path.basename(args.metadata_log_output_path))
        
        
        
        combine_master_files(args.input_dir, 
                             args.output_path, 
                             final_metadata_log_full_path,
                             args.soc_method, 
                             args.drop_source_name_final)
        
        desired_order = ['Date', 'Time', 'Time_Seconds', 'Ambient_Temperature_C', 'SoC_Percentage', 'Voltage_V', 'Current_A', 'Cell_Temperature_C', 'Capacity_Ah', 'Step_Number', 'Energy_Wh', 'Count_#']            
        reorder_csv_columns(args.output_path, args.output_path, desired_order)