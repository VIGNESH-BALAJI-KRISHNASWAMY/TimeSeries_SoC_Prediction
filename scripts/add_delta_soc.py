# import pandas as pd
# import os

# INPUT_FILENAME = 'FINAL_MASTER_DATASET_with_SOC.csv'

# def add_delta_soc_column(df):
    # if 'SoC_Percentage' not in df.columns:
        # raise ValueError("Missing required column: 'SoC_Percentage'")

    # if 'Source_File_Name' in df.columns:
        # df['Delta_SoC'] = df.groupby('Source_File_Name')['SoC_Percentage'].diff()
    # else:
        # df['Delta_SoC'] = df['SoC_Percentage'].diff()

    # # Move Delta_SoC to the end
    # columns = [col for col in df.columns if col != 'Delta_SoC']
    # columns.append('Delta_SoC')
    # return df[columns]

# def main():
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # input_path = os.path.join(script_dir, '..', 'output_data', INPUT_FILENAME)

    # if not os.path.exists(input_path):
        # print(f"File not found: {input_path}")
        # return

    # try:
        # print(f"Reading {input_path}...")
        # df = pd.read_csv(input_path)

        # print("Calculating Delta SoC...")
        # df = add_delta_soc_column(df)

        # df.to_csv(input_path, index=False)
        # print(f"Overwritten with Delta SoC added: {input_path}")

    # except Exception as e:
        # print(f"Error processing file: {e}")

# if __name__ == "__main__":
    # main()
import pandas as pd
import os
import numpy as np

# Constants
INPUT_FILENAME = 'FINAL_MASTER_DATASET_with_SOC.csv'
K = 0.005  # Voltage_Temp_Corrected gain
T_REF = 25  # Reference temperature in Celsius
ROLLING_WINDOW = 5
CAPACITY_WINDOW = 10

def add_features(df):
    # 1. Delta SoC
    if 'SoC_Percentage' in df.columns:
        if 'Source_File_Name' in df.columns:
            df['Delta_SoC'] = df.groupby('Source_File_Name')['SoC_Percentage'].diff().fillna(0)
        else:
            df['Delta_SoC'] = df['SoC_Percentage'].diff().fillna(0)

    # 2. Delta Voltage
    if 'Voltage_V' in df.columns:
        df['Delta_Voltage'] = df['Voltage_V'].diff().fillna(0)
        df['Rolling_Avg_Voltage_V'] = df['Voltage_V'].rolling(ROLLING_WINDOW, min_periods=1).mean()

    # 3. Delta Current
    if 'Current_A' in df.columns:
        df['Delta_Current'] = df['Current_A'].diff().fillna(0)
        df['Rolling_Avg_Current_A'] = df['Current_A'].rolling(ROLLING_WINDOW, min_periods=1).mean()

    # 4. Power
    if 'Voltage_V' in df.columns and 'Current_A' in df.columns:
        df['Power_W'] = df['Voltage_V'] * df['Current_A']

    # 5. Voltage_Temp_Corrected
    if 'Voltage_V' in df.columns and 'Cell_Temperature_C' in df.columns:
        df['Voltage_Temp_Corrected'] = df['Voltage_V'] + K * (T_REF - df['Cell_Temperature_C'])

    # 6. Delta Cell Temperature
    if 'Cell_Temperature_C' in df.columns:
        df['Delta_Cell_Temperature'] = df['Cell_Temperature_C'].diff().fillna(0)

    # 7. Cumulative Capacity (windowed integral approximation)
    if 'Current_A' in df.columns and 'Time_Seconds' in df.columns:
        df['Delta_t'] = df['Time_Seconds'].diff().fillna(0)
        df['Cumulative_Capacity_Window'] = (
            df['Current_A'].rolling(CAPACITY_WINDOW, min_periods=1).sum() * df['Delta_t'] / 3600
        )

    # 8. Interaction terms
    if 'Current_A' in df.columns and 'Cell_Temperature_C' in df.columns:
        df['Current_x_Cell_Temperature'] = df['Current_A'] * df['Cell_Temperature_C']

    if 'Voltage_V' in df.columns and 'Cell_Temperature_C' in df.columns:
        df['Voltage_x_Cell_Temperature'] = df['Voltage_V'] * df['Cell_Temperature_C']

    # 9. Last Current Sign Change
    if 'Current_A' in df.columns:
        sign = np.sign(df['Current_A'])
        prev_sign = sign.shift(1)
        df['Last_Current_Sign_Change'] = (sign > 0) != (prev_sign > 0)
        df['Last_Current_Sign_Change'] = df['Last_Current_Sign_Change'].astype(int).fillna(0)

    # Clean up
    if 'Delta_t' in df.columns and 'Cumulative_Capacity_Window' not in df.columns:
        df.drop(columns=['Delta_t'], inplace=True)

    return df

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, '..', 'output_data', INPUT_FILENAME)

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    try:
        print(f"Reading {input_path}...")
        df = pd.read_csv(input_path)

        print("Adding time-series derived features...")
        df = add_features(df)

        df.to_csv(input_path, index=False)
        print(f"Overwritten with features added: {input_path}")

    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    main()