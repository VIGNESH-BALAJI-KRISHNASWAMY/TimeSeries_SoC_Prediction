import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, confusion_matrix, ConfusionMatrixDisplay, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from PIL import Image
import os
import sys
import joblib
import datetime
import io
import shutil
import re
import xgboost as xgb
import tensorflow as tf
import random
import time

start_time_global = time.time()


# --- Tee class for simultaneous console and string buffer output ---
# This class redirects stdout to both the console and an in-memory buffer.
# This allows capturing the entire script's log output for inclusion in the PDF report.
class Tee:
    def __init__(self, primary_stream, secondary_stream):
        self.primary_stream = primary_stream
        self.secondary_stream = secondary_stream

    def write(self, data):
        self.primary_stream.write(data)
        self.secondary_stream.write(data)

    def flush(self):
        self.primary_stream.flush()
        self.secondary_stream.flush()

    def getvalue(self):
        return self.secondary_stream.getvalue()
# --- END NEW TEE CLASS ---


# --- Global Configuration Parameters ---
# This dictionary holds all configurable parameters for the entire pipeline.
# Modifying values here allows for systematic experimentation without changing code logic.
CONFIG = {
    'ENABLE_OVERSAMPLING': True,
    'MID_SOC_OVERSAMPLING_FACTOR': 3,
    'XGBOOST_MID_SOC_WEIGHT': 1.5,
    'ENABLE_REBALANCING_03_08': True, # Global rebalancing before splitting
    'USE_HUBER_LOSS': True, # Huber loss is safer than MSE (less sensitive to outliers).
    'ENABLE_RESIDUAL_XGBOOST': True, # Set to True to use XGBoost as a second-stage corrector

    # Dataset and Feature Configuration
    'DATASET_FILE': 'FINAL_MASTER_DATASET_With_SOC.csv', # Name of the input CSV file
    'COLUMN_VOLTAGE': 'Voltage_V',
    'COLUMN_CURRENT': 'Current_A',
    'COLUMN_CELL_TEMPERATURE': 'Cell_Temperature_C',
    'COLUMN_AMBIENT_TEMPERATURE': 'Ambient_Temperature_C',
    'COLUMN_CAPACITY_AH': 'Capacity_Ah',
    'COLUMN_ENERGY_WH': 'Energy_Wh',
    'COLUMN_TIME': 'Time_Seconds',
    'COLUMN_SOC': 'SoC_Percentage',
    'COLUMN_CYCLE_COUNT': 'Count_#',

    'SEQUENCE_LENGTH': 60, # Number of past time steps the LSTM will look at for each prediction (Increased for more context)

    # Advanced Feature Engineering Parameters
    # These parameters are used in creating new, more informative features.
    'TEMP_COMP_K': -0.005, # Temperature compensation factor for voltage (V/degC). Needs empirical tuning.
    'TEMP_COMP_REF_T': 25.0, # Reference temperature for voltage compensation (degC)
    'ROLLING_WINDOW_SIZE': 5, # Window size for rolling mean/std features (in timesteps)
    'CUMULATIVE_WINDOW_SIZE': 10, # Window size for cumulative capacity (Coulomb counting) (in timesteps)

    'INPUT_FEATURES_ORDER': [ # Keep all informative features
        'Current_A', 'Voltage_V', 'Cell_Temperature_C', 'Ambient_Temperature_C',
        'Time_Seconds', 'Capacity_Ah', 'Energy_Wh', 'Count_#',
        'Delta_Voltage', 'Delta_Current',
        'Power_W',
        'Voltage_Temp_Corrected',
        'Delta_Cell_Temperature',
        'Rolling_Avg_Voltage_V',
        'Rolling_Avg_Current_A',
        'Cumulative_Capacity_Window',
        'Current_x_Cell_Temperature', 'Voltage_x_Cell_Temperature', 'Last_Current_Sign_Change', #'Delta_SoC',
    ],

    # MODEL_HP defines max epochs for the final retraining (used if GridSearchCV is disabled)
    'MODEL_HP': {
        'LSTM_UNITS': 128,          # Increased capacity
        'DROPOUT_RATE': 0.2,        # Moderate dropout
        'EPOCHS': 200,              # Allow more epochs for convergence
        'BATCH_SIZE': 32,           # Smaller batch size can help generalization
        'EARLY_STOPPING_PATIENCE': 20, # Increased patience
        'L2_REG': 0.0001,           # Small L2 regularization
        'LEARNING_RATE': 0.0005,    # Stable learning rate
    },
    'ENABLE_GRID_SEARCH': True, # Set to False to skip GridSearchCV and use MODEL_HP directly

    # --- GridSearchCV Parameters (for hyperparameter search) ---
    'GRID_SEARCH_PARAMS': {
        'lstm_units': [64, 128],       # Reduced range for faster initial runs, expand on GPU
        'dropout_rate': [0.1, 0.2],    # Explore different dropout strengths
        'batch_size': [16, 32],        # Explore different batch sizes
        'epochs': [50, 100],           # Max epochs per CV fit (still limited for CPU)
        'l2_reg': [0.0001, 0.001],     # Explore different L2 regularization strengths
        'learning_rate': [0.0005, 0.001], # Explore a range of learning rates
    },
    'GRID_SEARCH_EPOCHS_PER_COMBINATION': 50, # Actual epochs per CV combination
    'GRID_SEARCH_CV_SPLITS': 3,               # Minimum splits for TimeSeriesSplit (5 is better if time permits)

    # XGBoost Corrector Configuration
    'XGBOOST_HP': { # Hyperparameters for the XGBoost corrector
        'n_estimators': 200,       # Number of boosting rounds (trees)
        'learning_rate': 0.01,      # Step size shrinkage
        'max_depth': 3,            # Maximum depth of a tree
        'subsample': 0.7,          # Subsample ratio of the training instance
        'colsample_bytree': 0.7,   # Subsample ratio of columns when constructing each tree
        'gamma': 0.1,
        'reg_lambda': 1.0,
        'reg_alpha': 0.1,
        'random_state': 42,
        'n_jobs': -1, # Use all available cores for XGBoost (can be set to 1 for debugging joblib errors)
        'early_stopping_rounds': 50 # Crucial for XGBoost to prevent overfitting
    },
    'XGBOOST_HIGH_ERROR_THRESHOLD': 0.0, # Increased threshold to filter larger errors for XGBoost (from 2.0)
    'XGBOOST_FEATURES': [ # Features for XGBoost. Must be in INPUT_FEATURES_ORDER.
        'Voltage_V', 'Current_A', 'Cell_Temperature_C', 'Ambient_Temperature_C',
        'Power_W', 'Voltage_Temp_Corrected', 'Delta_Cell_Temperature',
        'Rolling_Avg_Voltage_V', 'Rolling_Avg_Current_A', 'Cumulative_Capacity_Window',
        'Current_x_Cell_Temperature', 'Voltage_x_Cell_Temperature', 'Last_Current_Sign_Change', 'Time_Seconds',
    ],

    'ENABLE_OUTLIER_REMOVAL': True,
    'OUTLIER_COLUMNS_FOR_REMOVAL': [
        'Voltage_V', 'Current_A', 'Cell_Temperature_C', 'Ambient_Temperature_C',
    ],
    'OUTLIER_IQR_FACTOR': 2,
    'ENABLE_XGBOOST_CORRECTOR' : True,
    'ENABLE_F1_SCORE_CALCULATION': True,
    'F1_SCORE_SOC_BINS': {
        'bins': [0.0, 20.0, 80.0, 100.0 + 1e-9], # Bins for 0-100% SoC scale
        'labels': ['Low', 'Mid', 'High']
    },

    # Data Split Ratios Configuration
    'DATA_SPLIT_RATIOS': {
        'TRAIN_RATIO': 0.70, # % of data for training (from total sequences)
        'VAL_RATIO': 0.15,   # % of data for validation (from total sequences)
        'TEST_RATIO': 0.15,  # % of data for testing (from total sequences)
    },
    'TEST_SPLIT_STRATEGY': 'chronological', # Crucial for time series
    'STRATIFIED_TEST_SIZE_PERCENT': 10, # Only relevant if TEST_SPLIT_STRATEGY is 'stratified_random'
    'RANDOM_SEED_SPLIT': 42, # Fixed seed for reproducibility of splits and model initialization

    'MODEL_SAVE_PATH': 'soc_prediction_model.keras', # Renamed for clarity
    'SCALER_X_PATH': 'scaler_X.pkl',
    'SCALER_Y_PATH': 'scaler_Y.pkl',
    'XGBOOST_MODEL_SAVE_PATH': 'xgboost_corrector.json',

    'DEFAULT_REPORT_OUTPUT_DIR': 'soc_reports', # Renamed output directory
    'TEMP_PLOTS_SUBDIR_NAME': 'temp_plots',
    'PRINT_RAW_TABLE_OUTPUTS_TO_LOG': True,
    
    'NUM_ROLLING_PREDICTION_STEPS': 100,
}

# --- Global lists to store plot information for PDF generation ---
# Used to collect file paths and titles of all generated plots for the PDF report.
all_plots_info = []
toc_sections = [] # Stores information for the Table of Contents in the PDF

# --- ANSI escape code remover for clean log output in PDF ---
ANSI_ESCAPE_PATTERN = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
def remove_ansi_escape_codes(text):
    return ANSI_ESCAPE_PATTERN.sub('', text)

# --- Helper Functions ---

def create_lstm_sequences(features_scaled, target_scaled, sequence_length):
    """
    Transforms 2D scaled data into 3D sequences suitable for LSTM.
    Each sequence consists of `sequence_length` past feature data points,
    and the corresponding target is the SoC value immediately following that sequence.
    """
    X, y = [], []
    for i in range(len(features_scaled) - sequence_length):
        X.append(features_scaled[i:(i + sequence_length), :])
        y.append(target_scaled[i + sequence_length, 0])

    return np.array(X), np.array(y)


def remove_outliers_iqr(df, column_names, iqr_factor):
    """
    Removes outliers from specified columns of a DataFrame using the IQR method.
    Outliers are values below Q1 - iqr_factor*IQR or above Q3 + iqr_factor*IQR.
    Returns the DataFrame with outlier rows removed.
    """
    initial_rows = len(df)
    for col in column_names:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_factor * IQR
            upper_bound = Q3 + iqr_factor * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        else:
            print(f"Warning: Outlier removal skipped for '{col}' as it's not in the DataFrame.")
    
    df = df.reset_index(drop=True)
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        print(f"Removed {rows_removed} rows (outliers) from the dataset.")
    else:
        print("No outliers removed based on specified columns and IQR factor.")
    return df


def preprocess_new_raw_data_for_prediction(new_raw_df_segment, scaler_x_fitted, config):
    """
    Takes a segment of new raw data (DataFrame) and preprocesses it
    to create a single LSTM input sequence for prediction.
    This function must mirror the feature engineering and scaling steps
    done during training. It's crucial for consistent deployment.
    """
    try:
        df_processed_segment = new_raw_df_segment.copy()

        # Feature Engineering: These must perfectly match load_and_preprocess_data
        df_processed_segment['Delta_Voltage'] = df_processed_segment[config['COLUMN_VOLTAGE']].diff().fillna(0)
        df_processed_segment['Delta_Current'] = df_processed_segment[config['COLUMN_CURRENT']].diff().fillna(0)
        
        df_processed_segment['Rolling_Avg_Voltage_V'] = df_processed_segment[config['COLUMN_VOLTAGE']].rolling(window=config['ROLLING_WINDOW_SIZE'], min_periods=1).mean()
        df_processed_segment['Rolling_Avg_Current_A'] = df_processed_segment[config['COLUMN_CURRENT']].rolling(window=config['ROLLING_WINDOW_SIZE'], min_periods=1).mean()

        df_processed_segment['Power_W'] = df_processed_segment[config['COLUMN_VOLTAGE']] * df_processed_segment[config['COLUMN_CURRENT']]
        df_processed_segment['Voltage_Temp_Corrected'] = df_processed_segment[config['COLUMN_VOLTAGE']] + config['TEMP_COMP_K'] * (
            config['TEMP_COMP_REF_T'] - df_processed_segment[config['COLUMN_CELL_TEMPERATURE']])
        df_processed_segment['Delta_Cell_Temperature'] = df_processed_segment[config['COLUMN_CELL_TEMPERATURE']].diff().fillna(0)

        # Estimate time_step_interval from the segment itself
        time_step_interval_actual_sim = df_processed_segment[config['COLUMN_TIME']].diff().mean()
        if pd.isna(time_step_interval_actual_sim) or time_step_interval_actual_sim == 0:
            time_step_interval_actual_sim = 1.0 # Default if not enough data or interval is zero
        df_processed_segment['Cumulative_Capacity_Window'] = df_processed_segment[config['COLUMN_CURRENT']].rolling(window=config['CUMULATIVE_WINDOW_SIZE'], min_periods=1).sum() * (time_step_interval_actual_sim / 3600.0)

        df_processed_segment['Current_x_Cell_Temperature'] = df_processed_segment[config['COLUMN_CURRENT']] * df_processed_segment[config['COLUMN_CELL_TEMPERATURE']]
        df_processed_segment['Voltage_x_Cell_Temperature'] = df_processed_segment[config['COLUMN_VOLTAGE']] * df_processed_segment[config['COLUMN_CELL_TEMPERATURE']]
        df_processed_segment['Last_Current_Sign_Change'] = ((np.sign(df_processed_segment[config['COLUMN_CURRENT']]) > 0) !=(np.sign(df_processed_segment[config['COLUMN_CURRENT']].shift(1)) > 0)).astype(int)

        # Drop any NaNs from the beginning that result from feature engineering
        df_processed_segment = df_processed_segment.dropna().reset_index(drop=True)

        # Ensure the segment has at least SEQUENCE_LENGTH rows for the LSTM input
        if len(df_processed_segment) < config['SEQUENCE_LENGTH']:
            raise ValueError(f"Not enough processed data to form a sequence of length {config['SEQUENCE_LENGTH']}. "
                             f"Provided: {len(df_processed_segment)} rows after processing. Need at least {config['SEQUENCE_LENGTH']}.")
        
        # Select the last SEQUENCE_LENGTH rows for the LSTM input
        df_lstm_input_segment = df_processed_segment.tail(config['SEQUENCE_LENGTH'])

        # Check for missing required columns after feature engineering
        missing_columns = [col for col in config['INPUT_FEATURES_ORDER'] if col not in df_lstm_input_segment.columns]
        if missing_columns:
            raise ValueError(f"The following required features are missing after preprocessing: {missing_columns}")

        # Select features in the same order as trained (this order is critical!)
        features_for_scaling = df_lstm_input_segment[config['INPUT_FEATURES_ORDER']].values

        # Scale the features using the *fitted* scaler from training
        scaled_features = scaler_x_fitted.transform(features_for_scaling)

        # Reshape to (1, SEQUENCE_LENGTH, num_features) for the LSTM model
        lstm_input = scaled_features.reshape(1, config['SEQUENCE_LENGTH'], len(config['INPUT_FEATURES_ORDER']))

        if np.any(np.isnan(scaled_features)) or np.any(np.isinf(scaled_features)):
            raise ValueError("Preprocessed sequence contains NaN or Inf. Check raw input.")
        print("[DEBUG] Feature min range:\n", df_lstm_input_segment[config['INPUT_FEATURES_ORDER']].min())            
        print("[DEBUG] Feature max range:\n", df_lstm_input_segment[config['INPUT_FEATURES_ORDER']].max())       

        return lstm_input, df_lstm_input_segment.copy() # Return the processed segment used for LSTM

    except Exception as e:
        print(f"Preprocessing failed during prediction demonstration: {e}")
        raise

# --- PDF Report Generation Class ---
class PDFReport(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.page_body_start_y = 0
        self.set_left_margin(20)
        self.set_right_margin(20)
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        if self.page_no() > 1:
            self.set_y(10)
            self.set_font('Inter', 'B', 10)
            self.cell(0, 5, 'Battery SoC Estimation Project Report', border=0, align='R', new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def footer(self):
        self.set_y(-15)
        self.set_font('Inter', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C', new_x=XPos.RIGHT, new_y=YPos.TOP)

    def chapter_title(self, title, level=1):
        if level == 1:
            self.add_page()
            toc_sections.append({'title': title, 'page_number': self.page_no()})
            self.set_font('Inter', 'B', 16)
            self.ln(10)
            self.cell(0, 10, title, border=0, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_line_width(0.5)
            if self.get_y() < self.h - self.b_margin - 5:
                self.line(self.get_x(), self.get_y(), self.get_x() + self.w - 2*self.l_margin, self.get_y())
            self.ln(5)
            self.page_body_start_y = self.get_y()
        elif level == 2:
            self.set_font('Inter', 'B', 14)
            self.ln(5)
            self.cell(0, 8, title, border=0, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(2)
        elif level == 3:
            self.set_font('Inter', 'B', 12)
            self.ln(2)
            self.cell(0, 6, title, border=0, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.ln(1)

    def chapter_body(self, body):
        self.set_font('Inter', '', 10)
        parts = body.split('**')
        for i, part in enumerate(parts):
            if i % 2 == 1:
                self.set_font('Inter', 'B', 10)
                self.write(5, part)
            else:
                self.set_font('Inter', '', 10)
                self.write(5, part)
        self.ln(7)

# --- PDF Generation Function ---
def generate_pdf_report(log_content, plots_info, report_meta, output_pdf_path):
    pdf = PDFReport('P', 'mm', 'A4')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    font_regular = os.path.join(script_dir, 'Inter-Regular.ttf')
    font_bold = os.path.join(script_dir, 'Inter-Bold.ttf')
    font_italic = os.path.join(script_dir, 'Inter-Italic.ttf')
    font_bold_italic = os.path.join(script_dir, 'Inter-BoldItalic.ttf')

    default_font_set = False
    try:
        pdf.add_font('Inter', '', font_regular)
        pdf.add_font('Inter', 'B', font_bold)
        pdf.add_font('Inter', 'I', font_italic)
        pdf.add_font('Inter', 'BI', font_bold_italic)
        pdf.set_font('Inter', '', 12)
        default_font_set = True
    except RuntimeError:
        pdf.set_font('helvetica', '', 12)
        print("Warning: Inter font files not found at absolute paths. Using default Helvetica font for report.")
    
    current_font_family = 'Inter' if default_font_set else 'helvetica'


    # --- Title Page ---
    pdf.add_page()
    pdf.set_font(current_font_family, 'B', 24)
    pdf.cell(0, 10, 'Battery SoC Estimation Report', border=0, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)
    pdf.set_font(current_font_family, '', 12)
    for key, value in report_meta.items():
        if key == 'Report File':
            pdf.cell(0, 8, f'Report File: {os.path.basename(value)}', border=0, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        elif key not in ['Overall_MAE', 'Overall_RMSE', 'Overall_R2', 'Overall_F1',
                         'Single_Pred_Abs_Error', 'Single_Pred_Actual_SoC', 'Single_Pred_Predicted_SoC', 'Best_CV_MAE']:
            pdf.cell(0, 8, f'{key}: {value}', border=0, align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(20)
    pdf.set_font(current_font_family, 'I', 10)
    pdf.multi_cell(0, 5, 'Generated Automatically by SoC Prediction Pipeline Script. \n\nNote: Plots in this PDF are static. For interactive analysis, run the script in environments like Google Colab or Jupyter Notebook.', 0, 'C')


    # --- Table of Contents ---
    pdf.add_page()
    toc_sections.append({'title': 'Table of Contents', 'page_number': pdf.page_no()})
    pdf.set_font(current_font_family, 'B', 16)
    pdf.cell(0, 10, 'Table of Contents', border=0, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)
    
    # Manual TOC generation (fixed to account for dynamic content and new structure)
    pdf.set_font(current_font_family, '', 10)
    for section_info in toc_sections:
        title = section_info['title']
        page_number = section_info['page_number']
        
        line_height = 7
        max_line_width = pdf.w - pdf.l_margin - pdf.r_margin
        page_num_width = pdf.get_string_width(str(page_number))
        title_width = pdf.get_string_width(title)
        
        target_content_width = max_line_width - 5
        
        if (title_width + page_num_width + 5) < target_content_width:
            dots_count = int((target_content_width - title_width - page_num_width) / pdf.get_string_width('.')) if pdf.get_string_width('.') > 0 else 0
            dots = '.' * max(1, dots_count)
        else:
            dots = ''

        pdf.cell(0, line_height, f'{title} {dots} {page_number}', border=0, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)


    # --- Run Log Section ---
    pdf.start_section('Run Log', level=1)
    pdf.set_font(current_font_family, 'B', 16)
    pdf.cell(0, 10, 'Run Log', border=0, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)
    pdf.set_font(current_font_family, 'BI', 10)
    pdf.multi_cell(0, 4, "Note: This log contains the real-time console output during the script execution, including debug information and progress updates.")
    pdf.ln(2)
    pdf.set_font(current_font_family, '', 8)
    
    # Remove ANSI escape codes for clean PDF output
    clean_log_content = remove_ansi_escape_codes(log_content)
    pdf.multi_cell(0, 4, clean_log_content)


    # --- Plots Section ---
    plots_by_section_grouped = {}
    for plot_info in plots_info:
        section = plot_info.get('section', 'General Plots')
        if section not in plots_by_section_grouped:
            plots_by_section_grouped[section] = []
        plots_by_section_grouped[section].append(plot_info)

    section_order = [
        'Phase 1: EDA - Data Distribution & Outliers',
        'Phase 1: EDA - Feature Relationships',
        'Phase 1: Data Distribution - Train Set',
        'Phase 2: Training Diagnostics',
        'Phase 3: Model Evaluation - Core Performance',
        'Phase 3: Model Evaluation - Learned Relationships',
        'Phase 3: Error Analysis - Residuals',
        'Phase 3: Error Analysis - Classification Performance',
        'General Plots'
    ]

    for section_title in section_order:
        if section_title in plots_by_section_grouped:
            pdf.start_section(section_title, level=1)
            pdf.set_font(current_font_family, 'B', 16)
            pdf.cell(0, 10, f'{section_title}', border=0, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(5)

            for plot_info in plots_by_section_grouped[section_title]:
                required_height_for_plot = 100
                if pdf.get_y() + required_height_for_plot > (pdf.h - pdf.b_margin):
                    pdf.add_page()
                    pdf.set_font(current_font_family, 'B', 16)
                    pdf.cell(0, 10, f'{section_title} (Cont.)', border=0, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    pdf.ln(5)
                
                pdf.set_font(current_font_family, 'B', 12)
                pdf.multi_cell(0, 8, f'Plot: {plot_info["title"]}', 0, 'L')
                pdf.ln(2)

                image_path = plot_info['filepath']
                if os.path.exists(image_path):
                    try:
                        page_width = pdf.w - 2 * pdf.l_margin
                        
                        with Image.open(image_path) as img:
                            img_width_px, img_height_px = img.size

                        dpi = 300
                        img_width_mm = img_width_px * 25.4 / dpi
                        img_height_mm = img_height_px * 25.4 / dpi

                        scale_factor_width = page_width / img_width_mm
                        width_mm = page_width
                        height_mm = img_height_mm * scale_factor_width

                        if height_mm > (pdf.h - pdf.get_y() - pdf.b_margin - 5):
                            height_mm = (pdf.h - pdf.get_y() - pdf.b_margin - 5)
                            width_mm = img_width_mm * (height_mm / img_height_mm)

                        x_pos_img = (pdf.w - width_mm) / 2

                        pdf.image(image_path, x=x_pos_img, y=pdf.get_y(), w=width_mm, h=height_mm)
                        pdf.ln(height_mm + 5)
                    except Exception as e:
                        pdf.set_font(current_font_family, '', 10)
                        pdf.multi_cell(0, 10, f'Error loading or processing image {os.path.basename(image_path)}: {e}', 0, 'L')
                        pdf.ln(5)

                else:
                    pdf.set_font(current_font_family, '', 10)
                    pdf.cell(0, 10, f'Image file not found: {os.path.basename(image_path)}', border=0, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            del plots_by_section_grouped[section_title]

    for section_title in sorted(plots_by_section_grouped.keys()):
        if plots_by_section_grouped[section_title]:
            pdf.start_section(section_title, level=1)
            pdf.set_font(current_font_family, 'B', 16)
            pdf.cell(0, 10, f'{section_title}', border=0, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(5)
            for plot_info in plots_by_section_grouped[section_title]:
                required_height_for_plot = 100
                if pdf.get_y() + required_height_for_plot > (pdf.h - pdf.b_margin):
                    pdf.add_page()
                    pdf.set_font(current_font_family, 'B', 16)
                    pdf.cell(0, 10, f'{section_title} (Cont.)', border=0, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    pdf.ln(5)

                pdf.set_font(current_font_family, 'B', 12)
                pdf.multi_cell(0, 8, f'Plot: {plot_info["title"]}', 0, 'L')
                pdf.ln(2)
                image_path = plot_info['filepath']
                if os.path.exists(image_path):
                    try:
                        page_width = pdf.w - 2 * pdf.l_margin
                        with Image.open(image_path) as img:
                            img_width_px, img_height_px = img.size
                        dpi = 300
                        img_width_mm = img_width_px * 25.4 / dpi
                        img_height_mm = img_height_px * 25.4 / dpi
                        scale_factor_width = page_width / img_width_mm
                        width_mm = page_width
                        height_mm = img_height_mm * scale_factor_width
                        if height_mm > (pdf.h - pdf.get_y() - pdf.b_margin - 5):
                            height_mm = (pdf.h - pdf.get_y() - pdf.b_margin - 5)
                            width_mm = img_width_mm * (height_mm / img_height_mm)
                        x_pos_img = (pdf.w - width_mm) / 2
                        pdf.image(image_path, x=x_pos_img, y=pdf.get_y(), w=width_mm, h=height_mm)
                        pdf.ln(height_mm + 5)
                    except Exception as e:
                        pdf.set_font(current_font_family, '', 10)
                        pdf.multi_cell(0, 10, f'Error loading or processing image {os.path.basename(image_path)}: {e}', 0, 'L')
                        pdf.ln(5)
                else:
                    pdf.set_font(current_font_family, '', 10)
                    pdf.cell(0, 10, f'Image file not found: {os.path.basename(image_path)}', border=0, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    # --- Conclusion and Discussion Section ---
    pdf.start_section('Conclusion and Discussion', level=1)
    pdf.set_font(current_font_family, 'B', 16)
    pdf.cell(0, 10, 'Conclusion and Discussion', border=0, align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    # Retrieve metrics from report_meta
    overall_mae = report_meta.get('Overall_MAE', 'N/A')
    overall_rmse = report_meta.get('Overall_RMSE', 'N/A')
    overall_r2 = report_meta.get('Overall_R2', 'N/A')
    overall_f1 = report_meta.get('Overall_F1', 'N/A')
    single_abs_error = report_meta.get('Single_Pred_Abs_Error', 'N/A')
    single_actual_soc = report_meta.get('Single_Pred_Actual_SoC', 'N/A')
    single_predicted_soc = report_meta.get('Single_Pred_Predicted_SoC', 'N/A')
    best_cv_mae = report_meta.get('Best_CV_MAE', 'N/A')

    # Overall Performance Proof - FIX: Use f-strings before passing to chapter_body
    overall_performance_text = (
        "The model's overall performance was rigorously evaluated on an unseen test set, "
        "providing the **primary statistical proof** of its accuracy and generalization capabilities. "
        f"Key metrics obtained were: **Mean Absolute Error (MAE) of {overall_mae:.6f} (0-1 Scale)**, "
        f"**Root Mean Squared Error (RMSE) of {overall_rmse:.6f} (0-1 Scale)**, and an **R-squared ($R^2$) of {overall_r2:.4f}**. "
        f"The classification F1-score (weighted average for binned SoC) was **{overall_f1:.4f}**. "
        "These values indicate that the model reliably predicts the State of Charge with low average deviations, "
        "and effectively captures the variance in the true SoC data, while also performing well "
        "in categorizing SoC levels."
    )
    pdf.chapter_body(overall_performance_text)
    pdf.ln(2)

    # Operational Demonstration Correlation - FIX: Use f-strings before passing to chapter_body
    if isinstance(single_actual_soc, (int, float)):
        
        # Convert overall_mae to 0-1 scale for comparison 
        # (due to evaluate_model_and_generate_plots updates)
        # need to display the 0-100% equivalent for readability in the text
        overall_mae_100_percent_display = overall_mae * 100 if isinstance(overall_mae, (int,float)) else 'N/A'
        single_actual_soc_100_percent_display = single_actual_soc * 100 if isinstance(single_actual_soc, (int,float)) else 'N/A'
        single_predicted_soc_100_percent_display = single_predicted_soc * 100 if isinstance(single_predicted_soc, (int,float)) else 'N/A'
        single_abs_error_100_percent_display = single_abs_error * 100 if isinstance(single_abs_error, (int,float)) else 'N/A'
        
        operational_demo_text = (
            "Following this comprehensive evaluation, the single-point prediction demonstration "
            "served as a **critical operational validation** of the entire inference pipeline. "
            f"For the simulated data instance, with an **Actual SoC of {single_actual_soc_100_percent_display:.2f}% ({single_actual_soc:.6f} 0-1 Scale)**, "
            f"the model predicted **{single_predicted_soc_100_percent_display:.2f}% ({single_predicted_soc:.6f} 0-1 Scale)**, resulting in an "
            f"**Absolute Error of {single_abs_error_100_percent_display:.2f}% ({single_abs_error:.6f} 0-1 Scale)**. This error is "
            f"{'highly consistent with' if (isinstance(overall_mae, (int,float)) and abs(single_abs_error - overall_mae) < 0.005) else ('generally consistent with' if (isinstance(overall_mae, (int,float)) and abs(single_abs_error - overall_mae) < 0.02) else 'notably different from')} "
            f"the overall MAE of {overall_mae_100_percent_display:.2f}%SoC ({overall_mae:.6f} 0-1 Scale), demonstrating that the deployed model can successfully process new raw inputs "
            "and produce a plausible SoC estimate in a 'live' scenario."
            "\n\n**Note**: A significant deviation in the single-point demonstration (if observed) compared to the overall average "
            "metrics may indicate the simulated data point represents a specific edge case, or a region "
            "of the SoC curve where the model's performance might be more variable. "
            "Such cases warrant further investigation into data representativeness."
        )
        pdf.chapter_body(operational_demo_text)
    else:
        pdf.chapter_body(
            "The single-point prediction demonstration was performed to validate the operational pipeline. "
            "However, specific single-point error metrics could not be retrieved for inclusion in this summary. "
            "Please refer to the 'Run Log' section for the detailed output of the demonstration."
        )
    pdf.ln(5)

    # Hyperparameter Optimization Impact - FIX:  f-strings before passing to chapter_body
    if isinstance(best_cv_mae, (int, float)):
        # Convert best_cv_mae to 0-100% for display in text
        best_cv_mae_100_percent_display = best_cv_mae * 100 if isinstance(best_cv_mae, (int,float)) else 'N/A'
        hp_optimization_text = (
            "The **GridSearchCV** process was instrumental in identifying the optimal hyperparameters "
            "for the LSTM model. The **best Mean Absolute Error (MAE) achieved during cross-validation "
            f"was {best_cv_mae_100_percent_display:.2f}%SoC ({best_cv_mae:.6f} 0-1 Scale)**. This systematic search ensures that the model is configured "
            "to maximize accuracy, demonstrating that selecting proper hyperparameters "
            "is crucial for enhancing predictive performance. The final model was retrained with these "
            "best-found parameters, as detailed in the 'Run Log'."
        )
        pdf.chapter_body(hp_optimization_text)
    else:
        pdf.chapter_body(
            "Hyperparameter optimization using GridSearchCV was performed to identify the best LSTM configuration. "
            "Please refer to the 'Run Log' section for detailed cross-validation results and best parameters."
        )
    pdf.ln(5)

    # Repeating Trials Advice - No f-strings here, so direct pass is fine.
    pdf.chapter_body(
        "To further assess the model's robustness and the variability of its performance, "
        "it is recommended to **repeat the entire pipeline execution multiple times**. "
        "This can be achieved by varying the `RANDOM_SEED_SPLIT` in the configuration for each run. "
        "Collecting the MAE, RMSE, and R-squared values from each trial allows for calculation of "
        "mean and standard deviation of these metrics, providing a more statistically robust "
        "understanding of the model's expected performance range under different random initializations or data splits (if stratified)."
    )
    pdf.ln(5)

    # FIX: Removed 'F' from output parameter
    pdf.output(output_pdf_path)
    print(f"\nPDF report generated successfully at: {output_pdf_path}")


# --- Core Pipeline Functions (Moved to be defined before __main__) ---

def setup_environment_and_args(config):
    """
    Handles command-line arguments, sets up report paths, and initializes logging redirection (Tee class).
    Returns a dictionary of paths and run metadata.
    """
    time_step_interval = 1.0

    original_stdout_stream = sys.stdout
    sys.stdout = Tee(original_stdout_stream, io.StringIO())

    if len(sys.argv) < 2:
        print("Usage: python soc_prediction_pipeline.py <path_to_FINAL_MASTER_DATASET_With_SOC.csv> [output_directory_for_pdf]")
        print("Example: python soc_prediction_pipeline.py data/FINAL_MASTER_DATASET_With_SOC.csv")
        print("Example: python soc_prediction_pipeline.py data/FINAL_MASTER_DATASET_With_SOC.csv ./my_reports_folder")
        sys.stdout = original_stdout_stream
        sys.exit(1)

    dataset_file_path_arg = sys.argv[1]
    output_pdf_base_dir = sys.argv[2] if len(sys.argv) > 2 else config['DEFAULT_REPORT_OUTPUT_DIR']

    os.makedirs(output_pdf_base_dir, exist_ok=True)
    report_filename = f"SoC_Prediction_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    report_pdf_path_final = os.path.join(output_pdf_base_dir, report_filename)

    temp_plots_dir_actual = os.path.join(os.getcwd(), config['TEMP_PLOTS_SUBDIR_NAME'])
    os.makedirs(temp_plots_dir_actual, exist_ok=True)

    script_filename = os.path.basename(sys.argv[0])

    run_timestamp = datetime.datetime.now()
    report_meta = {
        'Report Title': 'Battery SoC Estimation Pipeline Run Report',
        'Generated By': script_filename,
        'Generation Time': run_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'Python Version': sys.version.split(' ')[0],
        'TensorFlow Version': 'Not available (check env)',
        'Sequence Length': config['SEQUENCE_LENGTH'],
        'Epochs': config['MODEL_HP']['EPOCHS'],
        'Batch Size': config['MODEL_HP']['BATCH_SIZE'],
        'Dataset File': config['DATASET_FILE'],
        'Report File': report_pdf_path_final
    }
    try:
        import tensorflow as tf
        report_meta['TensorFlow Version'] = tf.__version__
    except ImportError:
        report_meta['TensorFlow Version'] = 'TensorFlow not imported/found'

    return {
        'dataset_file_path': dataset_file_path_arg,
        'report_pdf_path_final': report_pdf_path_final,
        'temp_plots_dir_actual': temp_plots_dir_actual,
        'report_meta': report_meta,
        'original_stdout_stream': original_stdout_stream
    }


def load_and_preprocess_data(file_path, config, env_paths):
    """
    Phase 1: Loads data, performs initial checks, sorting, EDA plots, and feature engineering.
    Returns the processed DataFrame.
    """
    print("--- Phase 1: Data Preparation and Feature Engineering ---")
    print(f"Attempting to load dataset from: {file_path}")

    if not os.path.exists(file_path):
        print(f"Error: Dataset file '{file_path}' not found.")
        sys.stdout = env_paths['original_stdout_stream']
        sys.exit(1)

    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        
        # Convert SoC_Percentage from 0-1 range to 0-100 range if it's not already
        # Assume if max is <= 1.01 (e.g., 1.0 or 1.0134), it's a 0-1 scale that needs conversion.
        # Otherwise, assume it's already 0-100 (or invalid).
        if df[config['COLUMN_SOC']].max() <= 1.015 and df[config['COLUMN_SOC']].min() >= 0.0:
            df[config['COLUMN_SOC']] = df[config['COLUMN_SOC']] * 100
            df[config['COLUMN_SOC']] = df[config['COLUMN_SOC']].clip(lower=0.0, upper=100.0)
            print(f"INFO: Converted '{config['COLUMN_SOC']}' from 0-1 scale to 0-100 scale.")
        else:
            print(f"INFO: '{config['COLUMN_SOC']}' appears to be already on a 0-100 scale (max: {df[config['COLUMN_SOC']].max():.2f}), no conversion applied.")

        print(f"\n--- DEBUG: Original SoC_Percentage Column Range (after potential conversion) ---")
        print(f"Min SoC in raw data: {df[config['COLUMN_SOC']].min():.2f}")
        print(f"Max SoC in raw data: {df[config['COLUMN_SOC']].max():.2f}")
        print("-------------------------------------------------")
        print("Dataset loaded successfully.")
        print(f"Initial dataset shape: {df.shape}")
        print(f"Columns available: {df.columns.tolist()}")
        if config['PRINT_RAW_TABLE_OUTPUTS_TO_LOG']:
            print(f"\n--- DEBUG: Actual DataFrame Columns Loaded ---")
            print(df.columns.tolist())
            print("------------------------------------------------")
        print(f"\n--- DEBUG: Original SoC_Percentage Column Range ---")
        print(f"Min SoC in raw data: {df[config['COLUMN_SOC']].min()}")
        print(f"Max SoC in raw data: {df[config['COLUMN_SOC']].max()}")
        print("-------------------------------------------------")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.stdout = env_paths['original_stdout_stream']
        sys.exit(1)

    non_derived_input_cols = [col for col in config['INPUT_FEATURES_ORDER'] if col not in ['Delta_Voltage', 'Delta_Current', 'Power_W', 'Voltage_Temp_Corrected', 'Delta_Cell_Temperature', 'Rolling_Avg_Voltage_V', 'Rolling_Avg_Current_A', 'Cumulative_Capacity_Window', 'Current_x_Cell_Temperature', 'Voltage_x_Cell_Temperature', 'Last_Current_Sign_Change']]#, 'Delta_SoC']]
    required_cols_for_load = non_derived_input_cols + [config['COLUMN_SOC']]

    missing_columns = [col for col in required_cols_for_load if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns in the dataset: {missing_columns}")
        print("Please ensure your CSV file has columns named EXACTLY as specified in the script configuration.")
        sys.stdout = env_paths['original_stdout_stream']
        sys.exit(1)

    df = df.sort_values(by=config['COLUMN_TIME']).reset_index(drop=True)
    print(f"Data sorted by '{config['COLUMN_TIME']}' column.")

    if config['PRINT_RAW_TABLE_OUTPUTS_TO_LOG']:
        print("\nFirst 5 rows of the dataset before feature engineering:")
        print(df.head())
    print("\nMissing values before feature engineering (should ideally be 0 for these columns):")
    print(df[required_cols_for_load].isnull().sum())

    if config['ENABLE_OUTLIER_REMOVAL']:
        print(f"\n--- Applying Outlier Removal (IQR method) to: {config['OUTLIER_COLUMNS_FOR_REMOVAL']} ---")
        df = remove_outliers_iqr(df, config['OUTLIER_COLUMNS_FOR_REMOVAL'], config['OUTLIER_IQR_FACTOR'])
    
    if CONFIG.get('ENABLE_REBALANCING_03_08', False):
        print("\n--- Applying Global Rebalancing for Mid-SoC Region (0.3-0.8) ---")
        # Note: The 0.3-0.8 here should now implicitly refer to the 0-100 scale values if converted above.
        df_mid = df[(df[config['COLUMN_SOC']] >= 30.0) & (df[config['COLUMN_SOC']] <= 80.0)] # Adjusted for 0-100 scale
        current_mid_count = len(df_mid)
        target_mid_count = 2000

        if current_mid_count == 0:
            print("[WARNING] No mid-SoC samples found in dataset. Rebalancing skipped.")
        if current_mid_count < target_mid_count:
            num_to_replicate = target_mid_count - current_mid_count
            replicated_mid_samples = df_mid.sample(n=num_to_replicate, replace=True, random_state=config['RANDOM_SEED_SPLIT'])
            df = pd.concat([df, replicated_mid_samples]).sample(frac=1.0, random_state=config['RANDOM_SEED_SPLIT']).reset_index(drop=True)
            print(f"Replicated {num_to_replicate} mid-SoC samples. New total samples: {len(df)}")
        else:
            print(f"Mid-SoC region already has {current_mid_count} samples, no replication needed.")

    print("Outlier removal step complete.")
    
    print("\n--- Visualizing Data Distributions and Outliers with Boxplots (Raw Data) ---")
    columns_to_boxplot = [
        config['COLUMN_VOLTAGE'], config['COLUMN_CURRENT'], config['COLUMN_CELL_TEMPERATURE'],
        config['COLUMN_AMBIENT_TEMPERATURE'], config['COLUMN_CAPACITY_AH'], config['COLUMN_ENERGY_WH'],
        config['COLUMN_TIME'], config['COLUMN_SOC'], config['COLUMN_CYCLE_COUNT']
    ]

    num_cols_for_plot = 3
    for i, col in enumerate(columns_to_boxplot):
        if col in df.columns:
            plt.figure(figsize=(num_cols_for_plot * 5, 4))
            sns.boxplot(y=df[col], color='skyblue')
            plt.title(f'Phase 1: EDA - Boxplot of {col}')
            plt.ylabel(col)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plot_filename = os.path.join(env_paths['temp_plots_dir_actual'], f'boxplot_{col}.png')
            plt.savefig(plot_filename, dpi=300)
            plt.close()
            all_plots_info.append({'filepath': plot_filename, 'title': f'Boxplot of {col}', 'section': 'Phase 1: EDA - Data Distribution & Outliers'})
        else:
            print(f"Warning: Column '{col}' not found in DataFrame for boxplot visualization. Skipping.")
    print("\n--- Outlier analysis complete. Plots saved to temporary directory. ---")

    print("\n--- Visualizing Histograms of Raw Features (Raw Data) ---")
    columns_to_hist = [
        config['COLUMN_VOLTAGE'], config['COLUMN_CURRENT'], config['COLUMN_CELL_TEMPERATURE'],
        config['COLUMN_AMBIENT_TEMPERATURE'], config['COLUMN_CAPACITY_AH'], config['COLUMN_ENERGY_WH'],
        config['COLUMN_TIME'], config['COLUMN_SOC'], config['COLUMN_CYCLE_COUNT']
    ]
    
    num_cols_for_plot = 3
    for i, col in enumerate(columns_to_hist):
        if col in df.columns:
            plt.figure(figsize=(num_cols_for_plot * 5, 4))
            sns.histplot(df[col], kde=True, bins=50, color='lightcoral')
            plt.title(f'Phase 1: EDA - Histogram of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.tight_layout()
            plot_filename = os.path.join(env_paths['temp_plots_dir_actual'], f'histogram_{col}.png')
            plt.savefig(plot_filename, dpi=300)
            plt.close()
            all_plots_info.append({'filepath': plot_filename, 'title': f'Histogram of {col}', 'section': 'Phase 1: EDA - Data Distribution & Outliers'})
        else:
            print(f"Warning: Column '{col}' not found in DataFrame for histogram visualization. Skipping.")
    print("\n--- Histogram analysis complete. Plots saved to temporary directory. ---")


    print("\n--- Visualizing Correlation Heatmap ---")
    df_for_corr = df.copy()
    
    if 'Delta_Voltage' not in df_for_corr.columns:
        df_for_corr['Delta_Voltage'] = df_for_corr[config['COLUMN_VOLTAGE']].diff().fillna(0)
    if 'Delta_Current' not in df_for_corr.columns:
        df_for_corr['Delta_Current'] = df_for_corr[config['COLUMN_CURRENT']].diff().fillna(0)
    #if 'Delta_SoC' not in df_for_corr.columns:
     #   df_for_corr['Delta_SoC'] = df_for_corr[config['COLUMN_SOC']].diff().fillna(0)
    if 'Power_W' not in df_for_corr.columns:
        df_for_corr['Power_W'] = df_for_corr[config['COLUMN_VOLTAGE']] * df_for_corr[config['COLUMN_CURRENT']]
    if 'Voltage_Temp_Corrected' not in df_for_corr.columns:
        df_for_corr['Voltage_Temp_Corrected'] = df_for_corr[config['COLUMN_VOLTAGE']] + config['TEMP_COMP_K'] * (config['TEMP_COMP_REF_T'] - df_for_corr[config['COLUMN_CELL_TEMPERATURE']])
    if 'Delta_Cell_Temperature' not in df_for_corr.columns:
        df_for_corr['Delta_Cell_Temperature'] = df_for_corr[config['COLUMN_CELL_TEMPERATURE']].diff().fillna(0)
    if 'Rolling_Avg_Voltage_V' not in df_for_corr.columns:
        df_for_corr['Rolling_Avg_Voltage_V'] = df_for_corr[config['COLUMN_VOLTAGE']].rolling(window=config['ROLLING_WINDOW_SIZE'], min_periods=1).mean()
    if 'Rolling_Avg_Current_A' not in df_for_corr.columns:
        df_for_corr['Rolling_Avg_Current_A'] = df_for_corr[config['COLUMN_CURRENT']].rolling(window=config['ROLLING_WINDOW_SIZE'], min_periods=1).mean()
    
    time_step_interval_actual_corr = df_for_corr[config['COLUMN_TIME']].diff().mean()
    if pd.isna(time_step_interval_actual_corr) or time_step_interval_actual_corr == 0:
        time_step_interval_actual_corr = 1.0
    if 'Cumulative_Capacity_Window' not in df_for_corr.columns:
        df_for_corr['Cumulative_Capacity_Window'] = df_for_corr[config['COLUMN_CURRENT']].rolling(window=config['CUMULATIVE_WINDOW_SIZE'], min_periods=1).sum() * (time_step_interval_actual_corr / 3600.0)
    if 'Current_x_Cell_Temperature' not in df_for_corr.columns:
        df_for_corr['Current_x_Cell_Temperature'] = df_for_corr[config['COLUMN_CURRENT']] * df_for_corr[config['COLUMN_CELL_TEMPERATURE']]
    if 'Voltage_x_Cell_Temperature' not in df_for_corr.columns:
        df_for_corr['Voltage_x_Cell_Temperature'] = df_for_corr[config['COLUMN_VOLTAGE']] * df_for_corr[config['COLUMN_CELL_TEMPERATURE']]
    if 'Last_Current_Sign_Change' not in df_for_corr.columns:
        df_for_corr['Last_Current_Sign_Change'] = ((np.sign(df_for_corr[config['COLUMN_CURRENT']]) > 0) != (np.sign(df_for_corr[config['COLUMN_CURRENT']].shift(1)) > 0)).astype(int)

    df_for_corr = df_for_corr.dropna().reset_index(drop=True)

    all_corr_cols = [col for col in config['INPUT_FEATURES_ORDER'] if col in df_for_corr.columns] + [config['COLUMN_SOC']]
    all_corr_cols = list(dict.fromkeys(all_corr_cols))

    correlation_matrix = df_for_corr[all_corr_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Phase 1: EDA - Correlation Heatmap of Features and SoC')
    plt.tight_layout()
    plot_filename = os.path.join(env_paths['temp_plots_dir_actual'], 'correlation_heatmap.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    all_plots_info.append({'filepath': plot_filename, 'title': 'Correlation Heatmap of Features and SoC', 'section': 'Phase 1: EDA - Feature Relationships'})
    print("Correlation heatmap saved.")

    # --- Feature Engineering Calculations (Order Matters for Diffs/Rollings) ---
    df['Delta_Voltage'] = df[config['COLUMN_VOLTAGE']].diff()
    df['Delta_Current'] = df[config['COLUMN_CURRENT']].diff()
    #df['Delta_SoC'] = df[config['COLUMN_SOC']].diff().fillna(0)

    df['Rolling_Avg_Voltage_V'] = df[config['COLUMN_VOLTAGE']].rolling(window=config['ROLLING_WINDOW_SIZE'], min_periods=1).mean()
    df['Rolling_Avg_Current_A'] = df[config['COLUMN_CURRENT']].rolling(window=config['ROLLING_WINDOW_SIZE'], min_periods=1).mean()

    df['Power_W'] = df[config['COLUMN_VOLTAGE']] * df[config['COLUMN_CURRENT']]
    df['Voltage_Temp_Corrected'] = df[config['COLUMN_VOLTAGE']] + config['TEMP_COMP_K'] * (config['TEMP_COMP_REF_T'] - df[config['COLUMN_CELL_TEMPERATURE']])
    df['Delta_Cell_Temperature'] = df[config['COLUMN_CELL_TEMPERATURE']].diff()
    
    time_step_interval = df[config['COLUMN_TIME']].iloc[1] - df[config['COLUMN_TIME']].iloc[0] if len(df[config['COLUMN_TIME']]) > 1 else 1.0
    df['Cumulative_Capacity_Window'] = df[config['COLUMN_CURRENT']].rolling(window=config['CUMULATIVE_WINDOW_SIZE'], min_periods=1).sum() * (time_step_interval / 3600.0)

    df['Current_x_Cell_Temperature'] = df[config['COLUMN_CURRENT']] * df[config['COLUMN_CELL_TEMPERATURE']]
    df['Voltage_x_Cell_Temperature'] = df[config['COLUMN_VOLTAGE']] * df[config['COLUMN_CELL_TEMPERATURE']]
    df['Last_Current_Sign_Change'] = ((np.sign(df[config['COLUMN_CURRENT']]) > 0) != (np.sign(df[config['COLUMN_CURRENT']].shift(1)) > 0)).astype(int)


    df_processed = df.dropna().reset_index(drop=True)
    print(f"\nDataset shape after adding derived features and dropping initial NaN rows: {df_processed.shape}")
    if config['PRINT_RAW_TABLE_OUTPUTS_TO_LOG']:
        print("First 5 rows after feature engineering and dropping initial NaN:")
        print(df_processed.head())

    return df_processed


def prepare_for_lstm(df_processed, config, env_paths):
    """
    Phase 1.1 & 1.2 & 1.3: Handles feature selection, scaling (and saving scalers),
    sequence creation, and train/val/test split.
    Returns X_train, y_train, etc., and the scalers.
    """

    print(f"\n--- Phase 1.1: Feature Engineering ---")

    #df_processed['Delta_SoC'] = df_processed[config['COLUMN_SOC']].diff().fillna(0)
    df_processed['Is_Mid_SoC_Region'] = (
        (df_processed[config['COLUMN_SOC']] >= 30.0) & # Adjusted for 0-100 scale
        (df_processed[config['COLUMN_SOC']] <= 80.0)  # Adjusted for 0-100 scale
    ).astype(int)

    X_data_raw = df_processed[config['INPUT_FEATURES_ORDER']].values
    y_data_raw = df_processed[config['COLUMN_SOC']].values.reshape(-1, 1)
    print(f"\n--- DEBUG: y_data_raw (before scaling) ---")
    print(f"Min y_data_raw: {y_data_raw.min():.4f}")
    print(f"Max y_data_raw: {y_data_raw.max():.4f}")
    print("---------------------------------------------")

    print(f"Raw input features (X_data_raw) shape: {X_data_raw.shape}")
    print(f"Raw target (y_data_raw) shape: {y_data_raw.shape}")

    print(f"\n--- Phase 1.2: Creating LSTM Sequences (SEQUENCE_LENGTH={config['SEQUENCE_LENGTH']}) ---")
    X_sequences, y_sequences = create_lstm_sequences(X_data_raw, y_data_raw, config['SEQUENCE_LENGTH'])

    print(f"Shape of LSTM input sequences (X_sequences): {X_sequences.shape}")
    print(f"Shape of LSTM target values (y_sequences): {y_sequences.shape}")

    y_sequences_unscaled_for_split_debug = y_sequences.flatten() # Keep a copy before main scaling
    soc_bins = config['F1_SCORE_SOC_BINS']['bins']
    soc_labels = config['F1_SCORE_SOC_BINS']['labels']

    print("\n--- Phase 1.3: Splitting Data ---")
    total_samples = len(X_sequences)

    y_train_unscaled_pre_oversample = None
    y_val_unscaled_pre_oversample = None
    y_test_unscaled_pre_oversample = None

    if config['TEST_SPLIT_STRATEGY'] == 'chronological':
        train_val_split_idx = int(total_samples * (config['DATA_SPLIT_RATIOS']['TRAIN_RATIO'] + config['DATA_SPLIT_RATIOS']['VAL_RATIO']))
        X_train_val_temp = X_sequences[:train_val_split_idx]
        y_train_val_temp = y_sequences[:train_val_split_idx]
        X_test = X_sequences[train_val_split_idx:]
        y_test = y_sequences[train_val_split_idx:]

        train_total_samples = len(X_train_val_temp)
        val_size = int(train_total_samples * (
            config['DATA_SPLIT_RATIOS']['VAL_RATIO'] /
            (config['DATA_SPLIT_RATIOS']['TRAIN_RATIO'] + config['DATA_SPLIT_RATIOS']['VAL_RATIO'])
        ))
        train_split_idx = train_total_samples - val_size

        X_train = X_train_val_temp[:train_split_idx]
        y_train = y_train_val_temp[:train_split_idx]
        X_val = X_train_val_temp[train_split_idx:]
        y_val = y_train_val_temp[train_split_idx:]

        y_train_unscaled_pre_oversample = y_train.flatten()
        y_val_unscaled_pre_oversample = y_val.flatten()
        y_test_unscaled_pre_oversample = y_test.flatten()


    elif config['TEST_SPLIT_STRATEGY'] == 'stratified_random':
        y_binned = pd.cut(y_sequences_unscaled_for_split_debug, bins=soc_bins, labels=soc_labels, include_lowest=True, right=True)
        y_binned_array = y_binned.astype(str)
        y_binned_array[y_binned.isna()] = 'Unassigned_Strat'

        X_remaining, X_test, y_remaining, y_test = train_test_split(
            X_sequences, y_sequences,
            test_size=config['STRATIFIED_TEST_SIZE_PERCENT'] / 100.0,
            random_state=config['RANDOM_SEED_SPLIT'],
            stratify=y_binned_array
        )

        train_total_samples = len(X_remaining)
        val_size = int(train_total_samples * (
            config['DATA_SPLIT_RATIOS']['VAL_RATIO'] /
            (config['DATA_SPLIT_RATIOS']['TRAIN_RATIO'] + config['DATA_SPLIT_RATIOS']['VAL_RATIO'])
        ))
        train_split_idx = train_total_samples - val_size

        X_train = X_remaining[:train_split_idx]
        y_train = y_remaining[:train_split_idx]
        X_val = X_remaining[train_split_idx:]
        y_val = y_remaining[train_split_idx:]
        
        y_train_unscaled_pre_oversample = y_train.flatten()
        y_val_unscaled_pre_oversample = y_val.flatten()
        y_test_unscaled_pre_oversample = y_test.flatten()

    else:
        raise ValueError(f"Unknown TEST_SPLIT_STRATEGY: {config['TEST_SPLIT_STRATEGY']}")

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    

    X_train_2d = X_train.reshape(-1, X_train.shape[2])
    X_val_2d = X_val.reshape(-1, X_val.shape[2])
    X_test_2d = X_test.reshape(-1, X_test.shape[2])

    X_train_scaled = scaler_X.fit_transform(X_train_2d).reshape(X_train.shape)
    X_val_scaled = scaler_X.transform(X_val_2d).reshape(X_val.shape)
    X_test_scaled = scaler_X.transform(X_test_2d).reshape(X_test.shape)
    

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    print(f"\n--- DEBUG: Scaled y_data ranges ---")
    print(f"Min y_train_scaled: {y_train_scaled.min():.4f}, Max y_train_scaled: {y_train_scaled.max():.4f}")
    print(f"Min y_val_scaled: {y_val_scaled.min():.4f}, Max y_val_scaled: {y_val_scaled.max():.4f}")
    print(f"Min y_test_scaled: {y_test_scaled.min():.4f}, Max y_test_scaled: {y_test_scaled.max():.4f}")
    print("------------------------------------")

    X_train, y_train = X_train_scaled, y_train_scaled
    X_val, y_val = X_val_scaled, y_val_scaled
    X_test, y_test = X_test_scaled, y_test_scaled

    try:
        joblib.dump(scaler_X, config['SCALER_X_PATH'])
        joblib.dump(scaler_y, config['SCALER_Y_PATH'])
        print(f"Scalers saved to '{config['SCALER_X_PATH']}' and '{config['SCALER_Y_PATH']}'")
    except Exception as e:
        print(f"Error saving scalers: {e}")

    print("\n--- Visualizing SoC Distribution in Train, Validation, and Test Sets (Pre-Oversampling) ---")
    hist_plot_so_distribution(y_train_unscaled_pre_oversample, "Train Set SoC Distribution (Pre-Oversampling)", env_paths['temp_plots_dir_actual'], "Phase 1: Data Distribution - Train Set", soc_bins, soc_labels)
    hist_plot_so_distribution(y_val_unscaled_pre_oversample, "Validation Set SoC Distribution (Pre-Oversampling)", env_paths['temp_plots_dir_actual'], "Phase 1: Data Distribution - Train Set", soc_bins, soc_labels)
    hist_plot_so_distribution(y_test_unscaled_pre_oversample, "Test Set SoC Distribution (Pre-Oversampling)", env_paths['temp_plots_dir_actual'], "Phase 1: Data Distribution - Train Set", soc_bins, soc_labels)


    print(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation set shape: X={X_val.shape}, y={y_val.shape}")
    print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")

    if config.get("ENABLE_OVERSAMPLING", True):
        # The mask calculation should use the unscaled values to check SoC range
        # Note: y_train here is already scaled, so we need to inverse transform for the mask check.
        y_train_unscaled_for_mask = scaler_y.inverse_transform(y_train.reshape(-1,1)).flatten()
        mid_mask = (y_train_unscaled_for_mask >= 30.0) & \
                   (y_train_unscaled_for_mask <= 80.0)
        
        X_train_mid = X_train[mid_mask]
        y_train_mid = y_train[mid_mask]
        
        multiplier = config.get("MID_SOC_OVERSAMPLING_FACTOR", 3)
        
        if len(X_train_mid) == 0:
            print(f"\n [WARNING] No mid-SoC samples found in training set. Oversampling for mid region needed.")
        if multiplier > 1 and len(X_train_mid) > 0: # Added len(X_train_mid) > 0 check
            X_train = np.concatenate([X_train] + [X_train_mid] * (multiplier - 1))
            y_train = np.concatenate([y_train] + [y_train_mid] * (multiplier - 1))
            print(f"\n [DEBUG] Mid-SoC samples found: {len(X_train_mid)}. Oversampling x{multiplier}")
            print(f"\n [INFO] Mid-SoC oversampling applied. New train shape: {X_train.shape}")
        else:
            print("[INFO] Mid-SoC oversampling skipped (no mid-SoC samples or multiplier <= 1).")

    print(f"\n--- DEBUG: scaler_Y inverse_transform test ---")
    sample_y_scaled = np.array([0.0, 0.25, 0.5, 0.75, 1.0]).reshape(-1,1)
    sample_y_unscaled = scaler_y.inverse_transform(sample_y_scaled).flatten()
    print(f"\n--- DEBUG: scaler_Y inverse_transform test ---")
    print(f"Scaled values: {sample_y_scaled.flatten()}")
    print(f"Inverse transformed values: {sample_y_unscaled}")
    print("--------------------------------------------")

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y


# --- Core Pipeline Functions ---

def setup_environment_and_args(config):
    """
    Handles command-line arguments, sets up report paths, and initializes logging redirection (Tee class).
    Returns a dictionary of paths and run metadata.
    """
    # Set the default time_step_interval here, if not derived from data later
    time_step_interval = 1.0 # Default if data is empty or only one row

    # Setup for teeing console output (to screen and to string buffer)
    original_stdout_stream = sys.stdout
    sys.stdout = Tee(original_stdout_stream, io.StringIO())

    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python soc_prediction_pipeline.py <path_to_FINAL_MASTER_DATASET_With_SOC.csv> [output_directory_for_pdf]")
        print("Example: python soc_prediction_pipeline.py data/FINAL_MASTER_DATASET_With_SOC.csv")
        print("Example: python soc_prediction_pipeline.py data/FINAL_MASTER_DATASET_With_SOC.csv ./my_reports_folder")
        sys.stdout = original_stdout_stream # Restore stdout before exiting
        sys.exit(1)

    dataset_file_path_arg = sys.argv[1]
    output_pdf_base_dir = sys.argv[2] if len(sys.argv) > 2 else config['DEFAULT_REPORT_OUTPUT_DIR']

    # Create the full path for the PDF report
    os.makedirs(output_pdf_base_dir, exist_ok=True)
    report_filename = f"SoC_Prediction_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    report_pdf_path_final = os.path.join(output_pdf_base_dir, report_filename)

    # Temporary directory for plots (always relative to script is run)
    temp_plots_dir_actual = os.path.join(os.getcwd(), config['TEMP_PLOTS_SUBDIR_NAME'])
    os.makedirs(temp_plots_dir_actual, exist_ok=True)

    # Dynamically get the script's own filename
    script_filename = os.path.basename(sys.argv[0]) # sys.argv[0] is the script's path

    # Setup report metadata
    run_timestamp = datetime.datetime.now()
    report_meta = {
        'Report Title': 'Battery SoC Estimation Pipeline Run Report',
        'Generated By': script_filename,
        'Generation Time': run_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'Python Version': sys.version.split(' ')[0],
        'TensorFlow Version': 'Not available (check env)',
        'Sequence Length': config['SEQUENCE_LENGTH'],
        'Epochs': config['MODEL_HP']['EPOCHS'],
        'Batch Size': config['MODEL_HP']['BATCH_SIZE'],
        'Dataset File': config['DATASET_FILE'],
        'Report File': report_pdf_path_final
    }
    try: # Attempt to get TensorFlow version
        import tensorflow as tf
        report_meta['TensorFlow Version'] = tf.__version__
    except ImportError:
        report_meta['TensorFlow Version'] = 'TensorFlow not imported/found'

    return {
        'dataset_file_path': dataset_file_path_arg,
        'report_pdf_path_final': report_pdf_path_final,
        'temp_plots_dir_actual': temp_plots_dir_actual,
        'report_meta': report_meta,
        'original_stdout_stream': original_stdout_stream
    }


# This function remains unchanged as it's the core model creation.
# It's called by KerasRegressor.
def create_keras_model(lstm_units, dropout_rate, sequence_length, num_features, l2_reg=0.0, learning_rate=0.001, **kwargs):
    """
    Creates and compiles a Keras LSTM model based on provided hyperparameters.
    Used by KerasRegressor for GridSearchCV.
    """
    model = Sequential([
        Input(shape=(sequence_length, num_features)),
        LSTM(units=lstm_units, return_sequences=True, kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        LSTM(units=lstm_units, return_sequences=False, kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(units=1, activation='linear', kernel_regularizer=l2(l2_reg))
    ])
    if CONFIG.get('USE_HUBER_LOSS', False):
    	model.compile(
        	optimizer=Adam(learning_rate=learning_rate),
        	loss=Huber(delta=0.01),
        	metrics=['mae']
    	)
    else:
        model.compile(optimizer=Adam(learning_rate=learning_rate),loss='mean_squared_error',metrics=['mae'])
    return model


# NEW: Helper function to plot SoC distribution histogram
def hist_plot_so_distribution(so_data, title, save_dir, section_title, bins, labels):
    plt.figure(figsize=(8, 5))
    sns.histplot(so_data, bins=bins, kde=False, color='steelblue') # Use bins directly
    plt.title(title)
    plt.xlabel('SoC (%)')
    plt.ylabel('Frequency')
    
    # Ensure xticks match the bin edges and are interpretable
    tick_positions = [b for b in bins if b >= 0 and b <= 100] # Only show ticks within 0-100 range
    tick_labels = [f'{b:.0f}%' for b in tick_positions]
    plt.xticks(tick_positions, tick_labels)
    
    # Add counts for each bin as text on plot
    counts, _ = np.histogram(so_data, bins=bins)
    max_count = counts.max() if len(counts) > 0 else 1 # Avoid division by zero
    
    for i in range(len(labels)):
        if i < len(bins) - 1: # Ensure we don't go out of bounds for bins
            # Calculate mid-point of the bin for text placement
            mid_bin = (bins[i] + bins[i+1]) / 2
            
            if counts[i] > 0: # Only plot count if there are samples in the bin
                # Adjust y offset based on max count to avoid text overlap
                y_offset = max_count * 0.05
                plt.text(mid_bin, counts[i] + y_offset, str(counts[i]), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_filename = os.path.join(save_dir, f'{title.replace(" ", "_").replace(":", "").replace("__", "_").replace("(", "").replace(")", "")}.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    all_plots_info.append({'filepath': plot_filename, 'title': title, 'section': section_title})
    print(f"Plot '{title}' saved.")


def create_xgboost_model(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, **kwargs):
    """
    Creates and returns an XGBoost Regressor model.
    """
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        objective='reg:squarederror', # For regression
        n_jobs=n_jobs, # Use n_jobs passed from config
        **kwargs
    )
    return model


def build_and_train_model(X_train, y_train, X_val, y_val, config, scaler_X, scaler_Y):
    """
    Phase 2: Builds the Keras LSTM model, performs GridSearchCV (conditionally),
    and potentially trains/saves an XGBoost corrector model.
    Returns the trained LSTM model and its history object.
    """
    print("\n--- Phase 2: Building and Training the LSTM Model ---")
    num_features_in_lstm = X_train.shape[2]
    sequence_length = config['SEQUENCE_LENGTH']

    best_params = {} # Initialize best_params
    
    if config['ENABLE_GRID_SEARCH']: # <-- NEW CONDITIONAL LOGIC HERE
        # --- Stage 1: LSTM Base Model Training (with GridSearchCV) ---
        print("\n--- Training LSTM Base Model (Stage 1) via GridSearchCV ---")
        keras_model_wrapper = KerasRegressor(
            build_fn=create_keras_model,
            # Pass fixed parameters to create_keras_model via KerasRegressor's constructor
            sequence_length=sequence_length,
            num_features=num_features_in_lstm,
            model__lstm_units=config['GRID_SEARCH_PARAMS']['lstm_units'][0],
            model__dropout_rate=config['GRID_SEARCH_PARAMS']['dropout_rate'][0],
            model__l2_reg=config['GRID_SEARCH_PARAMS']['l2_reg'][0],
            model__learning_rate=config['GRID_SEARCH_PARAMS']['learning_rate'][0],
            callbacks=[EarlyStopping(monitor='val_loss', patience=config['MODEL_HP']['EARLY_STOPPING_PATIENCE'], restore_best_weights=True)],
            verbose=0
        )

        param_grid = {
            'model__lstm_units': config['GRID_SEARCH_PARAMS']['lstm_units'],
            'model__dropout_rate': config['GRID_SEARCH_PARAMS']['dropout_rate'],
            'batch_size': config['GRID_SEARCH_PARAMS']['batch_size'],
            'epochs': config['GRID_SEARCH_PARAMS']['epochs'],
            'model__l2_reg': config['GRID_SEARCH_PARAMS']['l2_reg'],
            'model__learning_rate': config['GRID_SEARCH_PARAMS']['learning_rate'],
        }

        tscv = TimeSeriesSplit(n_splits=config['GRID_SEARCH_CV_SPLITS'])

        print(f"\n--- Starting GridSearchCV for LSTM Hyperparameter Optimization ---")
        print(f"Grid Parameters: {param_grid}")
        print(f"Cross-Validation Splits (TimeSeriesSplit): {config['GRID_SEARCH_CV_SPLITS']}")
        print("WARNING: This process can be very time-consuming.")

        grid_search = GridSearchCV(
            estimator=keras_model_wrapper,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1,
            error_score='raise'
        )

        grid_result = grid_search.fit(X_train, y_train, validation_data=(X_val, y_val))

        print("\n====================================================================")
        print("--- Cross-Validation Results: Best LSTM Hyperparameters Found ---")
        print(f"Best Mean Absolute Error (MAE) from CV: {-grid_result.best_score_:.6f} (0-1 Scale)")
        # ... (print other best params) ...
        print("====================================================================")

        best_params = grid_result.best_params_
        config['Best_CV_MAE'] = -grid_result.best_score_ # Store best CV MAE in config for report (it's already 0-1)

    else: # <-- NEW ELSE BLOCK
        print("\n--- Skipping GridSearchCV. Training LSTM Base Model using MODEL_HP parameters ---")
        # Use parameters from MODEL_HP directly
        best_params = {
            'model__lstm_units': config['MODEL_HP']['LSTM_UNITS'],
            'model__dropout_rate': config['MODEL_HP']['DROPOUT_RATE'],
            'batch_size': config['MODEL_HP']['BATCH_SIZE'],
            'epochs': config['MODEL_HP']['EPOCHS'], # Note: This will be the max epochs for direct training
            'model__l2_reg': config['MODEL_HP']['L2_REG'],
            'model__learning_rate': config['MODEL_HP']['LEARNING_RATE'],
        }
        # Set Best_CV_MAE to N/A or a placeholder if GridSearchCV is skipped
        config['Best_CV_MAE'] = 'N/A (GridSearchCV skipped)'
        print(f"Using fixed parameters: {best_params}")

    # Retrain the best LSTM model with full epochs and save it
    print("\n--- Retraining Best LSTM Model for Detailed History ---")
    
    # Re-initialize the model with best parameters and a fresh compile to ensure callbacks work
    model_lstm = create_keras_model(
        lstm_units=best_params['model__lstm_units'],
        dropout_rate=best_params['model__dropout_rate'],
        sequence_length=sequence_length,
        num_features=num_features_in_lstm,
        l2_reg=best_params['model__l2_reg'],
        learning_rate=best_params['model__learning_rate']
    )
    
    history_lstm = model_lstm.fit(
        X_train, y_train,
        epochs=config['MODEL_HP']['EPOCHS'], # Use final model's EPOCHS from CONFIG
        batch_size=best_params['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(monitor='val_loss', patience=config['MODEL_HP']['EARLY_STOPPING_PATIENCE'], restore_best_weights=True),
                   ModelCheckpoint(config['MODEL_SAVE_PATH'], monitor='val_loss', save_best_only=True, mode='min', verbose=0)],
        verbose=1
    )
    print("Best LSTM model retraining complete and saved.")

    
    # Load the final best LSTM model (base model for stacking)
    final_lstm_model = load_model(config['MODEL_SAVE_PATH'])

    # --- Stage 2: XGBoost Corrector Training (If Enabled) ---
    if config['ENABLE_XGBOOST_CORRECTOR']:
        print("\n--- Training XGBoost Corrector (Stage 2) ---")
        
        # 1. Get LSTM predictions on the training data (still scaled 0-1)
        y_pred_lstm_train_scaled = final_lstm_model.predict(X_train)
        
        # Get the TRUE unscaled SoC values for residual calculation
        y_true_train_unscaled = scaler_Y.inverse_transform(y_train.reshape(-1,1)).flatten()

        # Get the RAW unscaled LSTM predictions (before clipping)
        y_pred_lstm_train_raw_unscaled = scaler_Y.inverse_transform(y_pred_lstm_train_scaled).flatten()

        # Calculate residuals using the TRUE unscaled values and RAW unscaled predictions
        # This is the absolute difference that XGBoost will learn to predict.
        residuals_train = y_true_train_unscaled - y_pred_lstm_train_raw_unscaled
        
        # --- NEW DEBUGGING PRINTS FOR RESIDUALS ---
        print(f"\n[DEBUG] Residuals_train (before filtering) stats:")
        print(f"  Min: {residuals_train.min():.4f}, Max: {residuals_train.max():.4f}, Mean: {residuals_train.mean():.4f}, Std: {residuals_train.std():.4f}")
        # --- END NEW DEBUGGING PRINTS ---

        # (Optional) Clip the LSTM predictions for display or other uses that expect valid SoC range
        y_pred_lstm_train_unscaled = np.clip(y_pred_lstm_train_raw_unscaled, 0, 100)
       
        # 3. Extract XGBoost features from the LAST TIMESTEP of LSTM's input sequences
        X_train_original_scale = scaler_X.inverse_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
        
        xgboost_feature_indices = {name: i for i, name in enumerate(config['INPUT_FEATURES_ORDER'])}
        
        # Prepare XGBoost features: take the last timestep's values for selected features
        X_train_xgboost_features = pd.DataFrame(
            X_train_original_scale[:, -1, [xgboost_feature_indices[f] for f in config['XGBOOST_FEATURES']]],
            columns=config['XGBOOST_FEATURES']
        )
        # Ensure XGBoost features are correctly named for its internal processes (no special chars/spaces)
        X_train_xgboost_features.columns = [re.sub(r'[^a-zA-Z0-9_]', '', col) for col in X_train_xgboost_features.columns]


        # 4. Calculate sample weights for XGBoost based on SoC bin imbalance
        # We're using a simpler approach of emphasizing mid-SoC directly as per discussion
        print(f"XGBoost Sample Weights (by bin based on XGBOOST_MID_SOC_WEIGHT):")
        sample_weights_xgboost = np.ones_like(y_true_train_unscaled)
        mid_mask_for_xgb_weights = (y_true_train_unscaled >= 30.0) & (y_true_train_unscaled <= 80.0) # Adjusted for 0-100 scale
        weight_multiplier = config.get("XGBOOST_MID_SOC_WEIGHT", 1.5)
        sample_weights_xgboost[mid_mask_for_xgb_weights] = weight_multiplier
        print(f"  - Mid-SoC (30.0-80.0%): weight x {weight_multiplier}")
        print(f"  - Other regions: weight x 1.0")

        # === Filter high-error residuals only ===
        # high_error_mask is a boolean array of shape (num_samples,)
        high_error_mask = np.abs(residuals_train) > config['XGBOOST_HIGH_ERROR_THRESHOLD']

        # Apply the mask to filter ROWS for XGBoost training data
        X_train_xgboost_features_filtered = X_train_xgboost_features[high_error_mask]
        residuals_train_filtered  = residuals_train[high_error_mask]
        sample_weights_xgboost_filtered  = sample_weights_xgboost[high_error_mask]

        # Prepare validation data for XGBoost early stopping
        y_pred_lstm_val_scaled = final_lstm_model.predict(X_val)
        y_true_val_unscaled = scaler_Y.inverse_transform(y_val.reshape(-1,1)).flatten()
        y_pred_lstm_val_raw_unscaled = scaler_Y.inverse_transform(y_pred_lstm_val_scaled).flatten()
        residuals_val = y_true_val_unscaled - y_pred_lstm_val_raw_unscaled
        
        # Apply the same high_error_mask logic to validation residuals for eval_set if desired,
        # but for simplicity and to ensure eval_set has enough data, we might not filter val_set.
        # If filtering val_set, ensure it's consistent with train_set filtering.
        # For now, we'll use unfiltered val_set for eval_set.
        # If you want to filter val_set, you'd need a high_error_mask_val based on residuals_val.
        
                # --- Prepare Validation Data for XGBoost early stopping (eval_set) ---
        # Get LSTM predictions on validation data (still scaled 0-1)
        y_pred_lstm_val_scaled = final_lstm_model.predict(X_val)

        # Get the TRUE unscaled SoC values for validation
        y_true_val_unscaled = scaler_Y.inverse_transform(y_val.reshape(-1,1)).flatten()

        # Get the RAW unscaled LSTM predictions for validation (before clipping)
        y_pred_lstm_val_raw_unscaled = scaler_Y.inverse_transform(y_pred_lstm_val_scaled).flatten()

        # Calculate residuals for validation set (Actual - Raw LSTM Prediction)
        residuals_val = y_true_val_unscaled - y_pred_lstm_val_raw_unscaled

        # Extract XGBoost features from the LAST TIMESTEP of LSTM's validation sequences
        X_val_original_scale = scaler_X.inverse_transform(X_val.reshape(-1, X_val.shape[2])).reshape(X_val.shape)
        
        # xgboost_feature_indices is already defined earlier in this function, no need to redefine.
        X_val_xgboost_features = pd.DataFrame(
            X_val_original_scale[:, -1, [xgboost_feature_indices[f] for f in config['XGBOOST_FEATURES']]],
            columns=config['XGBOOST_FEATURES']
        )
        # Clean column names for XGBoost
        X_val_xgboost_features.columns = [re.sub(r'[^a-zA-Z0-9_]', '', col) for col in X_val_xgboost_features.columns]
        # --- END Prepare Validation Data for XGBoost ---
        
        # --- NEW DEBUGGING PRINTS FOR FILTERED RESIDUALS ---
        print(f"\n[DEBUG] Residuals_train_filtered (after filtering) stats:")
        print(f"  Shape: {residuals_train_filtered.shape}")
        if residuals_train_filtered.size > 0: # Check if array is not empty
            print(f"  Min: {residuals_train_filtered.min():.4f}, Max: {residuals_train_filtered.max():.4f}, Mean: {residuals_train_filtered.mean():.4f}, Std: {residuals_train_filtered.std():.4f}")
        else:
            print("  No high-error residuals found after filtering. XGBoost training data is empty.")
        # --- END NEW DEBUGGING PRINTS ---
        
        # 5. Train XGBoost Regressor
        xgb_corrector_model = create_xgboost_model(**config['XGBOOST_HP'])

        if config.get('ENABLE_RESIDUAL_XGBOOST', False):
            print(" Using residual learning mode for XGBoost (predicting errors).")
            xgb_corrector_model.fit(X_train_xgboost_features_filtered, residuals_train_filtered,
                                    sample_weight=sample_weights_xgboost_filtered,
                                    eval_set=[(X_val_xgboost_features, residuals_val)],
                                    #early_stopping_rounds=config['XGBOOST_HP']['early_stopping_rounds'],
                                    verbose=False)
        else:
            print(" Using direct SoC prediction mode for XGBoost (predicting SoC).")
            xgb_corrector_model.fit(X_train_xgboost_features_filtered, y_true_train_unscaled[high_error_mask],
                                    sample_weight=sample_weights_xgboost_filtered,
                                    eval_set=[(X_val_xgboost_features, y_true_val_unscaled)],
                                    #early_stopping_rounds=config['XGBOOST_HP']['early_stopping_rounds'],
                                    verbose=False)

        # Save XGBoost model (XGBoost has its own save/load method)
        xgb_corrector_model.save_model(config['XGBOOST_MODEL_SAVE_PATH'])
        print(f"XGBoost Corrector model saved to: {config['XGBOOST_MODEL_SAVE_PATH']}")
        
        config['xgb_corrector_model_instance'] = xgb_corrector_model # Store instance for evaluation
        
    else: # If XGBoost Corrector is disabled
        print("\nXGBoost Corrector is DISABLED in CONFIG. Skipping Stage 2 training.")
        config['xgb_corrector_model_instance'] = None # Ensure it's explicitly None

    return final_lstm_model, history_lstm


def evaluate_model_and_generate_plots(model, history, X_test, y_test, scaler_X, scaler_Y, config, env_paths):
    """
    Phase 3: Evaluates the trained model, calculates metrics, and generates all evaluation/error plots.
    Adds plot info to all_plots_info.
    Returns overall MAE, RMSE, R2, and F1.
    """
    # Calculate the range of the original unscaled target data (typically 100 for 0-100% SoC)
    # This is needed to convert MAE/RMSE from 0-100 scale back to 0-1 scale
    unscaled_target_range = scaler_Y.data_max_[0] - scaler_Y.data_min_[0]
    if unscaled_target_range == 0: # Avoid division by zero if data is constant
        unscaled_target_range = 1.0 # Or use 100.0 if you expect 0-100% range
    
    
    print("\n--- Phase 3: Evaluating the Trained Model ---")
    
    # --- Get LSTM Base Predictions ---
    y_pred_lstm_scaled = model.predict(X_test)
    y_pred_lstm = scaler_Y.inverse_transform(y_pred_lstm_scaled).flatten()
    y_true_unscaled = scaler_Y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Initialize y_pred_final with LSTM's prediction
    y_pred_final = np.clip(y_pred_lstm, 0, 100) # Clip LSTM prediction for consistency in plots/metrics

    print(f"\n--- Debugging Evaluation Values (Post-Inverse Transform) ---")
    print(f"y_true_unscaled stats: Min={y_true_unscaled.min():.4f}, Max={y_true_unscaled.max():.4f}, Mean={y_true_unscaled.mean():.4f}")
    print(f"y_pred_lstm_unscaled (clipped) stats: Min={y_pred_final.min():.4f}, Max={y_pred_final.max():.4f}, Mean={y_pred_final.mean():.4f}")
    # Note: y_pred_final is initialized with clipped LSTM here. If XGBoost is enabled, it will be updated.
    
    overall_mae_lstm, overall_rmse_lstm, overall_r2_lstm, overall_f1_lstm = None, None, None, None
    overall_mae_combined, overall_rmse_combined, overall_r2_combined, overall_f1_combined = None, None, None, None


    # --- Evaluate LSTM Only ---
    print(f"\n--- Performance for Scenario: LSTM Only ---")
    # Use the clipped LSTM predictions for evaluation metrics
    overall_mae_lstm_100_percent = mean_absolute_error(y_true_unscaled, y_pred_lstm) # Use original y_pred_lstm for MAE calc
    overall_rmse_lstm_100_percent = np.sqrt(mean_squared_error(y_true_unscaled, y_pred_lstm))
    overall_r2_lstm = r2_score(y_true_unscaled, y_pred_lstm)

    # Convert to 0-1 scale
    overall_mae_lstm_0_1_scale = overall_mae_lstm_100_percent / unscaled_target_range
    overall_rmse_lstm_0_1_scale = overall_rmse_lstm_100_percent / unscaled_target_range

    print(f"Mean Absolute Error (MAE): {overall_mae_lstm_100_percent:.2f} %SoC ({overall_mae_lstm_0_1_scale:.6f} 0-1 Scale)")
    print(f"Root Mean Squared Error (RMSE): {overall_rmse_lstm_100_percent:.2f} %SoC ({overall_rmse_lstm_0_1_scale:.6f} 0-1 Scale)")
    print(f"R-squared (R^2): {overall_r2_lstm:.4f}")

    # F1-Score for LSTM Only
    if config['ENABLE_F1_SCORE_CALCULATION']:
        print(f"--- Classification Metrics for LSTM Only ---")
        bins = config['F1_SCORE_SOC_BINS']['bins']
        labels = config['F1_SCORE_SOC_BINS']['labels']

        y_true_clamped = np.clip(y_true_unscaled, bins[0], bins[-1]) # Clamp to bin range for robust binning
        y_pred_lstm_clamped = np.clip(y_pred_lstm, bins[0], bins[-1]) # Use clipped LSTM predictions for binning

        y_true_binned_lstm = pd.cut(y_true_clamped, bins=bins, labels=labels, include_lowest=True, right=True)
        y_pred_binned_lstm = pd.cut(y_pred_lstm_clamped, bins=bins, labels=labels, include_lowest=True, right=True)

        y_true_binned_array_lstm = y_true_binned_lstm.astype(str)
        y_pred_binned_array_lstm = y_pred_binned_lstm.astype(str)

        # Handle 'Unassigned' due to binning issues if any
        all_possible_labels_f1 = list(labels)
        if 'Unassigned' not in all_possible_labels_f1:
             if 'Unassigned' in np.unique(y_true_binned_array_lstm) or 'Unassigned' in np.unique(y_pred_binned_array_lstm):
                all_possible_labels_f1.append('Unassigned')

        true_bin_counts_lstm = pd.Series(y_true_binned_lstm).value_counts().sort_index()
        print(f"True samples per bin (for LSTM Only): {true_bin_counts_lstm.to_dict()}")

        try:
            valid_indices_lstm = (y_true_binned_array_lstm != 'Unassigned') & (y_pred_binned_array_lstm != 'Unassigned')
            y_true_f1_lstm = y_true_binned_array_lstm[valid_indices_lstm]
            y_pred_f1_lstm = y_pred_binned_array_lstm[valid_indices_lstm]
            
            # Filter labels for f1_score based on what's actually present in valid data
            f1_labels_actual = [label for label in labels if label in np.unique(y_true_f1_lstm)]

            if len(f1_labels_actual) == 0:
                print("Warning: No valid true labels for F1-score calculation (LSTM Only) after cleaning. F1-score skipped.")
                overall_f1_lstm = np.nan
            else:
                overall_f1_lstm = f1_score(y_true_f1_lstm, y_pred_f1_lstm, labels=f1_labels_actual, average='weighted', zero_division=0)
                print(f"F1-Score (binned SoC, weighted average): {overall_f1_lstm:.4f}")
        except ValueError as e:
            print(f"Could not calculate F1-score for LSTM Only: {e}. F1-score set to NaN.")
            overall_f1_lstm = np.nan

        cm_lstm = confusion_matrix(y_true_binned_array_lstm, y_pred_binned_array_lstm, labels=all_possible_labels_f1)
        disp_lstm = ConfusionMatrixDisplay(confusion_matrix=cm_lstm, display_labels=all_possible_labels_f1)
        
        plt.figure(figsize=(10, 8))
        disp_lstm.plot(cmap='Blues', values_format='d', ax=plt.gca())
        plt.title(f'Phase 3: Error Analysis - Confusion Matrix (LSTM Only Binned SoC)')
        plt.tight_layout()
        plot_filename = os.path.join(env_paths['temp_plots_dir_actual'], f'confusion_matrix_LSTM_Only.png')
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        all_plots_info.append({'filepath': plot_filename, 'title': f'Confusion Matrix (LSTM Only Binned SoC)', 'section': 'Phase 3: Error Analysis - Classification Performance'})
        print(f"Confusion Matrix plot for LSTM Only saved.")


    # --- Apply XGBoost Correction if Enabled ---
    if config['ENABLE_XGBOOST_CORRECTOR'] and os.path.exists(config['XGBOOST_MODEL_SAVE_PATH']):
        try:
            if 'xgb_corrector_model_instance' in config and config['xgb_corrector_model_instance'] is not None:
                xgb_corrector_model = config['xgb_corrector_model_instance']
            else:
                xgb_corrector_model = xgb.XGBRegressor()
                xgb_corrector_model.load_model(config['XGBOOST_MODEL_SAVE_PATH'])
                config['xgb_corrector_model_instance'] = xgb_corrector_model

            print("\n--- Performance for Scenario: LSTM + XGBoost Corrected ---")
            
            X_test_original_scale = scaler_X.inverse_transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)
            
            xgboost_feature_indices = {name: i for i, name in enumerate(config['INPUT_FEATURES_ORDER'])}
            
            X_test_xgboost_features = pd.DataFrame(
                X_test_original_scale[:, -1, [xgboost_feature_indices[f] for f in config['XGBOOST_FEATURES']]],
                columns=config['XGBOOST_FEATURES']
            )
            X_test_xgboost_features.columns = [re.sub(r'[^a-zA-Z0-9_]', '', col) for col in X_test_xgboost_features.columns]

            # Get raw unclipped LSTM predictions for residual calculation
            y_pred_lstm_raw_unscaled_test = scaler_Y.inverse_transform(y_pred_lstm_scaled).flatten()
            
            # Calculate residuals on test set for XGBoost prediction
            # This is the residual that XGBoost *should* predict for these test samples
            # residuals_test_actual = y_true_unscaled - y_pred_lstm_raw_unscaled_test # For debugging/analysis

            xgboost_prediction_raw = xgb_corrector_model.predict(X_test_xgboost_features)

            if config.get('ENABLE_RESIDUAL_XGBOOST', False):
                # Apply correction: LSTM prediction + XGBoost predicted residual
                y_pred_final = y_pred_lstm_raw_unscaled_test + xgboost_prediction_raw
                y_pred_final = np.clip(y_pred_final, 0, 100) # Clip final prediction to valid SoC range
                print("XGBoost correction applied as residual learning.")
                print(f"[DEBUG] LSTM base prediction (raw unclipped): {y_pred_lstm_raw_unscaled_test[:5]}")
                print(f"[DEBUG] XGBoost predicted residual: {xgboost_prediction_raw[:5]}")
                print(f"[DEBUG] Final corrected prediction (clipped): {y_pred_final[:5]}")
            else:
                y_pred_final = xgboost_prediction_raw # XGBoost directly predicts SoC
                y_pred_final = np.clip(y_pred_final, 0, 100) # Clip direct XGBoost prediction
                print("XGBoost correction applied as direct SoC prediction.")


            overall_mae_combined = mean_absolute_error(y_true_unscaled, y_pred_final)
            overall_rmse_combined = np.sqrt(mean_squared_error(y_true_unscaled, y_pred_final))
            overall_r2_combined = r2_score(y_true_unscaled, y_pred_final)

           # Convert to 0-1 scale
            overall_mae_combined_0_1_scale = overall_mae_combined / unscaled_target_range
            overall_rmse_combined_0_1_scale = overall_rmse_combined / unscaled_target_range

            print(f"Mean Absolute Error (MAE): {overall_mae_combined:.2f} %SoC ({overall_mae_combined_0_1_scale:.6f} 0-1 Scale)")
            print(f"Root Mean Squared Error (RMSE): {overall_rmse_combined:.2f} %SoC ({overall_rmse_combined_0_1_scale:.6f} 0-1 Scale)")
            print(f"R-squared (R^2): {overall_r2_combined:.4f}")


            # F1-Score for Combined Model
            if config['ENABLE_F1_SCORE_CALCULATION']:
                print(f"--- Classification Metrics for LSTM + XGBoost Corrected ---")
                bins = config['F1_SCORE_SOC_BINS']['bins']
                labels = config['F1_SCORE_SOC_BINS']['labels']

                y_pred_final_clamped = np.clip(y_pred_final, bins[0], bins[-1])
                y_true_binned_final = pd.cut(y_true_clamped, bins=bins, labels=labels, include_lowest=True, right=True)
                y_pred_binned_final = pd.cut(y_pred_final_clamped, bins=bins, labels=labels, include_lowest=True, right=True)

                y_true_binned_array_final = y_true_binned_final.astype(str)
                y_pred_binned_array_final = y_pred_binned_final.astype(str)

                all_possible_labels_f1_combined = list(labels)
                if 'Unassigned' not in all_possible_labels_f1_combined:
                     if 'Unassigned' in np.unique(y_true_binned_array_final) or 'Unassigned' in np.unique(y_pred_binned_array_final):
                        all_possible_labels_f1_combined.append('Unassigned')

                true_bin_counts_final = pd.Series(y_true_binned_final).value_counts().sort_index()
                print(f"True samples per bin (for LSTM + XGBoost Corrected): {true_bin_counts_final.to_dict()}")

                try:
                    valid_indices_final = (y_true_binned_array_final != 'Unassigned') & (y_pred_binned_array_final != 'Unassigned')
                    y_true_f1_final = y_true_binned_array_final[valid_indices_final]
                    y_pred_f1_final = y_pred_binned_array_final[valid_indices_final]

                    f1_labels_actual_final = [label for label in labels if label in np.unique(y_true_f1_final)]

                    if len(f1_labels_actual_final) == 0:
                        print("Warning: No valid true labels for F1-score calculation (Combined) after cleaning. F1-score skipped.")
                        overall_f1_combined = np.nan
                    else:
                        overall_f1_combined = f1_score(y_true_f1_final, y_pred_f1_final, labels=f1_labels_actual_final, average='weighted', zero_division=0)
                        print(f"F1-Score (binned SoC, weighted average): {overall_f1_combined:.4f}")
                except ValueError as e:
                    print(f"Could not calculate F1-score for Combined Model: {e}. F1-score set to NaN.")
                    overall_f1_combined = np.nan

                cm_final = confusion_matrix(y_true_binned_array_final, y_pred_binned_array_final, labels=all_possible_labels_f1_combined)
                disp_final = ConfusionMatrixDisplay(confusion_matrix=cm_final, display_labels=all_possible_labels_f1_combined)
                
                plt.figure(figsize=(10, 8))
                disp_final.plot(cmap='Blues', values_format='d', ax=plt.gca())
                plt.title(f'Phase 3: Error Analysis - Confusion Matrix (LSTM + XGBoost Corrected Binned SoC)')
                plt.tight_layout()
                plot_filename = os.path.join(env_paths['temp_plots_dir_actual'], f'confusion_matrix_LSTM_XGBoost_Corrected.png')
                plt.savefig(plot_filename, dpi=300)
                plt.close()
                all_plots_info.append({'filepath': plot_filename, 'title': f'Confusion Matrix (LSTM + XGBoost Corrected Binned SoC)', 'section': 'Phase 3: Error Analysis - Classification Performance'})
                print(f"Confusion Matrix plot for LSTM + XGBoost Corrected saved.")

        except Exception as e:
            print(f"Error applying XGBoost correction: {e}. Using raw LSTM predictions for evaluation.")
            # Fallback to LSTM-only metrics if XGBoost application fails
            overall_mae_combined_0_1_scale = overall_mae_lstm_0_1_scale
            overall_rmse_combined_0_1_scale = overall_rmse_lstm_0_1_scale
            overall_r2_combined = overall_r2_lstm
            overall_f1_combined = overall_f1_lstm
            y_pred_final = np.clip(y_pred_lstm, 0, 100) # Ensure y_pred_final is still clipped LSTM output for plots
    else:
        print("\nXGBoost Corrector is DISABLED or model not found. Using raw LSTM predictions for combined evaluation.")
        overall_mae_combined_0_1_scale = overall_mae_lstm_0_1_scale
        overall_rmse_combined_0_1_scale = overall_rmse_lstm_0_1_scale
        overall_r2_combined = overall_r2_lstm
        overall_f1_combined = overall_f1_lstm
        y_pred_final = np.clip(y_pred_lstm, 0, 100) # Ensure y_pred_final is clipped LSTM output for plots


    print("\n--- Visualizing Model Performance Plots ---")

    # Plot 1: Actual vs. Predicted SoC on Test Set (Time Series)
    plt.figure(figsize=(15, 6))
    plt.plot(y_true_unscaled, label='Actual SoC', color='#06D6A0', linewidth=2)
    plt.plot(y_pred_lstm, label=f'Predicted SoC (LSTM Only)', color='#FF6B6B', linestyle='--', linewidth=2)
    if config['ENABLE_XGBOOST_CORRECTOR'] and config.get('xgb_corrector_model_instance') is not None:
        plt.plot(y_pred_final, label=f'Predicted SoC (LSTM + XGBoost Corrected)', color='#118AB2', linestyle='-.', linewidth=2)
    plt.title(f'Phase 3: Model Evaluation - Actual vs. Predicted SoC (Time Series)')
    plt.xlabel('Time Step (Sequential Index in Test Set)')
    plt.ylabel('SoC (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filename = os.path.join(env_paths['temp_plots_dir_actual'], f'actual_vs_predicted_timeseries_all_models.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    all_plots_info.append({'filepath': plot_filename, 'title': f'Actual vs. Predicted SoC (Time Series) - All Models', 'section': 'Phase 3: Model Evaluation - Core Performance'})


    # Plot 2: Actual vs. Predicted SoC (Scatter Plot)
    # Scatter for LSTM Only
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_unscaled, y_pred_lstm, alpha=0.5, color='#FF6B6B')
    plt.plot([min(y_true_unscaled), max(y_true_unscaled)], [min(y_true_unscaled), max(y_true_unscaled)], 'k--', lw=2, label='Perfect Prediction Line')
    plt.title(f'Phase 3: Model Evaluation - Actual vs. Predicted SoC (Scatter Plot) - LSTM Only')
    plt.xlabel('Actual SoC (%)')
    plt.ylabel('Predicted SoC (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_filename = os.path.join(env_paths['temp_plots_dir_actual'], f'actual_vs_predicted_scatter_lstm_only.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    all_plots_info.append({'filepath': plot_filename, 'title': f'Actual vs. Predicted SoC (Scatter Plot) - LSTM Only', 'section': 'Phase 3: Model Evaluation - Core Performance'})

    # Scatter for Final (LSTM + XGBoost)
    if config['ENABLE_XGBOOST_CORRECTOR'] and config.get('xgb_corrector_model_instance') is not None:
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true_unscaled, y_pred_final, alpha=0.5, color='#118AB2')
        plt.plot([min(y_true_unscaled), max(y_true_unscaled)], [min(y_true_unscaled), max(y_true_unscaled)], 'k--', lw=2, label='Perfect Prediction Line')
        plt.title(f'Phase 3: Model Evaluation - Actual vs. Predicted SoC (Scatter Plot) - LSTM + XGBoost Corrected')
        plt.xlabel('Actual SoC (%)')
        plt.ylabel('Predicted SoC (%)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plot_filename = os.path.join(env_paths['temp_plots_dir_actual'], f'actual_vs_predicted_scatter_combined.png')
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        all_plots_info.append({'filepath': plot_filename, 'title': f'Actual vs. Predicted SoC (Scatter Plot) - LSTM + XGBoost Corrected', 'section': 'Phase 3: Model Evaluation - Core Performance'})


    # Plot 3: Training History Plots (Loss & Metrics over Epochs)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Phase 2: Training Diagnostics - Model Loss (MSE) over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    # Calculate Accuracy from MAE
    # Corrected Accuracy Calculation: Unscale MAE from 0-1 range to 0-100 range first.
    # scaler_Y.data_max_ - scaler_Y.data_min_ gives the range of the original unscaled target (e.g., 101.34 - 0 = 101.34)
    unscaled_target_range_for_plot = scaler_Y.data_max_[0] - scaler_Y.data_min_[0]

    # Convert MAE from Keras's 0-1 scaled range to the 0-100 actual SoC scale
    train_mae_on_100_scale = np.array(history.history['mae']) * unscaled_target_range_for_plot
    val_mae_on_100_scale = np.array(history.history['val_mae']) * unscaled_target_range_for_plot

    # Calculate Accuracy: 100 - Unscaled MAE (ensure it doesn't go below 0%)
    train_accuracy_epochs = np.maximum(0, 100 - train_mae_on_100_scale)
    val_accuracy_epochs = np.maximum(0, 100 - val_mae_on_100_scale)

    plt.plot(train_accuracy_epochs, label='Train Accuracy')
    plt.plot(val_accuracy_epochs, label='Validation Accuracy')
    plt.title('Phase 2: Training Diagnostics - Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100) # Keep y-axis from 0 to 100
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filename = os.path.join(env_paths['temp_plots_dir_actual'], 'training_history.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    all_plots_info.append({'filepath': plot_filename, 'title': 'Model Training History', 'section': 'Phase 2: Training Diagnostics'})

    # Plot 4: Predicted SoC vs. Individual Input Features (Characteristic Curves)
    print("\n--- Visualizing Predicted SoC vs. Input Features (Characteristic Curves) ---")
    X_test_original_scale = scaler_X.inverse_transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)
    feature_indices = {name: i for i, name in enumerate(config['INPUT_FEATURES_ORDER'])}

    plot_features_soc_vs = [
        config['COLUMN_VOLTAGE'], config['COLUMN_CURRENT'], config['COLUMN_CELL_TEMPERATURE'],
        config['COLUMN_AMBIENT_TEMPERATURE'], config['COLUMN_CAPACITY_AH'], config['COLUMN_ENERGY_WH'],
        config['COLUMN_CYCLE_COUNT'],
        'Power_W', 'Voltage_Temp_Corrected', 'Delta_Cell_Temperature',
        'Rolling_Avg_Voltage_V', 'Rolling_Avg_Current_A', 'Cumulative_Capacity_Window',
        'Current_x_Cell_Temperature', 'Voltage_x_Cell_Temperature', 'Last_Current_Sign_Change', 'Time_Seconds'#, 'Delta_SoC'
    ]

    for feature_name in plot_features_soc_vs:
        if feature_name in feature_indices:
            feature_for_plot = X_test_original_scale[:, -1, feature_indices[feature_name]]
            # Plot for Final Model
            if config['ENABLE_XGBOOST_CORRECTOR'] and config.get('xgb_corrector_model_instance') is not None:
                plt.figure(figsize=(10, 6))
                plt.scatter(feature_for_plot, y_pred_final, alpha=0.5, color='#118AB2', label='Predicted SoC (Final)')
                plt.scatter(feature_for_plot, y_true_unscaled, alpha=0.3, color='#06D6A0', label='Actual SoC')
                plt.title(f'Phase 3: Model Evaluation - Predicted SoC vs. Actual {feature_name} (Final)')
                plt.xlabel(f'Actual {feature_name}')
                plt.ylabel('SoC (%)')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plot_filename = os.path.join(env_paths['temp_plots_dir_actual'], f'predicted_soc_vs_{feature_name}_final.png')
                plt.savefig(plot_filename, dpi=300)
                plt.close()
                all_plots_info.append({'filepath': plot_filename, 'title': f'Predicted SoC vs. {feature_name} (Final)', 'section': 'Phase 3: Model Evaluation - Learned Relationships'})
            
            # Plot for LSTM Only
            plt.figure(figsize=(10, 6))
            plt.scatter(feature_for_plot, y_pred_lstm, alpha=0.5, color='#FF6B6B', label='Predicted SoC (LSTM Only)')
            plt.scatter(feature_for_plot, y_true_unscaled, alpha=0.3, color='#06D6A0', label='Actual SoC')
            plt.title(f'Phase 3: Model Evaluation - Predicted SoC vs. Actual {feature_name} (LSTM Only)')
            plt.xlabel(f'Actual {feature_name}')
            plt.ylabel('SoC (%)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plot_filename = os.path.join(env_paths['temp_plots_dir_actual'], f'predicted_soc_vs_{feature_name}_lstm_only.png')
            plt.savefig(plot_filename, dpi=300)
            plt.close()
            all_plots_info.append({'filepath': plot_filename, 'title': f'Predicted SoC vs. {feature_name} (LSTM Only)', 'section': 'Phase 3: Model Evaluation - Learned Relationships'})
        else:
            print(f"Warning: Feature '{feature_name}' not found in INPUT_FEATURES_ORDER for 'Predicted SoC vs. Feature' plot.")
        

    # Plot 5: Prediction Error (Residuals) vs. Individual Input Features
    print("\n--- Visualizing Prediction Error vs. Input Features ---")
    errors_final = y_true_unscaled - y_pred_final
    errors_lstm_only = y_true_unscaled - y_pred_lstm

    for feature_name in plot_features_soc_vs:
        if feature_name in feature_indices:
            feature_for_plot = X_test_original_scale[:, -1, feature_indices[feature_name]]
            # Plot for Final Model
            if config['ENABLE_XGBOOST_CORRECTOR'] and config.get('xgb_corrector_model_instance') is not None:
                plt.figure(figsize=(10, 6))
                plt.scatter(feature_for_plot, errors_final, alpha=0.5, color='#118AB2')
                plt.axhline(0, color='red', linestyle='--', linewidth=1, label='Zero Error')
                plt.title(f'Phase 3: Error Analysis - Prediction Error vs. Actual {feature_name} (Final)')
                plt.xlabel(f'Actual {feature_name}')
                plt.ylabel('Prediction Error (Actual - Predicted) %SoC')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plot_filename = os.path.join(env_paths['temp_plots_dir_actual'], f'error_vs_{feature_name}_final.png')
                plt.savefig(plot_filename, dpi=300)
                plt.close()
                all_plots_info.append({'filepath': plot_filename, 'title': f'Prediction Error vs. {feature_name} (Final)', 'section': 'Phase 3: Error Analysis - Residuals'})
            
            # Plot for LSTM Only
            plt.figure(figsize=(10, 6))
            plt.scatter(feature_for_plot, errors_lstm_only, alpha=0.5, color='#FF6B6B')
            plt.axhline(0, color='blue', linestyle=':', linewidth=1, label='Zero Error')
            plt.title(f'Phase 3: Error Analysis - Prediction Error vs. Actual {feature_name} (LSTM Only)')
            plt.xlabel(f'Actual {feature_name}')
            plt.ylabel('Prediction Error (Actual - Predicted) %SoC')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plot_filename = os.path.join(env_paths['temp_plots_dir_actual'], f'error_vs_{feature_name}_lstm_only.png')
            plt.savefig(plot_filename, dpi=300)
            plt.close()
            all_plots_info.append({'filepath': plot_filename, 'title': f'Prediction Error vs. {feature_name} (LSTM Only)', 'section': 'Phase 3: Error Analysis - Residuals'})

    # Plot 6: Error Distribution (Histogram of Residuals)
    print("\n--- Visualizing Error Distribution (Histogram of Residuals) ---")
    
    # Plot for Final Model
    if config['ENABLE_XGBOOST_CORRECTOR'] and config.get('xgb_corrector_model_instance') is not None:
        plt.figure(figsize=(8, 5))
        sns.histplot(errors_final, kde=True, bins=50, color='#118AB2', label='Final Model Errors')
        plt.axvline(errors_final.mean(), color='red', linestyle='--', label=f'Mean Final Error: {errors_final.mean():.2f}')
        plt.title('Phase 3: Error Analysis - Distribution of Prediction Errors (Final Model)')
        plt.xlabel('Prediction Error (Actual - Predicted) %SoC')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_filename = os.path.join(env_paths['temp_plots_dir_actual'], 'error_distribution_final.png')
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        all_plots_info.append({'filepath': plot_filename, 'title': 'Error Distribution (Final Model)', 'section': 'Phase 3: Error Analysis - Residuals'})

    # Plot for LSTM Only
    plt.figure(figsize=(8, 5))
    sns.histplot(errors_lstm_only, kde=True, bins=50, color='#FF6B6B', label='LSTM Only Errors')
    plt.axvline(errors_lstm_only.mean(), color='blue', linestyle=':', label=f'Mean LSTM Only Error: {errors_lstm_only.mean():.2f}')
    plt.title('Phase 3: Error Analysis - Distribution of Prediction Errors (LSTM Only)')
    plt.xlabel('Prediction Error (Actual - Predicted) %SoC')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filename = os.path.join(env_paths['temp_plots_dir_actual'], 'error_distribution_lstm_only.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    all_plots_info.append({'filepath': plot_filename, 'title': 'Error Distribution (LSTM Only)', 'section': 'Phase 3: Error Analysis - Residuals'})

    # Store overall and single prediction metrics in report_meta for PDF generation
    env_paths['report_meta']['Overall_MAE'] = overall_mae_combined_0_1_scale
    env_paths['report_meta']['Overall_RMSE'] = overall_rmse_combined_0_1_scale
    env_paths['report_meta']['Overall_R2'] = overall_r2_combined
    env_paths['report_meta']['Overall_F1'] = overall_f1_combined # F1 is already 0-1
    # Return the metrics for the combined model for the report, or LSTM-only if XGBoost was disabled/failed
    return overall_mae_combined, overall_rmse_combined, overall_r2_combined, overall_f1_combined

def demonstrate_rolling_prediction(df_original, model, scaler_X, scaler_Y, config, all_plots_info):
    """
    Phase 4: Demonstrates rolling prediction over a sequence of new data points.
    Generates a plot showing actual vs. predicted SoC over time for this rolling window.
    Does NOT return single-point metrics, as it's a series demonstration.
    """
    print(f"\n--- Phase 4: Demonstrating Rolling Prediction ({config['NUM_ROLLING_PREDICTION_STEPS']} steps) ---")

    num_prediction_steps = config['NUM_ROLLING_PREDICTION_STEPS']
    sequence_length = config['SEQUENCE_LENGTH']

    # It's best to use a segment from the actual dataset for a realistic rolling demo.
    # Ensure this segment is NOT part of your X_test, or is a completely new segment.
    # For this demo, we'll take a segment from the end of the original_df_from_load
    # that hasn't been used in training/validation/test.
    # This requires careful data management. For simplicity, we'll take a tail segment,
    # assuming it's beyond the train/val/test split if the full data was used for split.
    
    # Calculate a safe starting index for the rolling prediction segment
    # We need at least (sequence_length + num_prediction_steps) data points
    # to perform all rolling predictions.
    required_data_points = sequence_length + num_prediction_steps
    
    if len(df_original) < required_data_points:
        print(f"Warning: Not enough data in df_original ({len(df_original)} rows) for {num_prediction_steps} rolling predictions with sequence_length {sequence_length}. "
              f"Need at least {required_data_points} rows. Skipping rolling prediction.")
        return np.nan, np.nan, np.nan # Indicate no data generated

    # Take the last 'required_data_points' from the original DataFrame
    # This assumes these points are outside your main train/val/test split if split chronologically.
    # In a real scenario, you'd feed truly new data.
    rolling_segment_df = df_original.tail(required_data_points).copy().reset_index(drop=True)

    all_predicted_socs_100_percent = []
    all_actual_socs_100_percent = []

    # Iterate to make multiple predictions
    # The loop runs for num_prediction_steps, each time predicting one future point
    # based on the preceding 'sequence_length' points.
    for i in range(num_prediction_steps):
        # The current_window needs to be sequence_length points long for LSTM input
        # and it needs to be derived from data that includes enough history for feature engineering
        # (e.g., rolling averages, diffs).
        # So, we need to extract a window of (sequence_length + max_feature_history)
        # For simplicity, we'll assume current_segment_for_pred handles this in preprocess_new_raw_data_for_prediction.
        
        # Extract the segment that will form the input sequence for the current prediction
        # This segment must be long enough to produce one LSTM input sequence after feature engineering.
        # The 'preprocess_new_raw_data_for_prediction' function expects a segment that can yield
        # at least 'sequence_length' rows after its internal dropna and tail operations.
        # So, we pass a slice of data that includes the necessary history.
        
        # The input window for prediction is from index 'i' up to 'i + sequence_length' (inclusive of history)
        # The actual SoC for this prediction is at index 'i + sequence_length'
        
        current_segment_for_preprocess = rolling_segment_df.iloc[i : i + sequence_length + 1] # +1 to get the target's actual SoC
        
        if len(current_segment_for_preprocess) < sequence_length + 1:
            print(f"Warning: Insufficient data in rolling segment for prediction step {i}. Stopping rolling prediction.")
            break

        try:
            # Preprocess the current segment to get the LSTM input (shape 1, SEQUENCE_LENGTH, num_features)
            lstm_input_for_prediction, _ = preprocess_new_raw_data_for_prediction(
                current_segment_for_preprocess, scaler_X, config
            )

            # Get LSTM base prediction (scaled 0-1)
            predicted_soc_lstm_scaled = model.predict(lstm_input_for_prediction)
            predicted_soc_lstm_100_percent = scaler_Y.inverse_transform(predicted_soc_lstm_scaled)[0][0]

            predicted_soc_final_100_percent = predicted_soc_lstm_100_percent # Start with LSTM prediction

            # Apply XGBoost correction if enabled
            if config['ENABLE_XGBOOST_CORRECTOR'] and config.get('xgb_corrector_model_instance') is not None:
                xgb_corrector_model = config.get('xgb_corrector_model_instance')
                if xgb_corrector_model is None: # Fallback if not passed in config or needs reloading
                    xgb_corrector_model = xgb.XGBRegressor()
                    xgb_corrector_model.load_model(config['XGBOOST_MODEL_SAVE_PATH'])
                
                # Extract XGBoost features from the last timestep of the processed segment
                # This segment is what was actually fed to LSTM after preprocessing.
                # The preprocess_new_raw_data_for_prediction returns df_lstm_input_segment as its second output
                # which is the processed tail. We need to capture that.
                # Let's re-call preprocess_new_raw_data_for_prediction to get df_lstm_input_segment
                _, df_processed_segment_for_lstm_xgb = preprocess_new_raw_data_for_prediction(
                    current_segment_for_preprocess, scaler_X, config
                )

                last_timestep_features_xgb = df_processed_segment_for_lstm_xgb.tail(1)[config['XGBOOST_FEATURES']]
                last_timestep_features_xgb.columns = [re.sub(r'[^a-zA-Z0-9_]', '', col) for col in last_timestep_features_xgb.columns]

                xgb_prediction = xgb_corrector_model.predict(last_timestep_features_xgb)[0]
                
                if config.get('ENABLE_RESIDUAL_XGBOOST', False):
                    predicted_soc_final_100_percent = predicted_soc_lstm_100_percent + xgb_prediction
                else:
                    predicted_soc_final_100_percent = xgb_prediction
                
                predicted_soc_final_100_percent = np.clip(predicted_soc_final_100_percent, 0, 100) # Clip final prediction

            # Store predictions and actuals
            all_predicted_socs_100_percent.append(predicted_soc_final_100_percent)
            
            # The actual SoC for this prediction is the SoC at the end of the input sequence
            actual_soc_for_this_step = current_segment_for_preprocess[config['COLUMN_SOC']].iloc[-1]
            all_actual_socs_100_percent.append(actual_soc_for_this_step)

        except Exception as e:
            print(f"Error during rolling prediction step {i}: {e}. Stopping rolling prediction.")
            # Append NaN to maintain array length for plotting, if desired
            all_predicted_socs_100_percent.append(np.nan)
            all_actual_socs_100_percent.append(np.nan)
            break # Stop if an error occurs

    # --- Plot the rolling predictions ---
    if len(all_predicted_socs_100_percent) > 0:
        plt.figure(figsize=(15, 6))
        plt.plot(all_actual_socs_100_percent, label='Actual SoC', color='#06D6A0', linewidth=2)
        plt.plot(all_predicted_socs_100_percent, label='Predicted SoC (Rolling)', color='#118AB2', linestyle='-', linewidth=2)
        plt.title(f'Phase 4: Rolling Prediction Demonstration ({len(all_predicted_socs_100_percent)} steps)')
        plt.xlabel('Time Step (Relative Index in Rolling Window)')
        plt.ylabel('SoC (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_filename = os.path.join(config['TEMP_PLOTS_SUBDIR_NAME'], 'rolling_prediction_demo.png')
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        all_plots_info.append({'filepath': plot_filename, 'title': f'Rolling Prediction Demonstration', 'section': 'Phase 4: Operational Demonstration'})
        print(f"\nCompleted {len(all_predicted_socs_100_percent)} rolling predictions and saved plot.")
    else:
        print("\nNo rolling predictions were generated for plotting.")

    # For the report_meta, we'll set single point metrics to N/A or derive an average from rolling
    # For now, let's return N/A for single point metrics as this is a series demo.
    # If you need an average MAE for the rolling prediction, you can calculate it here.
    # e.g., rolling_mae = mean_absolute_error(all_actual_socs_100_percent, all_predicted_socs_100_percent)
    
    # Return NaNs for the single-point metrics as this function now does rolling prediction
    return np.nan, np.nan, np.nan

def demonstrate_prediction(df_original, model, scaler_X, scaler_Y, config):
    """
    Phase 4: Simulates new data and demonstrates prediction using the loaded model.
    Returns single_abs_error, single_actual_soc, single_predicted_soc for report.
    """
    print("\n--- Phase 4: Demonstrating Prediction for a New Module (Brain) ---")
    print(f"\nSimulating a new raw data segment of {config['SEQUENCE_LENGTH'] + 1} time steps for prediction...")

    # Use actual time step from original data for simulation if possible
    time_step_interval = df_original[config['COLUMN_TIME']].iloc[1] - df_original[config['COLUMN_TIME']].iloc[0] if len(df_original[config['COLUMN_TIME']]) > 1 else 1.0

    # Create a simulated data segment. Make it slightly more realistic by gradually decreasing SoC
    # and ensuring some current/voltage changes.
    simulated_new_data = {
        config['COLUMN_TIME']: np.arange(1, config['SEQUENCE_LENGTH'] + 2) * time_step_interval,
        config['COLUMN_VOLTAGE']: np.linspace(3.75, 3.60, config['SEQUENCE_LENGTH'] + 1) + np.random.rand(config['SEQUENCE_LENGTH'] + 1) * 0.005,
        config['COLUMN_CURRENT']: np.linspace(-5, -10, config['SEQUENCE_LENGTH'] + 1) + np.random.rand(config['SEQUENCE_LENGTH'] + 1) * 0.1,
        config['COLUMN_CELL_TEMPERATURE']: np.full(config['SEQUENCE_LENGTH'] + 1, 25.0) + np.random.rand(config['SEQUENCE_LENGTH'] + 1) * 0.2,
        config['COLUMN_AMBIENT_TEMPERATURE']: np.full(config['SEQUENCE_LENGTH'] + 1, 23.0) + np.random.rand(config['SEQUENCE_LENGTH'] + 1) * 0.1,
        config['COLUMN_CAPACITY_AH']: np.full(config['SEQUENCE_LENGTH'] + 1, 2.0) - np.arange(0, config['SEQUENCE_LENGTH'] + 1) * 0.0001,
        config['COLUMN_ENERGY_WH']: np.full(config['SEQUENCE_LENGTH'] + 1, 10.0) - np.arange(0, config['SEQUENCE_LENGTH'] + 1) * 0.01, # Energy decreasing with discharge
        config['COLUMN_CYCLE_COUNT']: np.full(config['SEQUENCE_LENGTH'] + 1, 50) + np.arange(0, config['SEQUENCE_LENGTH'] + 1) * 0.01,
        config['COLUMN_SOC']: np.linspace(80, 70, config['SEQUENCE_LENGTH'] + 1) + np.random.rand(config['SEQUENCE_LENGTH'] + 1) * 0.05 # Realistic SoC drop
    }
    new_raw_df_segment_for_pred = pd.DataFrame(simulated_new_data)

    print("\n--- Raw Simulated Input Data Segment (Last 5 Timesteps) ---") #
    print(new_raw_df_segment_for_pred.tail(5).to_string()) # Print tail of raw data for clarity

    try:
        print("\n--- Preprocessing for Prediction ---") #
        # Preprocess the new data using the general preprocessing function
        lstm_input_for_prediction, df_processed_segment_for_lstm = preprocess_new_raw_data_for_prediction(
            new_raw_df_segment_for_pred, scaler_X, config
        )
        
        print("\nDataFrame after Feature Engineering (Last 60 Timesteps - displaying tail):") #
        # Display tail of the processed DataFrame *before* scaling and reshaping for LSTM
        # Show a reasonable number of rows, e.g., last 5-10
        print(df_processed_segment_for_lstm.tail(min(10, len(df_processed_segment_for_lstm))).to_string())

        print(f"\nFeatures selected for LSTM input (in order): {config['INPUT_FEATURES_ORDER']}") #
        
        print("\n--- Scaled LSTM Input for Prediction ---") #
        print(f"Shape of LSTM input: {lstm_input_for_prediction.shape}")
        # Show a sample of the scaled input for the last sequence timestep (first few features)
        print("Sample of Scaled Input (last sequence timestep, first 5 features):")
        print(f"[[{', '.join([f'{x:.4f}' for x in lstm_input_for_prediction[0, -1, :5]])} ...]]")


        print("\n--- Prediction Results ---") #
        # Get the actual SoC value for the timestep being predicted from the simulated data
        actual_soc_for_prediction_100_percent = new_raw_df_segment_for_pred[config['COLUMN_SOC']].iloc[-1]
        print(f"Actual SoC for the predicted timestep: {actual_soc_for_prediction_100_percent:.2f}%")

        # Get LSTM base prediction
        predicted_soc_lstm_scaled = model.predict(lstm_input_for_prediction)
        # Print the raw scaled prediction for debugging
        print(f"LSTM-only Predicted SoC (scaled): {predicted_soc_lstm_scaled[0][0]:.6f}") # Access the scalar value
        predicted_soc_lstm_100_percent = scaler_Y.inverse_transform(predicted_soc_lstm_scaled)[0][0]
        print(f"LSTM-only Predicted SoC (unscaled): {predicted_soc_lstm_100_percent:.2f}%")

        predicted_soc_final_100_percent = predicted_soc_lstm_100_percent # Start with LSTM prediction

        # Apply XGBoost correction if enabled
        if config['ENABLE_XGBOOST_CORRECTOR'] and os.path.exists(config['XGBOOST_MODEL_SAVE_PATH']):
            try:
                # Load XGBoost model if not already loaded (should be in config['xgb_corrector_model_instance'])
                xgb_corrector_model = config.get('xgb_corrector_model_instance')
                if xgb_corrector_model is None: # Fallback if not passed in config or needs reloading
                    xgb_corrector_model = xgb.XGBRegressor()
                    xgb_corrector_model.load_model(config['XGBOOST_MODEL_SAVE_PATH'])
                
                print("\n--- XGBoost Correction (If Enabled) ---") #
                
                # Extract XGBoost features from the last timestep of the processed simulated segment
                last_timestep_features = df_processed_segment_for_lstm.tail(1)[config['XGBOOST_FEATURES']]
                last_timestep_features.columns = [re.sub(r'[^a-zA-Z0-9_]', '', col) for col in last_timestep_features.columns] # Clean column names
                
                print("Features extracted for XGBoost (from last timestep of sequence):") #
                print(last_timestep_features.to_string())

                xgb_prediction = xgb_corrector_model.predict(last_timestep_features)[0]
                
                if config.get('ENABLE_RESIDUAL_XGBOOST', False):
                    xgb_correction_value = xgb_prediction # This is the absolute residual predicted by XGBoost
                    predicted_soc_final_100_percent = predicted_soc_lstm_100_percent + xgb_correction_value
                    # Clip the final prediction to ensure it's within valid SoC range
                    predicted_soc_final_100_percent = np.clip(predicted_soc_final_100_percent, 0, 100)
                    print(f"\nXGBoost predicted correction (residual): {xgb_correction_value:.2f}%") #
                    print(f"Final Predicted SoC (LSTM + XGBoost): {predicted_soc_lstm_100_percent:.2f}% + {xgb_correction_value:.2f}% = {predicted_soc_final_100_percent:.2f}%")
                else:
                    # If XGBoost directly predicts SoC, its prediction is the final one
                    predicted_soc_final_100_percent = xgb_prediction
                    predicted_soc_final_100_percent = np.clip(predicted_soc_final_100_percent, 0, 100) # Clip direct XGBoost prediction
                    print(f"\nXGBoost predicted SoC directly: {xgb_prediction:.2f}%")
                    print(f"Final Predicted SoC (XGBoost Direct): {predicted_soc_final_100_percent:.2f}%")

            except Exception as e:
                print(f"Error applying XGBoost correction in demonstration: {e}. Using LSTM-only prediction.")
                predicted_soc_final_100_percent = np.clip(predicted_soc_lstm_100_percent, 0, 100) # Fallback to clipped LSTM-only if error
        else:
            print("\nXGBoost Corrector is DISABLED or model not found. Using LSTM-only prediction for final result.")
            predicted_soc_final_100_percent = np.clip(predicted_soc_lstm_100_percent, 0, 100) # Ensure clipped LSTM-only prediction

        # --- Single Prediction Error Metrics ---
        single_prediction_error_100_percent = actual_soc_for_prediction_100_percent - predicted_soc_final_100_percent
        single_abs_error_100_percent = abs(single_prediction_error_100_percent)
        single_squared_error_100_percent = single_prediction_error_100_percent**2

        # Calculate the range of the original unscaled target data
        unscaled_target_range = scaler_Y.data_max_[0] - scaler_Y.data_min_[0]
        if unscaled_target_range == 0:
            unscaled_target_range = 100.0 # Fallback

        # Convert all single point metrics to 0-1 scale for return/storage
        single_abs_error_0_1_scale = single_abs_error_100_percent / unscaled_target_range
        single_actual_soc_0_1_scale = actual_soc_for_prediction_100_percent / unscaled_target_range
        single_prediction_error_0_1_scale = single_prediction_error_100_percent / unscaled_target_range # Fix: Ensure this is defined
        single_predicted_soc_0_1_scale = predicted_soc_final_100_percent / unscaled_target_range
        
        print("\n--- Single Prediction Error Metrics ---")
        print(f"Error: {single_prediction_error_100_percent:.2f}%  (Actual - Predicted) ({single_prediction_error_0_1_scale:.6f} 0-1 Scale)")
        print(f"Absolute Error: {single_abs_error_100_percent:.2f}% ({single_abs_error_0_1_scale:.6f} 0-1 Scale)")
        print(f"Squared Error: {single_squared_error_100_percent:.4f} (SoC%)^2")
        print(f"MAE for this point: {single_abs_error_100_percent:.2f}%")
        print(f"MSE for this point: {single_squared_error_100_percent:.4f} (SoC%)^2")
        print(f"RMSE for this point: {np.sqrt(single_squared_error_100_percent):.2f}%")

        print(f"\nPredicted SoC for the next time step: {predicted_soc_final_100_percent:.2f}% ({single_predicted_soc_0_1_scale:.6f} 0-1 Scale)")

        return single_abs_error_0_1_scale, single_actual_soc_0_1_scale, single_predicted_soc_0_1_scale

    except ValueError as ve:
        print(f"Error during prediction preparation: {ve}")
        print("Please ensure the simulated or actual new data segment is long enough and properly formatted.")
        return np.nan, np.nan, np.nan # Return NaN if error occurs
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        print(f"Error details: {e}")
        return np.nan, np.nan, np.nan # Return NaN if error occurs


def cleanup_temp_files(temp_plots_dir_actual, original_stdout_stream):
    """
    Cleans up temporary plot directories after report generation.
    """
    if os.path.exists(temp_plots_dir_actual):
        try:
            shutil.rmtree(temp_plots_dir_actual)
            print(f"Cleaned up temporary plots directory: {temp_plots_dir_actual}")
        except Exception as e:
            print(f"Error cleaning up temporary plots directory: {e}")
            sys.stdout = original_stdout_stream


# --- Main Execution Flow ---
if __name__ == "__main__":
    # Set global random seeds for full reproducibility
    random_seed = CONFIG['RANDOM_SEED_SPLIT']
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    # For CUDA/cuDNN reproducibility if using GPU
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # 1. Setup Environment and Parse Arguments
    env_paths = setup_environment_and_args(CONFIG)
    captured_stdout_buffer = sys.stdout.secondary_stream

    print("--- Starting SoC Prediction Pipeline ---")

    # 2. Load and Preprocess Data
    original_df_from_load = None
    try:
        original_df_from_load = load_and_preprocess_data(env_paths['dataset_file_path'], CONFIG, env_paths)
    except SystemExit:
        cleanup_temp_files(env_paths['temp_plots_dir_actual'], env_paths['original_stdout_stream'])
        sys.exit(1)

    # 3. Prepare Data for LSTM (Scaling, Sequencing, Splitting)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_Y = \
        prepare_for_lstm(original_df_from_load, CONFIG, env_paths)

    # 4. Build and Train Model (with GridSearchCV)
    model, history = build_and_train_model(X_train, y_train, X_val, y_val, CONFIG, scaler_X, scaler_Y)

    # 5. Evaluate Model and Generate Plots
    overall_mae_0_1_scale, overall_rmse_0_1_scale, overall_r2, overall_f1 = \
        evaluate_model_and_generate_plots(model, history, X_test, y_test, scaler_X, scaler_Y, CONFIG, env_paths)

    # 6. Demonstrate Prediction (using the original dataframe for time step consistency)
    #single_abs_error_0_1_scale, single_actual_soc_0_1_scale, single_predicted_soc_0_1_scale = \
    #    demonstrate_prediction(original_df_from_load, model, scaler_X, scaler_Y, CONFIG)

    single_abs_error_0_1_scale = np.nan # Initialize to NaN
    single_actual_soc_0_1_scale = np.nan
    single_predicted_soc_0_1_scale = np.nan

    # Call the new rolling prediction function
    # It will generate its own plot and print its progress
    demonstrate_rolling_prediction(original_df_from_load, model, scaler_X, scaler_Y, CONFIG, all_plots_info)

    print("\n--- Full Pipeline Execution Complete ---")
    print("Trained model saved to:", CONFIG['MODEL_SAVE_PATH'])
    print("Scalers saved to:", CONFIG['SCALER_X_PATH'], "and", CONFIG['SCALER_Y_PATH'])
    if CONFIG['ENABLE_XGBOOST_CORRECTOR']:
        print("XGBoost corrector model saved to:", CONFIG['XGBOOST_MODEL_SAVE_PATH'])


    # Store overall and single prediction metrics in report_meta for PDF generation
    env_paths['report_meta']['Overall_MAE'] = overall_mae_0_1_scale
    env_paths['report_meta']['Overall_RMSE'] = overall_rmse_0_1_scale
    env_paths['report_meta']['Overall_R2'] = overall_r2
    env_paths['report_meta']['Overall_F1'] = overall_f1
    env_paths['report_meta']['Single_Pred_Abs_Error'] = single_abs_error_0_1_scale
    env_paths['report_meta']['Single_Pred_Actual_SoC'] = single_actual_soc_0_1_scale
    env_paths['report_meta']['Single_Pred_Predicted_SoC'] = single_predicted_soc_0_1_scale


    # Final steps for PDF generation and cleanup
    captured_log_content = captured_stdout_buffer.getvalue()
    sys.stdout = env_paths['original_stdout_stream'] # Restore stdout to console
    
    end_time_global = time.time()
    print("\n==================================================")
    print(f" Total execution time: {(end_time_global - start_time_global)/3600:.2f} hours")
    print("==================================================")

    print(f"\nGenerating PDF report: {env_paths['report_pdf_path_final']}")
    try:
        generate_pdf_report(captured_log_content, all_plots_info, env_paths['report_meta'],
                            env_paths['report_pdf_path_final'])
    except Exception as e:
        print(f"Failed to generate PDF report: {e}")
        print(f"Error details: {e}")

    cleanup_temp_files(env_paths['temp_plots_dir_actual'], env_paths['original_stdout_stream'])
