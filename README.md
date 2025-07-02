🔋 Time Series Li-ion Battery State of Charge (SoC) Prediction
This project delivers a robust machine learning pipeline for accurate State of Charge (SoC) estimation in Li-ion batteries, developed as a comprehensive university project within the 'Project Lab Embedded Systems' course. It is crucial for optimizing performance, extending lifespan, and ensuring safety in various applications.

Leveraging advanced deep learning techniques, this solution addresses the complex, non-linear dynamics of battery behavior under real-world operating conditions.

✨ Key Features & Technologies:
Advanced Data Engineering: Ingestion and fusion of fragmented time-series sensor data (69 CSV files) into a unified master dataset.

Domain-Informed Feature Engineering: Development of 19+ critical features, including:

Estimated Internal Resistance: Crucial for improving accuracy in flat OCV regions.

Temperature Compensation: Normalizing voltage for thermal effects.

Hysteresis Tracking: Accounting for charge/discharge path dependency.

Coulomb Counting: Cumulative capacity tracking.

Deep Learning Model: Implementation of a Long Short-Term Memory (LSTM) neural network for capturing complex temporal dependencies.

Hybrid Ensemble Approach: Integration of an XGBoost Regressor as a residual corrector to enhance LSTM predictions.

Robust Validation: Utilizes chronological TimeSeriesSplit cross-validation to ensure unbiased performance evaluation.

Hyperparameter Optimization: Employs GridSearchCV for systematic tuning of model parameters.

Automated Reporting: Generates comprehensive PDF diagnostic reports for every run, ensuring reproducibility and transparent analysis.

Scalable Architecture: Modular Python codebase designed for maintainability and future enhancements.

📊 Performance Highlights:
The hybrid LSTM-XGBoost model achieved significant accuracy improvements on unseen test data:

Reduced Mean Absolute Error (MAE): Achieved 3.46% SoC (from 4.62% for LSTM-only).

High R-squared (R 
2
 ): 0.9322, indicating excellent variance explanation.

Strong F1-Score: 0.9588 for binned SoC classification.

🛠️ Skills Demonstrated:
Machine Learning: Deep Learning (LSTM), Ensemble Methods (XGBoost), Regression, Classification.

Data Science: Data Acquisition, Data Cleaning (Outlier Detection - IQR), Feature Engineering, Time-Series Analysis, Data Normalization (MinMaxScaler), Exploratory Data Analysis (EDA), Statistical Analysis.

Model Optimization: Hyperparameter Tuning (GridSearchCV), Cross-Validation (TimeSeriesSplit), Regularization (Dropout, L2), Early Stopping.

Software Engineering: Modular Code Design, Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn, FPDF.

Problem Solving: Addressing challenges like data fragmentation, non-linearity, hysteresis, and model collapse.

🚀 Get Started:
To explore the code, replicate the results, or run your own experiments:

Clone this repository.

Install the required Python packages (pip install -r requirements.txt).

Place your FINAL_MASTER_DATASET_With_SOC.csv file in the designated data/ directory.

Run the main script: python scripts/HYBRID_Train.py data/FINAL_MASTER_DATASET_With_SOC.csv
