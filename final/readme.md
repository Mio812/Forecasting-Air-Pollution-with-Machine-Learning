# Air Quality Analysis & Prediction Project

This project implements a complete machine learning workflow based on the UCI Air Quality dataset. It includes data cleaning, Exploratory Data Analysis (EDA), Anomaly Detection, Time Series Regression (predicting pollutant concentrations), and Time Series Classification (predicting pollution levels).

## 1. Environment Requirements

This project is developed using **Python 3.8+**. Please ensure the following core libraries are installed:

*   **Data Manipulation**: `numpy`, `pandas`
*   **Visualization**: `matplotlib`, `seaborn`
*   **Machine Learning**: `scikit-learn`, `xgboost`
*   **Statistics**: `statsmodels`

### Installation
You can create a `requirements.txt` file with the following content:

```txt
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
statsmodels
```

Then, run the following command to install the dependencies:
```bash
pip install -r requirements.txt
```

## 2. File Structure

Before running the code, please ensure your project directory is structured as follows. **Specifically, the `data` folder and the raw CSV file must exist.**

```text
Project_Root/
│
├── data/
│   └── AirQualityUCI.csv       # [REQUIRED] Raw dataset from UCI (CSV format)
│
├── final.py                    # Main script
├── cleaned_air_quality.csv     # Backup of cleaned data (generated after running)
├── README.md                   # Project documentation
│
└── output/                     # [Auto-Generated] Directory for all results
    ├── anomalies_detected.csv  # Detected anomalies list
    │
    ├── regression/             # Regression task outputs
    │   ├── data/               # Summary metrics CSV (RMSE, R2, etc.)
    │   └── plots/
    │       ├── eda/            # EDA plots (Time series, Correlations, etc.)
    │       ├── diagnostics/    # Model diagnostics (Predicted vs Actual, Residuals)
    │       ├── feature_importance/ # Feature importance bar charts
    │       └── regression_rmse_comparison.png # RMSE comparison across models
    │
    └── classification/         # Classification task outputs
        ├── data/               # Summary metrics CSV (Accuracy, F1)
        └── plots/
            ├── confusion_matrices/ # Confusion Matrices
            ├── roc_curves/         # Multi-class ROC Curves
            ├── pr_curves/          # Precision-Recall Curves
            ├── feature_importance/ # Classification feature importance
            ├── logreg_coefficients/ # Logistic Regression coefficients visualization
            └── summary_bars/       # Bar charts comparing Accuracy/F1
```

### Data Acquisition
If `data/AirQualityUCI.csv` is missing, please download it from the [UCI Machine Learning Repository - Air Quality Data Set](https://archive.ics.uci.edu/ml/datasets/Air+Quality). Ensure the file is named `AirQualityUCI.csv`.

## 3. How to Run

1.  **Prepare Data**: Ensure `data/AirQualityUCI.csv` is placed correctly.
2.  **Execute Script**:
    Run the following command in your terminal:

    ```bash
    python final.py
    ```

3.  **Process**:
    *   The script will automatically clean the data.
    *   Perform EDA and save the plots.
    *   Run **Regression** tasks on 5 major pollutants (`CO`, `NMHC`, `C6H6`, `NOx`, `NO2`) for horizons of 1h, 6h, 12h, and 24h.
    *   Run **Classification** tasks on `CO(GT)` levels (Low/Medium/High).
    *   Upon completion, all results will be saved in the `output/` directory.

## 4. Output Explanation

### 4.1 Data Preprocessing & EDA (`output/regression/plots/eda/`)
*   **Strategy**: The code defaults to `ffill` (Forward Fill) to handle missing values and aligns data by timestamp.
*   **Plots**:
    *   `eda_correlation_matrix.png`: Heatmap showing correlations between pollutants and meteorological features (T, RH).
    *   `eda_intraday_pattern_*.png`: Average concentration trends over a 24-hour cycle.
    *   `eda_distributions.png`: Histograms of data distribution.

### 4.2 Regression Task (`output/regression/`)
The goal is to predict the exact future numerical value of pollutants.
*   **Models**: Linear Regression, Random Forest, XGBoost, Naive Baseline.
*   **Horizons**: Predicting 1, 6, 12, and 24 hours ahead.
*   **Key Files**:
    *   `data/regression_summary.csv`: Contains RMSE, MAE, and R2 scores for all models and horizons.
    *   `plots/diagnostics/`: Contains `Predicted vs Actual` scatter plots (closer to the red line is better) and Residual plots.
    *   `plots/feature_importance/`: Shows which features (e.g., past concentrations, temperature, hour of day) influenced the prediction most.

### 4.3 Classification Task (`output/classification/`)
The goal is to classify `CO(GT)` concentrations into three severity levels:
- **Low**: < 1.5
- **Medium**: 1.5 - 2.5
- **High**: > 2.5

*   **Models**: Logistic Regression, Random Forest, XGBoost.
*   **Key Files**:
    *   `plots/confusion_matrices/`: Visualizes where the model confused one class for another.
    *   `plots/roc_curves/`: Multi-class ROC curves; AUC closer to 1.0 indicates better performance.
    *   `plots/pr_curves/`: Precision-Recall curves, useful for evaluating performance on imbalanced classes.
    *   `plots/logreg_coefficients/`: Visualizes how specific features positively or negatively affect the probability of a specific class (Logistic Regression only).

### 4.4 Anomaly Detection
*   `anomalies_detected.csv`: A list of data points identified as anomalies using the Isolation Forest algorithm.

---
**Note**: The code uses `matplotlib.use('Agg')`, meaning no plot windows will pop up during execution. All visualizations are saved directly to the disk.