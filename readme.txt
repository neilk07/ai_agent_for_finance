# 📊 AI-Powered Financial Analyst Pro

A Streamlit web application designed for comprehensive financial data analysis, visualization, time series forecasting, classification, and reporting.

## 📝 Overview

This tool allows users to upload financial datasets in CSV format and perform a variety of analyses powered by Python libraries like Pandas, Scikit-learn, and Statsmodels. It aims to provide quick insights, identify trends, detect anomalies, build predictive models, and generate summary reports automatically.

## ✨ Features

*   **Data Loading & Exploration:**
    *   Upload CSV files easily.
    *   Display data preview, information (`df.info()`), and summary statistics (`df.describe()`).
*   **Data Preparation:**
    *   Identify missing values.
    *   Offer imputation options (Drop NaNs, Mean/Median fill, Forward fill).
*   **Visualization:**
    *   Correlation Heatmap for numeric features.
    *   Distribution plots (Histograms/KDE) for numeric variables.
    *   Time series line plots for selected columns.
*   **Time Series Analysis:**
    *   Automatic Date/Time column detection and conversion.
    *   Plotting of time series data.
    *   **Rolling Statistics:** Calculate and visualize rolling mean and standard deviation (volatility).
    *   **Decomposition:** Analyze Trend, Seasonality, and Residual components.
    *   **Stationarity Testing:** Perform Augmented Dickey-Fuller (ADF) test.
*   **Anomaly Detection:**
    *   Identify outliers using Z-Score or Interquartile Range (IQR) methods.
*   **Forecasting:**
    *   Predict future values using time series models:
        *   Simple Exponential Smoothing
        *   Holt's Linear Trend
        *   Holt-Winters Seasonal Method
    *   Visualize historical data alongside forecasts.
    *   Basic backtesting evaluation (MAE, RMSE).
*   **Classification:**
    *   Train models (Random Forest, Logistic Regression) to predict a categorical target variable.
    *   Select numeric features for training.
    *   Display Classification Report (Precision, Recall, F1-Score).
    *   Visualize Confusion Matrix.
    *   Show Feature Importances (for Random Forest).
*   **AI Insights:**
    *   Dynamically generated textual summary based on the analyses performed.
*   **Reporting:**
    *   Export a comprehensive analysis report, including key statistics, plots, and insights, to a PDF document.

## 🛠️ Requirements

*   Python 3.7+
*   Required Python packages listed in `requirements.txt`.

## 🚀 Installation

1.  **Clone the repository (or download the script):**
    ```bash
    git clone <your-repository-url> # Or simply download the .py file
    cd <repository-folder>
    ```
2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Usage

1.  Ensure your virtual environment is activated.
2.  Run the Streamlit application from your terminal:
    ```bash
    streamlit run financial_analyzer_pro.py # Replace with your actual script name
    ```
3.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
4.  Use the sidebar to upload your CSV file.
5.  Interact with the different analysis sections through the expanders in the main application window.
6.  Generate and download the PDF report using the button at the end.

## 📄 Input Data Format

*   The application expects data in a **CSV (Comma-Separated Values)** file.
*   For full functionality, the dataset should ideally contain:
    *   One or more **numeric columns** for statistical analysis, correlation, forecasting, and feature input for classification.
    *   At least one **date or datetime column** for time series analysis and forecasting. Ensure the format is recognizable by `pandas.to_datetime`.
    *   Optionally, a **categorical column** (with a relatively small number of unique values, e.g., < 15) to be used as the target for classification tasks.

## 💡 Potential Future Enhancements

*   Integration with a real LLM (e.g., GPT, Llama) for more nuanced insights.
*   Support for more data sources (Excel, databases, APIs).
*   More sophisticated forecasting models (ARIMA, SARIMA, Prophet).
*   Advanced feature engineering and scaling options.
*   Interactive plots using libraries like Plotly or Bokeh.
*   User accounts and saving analysis sessions.
*   Deployment options (e.g., Streamlit Cloud, Docker).

---