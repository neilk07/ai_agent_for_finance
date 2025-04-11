# --- (Continuing from app.py code above) ---
# 🧠 AI-Powered Financial Data Analyst with Forecasting, Classification, PDF Export and Enhanced Features

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO, BytesIO
from fpdf import FPDF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import traceback # For more detailed error logging

# --- Page Configuration ---
st.set_page_config(page_title="AI Financial Analyst", layout="wide", initial_sidebar_state="expanded")

# --- Helper Functions ---
@st.cache_data # Persist data across reruns for the same input file
def load_data(uploaded_file):
    """Loads data from uploaded CSV file."""
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

@st.cache_data
def safe_convert_to_datetime(df, col_name):
    """Safely converts a column to datetime, handling errors. Returns the converted series or None."""
    if col_name not in df.columns:
        st.error(f"Column '{col_name}' not found in DataFrame.")
        return None
    try:
        return pd.to_datetime(df[col_name], errors='coerce')
    except Exception as e:
        st.warning(f"Could not convert '{col_name}' to datetime: {e}. Please select a valid date/time column.")
        return None

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plots a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    return fig

# --- Function to Generate Enhanced Dynamic Insights (Rule-Based) ---
def generate_dynamic_insights(results):
    insight_text = "### 🤖 AI Insights Summary (Generated Interpretation)\n\n"

    # --- Overview ---
    insight_text += "**1. Dataset Overview:**\n"
    shape_str = results.get('data_shape', 'N/A')
    if shape_str != 'N/A':
        try:
            # Attempt to parse shape like '(rows, cols)'
            rows, cols = shape_str.strip('()').split(',')
            insight_text += f"- The dataset initially contains {rows.strip()} rows and {cols.strip()} columns.\n"
        except:
             insight_text += f"- Initial dataset dimensions: {shape_str}.\n" # Fallback

    imputation_method = results.get('imputation_method', "None (Keep As Is)")
    if imputation_method != "None (Keep As Is)":
         insight_text += f"- Missing data was addressed using the '{imputation_method}' method for subsequent analysis.\n"

    missing_info = results.get('missing', "No missing values.")
    if missing_info != "No missing values.":
        # Try to extract top columns from the string (simple approach)
        missing_lines = missing_info.strip().split('\n')
        top_missing_cols = [line.split('    ')[0].strip() for line in missing_lines[:3]] # Get names of top 3
        if top_missing_cols:
             insight_text += f"- Missing values were detected, primarily in columns like: {', '.join(top_missing_cols)}{'...' if len(missing_lines)>3 else ''}. Refer to Section 1 for details.\n"
        else:
             insight_text += "- Missing values were detected. Refer to Section 1 for details.\n"
    elif imputation_method == "None (Keep As Is)": # Only explicitly say none if none were handled *and* none were found initially
        insight_text += "- No missing values were detected in the initial scan.\n"

    # --- Correlation ---
    insight_text += "\n**2. Correlation Insights:**\n"
    if 'correlation_plot' in results:
        insight_text += "- A heatmap visualized the linear relationships between numeric variables.\n"
        high_corrs = results.get('high_correlations') # Expects a list like [('A', 'B', 0.8), ...]
        if high_corrs:
             insight_text += "- **Notable Strong Relationships Found (absolute correlation > 0.7):**\n"
             for col1, col2, corr_val in high_corrs:
                 relation_type = "positive" if corr_val > 0 else "negative"
                 insight_text += f"  - A strong {relation_type} linear correlation ({corr_val:.2f}) exists between '{col1}' and '{col2}'. This suggests they tend to move in the {'same' if relation_type == 'positive' else 'opposite'} direction.\n"
        elif 'correlation_checked' in results: # Check if analysis was run but no high correlations found
             insight_text += "- No strong linear correlations (absolute value > 0.7) were detected among the numeric variables analyzed.\n"
        insight_text += "- *Remember: Correlation does not imply causation.*\n"
    else:
        insight_text += "- Correlation analysis requires at least two numeric columns with variance.\n"

    # --- Time Series Observations ---
    insight_text += "\n**3. Time Series Observations"
    ts_index = results.get('ts_index')
    if ts_index:
        insight_text += f" (using '{ts_index}' as time index):**\n"
        resample_freq = results.get('resample_freq')
        if resample_freq:
            insight_text += f"- *Data was optionally resampled to '{resample_freq}' frequency using '{results.get('resample_agg', 'N/A')}' aggregation before analysis.*\n"

        ts_plots_available = 'time_series_plots' in results and results['time_series_plots']

        if ts_plots_available and 'Rolling Statistics' in results['time_series_plots']:
            insight_text += "- **Trend & Volatility:** Rolling statistics (like Moving Averages) were plotted to smooth short-term fluctuations and highlight trends. The rolling standard deviation visualizes the data's volatility (ups and downs) over time.\n"

        if ts_plots_available and 'Decomposition' in results['time_series_plots']:
            insight_text += "- **Seasonality & Trend:** Time series decomposition attempted to separate the data into an underlying trend, seasonal patterns (with a period of {results.get('decomp_period', 'N/A')}), and residual noise. Review the decomposition plot (Section 2) to understand these components.\n"

        adf_results = results.get('adf_results')
        if adf_results:
             insight_text += "- **Stationarity (ADF Test):** This statistical test checks if the time series has properties (like mean) that are constant over time.\n"
             for col, res_summary in adf_results.items():
                 insight_text += f"  - For '{col}': {res_summary}. "
                 if "Non-Stationary" in res_summary:
                     insight_text += "This often indicates trends or seasonality that might require transformations (like calculating differences or returns) for certain forecasting models.\n"
                 else:
                     insight_text += "This suggests the series might be suitable as-is for models assuming stationarity.\n"
        elif ts_index: # Only mention if TS analysis was attempted
             insight_text += "- Stationarity tests (ADF) were not performed or failed for selected columns.\n"
    else:
        insight_text += ":**\n- Time series analysis requires a valid date/time column selected as the index.\n"


    # --- Anomaly Detection ---
    insight_text += "\n**4. Anomaly Detection:**\n"
    anomaly_method = results.get('anomaly_method')
    if anomaly_method:
        insight_text += f"- Potential outliers (anomalies) were checked using the **{results.get('anomaly_method', 'N/A')}** method with threshold {results.get('anomaly_threshold', 'N/A')}.\n"
        anomalies_found = results.get('anomalies_found') # Expects dict {col: count}
        if anomalies_found:
            insight_text += "- Anomalies were flagged in the following columns: "
            col_list = list(anomalies_found.keys())
            insight_text += f"{', '.join(col_list)}.\n"
            counts_str = ", ".join([f"{col} ({count})" for col, count in anomalies_found.items()])
            insight_text += f"  - Number of anomalies found: {counts_str}.\n"
            insight_text += "- These data points are statistically unusual compared to the rest of the data in those columns and may warrant further investigation (check Section 3 for listed points).\n"
        else: # Method was run, but none found
            insight_text += "- No significant anomalies were detected with the current settings and selected method.\n"
    else:
        insight_text += "- Anomaly detection requires numeric columns and was not performed.\n"


    # --- Forecasting ---
    insight_text += "\n**5. Forecasting Insights:**\n"
    forecast_details = results.get('forecast_details')
    if forecast_details:
        insight_text += f"- {forecast_details}\n" # Uses the already descriptive text
        if 'forecast_plot' in results:
            insight_text += "- The forecast plot (Section 4) visually compares the predicted future values against the historical data.\n"

        mae = results.get('forecast_mae')
        rmse = results.get('forecast_rmse')
        if mae is not None and rmse is not None:
            insight_text += "- **Backtest Evaluation:** To estimate accuracy, the model's forecasts were compared against the most recent actual data:\n"
            insight_text += f"  - Mean Absolute Error (MAE) ≈ {mae:.3f}: On average, the forecast was off by this amount in the variable's units.\n"
            insight_text += f"  - Root Mean Squared Error (RMSE) ≈ {rmse:.3f}: This measure (also in the variable's units) penalizes larger errors more heavily.\n"
            insight_text += "  - Lower MAE/RMSE values generally indicate better forecast accuracy on the recent past data. Compare these errors to the scale of the data itself.\n"
        elif 'forecast_eval' in results: # Fallback if only the string exists
             insight_text += f"- **Backtest Evaluation Results:**\n{results['forecast_eval']}\n"
        else:
             insight_text += "- A backtest evaluation to estimate forecast accuracy was not performed or failed.\n"
    else:
        insight_text += "- Forecasting requires a valid time series index (from Section 2), numeric data, and successful model fitting.\n"

    # --- Classification ---
    insight_text += "\n**6. Classification Insights:**\n"
    clf_details = results.get('classification_details')
    if clf_details:
        insight_text += f"- {clf_details}\n" # Uses the descriptive text
        insight_text += "- The goal was to train a model to predict the category based on the selected features.\n"

        accuracy = results.get('classification_accuracy')
        if accuracy is not None:
            insight_text += f"- **Performance:** The model achieved an overall accuracy of approximately {accuracy:.1f}% on the unseen test data.\n"

        if 'classification_report' in results:
             insight_text += "- Key metrics like precision (accuracy of positive predictions) and recall (ability to find all positive instances) for each class are detailed in the Classification Report (Section 5). The Confusion Matrix visually shows correct vs. incorrect predictions for each class.\n"
             # Enhancement: Could parse the report string to highlight specific class performance if needed.

        feature_importances = results.get('feature_importances')
        if feature_importances:
             insight_text += "- **Key Drivers (Feature Importance for Random Forest):** The analysis identified the most influential numeric features for the model's predictions:\n"
             # Assume feature_importances is a pre-formatted string of top N features
             insight_text += f" ```\n{feature_importances}\n ```\n" # Use markdown code block
             insight_text += "  - Features listed higher had a greater impact on the prediction outcome according to this model.\n"
        elif 'classification_report' in results and 'Random Forest' in clf_details:
             insight_text += "- Feature importance analysis is typically available for Random Forest models.\n"
    else:
        insight_text += "- Classification requires a suitable categorical target column, numeric feature columns, and successful model training.\n"


    # --- Final Disclaimer ---
    insight_text += "\n**Disclaimer:** These insights are automatically generated based on statistical analysis and modeling. They provide a starting point, but should *always* be validated with domain expertise and context-specific knowledge. Patterns observed do not guarantee future results or imply causation."
    return insight_text

# --- PDF Report Generation (Minimal changes needed, uses the insight text) ---
def generate_pdf_report(report_data):
    """Generates a PDF report from collected analysis results."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Financial Data Analysis Report", 0, 1, 'C')
    pdf.ln(10)

    current_section = 1

    # --- Section: Overview ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"{current_section}. Dataset Overview", 0, 1)
    current_section += 1
    pdf.set_font("Arial", size=10)
    if 'data_shape' in report_data:
         pdf.multi_cell(0, 5, f"Original Shape: {report_data['data_shape']}")
    if 'imputation_method' in report_data and report_data['imputation_method'] != "None (Keep As Is)":
         pdf.multi_cell(0, 5, f"Imputation Method Applied: {report_data['imputation_method']}")
    if 'summary' in report_data:
        pdf.multi_cell(0, 5, f"Summary Statistics (Numeric):\n{report_data['summary']}")
    if 'missing' in report_data:
        pdf.multi_cell(0, 5, f"Missing Values Summary:\n{report_data['missing']}")
    pdf.ln(5)

    # --- Section: Correlation ---
    if 'correlation_plot' in report_data:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"{current_section}. Correlation Analysis", 0, 1)
        current_section += 1
        try:
            pdf.image(report_data['correlation_plot'], x=10, y=pdf.get_y(), w=180)
            pdf.ln(85) # Adjust spacing as needed based on plot height
        except Exception as e:
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 5, f"(Error embedding correlation plot: {e})")
            pdf.ln(5)

    # --- Section: Time Series ---
    if report_data.get('ts_index'): # Check if TS analysis was done
        if pdf.get_y() > 200: pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"{current_section}. Time Series Analysis", 0, 1)
        current_section += 1
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 5, f"Time Series Index: {report_data['ts_index']}")
        if 'resample_freq' in report_data:
             pdf.multi_cell(0, 5, f"Resampling Applied: Frequency='{report_data['resample_freq']}', Aggregation='{report_data.get('resample_agg', 'N/A')}'")
        pdf.ln(2)

        if 'time_series_plots' in report_data:
            for title, plot_bytes in report_data['time_series_plots'].items():
                pdf.set_font("Arial", 'B', 10)
                pdf.cell(0, 8, title, 0, 1)
                try:
                    if pdf.get_y() > 190: pdf.add_page() # Check space before plot
                    pdf.image(plot_bytes, x=10, y=pdf.get_y(), w=180)
                    pdf.ln(85) # Adjust spacing
                except Exception as e:
                    pdf.set_font("Arial", size=10)
                    pdf.multi_cell(0, 5, f"(Error embedding plot '{title}': {e})")
                    pdf.ln(5)
            pdf.ln(5)

    # --- Section: Anomaly Detection ---
    if 'anomaly_method' in report_data:
        if pdf.get_y() > 220: pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"{current_section}. Anomaly Detection", 0, 1)
        current_section += 1
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 5, f"Method Used: {report_data['anomaly_method']} (Threshold: {report_data.get('anomaly_threshold', 'N/A')})")
        if 'anomalies_found_details' in report_data and report_data['anomalies_found_details']:
            for col, anom_df_str in report_data['anomalies_found_details'].items():
                pdf.set_font("Arial", 'B', 10)
                pdf.multi_cell(0, 5, f"Potential Anomalies in '{col}' (Top 10 shown):")
                pdf.set_font("Courier", size=8) # Use monospace for data snippets
                pdf.multi_cell(0, 4, anom_df_str)
                pdf.ln(2)
        else:
             pdf.set_font("Arial", size=10)
             pdf.multi_cell(0, 5, "No significant anomalies detected with current settings.")
        pdf.ln(5)


    # --- Section: Forecasting ---
    if 'forecast_details' in report_data:
        if pdf.get_y() > 180: pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"{current_section}. Forecasting", 0, 1)
        current_section += 1
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 5, f"{report_data['forecast_details']}")
        if 'forecast_eval' in report_data:
            pdf.set_font("Arial", 'B', 10)
            pdf.multi_cell(0, 5, f"Backtest Evaluation:")
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 5, f"{report_data['forecast_eval']}") # Use the formatted string here

        if 'forecast_plot' in report_data:
            try:
                if pdf.get_y() > 190: pdf.add_page()
                pdf.image(report_data['forecast_plot'], x=10, y=pdf.get_y(), w=180)
                pdf.ln(85) # Adjust spacing
            except Exception as e:
                pdf.set_font("Arial", size=10)
                pdf.multi_cell(0, 5, f"(Error embedding forecast plot: {e})")
                pdf.ln(5)
        pdf.ln(5)


    # --- Section: Classification ---
    if 'classification_details' in report_data:
        if pdf.get_y() > 150: pdf.add_page() # Check space before adding report + matrix
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"{current_section}. Classification Results", 0, 1)
        current_section += 1
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 5, f"{report_data['classification_details']}")
        if 'classification_accuracy' in report_data:
             pdf.multi_cell(0, 5, f"Model Accuracy (Test Set): {report_data['classification_accuracy']:.1f}%")
        pdf.ln(2)

        if 'classification_report' in report_data:
            pdf.set_font("Arial", 'B', 10)
            pdf.multi_cell(0, 5, f"Classification Report:")
            pdf.set_font("Courier", size=8) # Use monospace for report
            pdf.multi_cell(0, 4, report_data['classification_report'])
            pdf.ln(5)

        if 'confusion_matrix_plot' in report_data:
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 10, "Confusion Matrix:", 0, 1)
            try:
                if pdf.get_y() > 200: pdf.add_page() # Add new page if needed
                pdf.image(report_data['confusion_matrix_plot'], x=10, y=pdf.get_y(), w=100)
                pdf.ln(60) # Adjust spacing
            except Exception as e:
                pdf.set_font("Arial", size=10)
                pdf.multi_cell(0, 5, f"(Error embedding confusion matrix: {e})")
                pdf.ln(5)

        if 'feature_importances' in report_data:
            pdf.set_font("Arial", 'B', 10)
            pdf.multi_cell(0, 5, f"Top Feature Importances (Random Forest):")
            pdf.set_font("Courier", size=8)
            pdf.multi_cell(0, 4, report_data['feature_importances'])
            pdf.ln(5)
        pdf.ln(5)

    # --- Section: AI Insights ---
    if 'ai_insights' in report_data: # Use the pre-generated text from the function
        if pdf.get_y() > 200: pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        # Remove markdown bold/headers for PDF text
        title_text = report_data['ai_insights'].split('\n')[0].replace("### ", "").replace("🤖", "").strip()
        pdf.cell(0, 10, f"{current_section}. {title_text}", 0, 1)
        current_section += 1
        pdf.set_font("Arial", size=10)
        # Split insights and remove markdown, then add to PDF
        insight_lines = report_data['ai_insights'].split('\n')[1:] # Skip title line
        for line in insight_lines:
             cleaned_line = line.replace("**", "").replace("```", "").strip()
             if cleaned_line: # Avoid empty lines
                 pdf.multi_cell(0, 5, cleaned_line)
        pdf.ln(5)

    return pdf.output(dest='S').encode('latin1')


# --- Global Variables & Session State ---
# Use clear naming for session state keys
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'dataframe' not in st.session_state:
    st.session_state.dataframe = None # Store the working dataframe
if 'current_file_id' not in st.session_state:
    st.session_state.current_file_id = None

# --- Sidebar ---
st.sidebar.title("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload Financial Data (CSV)", type=["csv"], key="file_uploader")

# Reset state if new file uploaded
if uploaded_file is not None:
     current_file_id = f"{uploaded_file.name}-{uploaded_file.size}"
     if st.session_state.current_file_id != current_file_id:
         st.info("New file detected, resetting analysis state.")
         st.session_state.analysis_results = {}
         st.session_state.data_loaded = False
         st.session_state.dataframe = None
         st.session_state.current_file_id = current_file_id
         # Force rerun after state reset for clean start
         st.rerun()

# --- Main App ---
st.title("📊 AI-Powered Financial Analyst")
st.markdown("""
Welcome! Upload your financial data (CSV). This tool performs EDA, time series analysis, forecasting, classification, and generates interpretable insights.
""")

if uploaded_file:
    # Load data only if not already loaded or if file changed
    if not st.session_state.data_loaded:
        df_original = load_data(uploaded_file)
        if df_original is not None:
            st.session_state.dataframe = df_original.copy()
            st.session_state.data_loaded = True
            st.session_state.analysis_results = {} # Clear results for new file
            st.session_state.analysis_results['data_shape'] = str(df_original.shape)
            st.success("File uploaded and initial data loaded successfully!")
            # Rerun to process the loaded data immediately
            st.rerun()
        else:
            st.error("Failed to load data from the uploaded file.")
            st.stop()

    # Work with the dataframe stored in session state
    if st.session_state.data_loaded and isinstance(st.session_state.dataframe, pd.DataFrame):
        df = st.session_state.dataframe # Use the working copy

        # --- 1. Data Exploration & Preparation ---
        with st.expander("1️⃣ Data Exploration & Preparation", expanded=True):
            st.write("### Preview of Original Dataset Head")
            # Show original head for reference, but analysis runs on potentially modified df
            st.dataframe(load_data(uploaded_file).head() if uploaded_file else "Upload file")

            st.write("### Working Dataset Information")
            st.markdown("Data types, non-null counts, and memory usage.")
            buffer = StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()
            st.text(info_str)

            st.write("### Summary Statistics (Working Data - Numeric Columns)")
            st.markdown("Key statistics like mean, median, min, max, and quartiles.")
            numeric_df_desc = df.select_dtypes(include=np.number)
            if not numeric_df_desc.empty:
                summary_stats = numeric_df_desc.describe()
                st.dataframe(summary_stats)
                st.session_state.analysis_results['summary'] = summary_stats.to_string()
            else:
                st.warning("No numeric columns found in the working data for summary statistics.")
                st.session_state.analysis_results.pop('summary', None)


            st.write("### Missing Values (Working Data)")
            st.markdown("Columns with missing data points and their counts.")
            missing_values = df.isnull().sum()
            missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
            if not missing_values.empty:
                st.dataframe(missing_values.to_frame(name='Missing Count'))
                st.session_state.analysis_results['missing'] = missing_values.to_string()

                impute_method = st.selectbox("Handle Missing Values for subsequent analysis?",
                                             ["None (Keep As Is)", "Drop Rows with NaNs", "Fill with Mean (Numeric)", "Fill with Median (Numeric)", "Forward Fill"],
                                             key="impute_select", help="Choose how to deal with rows containing missing values before performing analysis. Changes apply to subsequent steps.")
                st.session_state.analysis_results['imputation_method'] = impute_method

                # Apply imputation logic
                if impute_method != "None (Keep As Is)":
                    df_processed = df.copy()
                    rows_before = len(df_processed)
                    imputed_cols_list = [] # Keep track of imputed columns

                    if impute_method == "Drop Rows with NaNs":
                        df_processed = df_processed.dropna()
                    elif impute_method in ["Fill with Mean (Numeric)", "Fill with Median (Numeric)"]:
                        numeric_cols_na = df_processed.select_dtypes(include=np.number).columns
                        fill_type = impute_method.split(' ')[2].lower()
                        for col in numeric_cols_na:
                            if df_processed[col].isnull().any():
                                fill_value = df_processed[col].mean() if fill_type == "mean" else df_processed[col].median()
                                df_processed[col] = df_processed[col].fillna(fill_value)
                                imputed_cols_list.append(col)
                    elif impute_method == "Forward Fill":
                        df_processed = df_processed.ffill()

                    rows_after = len(df_processed)
                    st.session_state.dataframe = df_processed # Update the main df in session state
                    # Show feedback on imputation
                    if impute_method == "Drop Rows with NaNs":
                        st.info(f"Applied 'Drop Rows': {rows_before - rows_after} rows removed. New shape: {df_processed.shape}")
                    elif impute_method in ["Fill with Mean (Numeric)", "Fill with Median (Numeric)"]:
                         st.info(f"Applied '{impute_method}': Filled NaNs in {len(imputed_cols_list)} numeric columns: {', '.join(imputed_cols_list)}")
                    elif impute_method == "Forward Fill":
                         st.info(f"Applied 'Forward Fill'.")

                    # Rerun to ensure subsequent steps use the modified dataframe
                    st.warning("Imputation applied. Rerunning analysis with modified data...")
                    st.button("Acknowledge & Rerun") # Dummy button to trigger rerun easily
                    st.rerun() # Force rerun

            else:
                st.info("No missing values found in the working data.")
                st.session_state.analysis_results['missing'] = "No missing values."
                st.session_state.analysis_results['imputation_method'] = "None (Keep As Is)"


            st.write("### Correlation Heatmap (Numeric Columns)")
            st.markdown("Visualizes linear relationships between numeric variables. Ranges from -1 (strong negative) to +1 (strong positive).")
            st.session_state.analysis_results.pop('correlation_plot', None)
            st.session_state.analysis_results.pop('high_correlations', None)
            st.session_state.analysis_results.pop('correlation_checked', None)

            numeric_df_corr = df.select_dtypes(include=[np.number])
            # Exclude columns with zero variance
            numeric_df_corr = numeric_df_corr.loc[:, numeric_df_corr.std() > 0]

            if not numeric_df_corr.empty and len(numeric_df_corr.columns) > 1:
                corr_matrix = numeric_df_corr.corr()
                st.session_state.analysis_results['correlation_checked'] = True # Mark that we attempted correlation

                fig_corr, ax_corr = plt.subplots(figsize=(10, 7))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
                ax_corr.set_title("Correlation Matrix of Numeric Variables")
                st.pyplot(fig_corr)
                try:
                    corr_img = BytesIO()
                    fig_corr.savefig(corr_img, format='png', bbox_inches='tight')
                    plt.close(fig_corr)
                    corr_img.seek(0)
                    st.session_state.analysis_results['correlation_plot'] = corr_img

                    # Find and store high correlations for insights
                    high_corrs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i + 1, len(corr_matrix.columns)):
                            col1 = corr_matrix.columns[i]
                            col2 = corr_matrix.columns[j]
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) > 0.7: # Threshold for high correlation
                                high_corrs.append((col1, col2, corr_val))
                    if high_corrs:
                        st.session_state.analysis_results['high_correlations'] = sorted(high_corrs, key=lambda x: abs(x[2]), reverse=True)

                except Exception as e:
                    st.error(f"Failed to save correlation plot: {e}")

            else:
                st.warning("Correlation heatmap requires at least two numeric columns with non-zero variance in the working data.")


            st.write("### Distribution of Numeric Variables")
            st.markdown("Shows the distribution shape (histogram and density curve) for a selected numeric column.")
            numeric_cols_dist = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols_dist:
                col_to_plot = st.selectbox("Select numeric column for distribution plot", numeric_cols_dist, key="dist_select")
                if col_to_plot and col_to_plot in df:
                    fig_dist, ax_dist = plt.subplots()
                    sns.histplot(df[col_to_plot].dropna(), kde=True, ax=ax_dist)
                    ax_dist.set_title(f'Distribution of {col_to_plot}')
                    st.pyplot(fig_dist)
                    plt.close(fig_dist)
            else:
                st.info("No numeric columns available for distribution plots.")

        # --- 2. Time Series Analysis ---
        with st.expander("2️⃣ Time Series Analysis"):
            st.write("### Select Date/Time Column")
            st.markdown("Choose the column representing time for analysis.")
            potential_date_cols = df.columns.tolist()
            sorted_date_cols = sorted(potential_date_cols, key=lambda x: (
                0 if 'date' in x.lower() or 'time' in x.lower() else 1,
                0 if pd.api.types.is_datetime64_any_dtype(df[x]) else (1 if df[x].dtype == 'object' else 2), x))

            date_col = st.selectbox("Select the primary date/time column",
                                    sorted_date_cols, index=0, key="date_col_select",
                                    help="This column will be used as the index for time-based analysis.")

            df_ts = None # Initialize df_ts
            st.session_state.analysis_results.pop('ts_index', None)
            st.session_state.analysis_results.pop('time_series_plots', None)
            st.session_state.analysis_results.pop('adf_results', None)
            st.session_state.analysis_results.pop('resample_freq', None)
            st.session_state.analysis_results.pop('resample_agg', None)
            st.session_state.analysis_results.pop('decomp_period', None)

            if date_col:
                datetime_series = safe_convert_to_datetime(df, date_col)
                if datetime_series is not None and not datetime_series.isnull().all():
                    df_temp_ts = df.copy()
                    df_temp_ts[date_col] = datetime_series
                    df_temp_ts = df_temp_ts.dropna(subset=[date_col])
                    df_temp_ts = df_temp_ts.sort_values(by=date_col)

                    try:
                        df_ts_pre_resample = df_temp_ts.set_index(date_col)

                        # --- Optional Resampling Step ---
                        st.markdown("---")
                        st.write("#### Optional: Resample Time Series")
                        st.markdown("Resampling aggregates data to a fixed frequency (e.g., daily, weekly). This can help if your data has irregular timestamps or gaps, fixing frequency inference issues. **Warning:** This aggregates data and changes granularity.")
                        apply_resampling = st.checkbox("Apply Resampling?", key="apply_resample_cb")

                        df_ts = df_ts_pre_resample # Default to non-resampled

                        if apply_resampling:
                            resample_freq = st.selectbox("Target Frequency", ['D', 'W', 'M', 'Q', 'Y', 'H', 'T'], index=0, key="resample_freq_select",
                                                         help="'D'=Daily, 'W'=Weekly, 'M'=Monthly, 'Q'=Quarterly, 'Y'=Yearly, 'H'=Hourly, 'T'=Minutely")
                            resample_agg = st.selectbox("Aggregation Method", ['mean', 'sum', 'last', 'median', 'ohlc'], index=0, key="resample_agg_select",
                                                        help="How to combine data points within each new period (e.g., 'mean', 'sum', 'last' known value, 'ohlc' for price data).")
                            try:
                                if resample_agg == 'ohlc':
                                    # OHLC requires numeric columns; select them
                                    numeric_cols_for_ohlc = df_ts_pre_resample.select_dtypes(include=np.number).columns
                                    if not numeric_cols_for_ohlc.empty:
                                        df_ts = df_ts_pre_resample[numeric_cols_for_ohlc].resample(resample_freq).ohlc()
                                        # Flatten MultiIndex columns if OHLC was used
                                        if isinstance(df_ts.columns, pd.MultiIndex):
                                            df_ts.columns = [f'{col[0]}_{col[1]}' for col in df_ts.columns]
                                    else:
                                        st.warning("OHLC aggregation requires numeric columns. Resampling skipped.")
                                        df_ts = df_ts_pre_resample # Revert
                                else:
                                    # Apply standard aggregation to numeric columns only
                                    numeric_cols = df_ts_pre_resample.select_dtypes(include=np.number).columns
                                    non_numeric_cols = df_ts_pre_resample.select_dtypes(exclude=np.number).columns
                                    df_ts_numeric = df_ts_pre_resample[numeric_cols].resample(resample_freq).agg(resample_agg)
                                    # Optionally handle non-numeric (e.g., take the first/last value) - keeping it simple for now
                                    # df_ts_non_numeric = df_ts_pre_resample[non_numeric_cols].resample(resample_freq).first()
                                    # df_ts = pd.concat([df_ts_numeric, df_ts_non_numeric], axis=1)
                                    df_ts = df_ts_numeric # Focus on numeric for analysis

                                st.success(f"Resampled data to '{resample_freq}' frequency using '{resample_agg}'. New shape: {df_ts.shape}")
                                st.dataframe(df_ts.head())
                                st.session_state.analysis_results['resample_freq'] = resample_freq
                                st.session_state.analysis_results['resample_agg'] = resample_agg

                            except Exception as e:
                                st.error(f"Resampling failed: {e}. Using original time index.")
                                df_ts = df_ts_pre_resample # Revert if resampling fails
                                st.session_state.analysis_results.pop('resample_freq', None)
                                st.session_state.analysis_results.pop('resample_agg', None)
                        st.markdown("---")
                        # --- End Resampling Step ---


                        if df_ts.index.has_duplicates:
                            st.warning(f"Duplicate timestamps found after potential resampling in '{df_ts.index.name}'. Aggregating duplicates using mean. Consider checking original data or resampling logic.")
                            # Aggregate duplicates - essential before many statsmodels functions
                            df_ts = df_ts.groupby(df_ts.index).mean()

                        st.success(f"Using '{df_ts.index.name}' as time series index. Data points: {len(df_ts)}")
                        st.session_state.analysis_results['ts_index'] = df_ts.index.name
                        st.session_state.analysis_results['time_series_plots'] = {}
                        st.session_state.analysis_results['adf_results'] = {}

                        numeric_cols_ts = df_ts.select_dtypes(include=np.number).columns.tolist()

                        if numeric_cols_ts:
                            st.write("### Time Series Plot")
                            ts_col_select = st.multiselect("Select numeric columns to plot over time", numeric_cols_ts, default=numeric_cols_ts[0] if numeric_cols_ts else None, key="ts_plot_select")
                            if ts_col_select:
                                st.line_chart(df_ts[ts_col_select])

                            st.write("### Rolling Statistics")
                            st.markdown("Visualize trends (rolling mean) and volatility (rolling standard deviation).")
                            roll_col = st.selectbox("Select column for rolling statistics", numeric_cols_ts, key="roll_col_select")
                            max_roll_window = max(3, min(90, len(df_ts)//2))
                            if max_roll_window > 3 and roll_col and roll_col in df_ts:
                                roll_window = st.slider("Rolling window size (periods)", min_value=3, max_value=max_roll_window, value=min(14, max_roll_window), key="roll_window_slider")
                                if roll_window < len(df_ts[roll_col].dropna()):
                                    rolling_mean = df_ts[roll_col].rolling(window=roll_window).mean()
                                    rolling_std = df_ts[roll_col].rolling(window=roll_window).std()

                                    fig_roll, ax_roll = plt.subplots(figsize=(12, 6))
                                    ax_roll.plot(df_ts.index, df_ts[roll_col], label='Original', alpha=0.7)
                                    ax_roll.plot(rolling_mean.index, rolling_mean, label=f'{roll_window}-Period Rolling Mean', color='orange')
                                    ax_roll.plot(rolling_std.index, rolling_std, label=f'{roll_window}-Period Rolling Std Dev (Volatility)', color='red', linestyle='--')
                                    ax_roll.set_title(f'Rolling Statistics for {roll_col}')
                                    ax_roll.legend()
                                    st.pyplot(fig_roll)
                                    try:
                                        roll_img = BytesIO()
                                        fig_roll.savefig(roll_img, format='png', bbox_inches='tight')
                                        plt.close(fig_roll)
                                        roll_img.seek(0)
                                        st.session_state.analysis_results['time_series_plots']['Rolling Statistics'] = roll_img
                                    except Exception as e:
                                         st.error(f"Failed to save rolling stats plot: {e}")
                                else:
                                     st.warning(f"Select a column and ensure window size ({roll_window}) is smaller than the number of data points ({len(df_ts[roll_col].dropna()) if roll_col in df_ts else 0}).")
                            else:
                                st.info("Not enough data points for rolling statistics or no numeric column selected.")


                            st.write("### Time Series Decomposition")
                            st.markdown("Separates the time series into Trend, Seasonal, and Residual components.")
                            decomp_col = st.selectbox("Select column for decomposition", numeric_cols_ts, key='decomp_sel')
                            # Infer frequency or use resampled freq
                            inferred_freq = pd.infer_freq(df_ts.index) or st.session_state.analysis_results.get('resample_freq')
                            st.write(f"Attempting decomposition based on frequency: {inferred_freq if inferred_freq else 'Could not infer (data might be irregular or not resampled)'}")
                            suggested_period = 1 # Default if no freq
                            if inferred_freq:
                                freq_char = inferred_freq.split('-')[0][0].upper() # Get primary freq char (D, W, M etc)
                                if freq_char == 'D': suggested_period = 7
                                elif freq_char == 'W': suggested_period = 52 # Weeks in year
                                elif freq_char in ['M', 'Q']: suggested_period = 12 # Months/Quarters relate to year
                                elif freq_char == 'H': suggested_period = 24 # Hours in day

                            period = st.number_input("Seasonality Period (override)",
                                                     min_value=2, value=suggested_period if suggested_period > 1 else 7, key='decomp_period',
                                                     help="Number of periods in a seasonal cycle (e.g., 7 for daily data/weekly cycle, 12 for monthly data/yearly cycle). Must be >= 2.")

                            if decomp_col and decomp_col in df_ts and period >= 2:
                                decomp_data = df_ts[decomp_col].dropna()
                                if len(decomp_data) >= 2 * period:
                                    try:
                                        decomp_model_type = st.radio("Decomposition Model", ('additive', 'multiplicative'), key='decomp_model',
                                                                     help="'additive' assumes seasonal variations are constant, 'multiplicative' assumes they change proportionally.")
                                        result = seasonal_decompose(decomp_data, model=decomp_model_type, period=period)
                                        fig_decomp = result.plot()
                                        fig_decomp.set_size_inches(10, 8)
                                        plt.tight_layout()
                                        st.pyplot(fig_decomp)
                                        st.session_state.analysis_results['decomp_period'] = period # Store period used
                                        try:
                                            decomp_img = BytesIO()
                                            fig_decomp.savefig(decomp_img, format='png', bbox_inches='tight')
                                            plt.close(fig_decomp)
                                            decomp_img.seek(0)
                                            st.session_state.analysis_results['time_series_plots']['Decomposition'] = decomp_img
                                        except Exception as e:
                                            st.error(f"Failed to save decomposition plot: {e}")
                                    except Exception as e:
                                        st.error(f"Decomposition failed for '{decomp_col}': {e}")
                                else:
                                    st.warning(f"Not enough data points ({len(decomp_data)}) for decomposition with period {period}. Need at least {2 * period}.")
                            elif period < 2:
                                st.info("Select a column and ensure Seasonality Period is >= 2 for decomposition.")
                            else:
                                st.info("Select a column for decomposition.")


                            st.write("### Stationarity Test (Augmented Dickey-Fuller)")
                            st.markdown("Checks if the time series properties (like mean, variance) are constant over time. Important for some forecasting models.")
                            adf_col = st.selectbox("Select column for ADF Test", numeric_cols_ts, key='adf_sel')
                            if adf_col and adf_col in df_ts:
                                adf_data = df_ts[adf_col].dropna()
                                if len(adf_data) > 10: # Need reasonable number of points for ADF
                                    try:
                                        result = adfuller(adf_data)
                                        p_value = result[1]
                                        st.write(f"**Results for {adf_col}:**")
                                        st.write(f"- ADF Statistic: {result[0]:.4f}")
                                        st.write(f"- p-value: {p_value:.4f}")

                                        adf_summary = ""
                                        if p_value <= 0.05:
                                            st.success(f"Result suggests '{adf_col}' is likely stationary (p <= 0.05).")
                                            adf_summary = f"Likely Stationary (p={p_value:.3f})"
                                        else:
                                            st.warning(f"Result suggests '{adf_col}' is likely non-stationary (p > 0.05).")
                                            adf_summary = f"Likely Non-Stationary (p={p_value:.3f})"
                                        st.session_state.analysis_results['adf_results'][adf_col] = adf_summary

                                    except Exception as e:
                                        st.error(f"ADF test failed for {adf_col}: {e}")
                                else:
                                    st.warning(f"Not enough non-missing data points ({len(adf_data)}) in '{adf_col}' for ADF test.")
                        else:
                            st.warning("No numeric columns found in the data after setting and potentially resampling the time index.")
                    except Exception as e:
                        st.error(f"Error setting index or during time series analysis setup: {e}")
                        st.error(f"Traceback: {traceback.format_exc()}")
                        df_ts = None # Ensure df_ts is None if setup fails
                else:
                    st.warning(f"Could not convert '{date_col}' to a valid datetime index. Please check the column's format.")
                    df_ts = None
            else:
                st.info("Select a date/time column to enable time series analysis.")

        # --- 3. Anomaly Detection ---
        with st.expander("3️⃣ Anomaly Detection"):
            st.markdown("Identifies statistically unusual data points (outliers) in numeric columns.")
            numeric_cols_anom = df.select_dtypes(include=np.number).columns.tolist()
            st.session_state.analysis_results.pop('anomalies_found', None) # Store counts {col: count}
            st.session_state.analysis_results.pop('anomalies_found_details', None) # Store dataframes {col: df_string}
            st.session_state.analysis_results.pop('anomaly_method', None)
            st.session_state.analysis_results.pop('anomaly_threshold', None)

            if numeric_cols_anom:
                anom_method = st.radio("Select Anomaly Detection Method", ["Z-Score", "Interquartile Range (IQR)"], key="anom_method_radio")
                st.session_state.analysis_results['anomaly_method'] = anom_method

                anomalies_found_dict = {} # Store counts
                anomalies_detail_dict = {} # Store string representations of outlier dataframes
                threshold_value = None

                if anom_method == "Z-Score":
                    z_thresh = st.slider("Set Z-score threshold", 1.5, 5.0, 3.0, 0.1, key="z_slider", help="How many standard deviations away from the mean a point must be to be flagged.")
                    threshold_value = z_thresh
                    for col in numeric_cols_anom:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            col_data = df[col].dropna()
                            if len(col_data) > 1:
                                mean = col_data.mean()
                                std = col_data.std()
                                if std > 0:
                                    z_scores = (df[col] - mean) / std
                                    outlier_indices = df.index[np.abs(z_scores) > z_thresh]
                                    if not outlier_indices.empty:
                                        outliers_df = df.loc[outlier_indices, [col]]
                                        anomalies_found_dict[col] = len(outliers_df)
                                        anomalies_detail_dict[col] = outliers_df.to_string(max_rows=10)
                                elif std == 0:
                                     st.info(f"Column '{col}' has zero standard deviation, skipping Z-score.")

                elif anom_method == "IQR":
                    iqr_multiplier = st.slider("Set IQR Multiplier", 1.0, 3.0, 1.5, 0.1, key="iqr_slider", help="How many times the Interquartile Range away from Q1/Q3 a point must be to be flagged.")
                    threshold_value = iqr_multiplier
                    for col in numeric_cols_anom:
                         if pd.api.types.is_numeric_dtype(df[col]):
                            col_data = df[col].dropna()
                            if len(col_data) > 3: # Need a few points for quartiles
                                Q1 = col_data.quantile(0.25)
                                Q3 = col_data.quantile(0.75)
                                IQR = Q3 - Q1
                                if IQR > 0:
                                    lower_bound = Q1 - iqr_multiplier * IQR
                                    upper_bound = Q3 + iqr_multiplier * IQR
                                    outlier_indices = df.index[(df[col] < lower_bound) | (df[col] > upper_bound)]
                                    if not outlier_indices.empty:
                                         outliers_df = df.loc[outlier_indices, [col]]
                                         anomalies_found_dict[col] = len(outliers_df)
                                         anomalies_detail_dict[col] = outliers_df.to_string(max_rows=10)
                                elif IQR == 0:
                                    st.info(f"Column '{col}' has zero Interquartile Range, skipping IQR method.")

                st.session_state.analysis_results['anomaly_threshold'] = threshold_value

                if anomalies_found_dict:
                    st.write("#### Anomalies Detected:")
                    st.session_state.analysis_results['anomalies_found'] = anomalies_found_dict
                    st.session_state.analysis_results['anomalies_found_details'] = anomalies_detail_dict
                    counts_str = ", ".join([f"{col} ({count})" for col, count in anomalies_found_dict.items()])
                    st.info(f"Found anomalies in: {counts_str}")
                    # Optionally display the detailed dataframes in the app
                    show_details = st.checkbox("Show details of detected anomalies?", key="anom_details_cb")
                    if show_details:
                         for col, anom_df_str in anomalies_detail_dict.items():
                            st.write(f"**{col}** (Top 10 shown):")
                            st.text(anom_df_str)
                else:
                    st.info("No significant anomalies detected with the current settings.")
            else:
                st.warning("No numeric columns available in the working data for anomaly detection.")

        # --- 4. Forecasting ---
        with st.expander("4️⃣ Forecasting"):
            st.markdown("Predict future values based on historical time series data.")
            st.session_state.analysis_results.pop('forecast_details', None)
            st.session_state.analysis_results.pop('forecast_eval', None)
            st.session_state.analysis_results.pop('forecast_plot', None)
            st.session_state.analysis_results.pop('forecast_mae', None)
            st.session_state.analysis_results.pop('forecast_rmse', None)

            # Check if Time Series index was set up correctly (df_ts exists and is valid)
            if 'df_ts' in locals() and isinstance(df_ts, pd.DataFrame) and isinstance(df_ts.index, pd.DatetimeIndex):
                numeric_cols_forecast = df_ts.select_dtypes(include=np.number).columns.tolist()
                if numeric_cols_forecast:
                    st.write("### Time Series Forecasting Setup")
                    forecast_col = st.selectbox("Select column to forecast", numeric_cols_forecast, key="fc_col_select")
                    forecast_periods = st.number_input("Number of periods to forecast", min_value=1, max_value=365, value=30, key="fc_periods")
                    forecast_model_type = st.selectbox("Select Forecasting Model",
                                                        ["Simple Exponential Smoothing", "Holt's Linear Trend", "Holt-Winters Seasonal"],
                                                        key="fc_model_select")

                    if forecast_col and forecast_periods > 0:
                        train_data = df_ts[forecast_col].dropna()
                        st.info(f"Attempting to forecast '{forecast_col}' using data indexed by '{df_ts.index.name}'. Training data length: {len(train_data)}")

                        if len(train_data) > 5: # Need a minimum number of points
                            model = None
                            model_fitted = None
                            model_params = {} # Store parameters used

                            try:
                                if forecast_model_type == "Simple Exponential Smoothing":
                                    model = ExponentialSmoothing(train_data, trend=None, seasonal=None)
                                    model_fitted = model.fit()
                                elif forecast_model_type == "Holt's Linear Trend":
                                    model = ExponentialSmoothing(train_data, trend='add', seasonal=None)
                                    model_fitted = model.fit()
                                elif forecast_model_type == "Holt-Winters Seasonal":
                                    # Use period determined during decomposition if available
                                    default_seasonal_period = st.session_state.analysis_results.get('decomp_period', 7)
                                    season_period_fc = st.number_input("Seasonality Period for Holt-Winters",
                                                                       min_value=2, value=max(2, default_seasonal_period), # Ensure period >= 2
                                                                       key="fc_hw_period")
                                    if len(train_data) > 2 * season_period_fc:
                                         seasonal_type = st.radio("Seasonal Type", ('add', 'mul'), key='fc_seasonal_type', help="'add' for constant seasonal effect, 'mul' for proportional.")
                                         model_params['seasonal_periods'] = season_period_fc
                                         model_params['seasonal'] = seasonal_type
                                         model = ExponentialSmoothing(train_data, trend='add', seasonal=seasonal_type, seasonal_periods=season_period_fc)
                                         model_fitted = model.fit()
                                    else:
                                        st.error(f"Not enough data ({len(train_data)}) for Holt-Winters with period {season_period_fc}. Need at least {2 * season_period_fc}.")
                                        model_fitted = None

                                if model_fitted:
                                    st.write("---")
                                    st.write("### Forecast Results")
                                    forecast = model_fitted.forecast(forecast_periods)

                                    # Create future index based on training data index frequency
                                    last_date = train_data.index[-1]
                                    freq = pd.infer_freq(train_data.index) or st.session_state.analysis_results.get('resample_freq')

                                    if not freq:
                                        st.warning("Could not infer frequency from time index, even after potential resampling. Assuming daily ('D'). Forecast index might be inaccurate.", icon="⚠️")
                                        freq = 'D' # Default to daily if inference still fails

                                    try:
                                         future_index = pd.date_range(start=last_date, periods=forecast_periods + 1, freq=freq)[1:] # Generate N periods *after* last_date
                                    except ValueError as e:
                                         st.error(f"Error generating future date range with frequency '{freq}': {e}. Trying index arithmetic (less robust).")
                                         time_diff = train_data.index[-1] - train_data.index[-2] if len(train_data.index) > 1 else pd.Timedelta(days=1)
                                         future_index = pd.Index([last_date + time_diff * i for i in range(1, forecast_periods + 1)])

                                    forecast_df = pd.DataFrame({'Forecast': forecast}, index=future_index)

                                    st.dataframe(forecast_df)

                                    fig_fc, ax_fc = plt.subplots(figsize=(12, 6))
                                    ax_fc.plot(train_data.index, train_data, label='Historical Data')
                                    ax_fc.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='red', linestyle='--')
                                    ax_fc.set_title(f'Forecast for {forecast_col} ({forecast_model_type})')
                                    ax_fc.legend()
                                    st.pyplot(fig_fc)

                                    # Store forecast details and plot for report
                                    st.session_state.analysis_results['forecast_details'] = f"Forecast generated for '{forecast_col}' using {forecast_model_type} for the next {forecast_periods} periods."
                                    try:
                                        fc_img = BytesIO()
                                        fig_fc.savefig(fc_img, format='png', bbox_inches='tight')
                                        plt.close(fig_fc)
                                        fc_img.seek(0)
                                        st.session_state.analysis_results['forecast_plot'] = fc_img
                                    except Exception as e:
                                        st.error(f"Failed to save forecast plot: {e}")

                                    # Simple Backtest Evaluation
                                    holdout_periods = min(max(1, forecast_periods // 2) , len(train_data) // 5) # Use max 20% or half forecast period for holdout
                                    if holdout_periods > 1 and len(train_data) > holdout_periods + 5: # Ensure enough data for train/test split
                                        st.write(f"--- \n#### Simple Backtest Evaluation (holding out last {holdout_periods} periods):")
                                        train_eval = train_data[:-holdout_periods]
                                        actual_eval = train_data[-holdout_periods:]

                                        model_eval_fitted = None
                                        try:
                                            # Refit model on the shorter training data
                                            if forecast_model_type == "Simple Exponential Smoothing":
                                                 model_eval_fitted = ExponentialSmoothing(train_eval, trend=None, seasonal=None).fit()
                                            elif forecast_model_type == "Holt's Linear Trend":
                                                 model_eval_fitted = ExponentialSmoothing(train_eval, trend='add', seasonal=None).fit()
                                            elif forecast_model_type == "Holt-Winters Seasonal":
                                                # Check if enough data for seasonal period on eval set
                                                hw_period = model_params.get('seasonal_periods')
                                                if hw_period and len(train_eval) > 2 * hw_period:
                                                     model_eval_fitted = ExponentialSmoothing(train_eval, trend='add', seasonal=model_params.get('seasonal'), seasonal_periods=hw_period).fit()
                                                else: model_eval_fitted = None # Cannot fit on eval data

                                            if model_eval_fitted:
                                                forecast_eval = model_eval_fitted.forecast(holdout_periods)
                                                mae = mean_absolute_error(actual_eval, forecast_eval)
                                                rmse = np.sqrt(mean_squared_error(actual_eval, forecast_eval))
                                                eval_text = f"- Mean Absolute Error (MAE): {mae:.4f}\n- Root Mean Squared Error (RMSE): {rmse:.4f}"
                                                st.text(eval_text)
                                                # Store metrics for insights
                                                st.session_state.analysis_results['forecast_eval'] = eval_text
                                                st.session_state.analysis_results['forecast_mae'] = mae
                                                st.session_state.analysis_results['forecast_rmse'] = rmse
                                            else:
                                                st.warning("Could not perform backtest evaluation (model fitting failed on evaluation training set or insufficient data).")
                                        except Exception as e:
                                            st.error(f"Backtest evaluation failed: {e}")
                                    elif holdout_periods <= 1:
                                         st.info("Not enough data or forecast periods to perform a meaningful backtest evaluation.")

                            except Exception as e:
                                st.error(f"Forecasting process failed: {e}")
                                st.error(f"Traceback: {traceback.format_exc()}")
                        else:
                            st.warning(f"Not enough data points ({len(train_data)}) in '{forecast_col}' to perform forecasting. Need > 5.")
                else:
                    st.warning("No numeric columns available in the time series data for forecasting.")
            else:
                st.warning("Forecasting requires a valid datetime index. Please select/configure a valid date/time column in Section 2.")

        # --- 5. Classification Task ---
        with st.expander("5️⃣ Classification"):
            st.markdown("Build a model to predict a categorical target variable based on numeric features.")
            st.session_state.analysis_results.pop('classification_details', None)
            st.session_state.analysis_results.pop('classification_report', None)
            st.session_state.analysis_results.pop('confusion_matrix_plot', None)
            st.session_state.analysis_results.pop('feature_importances', None)
            st.session_state.analysis_results.pop('classification_accuracy', None)

            st.write("### Classification Setup")
            # Identify potential targets (object/category/bool or low-cardinality int)
            potential_targets = []
            ts_index_name = st.session_state.analysis_results.get('ts_index') # Get TS index name if set
            for col in df.columns:
                 if col != ts_index_name: # Exclude the time series index column if it exists
                    n_unique = df[col].nunique()
                    is_candidate = pd.api.types.is_object_dtype(df[col]) or \
                                   pd.api.types.is_categorical_dtype(df[col]) or \
                                   pd.api.types.is_bool_dtype(df[col]) or \
                                   (pd.api.types.is_integer_dtype(df[col]) and n_unique < 15)

                    if is_candidate and n_unique > 1 and n_unique < 15: # Exclude constant columns and high cardinality ints
                        potential_targets.append(col)

            if not potential_targets:
                st.warning("No suitable categorical target columns (with 2-14 unique values) found in the working data (excluding time index).")
            else:
                target_column = st.selectbox("Select Target Variable (Categorical)", potential_targets, key="clf_target_select")

                if target_column:
                    # Features: numeric columns, excluding the target
                    available_features = df.select_dtypes(include=np.number).columns.tolist()
                    if target_column in available_features: # Should not happen often but check
                        available_features.remove(target_column)
                    # Also remove the time series index if it happens to be numeric (unlikely but possible)
                    if ts_index_name and ts_index_name in available_features:
                        available_features.remove(ts_index_name)

                    if not available_features:
                        st.warning("No suitable numeric feature columns found in the working data for classification.")
                    else:
                        selected_features = st.multiselect("Select Feature Columns (Numeric)", available_features, default=available_features, key="clf_features_select")

                        if selected_features:
                            clf_model_type = st.selectbox("Select Classification Model", ["Random Forest", "Logistic Regression"], key="clf_model_select")
                            st.write("---")
                            st.write("### Classification Results")

                            try:
                                cols_to_use = [target_column] + selected_features
                                df_clf = df[cols_to_use].copy()
                                initial_rows = len(df_clf)
                                df_clf = df_clf.dropna() # Drop rows with NaNs *only in selected columns*
                                final_rows = len(df_clf)

                                st.info(f"Using columns: Target='{target_column}', Features={', '.join(selected_features)}")
                                st.info(f"Data points available for classification after dropping NaNs in selected columns: {final_rows} (from {initial_rows})")

                                if final_rows < 30: # Increased threshold for meaningful split/training
                                    st.warning(f"Very few data points ({final_rows}) available. Classification may be unreliable or fail.")
                                else:
                                    X = df_clf[selected_features]
                                    y_raw = df_clf[target_column]

                                    # Encode target labels robustly
                                    y_encoded = y_raw.astype('category')
                                    target_classes_map = dict(enumerate(y_encoded.cat.categories))
                                    y = y_encoded.cat.codes # Numerical labels for sklearn

                                    st.write("Target Variable Class Distribution:")
                                    st.dataframe(y_raw.value_counts(normalize=True).map("{:.1%}".format).to_frame(name='Percentage'))

                                    num_classes = len(target_classes_map)
                                    if num_classes < 2:
                                        st.error("Target variable has only one class after filtering. Cannot perform classification.")
                                    else:
                                        # Train/Test Split
                                        stratify_param = y if num_classes > 1 else None
                                        try:
                                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify_param)
                                        except ValueError as e:
                                            st.warning(f"Could not stratify train/test split (likely due to few samples in a class). Performing regular split. Error: {e}")
                                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

                                        st.info(f"Training model: {clf_model_type} on {len(X_train)} samples, Testing on {len(X_test)} samples.")

                                        # Define and Train Model with Hyperparameters
                                        if clf_model_type == "Random Forest":
                                            n_estimators = st.slider("Number of Trees (n_estimators)", 50, 500, 100, 50, key='rf_n_estimators')
                                            max_depth = st.slider("Max Depth of Trees (max_depth)", 3, 30, 10, 1, key='rf_max_depth', help="Controls complexity. Lower values prevent overfitting.")
                                            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, class_weight='balanced')
                                        elif clf_model_type == "Logistic Regression":
                                            log_reg_c = st.number_input("Regularization Strength (C)", 0.01, 10.0, 1.0, 0.1, key='lr_c', help="Lower values = stronger regularization (simpler model).")
                                            clf = LogisticRegression(C=log_reg_c, random_state=42, class_weight='balanced', max_iter=1000, solver='liblinear')

                                        clf.fit(X_train, y_train)
                                        preds = clf.predict(X_test)
                                        pred_proba = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None # Get probabilities if available

                                        # --- Display Results ---
                                        # Convert numerical predictions/test back to original labels for report/CM
                                        pred_labels = pd.Series(preds).map(target_classes_map).astype(y_encoded.dtype)
                                        y_test_labels = pd.Series(y_test).map(target_classes_map).astype(y_encoded.dtype)
                                        target_class_names = [str(c) for c in target_classes_map.values()]

                                        clf_details_text = f"Model: {clf_model_type} predicting '{target_column}' using {len(selected_features)} features."
                                        st.session_state.analysis_results['classification_details'] = clf_details_text

                                        # Accuracy
                                        accuracy = accuracy_score(y_test, preds)
                                        st.metric(label="Model Accuracy (Test Set)", value=f"{accuracy:.2%}")
                                        st.session_state.analysis_results['classification_accuracy'] = accuracy * 100 # Store as percentage

                                        # Classification Report
                                        report = classification_report(y_test_labels, pred_labels, target_names=target_class_names, zero_division=0)
                                        st.text("Classification Report:")
                                        st.text(report)
                                        st.session_state.analysis_results['classification_report'] = report

                                        # Confusion Matrix
                                        st.write("#### Confusion Matrix")
                                        st.markdown("Rows = Actual Class, Columns = Predicted Class")
                                        try:
                                            fig_cm = plot_confusion_matrix(y_test_labels, pred_labels, classes=target_class_names)
                                            st.pyplot(fig_cm)
                                            cm_img = BytesIO()
                                            fig_cm.savefig(cm_img, format='png', bbox_inches='tight')
                                            plt.close(fig_cm)
                                            cm_img.seek(0)
                                            st.session_state.analysis_results['confusion_matrix_plot'] = cm_img
                                        except Exception as e:
                                            st.error(f"Failed to plot/save confusion matrix: {e}")


                                        # Feature Importances (for RF)
                                        if clf_model_type == "Random Forest" and hasattr(clf, 'feature_importances_'):
                                            st.write("#### Feature Importances (Random Forest)")
                                            st.markdown("Shows the relative importance of each feature in making predictions.")
                                            importances = pd.Series(clf.feature_importances_, index=selected_features).sort_values(ascending=False)
                                            st.bar_chart(importances)
                                            # Store top N importances string for report/insights
                                            top_n = min(5, len(importances))
                                            importances_str = importances.head(top_n).to_string(float_format="{:.3f}".format)
                                            st.session_state.analysis_results['feature_importances'] = importances_str

                            except Exception as e:
                                st.error(f"Classification process failed: {e}")
                                st.error(f"Traceback: {traceback.format_exc()}")
                        else:
                            st.warning("Please select at least one numeric feature column.")
                else:
                    st.info("Select a target column to enable classification.")


        # --- 6. AI Insights & Reporting ---
        with st.expander("6️⃣ AI Insights & Reporting"):
            st.markdown("---")
            st.write("### 🤖 AI Insights Summary & Report Export")
            st.markdown("Automatically generated interpretation based on the analysis performed above.")

            # Generate insights based *only* on session_state results
            final_insights = generate_dynamic_insights(st.session_state.analysis_results)
            st.markdown(final_insights)
            # Store the generated markdown itself in session state for the PDF
            st.session_state.analysis_results['ai_insights'] = final_insights


            st.markdown("---")
            st.write("### Export Full Report")
            if st.button("🗕️ Generate PDF Report", key="pdf_button"):
                with st.spinner("Generating PDF... This may take a moment."):
                    try:
                        # Prepare data for PDF generation (pass relevant results)
                        report_data_for_pdf = {}
                        # Copy relevant keys from session state
                        keys_to_copy = [
                            'data_shape', 'imputation_method', 'summary', 'missing',
                            'correlation_plot', 'high_correlations', 'correlation_checked', # Added correlation details
                            'ts_index', 'resample_freq', 'resample_agg', # Added resampling info
                            'time_series_plots', 'decomp_period', 'adf_results', # Added decomp period
                            'anomaly_method', 'anomaly_threshold', # Added threshold
                            'anomalies_found', 'anomalies_found_details', # Added details for PDF
                            'forecast_details', 'forecast_eval', 'forecast_mae', 'forecast_rmse', # Added specific metrics
                            'forecast_plot', 'classification_details', 'classification_accuracy', # Added accuracy
                            'classification_report', 'confusion_matrix_plot',
                            'feature_importances', 'ai_insights' # Pass the generated markdown insight text
                        ]
                        for key in keys_to_copy:
                            if key in st.session_state.analysis_results:
                                report_data_for_pdf[key] = st.session_state.analysis_results[key]

                        pdf_bytes = generate_pdf_report(report_data_for_pdf)
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=pdf_bytes,
                            file_name="financial_analysis_report.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"Failed to generate PDF report: {e}")
                        st.error(f"Traceback: {traceback.format_exc()}")

    elif not st.session_state.data_loaded and uploaded_file:
         # This case might happen if loading failed after upload
         st.warning("Data loading seems to have failed. Please check the file format and try uploading again.")
    elif not uploaded_file:
         st.info(" PLease Upload a CSV file using the sidebar to begin analysis.")
else:
    st.info(" PLease Upload a CSV file using the sidebar to begin analysis.")