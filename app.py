# 🧠 AI-Powered Financial Data Analyst with Forecasting, Classification, PDF Export and Enhanced Features

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO, BytesIO
from fpdf import FPDF  # Standardized to fpdf2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import traceback # For more detailed error logging if needed

# --- Page Configuration ---
st.set_page_config(page_title="AI Financial Analyst", layout="wide", initial_sidebar_state="expanded")

# --- Helper Functions ---
@st.cache_data # Persist data across reruns for the same input file
def load_data(uploaded_file):
    """Loads data from uploaded CSV file."""
    try:
        # Reset the file pointer just in case
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        # Attempt basic cleaning: remove leading/trailing whitespace from headers
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

# Use st.cache_data for functions that return data/objects to be reused
@st.cache_data
def safe_convert_to_datetime(df, col_name):
    """Safely converts a column to datetime, handling errors. Returns the converted series or None."""
    if col_name not in df.columns:
        st.error(f"Column '{col_name}' not found in DataFrame.")
        return None
    try:
        # Make a copy to avoid modifying the cached DataFrame directly if conversion is done inplace
        return pd.to_datetime(df[col_name], errors='coerce')
    except Exception as e:
        st.warning(f"Could not convert '{col_name}' to datetime: {e}. Please select a valid date/time column.")
        return None

# Use st.cache_resource for things like ML models or plotting objects if they are expensive to create
# For simple plots like this, caching might be overkill unless generation is very slow.
# Not caching plots for now as they depend on interactive selections.
def plot_confusion_matrix(y_true, y_pred, classes):
    """Plots a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    return fig

def generate_pdf_report(report_data):
    """Generates a PDF report from collected analysis results."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Financial Data Analysis Report", 0, 1, 'C')
    pdf.ln(10)

    # --- Section 1: Overview ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Dataset Overview", 0, 1)
    pdf.set_font("Arial", size=10)
    if 'data_shape' in report_data:
         pdf.multi_cell(0, 5, f"Original Shape: {report_data['data_shape']}\n")
    if 'imputation_method' in report_data and report_data['imputation_method'] != "None (Keep As Is)":
         pdf.multi_cell(0, 5, f"Imputation Method Applied: {report_data['imputation_method']}\n")
    if 'summary' in report_data:
        pdf.multi_cell(0, 5, f"Summary Statistics:\n{report_data['summary']}\n")
    if 'missing' in report_data:
        pdf.multi_cell(0, 5, f"Missing Values Summary:\n{report_data['missing']}\n")
    pdf.ln(5)

    # --- Section 2: Correlation ---
    if 'correlation_plot' in report_data:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "2. Correlation Analysis", 0, 1)
        try:
            pdf.image(report_data['correlation_plot'], x=10, y=pdf.get_y(), w=180)
            pdf.ln(85) # Adjust spacing as needed based on plot height
        except Exception as e:
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 5, f"(Error embedding correlation plot: {e})")
            pdf.ln(5)

    # --- Section 3: Time Series ---
    if 'time_series_plots' in report_data and report_data['time_series_plots']:
        pdf.add_page() # Add new page for potentially many plots
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "3. Time Series Analysis", 0, 1)
        pdf.set_font("Arial", size=10)
        if 'ts_index' in report_data:
            pdf.multi_cell(0, 5, f"Time Series Index: {report_data['ts_index']}\n")
        for title, plot_bytes in report_data['time_series_plots'].items():
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 10, title, 0, 1)
            try:
                # Check if plot needs more space vertically
                if pdf.get_y() > 200: # Roughly check if near bottom
                    pdf.add_page()
                pdf.image(plot_bytes, x=10, y=pdf.get_y(), w=180)
                pdf.ln(85) # Adjust spacing
            except Exception as e:
                pdf.set_font("Arial", size=10)
                pdf.multi_cell(0, 5, f"(Error embedding plot '{title}': {e})")
                pdf.ln(5)
        pdf.ln(5)

    # --- Section 4: Anomaly Detection ---
    if 'anomalies_found' in report_data and report_data['anomalies_found']:
        if pdf.get_y() > 220: pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "4. Anomaly Detection", 0, 1)
        pdf.set_font("Arial", size=10)
        if 'anomaly_method' in report_data:
             pdf.multi_cell(0, 5, f"Method Used: {report_data['anomaly_method']}\n")
        for col, anom_df_str in report_data['anomalies_found'].items():
            pdf.set_font("Arial", 'B', 10)
            pdf.multi_cell(0, 5, f"Anomalies in '{col}':")
            pdf.set_font("Arial", size=8)
            pdf.multi_cell(0, 4, anom_df_str) # Display string representation
            pdf.ln(2)
        pdf.ln(5)


    # --- Section 5: Forecasting ---
    if 'forecast_plot' in report_data:
        if pdf.get_y() > 180: pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "5. Forecasting", 0, 1)
        pdf.set_font("Arial", size=10)
        if 'forecast_details' in report_data:
            pdf.multi_cell(0, 5, f"{report_data['forecast_details']}\n")
        if 'forecast_eval' in report_data:
            pdf.multi_cell(0, 5, f"Backtest Evaluation:\n{report_data['forecast_eval']}\n")

        try:
            pdf.image(report_data['forecast_plot'], x=10, y=pdf.get_y(), w=180)
            pdf.ln(85) # Adjust spacing
        except Exception as e:
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 5, f"(Error embedding forecast plot: {e})")
            pdf.ln(5)


    # --- Section 6: Classification ---
    if 'classification_report' in report_data:
        if pdf.get_y() > 150: pdf.add_page() # Check space before adding report + matrix
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "6. Classification Results", 0, 1)
        pdf.set_font("Arial", size=10)
        if 'classification_details' in report_data:
             pdf.multi_cell(0, 5, f"{report_data['classification_details']}\n")

        pdf.set_font("Arial", 'B', 10)
        pdf.multi_cell(0, 5, f"Classification Report:\n")
        pdf.set_font("Courier", size=8) # Use monospace for report
        pdf.multi_cell(0, 4, report_data['classification_report'])
        pdf.ln(5)

        if 'confusion_matrix_plot' in report_data:
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 10, "Confusion Matrix:", 0, 1)
            try:
                # Check if plot needs more space vertically
                if pdf.get_y() > 200: # Add new page if needed
                    pdf.add_page()
                pdf.image(report_data['confusion_matrix_plot'], x=10, y=pdf.get_y(), w=100)
                pdf.ln(60) # Adjust spacing
            except Exception as e:
                pdf.set_font("Arial", size=10)
                pdf.multi_cell(0, 5, f"(Error embedding confusion matrix: {e})")
                pdf.ln(5)
        if 'feature_importances' in report_data:
            pdf.set_font("Arial", 'B', 10)
            pdf.multi_cell(0, 5, f"Top Feature Importances:\n{report_data['feature_importances']}")
            pdf.ln(5)

    # --- Section 7: AI Insights ---
    if 'ai_insights' in report_data:
        if pdf.get_y() > 200: pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "7. AI Insights Summary", 0, 1)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 5, report_data['ai_insights']) # Use the pre-generated text
        pdf.ln(5)

    return pdf.output(dest='S').encode('latin1')

# --- Function to Generate Dynamic Insights ---
def generate_dynamic_insights(results):
    insight_text = "Insights Summary based on Analysis:\n\n"
    if 'data_shape' in results:
        insight_text += f"**Dataset Overview:**\n- Initial dataset dimensions: {results['data_shape']}.\n"
        if 'imputation_method' in results and results['imputation_method'] != "None (Keep As Is)":
             insight_text += f"- Missing values handled using: {results['imputation_method']}.\n"
    if 'missing' in results and results['missing'] != "No missing values.":
        insight_text += f"- Missing values were detected. Summary:\n{results['missing']}\n"
    elif 'missing' in results:
         insight_text += "- No missing values were detected in the initial scan.\n"

    if 'correlation_plot' in results:
        insight_text += "**Correlation Insights:**\n- Heatmap generated. Look for strong positive/negative correlations between numeric variables.\n"

    if 'ts_index' in results:
        insight_text += f"**Time Series Observations (Index: {results['ts_index']}):**\n"
        if 'time_series_plots' in results:
            if 'Rolling Statistics' in results['time_series_plots']:
                insight_text += "- Rolling mean & standard deviation plotted, indicating trends and volatility periods.\n"
            if 'Decomposition' in results['time_series_plots']:
                insight_text += "- Time series decomposition (trend, seasonal, residual) performed. Check plots for patterns.\n"
        if 'adf_results' in results:
             insight_text += "- Stationarity (ADF Test) results:\n"
             for col, res in results['adf_results'].items():
                 insight_text += f"  - {col}: {res}\n"

    if 'anomalies_found' in results and results['anomalies_found']:
        insight_text += f"**Anomaly Detection ({results.get('anomaly_method', 'N/A')}):**\n"
        insight_text += f"- Potential outliers detected in columns: {', '.join(results['anomalies_found'].keys())}. Review data points listed in Section 3.\n"
    elif 'anomaly_method' in results: # Check if detection was run but none found
        insight_text += f"**Anomaly Detection ({results.get('anomaly_method', 'N/A')}):**\n- No significant anomalies detected with current settings.\n"


    if 'forecast_details' in results:
        insight_text += f"**Forecasting Summary:**\n- {results['forecast_details']}.\n"
        if 'forecast_eval' in results:
            insight_text += f"- Backtest Evaluation: {results['forecast_eval']}\n"

    if 'classification_details' in results:
        insight_text += f"**Classification Performance:**\n- {results['classification_details']}.\n"
        insight_text += "- Performance metrics (precision, recall, F1) in report. Confusion matrix shows prediction accuracy per class.\n"
        if 'feature_importances' in results:
            insight_text += f"- Key features driving predictions:\n{results['feature_importances']}\n"

    insight_text += "\n**Disclaimer:** Insights are auto-generated. Validate findings with domain expertise."
    return insight_text


# --- Global Variables & Session State ---
# Use clear naming for session state keys
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'dataframe' not in st.session_state:
    st.session_state.dataframe = None # Store the working dataframe

# --- Sidebar ---
st.sidebar.title("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload Financial Data (CSV)", type=["csv"], key="file_uploader")

# Reset state if new file uploaded
if uploaded_file is not None:
     # Check if it's a new file instance; file_uploader resets state on new upload
     # We need a robust way to know if it's *really* a new file or just a rerun
     # A simple approach: Store filename and size, reset if they change.
     current_file_id = f"{uploaded_file.name}-{uploaded_file.size}"
     if st.session_state.get('current_file_id') != current_file_id:
         st.session_state.analysis_results = {}
         st.session_state.data_loaded = False
         st.session_state.dataframe = None
         st.session_state.current_file_id = current_file_id
         st.info("New file detected, resetting analysis state.")


# --- Main App ---
st.title("📊 AI-Powered Financial Analyst")
st.markdown("""
Welcome! Upload your financial data in CSV format. This tool performs exploratory data analysis, time series analysis, forecasting, and classification.
""")

if uploaded_file:
    # Load data only if not already loaded or if file changed
    if not st.session_state.data_loaded:
        df_original = load_data(uploaded_file)
        if df_original is not None:
            st.session_state.dataframe = df_original.copy()
            st.session_state.data_loaded = True
            st.session_state.analysis_results['data_shape'] = str(df_original.shape)
            st.success("File uploaded and initial data loaded successfully!")
        else:
            st.error("Failed to load data from the uploaded file.")
            st.stop() # Stop execution if loading failed

    # Work with the dataframe stored in session state
    if st.session_state.data_loaded and st.session_state.dataframe is not None:
        df = st.session_state.dataframe # Use the working copy

        # --- 1. Data Exploration & Preparation ---
        with st.expander("1️⃣ Data Exploration & Preparation", expanded=True):
            st.write("### Preview of Original Dataset Head")
            # Show original head for reference, but analysis runs on potentially modified df
            st.dataframe(load_data(uploaded_file).head() if uploaded_file else "Upload file")

            st.write("### Working Dataset Information")
            buffer = StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()
            st.text(info_str)

            st.write("### Summary Statistics (Working Data)")
            numeric_df_desc = df.select_dtypes(include=np.number)
            if not numeric_df_desc.empty:
                summary_stats = numeric_df_desc.describe()
                st.dataframe(summary_stats)
                st.session_state.analysis_results['summary'] = summary_stats.to_string()
            else:
                st.warning("No numeric columns found in the working data for summary statistics.")
                if 'summary' in st.session_state.analysis_results: del st.session_state.analysis_results['summary']


            st.write("### Missing Values (Working Data)")
            missing_values = df.isnull().sum()
            missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
            st.session_state.analysis_results['missing'] = missing_values.to_string() if not missing_values.empty else "No missing values."

            if not missing_values.empty:
                st.dataframe(missing_values.to_frame(name='Missing Count'))
                impute_method = st.selectbox("Handle Missing Values for subsequent analysis?",
                                             ["None (Keep As Is)", "Drop Rows with NaNs", "Fill with Mean (Numeric)", "Fill with Median (Numeric)", "Forward Fill"],
                                             key="impute_select")
                st.session_state.analysis_results['imputation_method'] = impute_method

                # Apply imputation based on selection - IMPORTANT: Modify df for subsequent steps
                if impute_method != "None (Keep As Is)":
                    df_processed = df.copy() # Create a copy to apply imputation
                    if impute_method == "Drop Rows with NaNs":
                        rows_before = len(df_processed)
                        df_processed = df_processed.dropna()
                        rows_after = len(df_processed)
                        st.write(f"Shape after dropping NaNs: {df_processed.shape}. ({rows_before - rows_after} rows removed)")
                    elif impute_method in ["Fill with Mean (Numeric)", "Fill with Median (Numeric)"]:
                        numeric_cols_na = df_processed.select_dtypes(include=np.number).columns
                        imputed_cols = []
                        for col in numeric_cols_na:
                            if df_processed[col].isnull().any():
                                fill_value = df_processed[col].mean() if impute_method == "Fill with Mean (Numeric)" else df_processed[col].median()
                                df_processed[col] = df_processed[col].fillna(fill_value)
                                imputed_cols.append(col)
                        st.write(f"{len(imputed_cols)} numeric columns filled with {impute_method.split(' ')[2]}: {', '.join(imputed_cols)}")
                    elif impute_method == "Forward Fill":
                        df_processed = df_processed.ffill()
                        st.write("Forward fill applied to all columns.")

                    st.session_state.dataframe = df_processed # Update the main df in session state
                    df = df_processed # Update local df variable as well
                    st.write("Preview after potential imputation:")
                    st.dataframe(df.head())
                else:
                    # If user selected "None", make sure we are using the original loaded data
                    # This happens implicitly because df was loaded from session state
                    pass
            else:
                st.info("No missing values found in the working data.")
                st.session_state.analysis_results['imputation_method'] = "None (Keep As Is)"


            st.write("### Correlation Heatmap (Numeric Columns)")
            numeric_df_corr = df.select_dtypes(include=[np.number])
            if not numeric_df_corr.empty and len(numeric_df_corr.columns) > 1:
                corr_matrix = numeric_df_corr.corr()
                fig_corr, ax_corr = plt.subplots(figsize=(10, 7))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
                st.pyplot(fig_corr)
                try:
                    corr_img = BytesIO()
                    fig_corr.savefig(corr_img, format='png', bbox_inches='tight')
                    plt.close(fig_corr)
                    corr_img.seek(0)
                    st.session_state.analysis_results['correlation_plot'] = corr_img
                except Exception as e:
                    st.error(f"Failed to save correlation plot: {e}")
                    if 'correlation_plot' in st.session_state.analysis_results: del st.session_state.analysis_results['correlation_plot']

            else:
                st.warning("Correlation heatmap requires at least two numeric columns in the working data.")
                if 'correlation_plot' in st.session_state.analysis_results: del st.session_state.analysis_results['correlation_plot']


            st.write("### Distribution of Numeric Variables")
            numeric_cols_dist = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols_dist:
                col_to_plot = st.selectbox("Select numeric column for distribution plot", numeric_cols_dist, key="dist_select")
                if col_to_plot:
                    fig_dist, ax_dist = plt.subplots()
                    sns.histplot(df[col_to_plot].dropna(), kde=True, ax=ax_dist)
                    ax_dist.set_title(f'Distribution of {col_to_plot}')
                    st.pyplot(fig_dist)
                    plt.close(fig_dist) # Close plot to free memory
            else:
                st.info("No numeric columns available for distribution plots.")

        # --- 2. Time Series Analysis ---
        with st.expander("2️⃣ Time Series Analysis"):
            st.write("### Select Date/Time Column")
            # Prioritize columns that look like dates based on name or dtype
            potential_date_cols = df.columns.tolist()
            # Simple heuristic: prioritize columns with 'date', 'time' in name, or object/datetime64 dtype
            sorted_date_cols = sorted(potential_date_cols, key=lambda x: (
                0 if 'date' in x.lower() or 'time' in x.lower() else 1,
                0 if pd.api.types.is_datetime64_any_dtype(df[x]) else (1 if df[x].dtype == 'object' else 2),
                x # Alphabetical tie-breaker
                ))

            date_col = st.selectbox("Select the primary date/time column for time series analysis",
                                    sorted_date_cols, index=0, key="date_col_select")

            df_ts = None # Initialize df_ts
            st.session_state.analysis_results.pop('ts_index', None) # Clear previous index
            st.session_state.analysis_results.pop('time_series_plots', None) # Clear previous plots
            st.session_state.analysis_results.pop('adf_results', None) # Clear previous ADF results


            if date_col:
                datetime_series = safe_convert_to_datetime(df, date_col)
                if datetime_series is not None and not datetime_series.isnull().all():
                    # Create a temporary copy for TS operations
                    df_temp_ts = df.copy()
                    df_temp_ts[date_col] = datetime_series
                    df_temp_ts = df_temp_ts.dropna(subset=[date_col]) # Drop rows where date conversion failed
                    df_temp_ts = df_temp_ts.sort_values(by=date_col) # Ensure chronological order
                    try:
                        df_ts = df_temp_ts.set_index(date_col)
                        # Check for duplicate indices, which can cause issues
                        if df_ts.index.has_duplicates:
                            st.warning(f"Duplicate timestamps found in '{date_col}'. Aggregating using mean. Consider preprocessing your data for duplicates.")
                            # Example aggregation: df_ts = df_ts.groupby(df_ts.index).mean() # Choose appropriate aggregation
                            # For now, let's just warn and proceed, statsmodels might handle some cases
                        st.success(f"Successfully set '{date_col}' as time series index. {len(df_ts)} rows.")
                        st.session_state.analysis_results['ts_index'] = date_col
                        st.session_state.analysis_results['time_series_plots'] = {} # Initialize dict for plots
                        st.session_state.analysis_results['adf_results'] = {} # Initialize dict for ADF results

                        numeric_cols_ts = df_ts.select_dtypes(include=np.number).columns.tolist()

                        if numeric_cols_ts:
                            st.write("### Time Series Plot")
                            ts_col_select = st.multiselect("Select numeric columns to plot over time", numeric_cols_ts, default=numeric_cols_ts[0] if numeric_cols_ts else None, key="ts_plot_select")
                            if ts_col_select:
                                st.line_chart(df_ts[ts_col_select])

                            st.write("### Rolling Statistics")
                            roll_col = st.selectbox("Select column for rolling statistics", numeric_cols_ts, key="roll_col_select")
                            # Adjust max window size dynamically
                            max_roll_window = max(3, min(90, len(df_ts)//2))
                            if max_roll_window > 3:
                                roll_window = st.slider("Select rolling window size (periods)", min_value=3, max_value=max_roll_window, value=min(14, max_roll_window), key="roll_window_slider")
                                if roll_col and roll_window and roll_window < len(df_ts[roll_col].dropna()):
                                    # Calculate on the fly, don't add to df_ts permanently unless needed elsewhere
                                    rolling_mean = df_ts[roll_col].rolling(window=roll_window).mean()
                                    rolling_std = df_ts[roll_col].rolling(window=roll_window).std()

                                    fig_roll, ax_roll = plt.subplots(figsize=(12, 6))
                                    ax_roll.plot(df_ts.index, df_ts[roll_col], label='Original', alpha=0.7)
                                    ax_roll.plot(rolling_mean.index, rolling_mean, label=f'Rolling Mean ({roll_window} periods)', color='orange')
                                    ax_roll.plot(rolling_std.index, rolling_std, label=f'Rolling Std Dev ({roll_window} periods)', color='red', linestyle='--')
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
                                     st.warning(f"Select a column and ensure window size ({roll_window}) is smaller than the number of data points ({len(df_ts[roll_col].dropna())}).")
                            else:
                                st.info("Not enough data points for rolling statistics.")


                            st.write("### Time Series Decomposition (Trend, Seasonality, Residuals)")
                            decomp_col = st.selectbox("Select column for decomposition", numeric_cols_ts, key='decomp_sel')
                            # Infer frequency - this can be tricky
                            inferred_freq = pd.infer_freq(df_ts.index)
                            st.write(f"Inferred frequency: {inferred_freq if inferred_freq else 'Could not infer frequency (data might be irregular)'}")
                            # Suggest period based on inferred freq
                            suggested_period = 7 if inferred_freq == 'D' else (12 if inferred_freq and ('M' in inferred_freq or 'Q' in inferred_freq) else (24 if inferred_freq == 'H' else 1))

                            period = st.number_input("Seasonality Period (e.g., 7 for daily/weekly, 12 for monthly/yearly)",
                                                     min_value=1, value=suggested_period, key='decomp_period')

                            if decomp_col and period > 1:
                                decomp_data = df_ts[decomp_col].dropna()
                                if len(decomp_data) >= 2 * period:
                                    try:
                                        # Determine if model should be additive or multiplicative
                                        decomp_model_type = st.radio("Decomposition Model", ('additive', 'multiplicative'), key='decomp_model')
                                        result = seasonal_decompose(decomp_data, model=decomp_model_type, period=period)
                                        fig_decomp = result.plot()
                                        fig_decomp.set_size_inches(10, 8)
                                        plt.tight_layout()
                                        st.pyplot(fig_decomp)
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
                            elif period <= 1:
                                st.info("Select a column and ensure Period > 1 for decomposition.")
                            else:
                                st.info("Select a column for decomposition.")


                            st.write("### Stationarity Test (Augmented Dickey-Fuller)")
                            adf_col = st.selectbox("Select column for ADF Test", numeric_cols_ts, key='adf_sel')
                            if adf_col:
                                adf_data = df_ts[adf_col].dropna()
                                if len(adf_data) > 5: # Need some data points for ADF
                                    try:
                                        result = adfuller(adf_data)
                                        p_value = result[1]
                                        st.write(f"**Results for {adf_col}:**")
                                        st.write(f"- ADF Statistic: {result[0]:.4f}")
                                        st.write(f"- p-value: {p_value:.4f}")
                                        # st.write('Critical Values:')
                                        # for key, value in result[4].items():
                                        #     st.write(f'\t{key}: {value:.4f}')

                                        adf_summary = ""
                                        if p_value <= 0.05:
                                            st.success(f"Result suggests the time series '{adf_col}' is likely stationary (p <= 0.05).")
                                            adf_summary = f"Likely Stationary (p={p_value:.3f})"
                                        else:
                                            st.warning(f"Result suggests the time series '{adf_col}' is likely non-stationary (p > 0.05). Consider differencing.")
                                            adf_summary = f"Likely Non-Stationary (p={p_value:.3f})"
                                        st.session_state.analysis_results['adf_results'][adf_col] = adf_summary

                                    except Exception as e:
                                        st.error(f"ADF test failed for {adf_col}: {e}")
                                else:
                                    st.warning(f"Not enough non-missing data points in '{adf_col}' for ADF test.")
                        else:
                            st.warning("No numeric columns found in the data after setting the time index.")
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
            numeric_cols_anom = df.select_dtypes(include=np.number).columns.tolist()
            st.session_state.analysis_results.pop('anomalies_found', None) # Clear previous results
            st.session_state.analysis_results.pop('anomaly_method', None)

            if numeric_cols_anom:
                st.write("### Detect Outliers in Numeric Columns")
                anom_method = st.radio("Select Anomaly Detection Method", ["Z-Score", "Interquartile Range (IQR)"], key="anom_method_radio")
                st.session_state.analysis_results['anomaly_method'] = anom_method

                anomalies_found_dict = {} # Store anomalies as dict {col: dataframe_string}

                if anom_method == "Z-Score":
                    z_thresh = st.slider("Set Z-score threshold", 1.5, 5.0, 3.0, 0.1, key="z_slider")
                    for col in numeric_cols_anom:
                        # Ensure column is numeric (redundant check, but safe)
                        if pd.api.types.is_numeric_dtype(df[col]):
                            col_data = df[col].dropna()
                            if len(col_data) > 1:
                                mean = col_data.mean()
                                std = col_data.std()
                                if std > 0:
                                    z_scores = (df[col] - mean) / std # Calculate on original df column
                                    # Identify outliers using the original DataFrame index
                                    outlier_indices = df.index[np.abs(z_scores) > z_thresh]
                                    outliers = df.loc[outlier_indices, [col]] # Select only the relevant column
                                    if not outliers.empty:
                                        # Store string representation for PDF
                                        anomalies_found_dict[col] = outliers.to_string(max_rows=10)
                                elif std == 0:
                                     st.info(f"Column '{col}' has zero standard deviation, skipping Z-score.")
                            else:
                                st.info(f"Column '{col}' has insufficient data for Z-score.")


                elif anom_method == "IQR":
                    iqr_multiplier = st.slider("Set IQR Multiplier", 1.0, 3.0, 1.5, 0.1, key="iqr_slider")
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
                                    # Identify outliers using the original DataFrame index
                                    outlier_indices = df.index[(df[col] < lower_bound) | (df[col] > upper_bound)]
                                    outliers = df.loc[outlier_indices, [col]] # Select only the relevant column
                                    if not outliers.empty:
                                         anomalies_found_dict[col] = outliers.to_string(max_rows=10)
                                elif IQR == 0:
                                    st.info(f"Column '{col}' has zero Interquartile Range, skipping IQR method.")
                            else:
                                st.info(f"Column '{col}' has insufficient data for IQR.")


                if anomalies_found_dict:
                    st.write("#### Anomalies Detected:")
                    st.session_state.analysis_results['anomalies_found'] = anomalies_found_dict
                    for col, anom_df_str in anomalies_found_dict.items():
                        st.write(f"**{col}** (Top 10 shown):")
                        st.text(anom_df_str) # Show string representation in app too
                else:
                    st.info("No anomalies detected with the current settings.")
            else:
                st.warning("No numeric columns available in the working data for anomaly detection.")

        # --- 4. Forecasting ---
        with st.expander("4️⃣ Forecasting"):
            st.session_state.analysis_results.pop('forecast_details', None)
            st.session_state.analysis_results.pop('forecast_eval', None)
            st.session_state.analysis_results.pop('forecast_plot', None)

            # Check if Time Series index was set up correctly
            if df_ts is not None and isinstance(df_ts.index, pd.DatetimeIndex):
                numeric_cols_forecast = df_ts.select_dtypes(include=np.number).columns.tolist()
                if numeric_cols_forecast:
                    st.write("### Time Series Forecasting")
                    forecast_col = st.selectbox("Select column to forecast", numeric_cols_forecast, key="fc_col_select")
                    forecast_periods = st.number_input("Number of periods to forecast", min_value=1, max_value=365, value=30, key="fc_periods")
                    forecast_model_type = st.selectbox("Select Forecasting Model",
                                                        ["Simple Exponential Smoothing", "Holt's Linear Trend", "Holt-Winters Seasonal"],
                                                        key="fc_model_select")

                    if forecast_col and forecast_periods > 0:
                        # Use the time series dataframe created in Section 2
                        train_data = df_ts[forecast_col].dropna()
                        st.write(f"Using data from column '{forecast_col}' indexed by '{df_ts.index.name}'. Training data length: {len(train_data)}")

                        if len(train_data) > 5: # Need a minimum number of points
                            model = None
                            model_fitted = None
                            try:
                                if forecast_model_type == "Simple Exponential Smoothing":
                                    model = ExponentialSmoothing(train_data, trend=None, seasonal=None)
                                    model_fitted = model.fit()
                                elif forecast_model_type == "Holt's Linear Trend":
                                    model = ExponentialSmoothing(train_data, trend='add', seasonal=None)
                                    model_fitted = model.fit()
                                elif forecast_model_type == "Holt-Winters Seasonal":
                                    season_period_fc = st.number_input("Seasonality Period for Holt-Winters",
                                                                       min_value=2, value=period if 'period' in locals() and period > 1 else max(2, suggested_period), # Use period from decomp if available
                                                                       key="fc_hw_period")
                                    # Check data length against seasonality period
                                    if len(train_data) > 2 * season_period_fc:
                                         # Choose seasonal type (add/mul)
                                         seasonal_type = st.radio("Seasonal Type", ('add', 'mul'), key='fc_seasonal_type')
                                         model = ExponentialSmoothing(train_data, trend='add', seasonal=seasonal_type, seasonal_periods=season_period_fc)
                                         model_fitted = model.fit()
                                    else:
                                        st.error(f"Not enough data ({len(train_data)}) for Holt-Winters with period {season_period_fc}. Need at least {2 * season_period_fc}.")
                                        model_fitted = None

                                if model_fitted:
                                    forecast = model_fitted.forecast(forecast_periods)
                                    # Create future index based on training data index
                                    last_date = train_data.index[-1]
                                    freq = pd.infer_freq(train_data.index)
                                    if not freq:
                                        st.warning("Could not infer frequency from time index. Assuming daily ('D'). Forecast index might be inaccurate.")
                                        freq = 'D' # Default to daily if inference fails
                                    # Ensure offset alias is valid
                                    try:
                                         future_index = pd.date_range(start=last_date, periods=forecast_periods + 1, freq=freq)[1:] # Generate N periods *after* last_date
                                    except ValueError as e:
                                         st.error(f"Error generating future date range with frequency '{freq}': {e}. Trying without frequency.")
                                         # Fallback: Calculate time difference and add manually (less robust)
                                         time_diff = train_data.index[-1] - train_data.index[-2] if len(train_data.index) > 1 else pd.Timedelta(days=1)
                                         future_index = [last_date + time_diff * i for i in range(1, forecast_periods + 1)]


                                    forecast_df = pd.DataFrame({'Forecast': forecast}, index=future_index)

                                    st.write("#### Forecasted Values:")
                                    st.dataframe(forecast_df)

                                    fig_fc, ax_fc = plt.subplots(figsize=(12, 6))
                                    ax_fc.plot(train_data.index, train_data, label='Historical Data')
                                    ax_fc.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='red', linestyle='--')
                                    ax_fc.set_title(f'Forecast for {forecast_col} using {forecast_model_type}')
                                    ax_fc.legend()
                                    st.pyplot(fig_fc)

                                    # Store forecast details and plot for report
                                    st.session_state.analysis_results['forecast_details'] = f"Forecast for '{forecast_col}' using {forecast_model_type} for next {forecast_periods} periods."
                                    try:
                                        fc_img = BytesIO()
                                        fig_fc.savefig(fc_img, format='png', bbox_inches='tight')
                                        plt.close(fig_fc)
                                        fc_img.seek(0)
                                        st.session_state.analysis_results['forecast_plot'] = fc_img
                                    except Exception as e:
                                        st.error(f"Failed to save forecast plot: {e}")


                                    # Simple Backtest Evaluation (optional)
                                    holdout_periods = min(forecast_periods, len(train_data) // 5) # Use max 20% or forecast_periods for holdout
                                    if holdout_periods > 1:
                                        st.write(f"--- \n#### Simple Backtest Evaluation (holding out last {holdout_periods} periods):")
                                        train_eval = train_data[:-holdout_periods]
                                        actual_eval = train_data[-holdout_periods:]

                                        model_eval = None
                                        model_eval_fitted = None
                                        # Refit model on the shorter training data
                                        try:
                                            if forecast_model_type == "Simple Exponential Smoothing":
                                                 model_eval = ExponentialSmoothing(train_eval, trend=None, seasonal=None)
                                                 model_eval_fitted = model_eval.fit()
                                            elif forecast_model_type == "Holt's Linear Trend":
                                                 model_eval = ExponentialSmoothing(train_eval, trend='add', seasonal=None)
                                                 model_eval_fitted = model_eval.fit()
                                            elif forecast_model_type == "Holt-Winters Seasonal":
                                                if len(train_eval) > 2 * season_period_fc:
                                                     model_eval = ExponentialSmoothing(train_eval, trend='add', seasonal=seasonal_type, seasonal_periods=season_period_fc) # Use same seasonal type
                                                     model_eval_fitted = model_eval.fit()
                                                else: model_eval_fitted = None # Cannot fit on eval data

                                            if model_eval_fitted:
                                                forecast_eval = model_eval_fitted.forecast(holdout_periods)
                                                mae = mean_absolute_error(actual_eval, forecast_eval)
                                                rmse = np.sqrt(mean_squared_error(actual_eval, forecast_eval))
                                                eval_text = f"- Mean Absolute Error (MAE): {mae:.4f}\n- Root Mean Squared Error (RMSE): {rmse:.4f}"
                                                st.text(eval_text)
                                                st.session_state.analysis_results['forecast_eval'] = eval_text
                                            else:
                                                st.warning("Could not perform backtest evaluation (model fitting failed on evaluation training set).")
                                        except Exception as e:
                                            st.error(f"Backtest evaluation failed: {e}")


                            except Exception as e:
                                st.error(f"Forecasting failed: {e}")
                                st.error(f"Traceback: {traceback.format_exc()}")
                        else:
                            st.warning(f"Not enough data points ({len(train_data)}) in '{forecast_col}' to perform forecasting. Need > 5.")
                else:
                    st.warning("No numeric columns available in the time series data for forecasting.")
            else:
                st.warning("Forecasting requires a valid datetime index. Please select a valid date/time column in Section 2.")

        # --- 5. Classification Task ---
        with st.expander("5️⃣ Classification"):
            st.session_state.analysis_results.pop('classification_details', None)
            st.session_state.analysis_results.pop('classification_report', None)
            st.session_state.analysis_results.pop('confusion_matrix_plot', None)
            st.session_state.analysis_results.pop('feature_importances', None)

            st.write("### Predict a Categorical Target Variable")
             # Identify potential targets (object or int type with < 15 unique values, excluding the date column if set)
            potential_targets = []
            ts_index_name = st.session_state.analysis_results.get('ts_index')
            for col in df.columns:
                 if col != ts_index_name: # Exclude the time series index column
                    n_unique = df[col].nunique()
                    # Check if dtype is object, category, boolean or integer-like
                    is_candidate = pd.api.types.is_object_dtype(df[col]) or \
                                   pd.api.types.is_categorical_dtype(df[col]) or \
                                   pd.api.types.is_bool_dtype(df[col]) or \
                                   (pd.api.types.is_integer_dtype(df[col]) and n_unique < 15) # Allow integers if low cardinality

                    if is_candidate and n_unique > 1 and n_unique < 15: # Exclude constant columns too
                        potential_targets.append(col)


            if not potential_targets:
                st.warning("No suitable categorical target columns (object/category/bool/low-cardinality int with 2-14 unique values) found in the working data.")
            else:
                target_column = st.selectbox("Select target (categorical) column for classification", potential_targets, key="clf_target_select")

                if target_column:
                    # Features: numeric columns, excluding the target
                    available_features = df.select_dtypes(include=np.number).columns.tolist()
                    if target_column in available_features:
                        available_features.remove(target_column) # Should not happen often based on target selection logic, but good check

                    if not available_features:
                        st.warning("No suitable numeric feature columns found in the working data for classification.")
                    else:
                        selected_features = st.multiselect("Select feature columns (numeric)", available_features, default=available_features, key="clf_features_select")

                        if selected_features:
                            clf_model_type = st.selectbox("Select Classification Model", ["Random Forest", "Logistic Regression"], key="clf_model_select")
                            st.write("---")

                            try:
                                # Prepare data: select columns and drop rows with ANY NaNs in selected cols
                                cols_to_use = [target_column] + selected_features
                                df_clf = df[cols_to_use].copy() # Work on a copy
                                initial_rows = len(df_clf)
                                df_clf = df_clf.dropna()
                                final_rows = len(df_clf)

                                st.write(f"Using columns: {', '.join(cols_to_use)}")
                                st.write(f"Data points after dropping NaNs in selected columns: {final_rows} (from {initial_rows})")

                                if final_rows < 20: # Need a reasonable number of samples
                                    st.warning(f"Very few data points ({final_rows}) available after handling missing values. Classification may be unreliable or fail.")
                                else:
                                    X = df_clf[selected_features]
                                    y_raw = df_clf[target_column]

                                    # Encode target labels
                                    y_encoded = y_raw.astype('category')
                                    target_classes_map = dict(enumerate(y_encoded.cat.categories))
                                    y = y_encoded.cat.codes # Numerical labels for sklearn

                                    # Display class distribution
                                    st.write("Target variable class distribution:")
                                    st.dataframe(y_raw.value_counts().to_frame(name='Count'))


                                    # Check if target is binary or multiclass
                                    num_classes = len(target_classes_map)
                                    if num_classes < 2:
                                        st.error("Target variable has only one class after filtering. Cannot perform classification.")
                                    else:
                                        stratify_param = y if num_classes > 1 else None # Stratify only if > 1 class

                                        try:
                                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify_param)
                                        except ValueError as e:
                                            st.warning(f"Could not stratify train/test split (perhaps too few samples in a class?). Performing regular split. Error: {e}")
                                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


                                        st.write(f"Training model: {clf_model_type} on {len(X_train)} samples, Testing on {len(X_test)} samples.")

                                        # Define and Train Model
                                        if clf_model_type == "Random Forest":
                                            # Add hyperparameters
                                            n_estimators = st.slider("Number of Trees (n_estimators)", 50, 500, 100, 50, key='rf_n_estimators')
                                            max_depth = st.slider("Max Depth of Trees (max_depth)", 3, 30, 10, 1, key='rf_max_depth')
                                            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, class_weight='balanced')
                                        elif clf_model_type == "Logistic Regression":
                                            # Add hyperparameters
                                            log_reg_c = st.number_input("Regularization Strength (C)", 0.01, 10.0, 1.0, 0.1, key='lr_c')
                                            clf = LogisticRegression(C=log_reg_c, random_state=42, class_weight='balanced', max_iter=1000, solver='liblinear') # liblinear good for smaller datasets

                                        clf.fit(X_train, y_train)
                                        preds = clf.predict(X_test)

                                        # Convert numerical predictions back to original labels for report
                                        pred_labels = pd.Series(preds).map(target_classes_map).astype(y_encoded.dtype) # Match dtype
                                        y_test_labels = pd.Series(y_test).map(target_classes_map).astype(y_encoded.dtype)
                                        target_class_names = [str(c) for c in target_classes_map.values()]


                                        # --- Display Results ---
                                        st.write(f"### Classification Results ({clf_model_type})")
                                        clf_details = f"Model: {clf_model_type} predicting '{target_column}' using features: {', '.join(selected_features)}."
                                        st.session_state.analysis_results['classification_details'] = clf_details

                                        # Classification Report
                                        report = classification_report(y_test_labels, pred_labels, target_names=target_class_names, zero_division=0)
                                        st.text("Classification Report:")
                                        st.text(report)
                                        st.session_state.analysis_results['classification_report'] = report

                                        # Confusion Matrix
                                        st.write("#### Confusion Matrix")
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
                                            importances = pd.Series(clf.feature_importances_, index=selected_features).sort_values(ascending=False)
                                            st.bar_chart(importances)
                                            # Store top N importances for report
                                            top_n = min(5, len(importances))
                                            importances_str = importances.head(top_n).to_string()
                                            st.session_state.analysis_results['feature_importances'] = importances_str

                            except Exception as e:
                                st.error(f"Classification failed: {e}")
                                st.error(f"Traceback: {traceback.format_exc()}")
                        else:
                            st.warning("Please select at least one feature column.")
                else:
                    st.info("Select a target column to enable classification.")


        # --- 6. AI Insights & Reporting ---
        with st.expander("6️⃣ AI Insights & Reporting"):
            st.markdown("---")
            st.write("### 🤖 AI Insights Summary")
            # Generate insights based *only* on session_state results
            final_insights = generate_dynamic_insights(st.session_state.analysis_results)
            st.markdown(final_insights)
            # Store the generated text itself in session state for the PDF
            st.session_state.analysis_results['ai_insights_text'] = final_insights.replace("**", "").replace("### ", "") # Remove markdown for PDF


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
                            'correlation_plot', 'ts_index', 'time_series_plots', 'adf_results',
                            'anomaly_method', 'anomalies_found', 'forecast_details',
                            'forecast_eval', 'forecast_plot', 'classification_details',
                            'classification_report', 'confusion_matrix_plot',
                            'feature_importances', 'ai_insights_text' # Use the cleaned text
                        ]
                        for key in keys_to_copy:
                            if key in st.session_state.analysis_results:
                                report_data_for_pdf[key] = st.session_state.analysis_results[key]

                         # Rename ai_insights_text back to ai_insights for the PDF function
                        if 'ai_insights_text' in report_data_for_pdf:
                            report_data_for_pdf['ai_insights'] = report_data_for_pdf.pop('ai_insights_text')


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

    elif not st.session_state.data_loaded:
        st.warning("Data could not be loaded. Please check the file and upload again.")
else:
    st.info(" PLease Upload a CSV file using the sidebar to begin analysis.")