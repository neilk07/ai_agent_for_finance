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
from sklearn.linear_model import LogisticRegression  # Removed unused LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- Page Configuration ---
st.set_page_config(page_title="AI Financial Analyst Pro", layout="wide", initial_sidebar_state="expanded")

# --- Helper Functions ---
@st.cache_data
def load_data(uploaded_file):
    """Loads data from uploaded CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

def safe_convert_to_datetime(df, col_name):
    """Safely converts a column to datetime, handling errors."""
    try:
        df[col_name] = pd.to_datetime(df[col_name])
        return True
    except Exception as e:
        st.warning(f"Could not convert '{col_name}' to datetime: {e}. Please select a valid date/time column.")
        return False

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

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Dataset Overview", 0, 1)
    pdf.set_font("Arial", size=10)
    if 'summary' in report_data:
        pdf.multi_cell(0, 5, f"Summary Statistics:\n{report_data['summary']}\n")
    if 'missing' in report_data:
        pdf.multi_cell(0, 5, f"Missing Values Summary:\n{report_data['missing']}\n")
    pdf.ln(5)

    if 'correlation_plot' in report_data:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "2. Correlation Analysis", 0, 1)
        try:
            pdf.image(report_data['correlation_plot'], x=10, y=None, w=180)
            pdf.ln(85)
        except Exception as e:
            st.error(f"Failed to embed correlation plot in PDF: {e}")
            pdf.multi_cell(0, 5, f"(Could not embed correlation plot: {e})")
            pdf.ln(5)

    if 'time_series_plots' in report_data:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "3. Time Series Analysis", 0, 1)
        for title, plot_bytes in report_data['time_series_plots'].items():
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 10, title, 0, 1)
            try:
                pdf.image(plot_bytes, x=10, y=None, w=180)
                pdf.ln(85)
            except Exception as e:
                pdf.multi_cell(0, 5, f"(Could not embed plot: {e})")
                pdf.ln(5)

    if 'classification_report' in report_data:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "4. Classification Results", 0, 1)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 5, f"Classification Report:\n{report_data['classification_report']}\n")
        if 'confusion_matrix_plot' in report_data:
            pdf.set_font("Arial", 'B', 10)
            pdf.cell(0, 10, "Confusion Matrix:", 0, 1)
            try:
                pdf.image(report_data['confusion_matrix_plot'], x=10, y=None, w=100)
                pdf.ln(60)
            except Exception as e:
                pdf.multi_cell(0, 5, f"(Could not embed confusion matrix: {e})")
                pdf.ln(5)

    if 'ai_insights' in report_data:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "5. AI Insights", 0, 1)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 5, report_data['ai_insights'])
        pdf.ln(5)

    return pdf.output(dest='S').encode('latin1')

# --- Global Variables & Session State ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'anomalies_found' not in st.session_state:
    st.session_state.anomalies_found = {}

# --- Sidebar ---
st.sidebar.title("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload Financial Data (CSV)", type=["csv"])

# --- Main App ---
st.title("📊 AI-Powered Financial Analyst Pro")
st.markdown("""
Welcome! Upload your financial data in CSV format. This tool performs exploratory data analysis, time series analysis, forecasting, and classification.
""")

if uploaded_file:
    df_original = load_data(uploaded_file)

    if df_original is not None:
        st.success("File uploaded successfully!")
        df = df_original.copy()

        # --- 1. Data Exploration & Preparation ---
        with st.expander("1️⃣ Data Exploration & Preparation", expanded=True):
            st.write("### Preview of Dataset")
            st.dataframe(df.head())

            st.write("### Dataset Information")
            buffer = StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()
            st.text(info_str)

            st.write("### Summary Statistics")
            summary_stats = df.describe()
            st.dataframe(summary_stats)
            st.session_state.analysis_results['summary'] = summary_stats.to_string()

            st.write("### Missing Values")
            missing_values = df.isnull().sum()
            missing_values = missing_values[missing_values > 0]
            if not missing_values.empty:
                st.dataframe(missing_values.to_frame(name='Missing Count'))
                st.session_state.analysis_results['missing'] = missing_values.to_string()
                impute_method = st.selectbox("Handle Missing Values (for analysis)?",
                                             ["None (Keep As Is)", "Drop Rows with NaNs", "Fill with Mean (Numeric)", "Fill with Median (Numeric)", "Forward Fill (Good for Time Series)"])
                if impute_method != "None (Keep As Is)":
                    if impute_method == "Drop Rows with NaNs":
                        df = df.dropna()
                        st.write(f"Shape after dropping NaNs: {df.shape}")
                    elif impute_method == "Fill with Mean (Numeric)":
                        numeric_cols_na = df.select_dtypes(include=np.number).columns
                        cols_to_impute = [col for col in numeric_cols_na if df[col].isnull().any()]
                        for col in cols_to_impute:
                            df[col] = df[col].fillna(df[col].mean())
                        st.write("Numeric columns filled with mean.")
                    elif impute_method == "Fill with Median (Numeric)":
                        numeric_cols_na = df.select_dtypes(include=np.number).columns
                        cols_to_impute = [col for col in numeric_cols_na if df[col].isnull().any()]
                        for col in cols_to_impute:
                            df[col] = df[col].fillna(df[col].median())
                        st.write("Numeric columns filled with median.")
                    elif impute_method == "Forward Fill (Good for Time Series)":
                        df = df.ffill()
                        st.write("Forward fill applied.")
                    st.write("Preview after imputation:")
                    st.dataframe(df.head())
            else:
                st.info("No missing values found.")
                st.session_state.analysis_results['missing'] = "No missing values."

            st.write("### Correlation Heatmap")
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty and len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                fig_corr, ax_corr = plt.subplots(figsize=(10, 7))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
                st.pyplot(fig_corr)
                corr_img = BytesIO()
                fig_corr.savefig(corr_img, format='png', bbox_inches='tight')
                plt.close(fig_corr)
                corr_img.seek(0)
                st.session_state.analysis_results['correlation_plot'] = corr_img
            else:
                st.warning("Correlation heatmap requires at least two numeric columns.")

            st.write("### Distribution of Numeric Variables")
            numeric_cols_dist = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols_dist:
                col_to_plot = st.selectbox("Select numeric column for distribution plot", numeric_cols_dist)
                fig_dist, ax_dist = plt.subplots()
                sns.histplot(df[col_to_plot], kde=True, ax=ax_dist)
                ax_dist.set_title(f'Distribution of {col_to_plot}')
                st.pyplot(fig_dist)
            else:
                st.info("No numeric columns available for distribution plots.")

        # --- 2. Time Series Analysis ---
        with st.expander("2️⃣ Time Series Analysis"):
            st.write("### Select Date/Time Column")
            potential_date_cols = df.columns
            date_col = st.selectbox("Select the primary date/time column for time series analysis",
                                    potential_date_cols, index=0)

            if date_col and safe_convert_to_datetime(df, date_col):
                try:
                    df_ts = df.set_index(date_col).copy()
                    st.success(f"'{date_col}' set as time series index.")
                    numeric_cols_ts = df_ts.select_dtypes(include=np.number).columns.tolist()
                    st.session_state.analysis_results['time_series_plots'] = {}

                    if numeric_cols_ts:
                        st.write("### Time Series Plot")
                        ts_col_select = st.multiselect("Select numeric columns to plot over time", numeric_cols_ts, default=numeric_cols_ts[0] if numeric_cols_ts else None)
                        if ts_col_select:
                            st.line_chart(df_ts[ts_col_select])

                        st.write("### Rolling Statistics")
                        roll_col = st.selectbox("Select column for rolling statistics", numeric_cols_ts)
                        roll_window = st.slider("Select rolling window size (periods)", min_value=3, max_value=min(90, len(df_ts)//2), value=14)
                        if roll_col and roll_window:
                            df_ts[f'{roll_col}_Rolling_Mean'] = df_ts[roll_col].rolling(window=roll_window).mean()
                            df_ts[f'{roll_col}_Rolling_Std'] = df_ts[roll_col].rolling(window=roll_window).std()
                            fig_roll, ax_roll = plt.subplots(figsize=(12, 6))
                            ax_roll.plot(df_ts.index, df_ts[roll_col], label='Original', alpha=0.7)
                            ax_roll.plot(df_ts.index, df_ts[f'{roll_col}_Rolling_Mean'], label=f'Rolling Mean ({roll_window} periods)', color='orange')
                            ax_roll.plot(df_ts.index, df_ts[f'{roll_col}_Rolling_Std'], label=f'Rolling Std Dev ({roll_window} periods)', color='red', linestyle='--')
                            ax_roll.set_title(f'Rolling Statistics for {roll_col}')
                            ax_roll.legend()
                            st.pyplot(fig_roll)
                            roll_img = BytesIO()
                            fig_roll.savefig(roll_img, format='png', bbox_inches='tight')
                            plt.close(fig_roll)
                            roll_img.seek(0)
                            st.session_state.analysis_results['time_series_plots']['Rolling Statistics'] = roll_img

                        st.write("### Time Series Decomposition (Trend, Seasonality, Residuals)")
                        decomp_col = st.selectbox("Select column for decomposition", numeric_cols_ts, key='decomp_sel')
                        inferred_freq = pd.infer_freq(df_ts.index)
                        period = st.number_input("Seasonality Period (e.g., 7 for daily data/weekly seasonality, 12 for monthly/yearly)",
                                                value=7 if inferred_freq == 'D' else (12 if inferred_freq and 'M' in inferred_freq else 1))
                        if decomp_col and period > 1:
                            try:
                                if len(df_ts[decomp_col].dropna()) > 2 * period:
                                    result = seasonal_decompose(df_ts[decomp_col].dropna(), model='additive', period=period)
                                    fig_decomp = result.plot()
                                    fig_decomp.set_size_inches(10, 8)
                                    plt.tight_layout()
                                    st.pyplot(fig_decomp)
                                    decomp_img = BytesIO()
                                    fig_decomp.savefig(decomp_img, format='png', bbox_inches='tight')
                                    plt.close(fig_decomp)
                                    decomp_img.seek(0)
                                    st.session_state.analysis_results['time_series_plots']['Decomposition'] = decomp_img
                                else:
                                    st.warning(f"Not enough data points ({len(df_ts[decomp_col].dropna())}) for decomposition with period {period}. Need at least {2*period}.")
                            except Exception as e:
                                st.error(f"Decomposition failed: {e}")
                        else:
                            st.info("Select a column and ensure Period > 1 for decomposition.")

                        st.write("### Stationarity Test (Augmented Dickey-Fuller)")
                        adf_col = st.selectbox("Select column for ADF Test", numeric_cols_ts, key='adf_sel')
                        if adf_col:
                            try:
                                result = adfuller(df_ts[adf_col].dropna())
                                st.write(f'ADF Statistic for {adf_col}: {result[0]:.4f}')
                                st.write(f'p-value: {result[1]:.4f}')
                                st.write('Critical Values:')
                                for key, value in result[4].items():
                                    st.write(f'\t{key}: {value:.4f}')
                                if result[1] <= 0.05:
                                    st.success(f"Result suggests the time series '{adf_col}' is likely stationary (p <= 0.05).")
                                else:
                                    st.warning(f"Result suggests the time series '{adf_col}' is likely non-stationary (p > 0.05). Consider differencing.")
                            except Exception as e:
                                st.error(f"ADF test failed for {adf_col}: {e}")
                    else:
                        st.warning("No numeric columns found in the data after setting the time index.")
                except Exception as e:
                    st.error(f"Error setting index or during time series analysis: {e}")
            else:
                st.warning("Please select a valid date/time column for time series analysis.")

        # --- 3. Anomaly Detection ---
        with st.expander("3️⃣ Anomaly Detection"):
            numeric_cols_anom = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols_anom:
                st.write("### Detect Outliers")
                anom_method = st.radio("Select Anomaly Detection Method", ["Z-Score", "Interquartile Range (IQR)"])
                st.session_state.anomalies_found.clear()

                if anom_method == "Z-Score":
                    z_thresh = st.slider("Set Z-score threshold", 1.5, 5.0, 3.0, 0.1)
                    for col in numeric_cols_anom:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            mean = df[col].mean()
                            std = df[col].std()
                            if std > 0:
                                z_scores = (df[col] - mean) / std
                                outliers = df[np.abs(z_scores) > z_thresh]
                                if not outliers.empty:
                                    st.session_state.anomalies_found[col] = outliers[[col]]
                            else:
                                st.info(f"Column '{col}' has zero standard deviation, skipping Z-score.")

                elif anom_method == "IQR":
                    iqr_multiplier = st.slider("Set IQR Multiplier", 1.0, 3.0, 1.5, 0.1)
                    for col in numeric_cols_anom:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - iqr_multiplier * IQR
                            upper_bound = Q3 + iqr_multiplier * IQR
                            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                            if not outliers.empty:
                                st.session_state.anomalies_found[col] = outliers[[col]]

                if st.session_state.anomalies_found:
                    st.write("#### Anomalies Detected:")
                    for col, anom_df in st.session_state.anomalies_found.items():
                        st.write(f"**{col}** (found {len(anom_df)}):")
                        st.dataframe(anom_df)
                else:
                    st.info("No anomalies detected with the current settings.")
            else:
                st.warning("No numeric columns available for anomaly detection.")

        # --- 4. Forecasting ---
        with st.expander("4️⃣ Forecasting"):
            numeric_cols_forecast = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols_forecast and date_col and date_col in df.columns:
                st.write("### Time Series Forecasting")
                forecast_col = st.selectbox("Select column to forecast", numeric_cols_forecast)
                forecast_periods = st.number_input("Number of periods to forecast", min_value=1, max_value=365, value=30)
                forecast_model_type = st.selectbox("Select Forecasting Model", ["Simple Exponential Smoothing", "Holt's Linear Trend", "Holt-Winters Seasonal"])

                if forecast_col and forecast_periods > 0:
                    if 'df_ts' in locals() and isinstance(df_ts.index, pd.DatetimeIndex):
                        train_data = df_ts[forecast_col].dropna()
                        st.write(f"Using data from column '{forecast_col}' indexed by '{df_ts.index.name}'. Training data length: {len(train_data)}")
                    else:
                        st.warning("Forecasting requires a datetime index (Section 2). Using numeric data, results may be unreliable.")
                        train_data = df[forecast_col].dropna()

                    if len(train_data) > 5:
                        try:
                            if forecast_model_type == "Simple Exponential Smoothing":
                                model = ExponentialSmoothing(train_data, trend=None, seasonal=None).fit()
                            elif forecast_model_type == "Holt's Linear Trend":
                                model = ExponentialSmoothing(train_data, trend='add', seasonal=None).fit()
                            elif forecast_model_type == "Holt-Winters Seasonal":
                                season_period_fc = st.number_input("Seasonality Period for Holt-Winters", value=period if 'period' in locals() else 7, min_value=2)
                                if len(train_data) > 2 * season_period_fc:
                                    model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=season_period_fc).fit()
                                else:
                                    st.error(f"Not enough data ({len(train_data)}) for Holt-Winters with period {season_period_fc}. Need at least {2*season_period_fc}.")
                                    model = None
                            else:
                                st.error("Model not yet implemented")
                                model = None

                            if model:
                                forecast = model.forecast(forecast_periods)
                                forecast_df = pd.DataFrame({'Forecast': forecast})
                                if isinstance(train_data.index, pd.DatetimeIndex):
                                    last_date = train_data.index[-1]
                                    freq = pd.infer_freq(train_data.index) or 'D'
                                    future_index = pd.date_range(start=last_date + pd.Timedelta(days=1 if freq == 'D' else '1' + freq), periods=forecast_periods, freq=freq)
                                    forecast_df.index = future_index
                                else:
                                    st.warning("No datetime index; using numeric indices.")
                                    last_numeric_index = len(train_data) - 1
                                    future_index = pd.RangeIndex(start=last_numeric_index + 1, stop=last_numeric_index + 1 + forecast_periods)
                                    forecast_df.index = future_index

                                st.write("#### Forecasted Values:")
                                st.dataframe(forecast_df)

                                fig_fc, ax_fc = plt.subplots(figsize=(12, 6))
                                ax_fc.plot(train_data.index, train_data, label='Historical Data')
                                ax_fc.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='red', linestyle='--')
                                ax_fc.set_title(f'Forecast for {forecast_col} using {forecast_model_type}')
                                ax_fc.legend()
                                st.pyplot(fig_fc)

                                holdout_periods = min(forecast_periods, len(train_data) // 5)
                                if holdout_periods > 1:
                                    train_eval = train_data[:-holdout_periods]
                                    actual_eval = train_data[-holdout_periods:]
                                    if forecast_model_type == "Simple Exponential Smoothing":
                                        model_eval = ExponentialSmoothing(train_eval, trend=None, seasonal=None).fit()
                                    elif forecast_model_type == "Holt's Linear Trend":
                                        model_eval = ExponentialSmoothing(train_eval, trend='add', seasonal=None).fit()
                                    elif forecast_model_type == "Holt-Winters Seasonal":
                                        if len(train_eval) > 2 * season_period_fc:
                                            model_eval = ExponentialSmoothing(train_eval, trend='add', seasonal='add', seasonal_periods=season_period_fc).fit()
                                        else:
                                            model_eval = None
                                    else:
                                        model_eval = None

                                    if model_eval:
                                        forecast_eval = model_eval.forecast(holdout_periods)
                                        mae = mean_absolute_error(actual_eval, forecast_eval)
                                        rmse = np.sqrt(mean_squared_error(actual_eval, forecast_eval))
                                        st.write("#### Simple Backtest Evaluation (on last {} periods):".format(holdout_periods))
                                        st.write(f"- Mean Absolute Error (MAE): {mae:.4f}")
                                        st.write(f"- Root Mean Squared Error (RMSE): {rmse:.4f}")
                        except Exception as e:
                            st.error(f"Forecasting failed: {e}")
                    else:
                        st.warning("Not enough data points in the selected column to perform forecasting.")
            else:
                st.warning("Forecasting requires numeric columns and a selected date column (from Section 2).")

        # --- 5. Classification Task ---
        with st.expander("5️⃣ Classification"):
            st.write("### Predict a Categorical Target Variable")
            potential_targets = [col for col in df.columns if df[col].nunique() < 15 and (df[col].dtype == 'object' or pd.api.types.is_integer_dtype(df[col]))]
            if not potential_targets:
                st.warning("No suitable categorical target columns (with < 15 unique values) found.")
            else:
                target_column = st.selectbox("Select target (categorical) column for classification", potential_targets)

                if target_column:
                    available_features = df.select_dtypes(include=np.number).columns.tolist()
                    if target_column in available_features:
                        available_features.remove(target_column)

                    if not available_features:
                        st.warning("No suitable numeric feature columns found for classification.")
                    else:
                        selected_features = st.multiselect("Select feature columns (numeric)", available_features, default=available_features)

                        if selected_features:
                            clf_model_type = st.selectbox("Select Classification Model", ["Random Forest", "Logistic Regression"])
                            try:
                                df_clf = df[[target_column] + selected_features].dropna()
                                if len(df_clf) < 10:
                                    st.warning(f"Very few data points ({len(df_clf)}) after dropping NaNs. Classification may be unreliable.")
                                else:
                                    X = df_clf[selected_features]
                                    y = df_clf[target_column]
                                    target_classes = y.astype('category').cat.categories
                                    y = y.astype('category').cat.codes

                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

                                    if clf_model_type == "Random Forest":
                                        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                                    elif clf_model_type == "Logistic Regression":
                                        clf = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)

                                    clf.fit(X_train, y_train)
                                    preds = clf.predict(X_test)

                                    st.write(f"### Classification Results ({clf_model_type})")
                                    report = classification_report(y_test, preds, target_names=[str(c) for c in target_classes])
                                    st.text("Classification Report:")
                                    st.text(report)
                                    st.session_state.analysis_results['classification_report'] = report

                                    st.write("#### Confusion Matrix")
                                    fig_cm = plot_confusion_matrix(y_test, preds, classes=[str(c) for c in target_classes])
                                    st.pyplot(fig_cm)
                                    cm_img = BytesIO()
                                    fig_cm.savefig(cm_img, format='png', bbox_inches='tight')
                                    plt.close(fig_cm)
                                    cm_img.seek(0)
                                    st.session_state.analysis_results['confusion_matrix_plot'] = cm_img

                                    if clf_model_type == "Random Forest" and hasattr(clf, 'feature_importances_'):
                                        st.write("#### Feature Importances")
                                        importances = pd.Series(clf.feature_importances_, index=selected_features).sort_values(ascending=False)
                                        st.bar_chart(importances)
                            except Exception as e:
                                st.error(f"Classification failed: {e}")
                        else:
                            st.warning("Please select at least one feature column.")

        # --- 6. AI Insights & Reporting ---
        with st.expander("6️⃣ AI Insights & Reporting"):
            def generate_dynamic_insights(results):
                insight_text = "### 🤖 AI-Generated Insights Summary\n\n"
                if 'summary' in results:
                    insight_text += "**Dataset Overview:**\n"
                    insight_text += "- The dataset contains features with varying scales and distributions (refer to summary statistics and distribution plots).\n"
                if 'missing' in results and results['missing'] != "No missing values.":
                    insight_text += f"- Missing values were detected and potentially handled via the selected method. Key columns with missing data:\n{results['missing']}\n"
                else:
                    insight_text += "- No missing values were detected in the initial scan.\n"

                if 'correlation_plot' in results:
                    insight_text += "**Correlation Insights:**\n"
                    insight_text += "- The heatmap visualizes linear relationships between numeric variables. Look for strong positive or negative correlations.\n"

                if 'time_series_plots' in results and results['time_series_plots']:
                    insight_text += "**Time Series Observations:**\n"
                    if 'Rolling Statistics' in results['time_series_plots']:
                        insight_text += "- Rolling standard deviation (volatility) analysis shows periods of higher/lower fluctuation.\n"
                    if 'Decomposition' in results['time_series_plots']:
                        insight_text += "- Decomposition analysis attempted to separate trend, seasonality, and residuals. Review these components for underlying patterns.\n"

                if st.session_state.anomalies_found:
                    insight_text += "**Anomaly Detection:**\n"
                    insight_text += f"- Anomalies (outliers) were detected in columns: {', '.join(st.session_state.anomalies_found.keys())} using the {anom_method} method. These data points might warrant further investigation.\n"

                if 'forecast_df' in locals() and 'Forecast' in forecast_df.columns:
                    insight_text += "**Forecasting Summary:**\n"
                    insight_text += f"- A forecast for '{forecast_col}' was generated using the {forecast_model_type} model for the next {forecast_periods} periods. Check the plot for trend and predicted values.\n"

                if 'classification_report' in results:
                    insight_text += "**Classification Performance:**\n"
                    insight_text += f"- A {clf_model_type} model was trained to predict '{target_column}'.\n"
                    insight_text += "- Key performance metrics (precision, recall, F1-score) are detailed in the classification report. Evaluate based on the specific goals (e.g., high recall for fraud detection).\n"
                    if 'confusion_matrix_plot' in results:
                        insight_text += "- The confusion matrix shows correct and incorrect predictions for each class.\n"
                    if clf_model_type == "Random Forest" and 'importances' in locals():
                        insight_text += f"- Top features influencing the prediction appear to be: {', '.join(importances.head(3).index)}.\n"

                insight_text += "\n**Disclaimer:** These insights are auto-generated based on statistical analysis and ML models. Always validate findings with domain expertise."
                return insight_text

            st.markdown("---")
            final_insights = generate_dynamic_insights(st.session_state.analysis_results)
            st.markdown(final_insights)
            st.session_state.analysis_results['ai_insights'] = final_insights

            st.markdown("---")
            st.write("### Export Full Report")
            if st.button("🗕️ Generate PDF Report"):
                with st.spinner("Generating PDF..."):
                    pdf_bytes = generate_pdf_report(st.session_state.analysis_results)
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name="financial_analysis_report.pdf",
                        mime="application/pdf"
                    )
    else:
        st.error("Could not load the data file.")
else:
    st.info("👆 Upload a CSV file using the sidebar to begin analysis.")