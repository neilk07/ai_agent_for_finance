# 🧠 AI-Powered Financial Data Analyst with Forecasting, Classification, PDF Export and Free LLM Option

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from fpdf import FPDF
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="AI Financial Analyst", layout="wide")

# --- App Title ---
st.title("📊 AI-Powered Financial Data Analyst")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload a financial dataset (CSV)", type=["csv"])

# --- Load & Show Data ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset", df.head())

    # Option 1: Fill NaN values with column mean
    df = df.fillna(df.mean())

    # Option 2: Drop rows with NaN values
    # df = df.dropna()

    # Summary statistics
    summary = df.describe().to_string()
    missing = df.isnull().sum().to_string()

    st.write("### Missing Values")
    st.text(missing)

    # Correlation heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Line chart
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if numeric_cols:
        col1 = st.selectbox("Select column for line plot", numeric_cols)
        st.line_chart(df[col1])

    # --- Time Series Analysis ---
    st.write("### ⏱️ Time Series Correlation Analysis")
    date_col = st.selectbox("Select date/time column", df.columns, index=0)

    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        st.write("### Time Series Data Preview")
        st.line_chart(df[numeric_cols])

        # Rolling correlation matrix over time
        st.write("### Rolling Correlation Matrix")
        window = st.slider("Select rolling window size (days)", min_value=3, max_value=60, value=14)

        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].rolling(window=window).corr().dropna()
            st.write("Rolling correlations computed. Sample preview:")
            st.dataframe(corr_matrix.head(20))

    except Exception as e:
        st.warning(f"Could not convert {date_col} to datetime or analyze time series: {e}")

    # --- Anomaly Detection ---
    st.write("### ⚡ Anomaly Detection (Z-Score Based)")
    z_thresh = st.slider("Set Z-score threshold", 2.0, 5.0, 3.0)
    anomalies = {}
    for col in numeric_cols:
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        outliers = df[np.abs(z_scores) > z_thresh]
        if not outliers.empty:
            anomalies[col] = outliers[[col]]
            st.write(f"#### Anomalies in `{col}`")
            st.dataframe(outliers[[col]])

    # --- Forecasting ---
    st.write("### ⚡ Forecasting")
    if numeric_cols:
        forecast_col = st.selectbox("Select column to forecast", numeric_cols)
        df['index'] = range(len(df))
        X = df[['index']]
        y = df[forecast_col]
        model = LinearRegression()
        model.fit(X, y)
        future_idx = np.array(range(len(df), len(df) + 10)).reshape(-1, 1)
        future_preds = model.predict(future_idx)
        st.write("#### Forecast for next 10 steps:")
        st.line_chart(pd.Series(np.append(y.values, future_preds)))

    # --- Classification ---
    st.write("### 🎓 Classification Task")
    target_column = st.selectbox("Select target (categorical) column for classification", df.columns)
    if df[target_column].nunique() <= 10:
        try:
            X = df.drop(columns=[target_column]).select_dtypes(include=['float64', 'int64']).dropna()
            y = df[target_column].loc[X.index]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = RandomForestClassifier()
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            report = classification_report(y_test, preds)
            st.text("Classification Report:")
            st.text(report)
        except Exception as e:
            st.warning(f"Classification not successful: {e}")

    # Free AI Summary (Offline Simulation)
    def generate_insights(summary, missing):
        return f"""
        This dataset shows summary statistics such as:

        {summary}

        Missing values observed:
        {missing}

        Key Observations:
        - Check variables with high standard deviation.
        - Handle missing values through imputation or filtering.
        - Explore correlations shown in the heatmap.
        - Time series correlation analysis may reveal lagging or leading indicators.
        - Anomaly detection highlights unusual data points that may need further investigation.
        - Forecasting provides a predictive outlook for key variables.
        - Classification enables categorical analysis based on financial predictors.
        """

    insights = generate_insights(summary, missing)
    st.write("### 🤖 AI Insights (Free Simulated)")
    st.text(insights)

    # Example (Gemini or DeepSeek API - pseudo code)
    #import requests


    #def call_llm(prompt):
    #    api_key = st.secrets["API_KEY"]  # Add in Streamlit secrets
    #    headers = {"Authorization": f"Bearer {api_key}"}
    #    payload = {"prompt": prompt, "model": "gemini-pro"}  # Or "deepseek-chat"
    #    response = requests.post("https://api.your-llm-provider.com/v1/chat", headers=headers, json=payload)
    #   return response.json().get("content")


    # Export PDF with charts and tables
    def create_pdf(summary, missing, insights):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, "Financial Data Summary Report\n")
        pdf.multi_cell(0, 10, f"Summary Stats:\n{summary}\n")
        pdf.multi_cell(0, 10, f"Missing Values:\n{missing}\n")
        pdf.multi_cell(0, 10, f"Insights:\n{insights}\n")

        # Save basic correlation heatmap to image
        plt.figure(figsize=(8, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
        heatmap_img = BytesIO()
        plt.savefig(heatmap_img, format='png')
        plt.close()
        heatmap_img.seek(0)
        pdf.image(heatmap_img, x=10, y=None, w=180)

        # Summary table as image
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=df.describe().round(2).values,
                 colLabels=df.describe().columns,
                 rowLabels=df.describe().index,
                 loc='center')
        table_img = BytesIO()
        plt.savefig(table_img, format='png')
        plt.close()
        table_img.seek(0)
        pdf.image(table_img, x=10, y=None, w=180)

        return pdf.output(dest='S').encode('latin1')

    if st.button("🗕️ Export Report to PDF"):
        pdf_bytes = create_pdf(summary, missing, insights)
        st.download_button("Download PDF Report", data=pdf_bytes, file_name="financial_report.pdf")

else:
    st.info("Please upload a CSV file to begin analysis.")
