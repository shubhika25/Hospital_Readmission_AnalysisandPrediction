# --- app.py ---
"""
Streamlit App for Hospital Readmission Prediction
Polished UI + KPI CSV/PDF Export + Insights + Modeling + Prediction
"""
from networkx import k_components
from ydata_profiling import ProfileReport
from advanced_insights import (
    plot_age_group_bar,
    plot_top_diag_bar,
    plot_visitor_pie,
    plot_los_bar,
    plot_treatment_cluster_bar,
    plot_correlation_heatmap
)
import streamlit as st
import pandas as pd
import os
from io import BytesIO
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ---- Advanced Insights ----
from advanced_insights import (
    kpis,
    generate_insights_report,
    generate_insights_pdf
)

# ---- ML Logic ----
from modeling import (
    load_model,
    train_and_save_model,
    prepare_data,
    simple_impute_and_encode,
    MODEL_PATH
)

from sklearn.metrics import roc_auc_score, confusion_matrix

# PDF export helper
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ------------------------------------------------------------------
# PDF GENERATOR (KPI ONLY)
# ------------------------------------------------------------------
def generate_kpi_pdf(kpi_dict):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "Readmission KPI Report")

    c.setFont("Helvetica", 12)
    y = 720
    for key, value in kpi_dict.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 20

    c.save()
    return buffer.getvalue()


# ------------------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Readmission Analytics Dashboard",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
.metric-card {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e1e1e1;
    box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    text-align: center;
    margin-bottom: 15px;
}
.section-title {
    font-size: 26px;
    font-weight: 600;
    margin-top: 15px;
    color: #333333;
    padding-bottom: 5px;
    border-bottom: 2px solid #d3d3d3;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🏥 Hospital Readmission — Analytics & Prediction Dashboard")

# ------------------------------------------------------------------
# SIDEBAR DATA LOAD
# ------------------------------------------------------------------
st.sidebar.header("📂 Data Source")

use_sample = st.sidebar.checkbox("Use sample_data/readmission.csv", value=True)
uploaded_file = st.sidebar.file_uploader("Or upload your CSV", type=["csv"])


@st.cache_data
def load_data(use_sample, uploaded):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    elif use_sample and os.path.exists("sample_data/readmission.csv"):
        return pd.read_csv("sample_data/readmission.csv")
    return pd.DataFrame()


df = load_data(use_sample, uploaded_file)

if df.empty:
    st.warning("⚠ No data loaded. Upload CSV or use sample data.")
    st.stop()

# ------------------------------------------------------------------
# MAIN TABS
# ------------------------------------------------------------------
tabs = st.tabs(["🏥 Overview", "📈 Advanced Insights", "🤖 Modeling", "🔮 Prediction", "EDA", "Anomaly Detection"])

# ------------------------------------------------------------------
# TAB 1 — OVERVIEW
# ------------------------------------------------------------------
with tabs[0]:
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

    st.write("### Sample Rows")
    st.dataframe(df.head())

    st.write("### Key KPIs")
    k = kpis(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='metric-card'><h4>Total Samples</h4><h2>{k['total_samples']}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><h4>Readmission Rate</h4><h2>{k['readmission_rate']}%</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><h4>Positive Cases</h4><h2>{k['pos_count']}</h2></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-card'><h4>Negative Cases</h4><h2>{k['neg_count']}</h2></div>", unsafe_allow_html=True)

    # Downloads
    st.download_button("📥 Download KPI Report (CSV)",
                       data=pd.DataFrame([k]).to_csv(index=False),
                       file_name="kpi_report.csv")

    pdf_bytes = generate_kpi_pdf(k)
    st.download_button("📄 Download KPI Report (PDF)",
                       data=pdf_bytes,
                       file_name="kpi_report.pdf",
                       mime="application/pdf")

# ------------------------------------------------------------------
# TAB 2 — ADVANCED INSIGHTS
# ------------------------------------------------------------------
with tabs[1]:
    st.markdown('<div class="section-title">Advanced Insights</div>', unsafe_allow_html=True)

    report = generate_insights_report(df)

    st.subheader("📌 Overview Metrics")
    st.json(report["overview"])

    st.subheader("📌 Readmission by Age Group")
    st.dataframe(report["age_group"])

    st.subheader("📌 Multi vs Single Diagnosis")
    st.dataframe(report["multi_vs_single"])

    st.subheader("📌 Frequent Visitor Buckets")
    st.dataframe(report["visitor_buckets"])

    st.subheader("📌 Top Diagnosis Categories")
    st.dataframe(report["top_diag"])

    st.subheader("📌 High-Risk Diagnosis Pairs")
    st.dataframe(report["diag_pairs"])

    st.subheader("📌 Diabetes Breakdown")
    st.dataframe(report["diabetes"]["diabetes_med"])
    st.dataframe(report["diabetes"]["glucose_test"])

    st.subheader("📌 LOS Buckets")
    st.dataframe(report["los_buckets"])

    st.subheader("📌 Treatment Clusters")
    st.dataframe(report["treatment_clusters"])

    st.subheader("📌 Recommendations")
    for r in report["recommendations"]:
        st.write("•", r)

    pdf_bytes = generate_insights_pdf(report)
    st.download_button("📄 Download Full Executive Insights Report (PDF — no charts)",
                       data=pdf_bytes,
                       file_name="readmission_insights.pdf",
                       mime="application/pdf")
    fig = plot_age_group_bar(df)
    st.pyplot(fig)

    fig = plot_top_diag_bar(df)
    st.pyplot(fig)

    fig = plot_visitor_pie(df)
    st.pyplot(fig)

    fig = plot_los_bar(df)
    st.pyplot(fig)

    fig = plot_treatment_cluster_bar(df)
    st.pyplot(fig)

    fig = plot_correlation_heatmap(df)
    st.pyplot(fig)

# ------------------------------------------------------------------
# TAB 3 — MODELING
# ------------------------------------------------------------------
with tabs[2]:
    st.markdown('<div class="section-title">Model Training & Evaluation</div>', unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        if st.button("Train RandomForest Model"):
            train_and_save_model(df)
            st.success("Model trained & saved successfully!")

    with colB:
        if st.button("Load Saved Model"):
            if load_model():
                st.success("Model loaded.")
            else:
                st.error("No saved model found.")

    model_obj = load_model()
    if model_obj:
        X, y = prepare_data(df)
        X_proc = simple_impute_and_encode(X)

        # Add missing columns
        for col in model_obj["columns"]:
            if col not in X_proc.columns:
                X_proc[col] = 0

        X_proc = X_proc[model_obj["columns"]]
        X_scaled = model_obj["scaler"].transform(X_proc)

        proba = model_obj["model"].predict_proba(X_scaled)[:, 1]
        pred = model_obj["model"].predict(X_scaled)

        st.write("### ROC-AUC:", round(roc_auc_score(y, proba), 3))

        cm = confusion_matrix(y, pred)
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(cm, cmap="Blues")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

# ------------------------------------------------------------------
# TAB 4 — PREDICTION
# ------------------------------------------------------------------
with tabs[3]:
    st.markdown('<div class="section-title">Make a Prediction</div>', unsafe_allow_html=True)

    model_obj = load_model()
    if not model_obj:
        st.warning("Train a model first.")
        st.stop()

    st.write("Fill in sample inputs:")

    sample = df.drop(columns=[df.columns[-1]]).iloc[0]
    user_input = {}

    for col in df.columns[:15]:
        if col == "readmitted":
            continue
        user_input[col] = st.text_input(col, value=str(sample.get(col, "")))

    if st.button("Predict Readmission"):
        input_df = pd.DataFrame([user_input])

        # convert numerics
        for c in input_df.columns:
            try:
                input_df[c] = pd.to_numeric(input_df[c])
            except:
                pass

        X_proc = simple_impute_and_encode(input_df)

        for col in model_obj["columns"]:
            if col not in X_proc.columns:
                X_proc[col] = 0

        X_proc = X_proc[model_obj["columns"]]
        X_scaled = model_obj["scaler"].transform(X_proc)

        prob = model_obj["model"].predict_proba(X_scaled)[0][1]
        pred = int(model_obj["model"].predict(X_scaled)[0])

        st.write({"prediction": pred, "probability": float(prob)})

        if pred == 1:
            st.error(f"High readmission risk — probability {prob:.2f}")
        else:
            st.success(f"Low readmission risk — probability {prob:.2f}")
# ------------------------------------------------------------------
# TAB 3 — FULL AUTOMATED EDA (YData Profiling)
# ------------------------------------------------------------------

with tabs[4]:
    st.markdown('<div class="section-title">Full EDA — YData Profiling</div>', unsafe_allow_html=True)

    st.write("Generate a complete automated EDA report for your dataset.")

    if st.button("🚀 Generate Full EDA Report (HTML)"):
        with st.spinner("Generating EDA Profile Report... this may take 10–20 seconds"):
            profile = ProfileReport(df, title="Hospital Readmission Profiling Report", explorative=True)

            # Save HTML to memory
            html_bytes = profile.to_html().encode('utf-8')

            # Store in session for display
            st.session_state["profile_html"] = html_bytes

        st.success("✅ EDA Report generated successfully!")

    # Show report inside app
    if "profile_html" in st.session_state:
        st.subheader("📄 EDA Report Preview")
        components.html(
            st.session_state["profile_html"].decode('utf-8'),
            height=900,
            scrolling=True
        )

        # Download button
        st.download_button(
            label="📥 Download Full EDA Report (HTML)",
            data=st.session_state["profile_html"],
            file_name="eda_report.html",
            mime="text/html"
        )
# ----------------------------------------------------
# TAB: Anomaly Detection
# ----------------------------------------------------
with tabs[5]:
    st.header("🔍 Anomaly Detection")

    if df is None:
        st.info("Upload a CSV from the sidebar to begin anomaly detection.")
    else:
        st.subheader("Select Numeric Columns")

        num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
        selected_cols = st.multiselect(
            "Choose features for anomaly detection:",
            num_cols,
            default=num_cols[:5]  # auto-select a few
        )

        if len(selected_cols) < 2:
            st.warning("Select at least 2 numeric columns.")
        else:
            from sklearn.ensemble import IsolationForest

            iso = IsolationForest(
                contamination=0.05,
                random_state=42
            )

            model_data = df[selected_cols].fillna(df[selected_cols].median())
            preds = iso.fit_predict(model_data)

            df['anomaly'] = preds
            anomalies = df[df['anomaly'] == -1]
            normal = df[df['anomaly'] == 1]

            st.success(f"Detected **{len(anomalies)} anomalies** out of {len(df)} rows.")

            st.subheader("📄 Anomalous Rows")
            st.dataframe(anomalies.head(20))

            # Download button
            csv = anomalies.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download anomalies.csv",
                csv,
                "anomalies.csv",
                "text/csv"
            )

            # Scatter plot (2D)
            import matplotlib.pyplot as plt

            col1, col2 = selected_cols[:2]
            fig, ax = plt.subplots()
            ax.scatter(normal[col1], normal[col2], alpha=0.3)
            ax.scatter(anomalies[col1], anomalies[col2], alpha=0.9)
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
            st.pyplot(fig)
