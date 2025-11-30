import streamlit as st
import pandas as pd
import altair as alt

def show_eda(df):

    st.title("📊 Exploratory Data Analysis – Hospital Readmission Dataset")

    # Ensure binary column exists
    df["readmit_binary"] = df["readmitted"].ne("NO").astype(int)

    # -----------------------------------------------------------
    # 1. BASIC DATA INSIGHTS
    # -----------------------------------------------------------
    st.header("1. Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Unique Patients", df["patient_nbr"].nunique())
    col3.metric("Overall Readmission Rate", 
                f"{round(df['readmit_binary'].mean() * 100, 2)} %")

    st.subheader("Preview")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe(include="all"))

    st.markdown("---")

    # -----------------------------------------------------------
    # 2. Readmission by Gender
    # -----------------------------------------------------------
    st.header("2. Readmission Rate by Gender")

    gender_df = (
        df.groupby("gender")["readmit_binary"]
        .mean()
        .reset_index(name="readmission_rate")
    )

    st.altair_chart(
        alt.Chart(gender_df)
        .mark_bar()
        .encode(
            x="gender:N",
            y="readmission_rate:Q",
            tooltip=["gender", "readmission_rate"]
        ),
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------------------------------------
    # 3. Readmission by Race
    # -----------------------------------------------------------
    st.header("3. Readmission Rate by Race")

    race_df = (
        df.groupby("race")["readmit_binary"]
        .mean()
        .reset_index(name="readmission_rate")
    )

    st.altair_chart(
        alt.Chart(race_df)
        .mark_bar()
        .encode(
            x="race:N",
            y="readmission_rate:Q",
            tooltip=["race", "readmission_rate"]
        ),
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------------------------------------
    # 4. Gender × Race × Age Interaction
    # -----------------------------------------------------------
    st.header("4. Gender × Race × Age Group Interaction")

    grp = (
        df.groupby(["gender", "race", "age"])["readmit_binary"]
        .mean()
        .reset_index(name="readmission_rate")
    )

    st.altair_chart(
        alt.Chart(grp)
        .mark_circle(size=150)
        .encode(
            x="race:N",
            y="readmission_rate:Q",
            color="gender:N",
            column="age:N",
            tooltip=["gender", "race", "age", "readmission_rate"]
        ),
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------------------------------------
    # 5. Readmission by Primary Diagnosis (diag_1)
    # -----------------------------------------------------------
    st.header("5. Readmission Rate by Primary Diagnosis (diag_1)")

    diag_df = (
        df.groupby("diag_1")["readmit_binary"]
        .mean()
        .reset_index(name="readmission_rate")
        .sort_values("readmission_rate", ascending=False)
        .head(20)
    )

    st.altair_chart(
        alt.Chart(diag_df)
        .mark_bar()
        .encode(
            x="diag_1:N",
            y="readmission_rate:Q",
            tooltip=["diag_1", "readmission_rate"]
        ),
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------------------------------------
    # 6. Visit/Procedure Counts (melted view)
    # -----------------------------------------------------------
    st.header("6. Average Count of Visits / Procedures")

    numeric_cols = [
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
    ]

    melted = df.melt(
        id_vars=["readmit_binary"],
        value_vars=numeric_cols,
        var_name="metric",
        value_name="value"
    )

    st.altair_chart(
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x="metric:N",
            y="mean(value):Q",
            color="metric:N",
            tooltip=["metric", "mean(value)"]
        ),
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------------------------------------
    # 7. Readmission by Age Group
    # -----------------------------------------------------------
    st.header("7. Readmission Rate by Age Group")

    age_df = (
        df.groupby("age")["readmit_binary"]
        .mean()
        .reset_index(name="readmission_rate")
        .sort_values("age")
    )

    st.altair_chart(
        alt.Chart(age_df)
        .mark_bar()
        .encode(
            x="age:N",
            y="readmission_rate:Q",
            tooltip=["age", "readmission_rate"]
        ),
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------------------------------------
    # 8. Length of Stay vs Readmission
    # -----------------------------------------------------------
    st.header("8. Length of Stay (time_in_hospital) vs Readmission")

    los_df = (
        df.groupby("time_in_hospital")["readmit_binary"]
        .mean()
        .reset_index(name="readmission_rate")
    )

    st.altair_chart(
        alt.Chart(los_df)
        .mark_line(point=True)
        .encode(
            x="time_in_hospital:Q",
            y="readmission_rate:Q",
            tooltip=["time_in_hospital", "readmission_rate"]
        ),
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------------------------------------
    # 9. Medical Specialty vs Readmission
    # -----------------------------------------------------------
    st.header("9. Readmission Rate by Medical Specialty")

    spec_df = (
        df.groupby("medical_specialty")["readmit_binary"]
        .mean()
        .reset_index(name="readmission_rate")
        .sort_values("readmission_rate", ascending=False)
        .head(20)
    )

    st.altair_chart(
        alt.Chart(spec_df)
        .mark_bar()
        .encode(
            y="medical_specialty:N",
            x="readmission_rate:Q",
            tooltip=["medical_specialty", "readmission_rate"]
        ),
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------------------------------------
    # 10. Emergency Visits vs Readmission
    # -----------------------------------------------------------
    st.header("10. Emergency Visits vs Readmission")

    em_df = (
        df.groupby("number_emergency")["readmit_binary"]
        .mean()
        .reset_index(name="readmission_rate")
    )

    st.altair_chart(
        alt.Chart(em_df)
        .mark_line(point=True)
        .encode(
            x="number_emergency:Q",
            y="readmission_rate:Q",
            tooltip=["number_emergency", "readmission_rate"]
        ),
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------------------------------------
    # 11. Medication Load vs Readmission
    # -----------------------------------------------------------
    st.header("11. Medication Count vs Readmission")

    med_df = (
        df.groupby("num_medications")["readmit_binary"]
        .mean()
        .reset_index(name="readmission_rate")
    )

    st.altair_chart(
        alt.Chart(med_df)
        .mark_bar()
        .encode(
            x="num_medications:Q",
            y="readmission_rate:Q",
            tooltip=["num_medications", "readmission_rate"]
        ),
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------------------------------------
    # 12. Glucose & A1C vs Readmission
    # -----------------------------------------------------------
    st.header("12. Glucose & A1C Abnormality vs Readmission")

    glu_df = (
        df.groupby("max_glu_serum")["readmit_binary"]
        .mean()
        .reset_index(name="readmission_rate")
    )

    a1c_df = (
        df.groupby("A1Cresult")["readmit_binary"]
        .mean()
        .reset_index(name="readmission_rate")
    )

    col1, col2 = st.columns(2)

    col1.subheader("Max Glucose Serum")
    col1.altair_chart(
        alt.Chart(glu_df)
        .mark_bar()
        .encode(
            x="max_glu_serum:N",
            y="readmission_rate:Q",
            tooltip=["max_glu_serum", "readmission_rate"]
        ),
        use_container_width=True
    )

    col2.subheader("A1C Result")
    col2.altair_chart(
        alt.Chart(a1c_df)
        .mark_bar()
        .encode(
            x="A1Cresult:N",
            y="readmission_rate:Q",
            tooltip=["A1Cresult", "readmission_rate"]
        ),
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------------------------------------
    # 13. Diabetes Medication Impact
    # -----------------------------------------------------------
    st.header("13. Diabetes Medication & Change Impact")

    change_df = (
        df.groupby("change")["readmit_binary"]
        .mean()
        .reset_index(name="readmission_rate")
    )

    medtype_df = (
        df.groupby("diabetesMed")["readmit_binary"]
        .mean()
        .reset_index(name="readmission_rate")
    )

    colA, colB = st.columns(2)

    colA.subheader("Medication Change")
    colA.altair_chart(
        alt.Chart(change_df)
        .mark_bar()
        .encode(
            x="change:N",
            y="readmission_rate:Q",
            tooltip=["change", "readmission_rate"]
        ),
        use_container_width=True
    )

    colB.subheader("Diabetes Medication")
    colB.altair_chart(
        alt.Chart(medtype_df)
        .mark_bar()
        .encode(
            x="diabetesMed:N",
            y="readmission_rate:Q",
            tooltip=["diabetesMed", "readmission_rate"]
        ),
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------------------------------------
    # 14. Procedures Count vs Readmission
    # -----------------------------------------------------------
    st.header("14. Procedure Count vs Readmission")

    proc_df = (
        df.groupby("num_procedures")["readmit_binary"]
        .mean()
        .reset_index(name="readmission_rate")
    )

    st.altair_chart(
        alt.Chart(proc_df)
        .mark_line(point=True)
        .encode(
            x="num_procedures:Q",
            y="readmission_rate:Q",
            tooltip=["num_procedures", "readmission_rate"]
        ),
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------------------------------------
    # 15. Comorbidity Count vs Readmission
    # -----------------------------------------------------------
    st.header("15. Comorbidity Count (diag_1, diag_2, diag_3)")

    df["diag_count"] = df[["diag_1", "diag_2", "diag_3"]].notna().sum(axis=1)

    diag_cnt_df = (
        df.groupby("diag_count")["readmit_binary"]
        .mean()
        .reset_index(name="readmission_rate")
    )

    st.altair_chart(
        alt.Chart(diag_cnt_df)
        .mark_bar()
        .encode(
            x="diag_count:O",
            y="readmission_rate:Q",
            tooltip=["diag_count", "readmission_rate"]
        ),
        use_container_width=True
    )

    st.success("EDA Loaded Successfully.")
