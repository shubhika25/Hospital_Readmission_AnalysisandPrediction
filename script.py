import pandas as pd
import pyodbc

df = pd.read_csv("hospital_readmissions.csv")

conn = pyodbc.connect(
     "Driver={SQL Server};"
    "Server=localhost\\SQLEXPRESS;"
    "Database=hospital_readmission;"
    "Trusted_Connection=yes;"
)

cursor = conn.cursor()

# Load into staging table
for index, row in df.iterrows():
    cursor.execute("""
        INSERT INTO staging.readmissions (
            age,
            time_in_hospital,
            n_lab_procedures,
            n_procedures,
            n_medications,
            n_outpatient,
            n_inpatient,
            n_emergency,
            medical_specialty,
            diag_1,
            diag_2,
            diag_3,
            glucose_test,
            A1Ctest,
            change,
            diabetes_med,
            readmitted
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, row.age,
        row.time_in_hospital,
        row.n_lab_procedures,
        row.n_procedures,
        row.n_medications,
        row.n_outpatient,
        row.n_inpatient,
        row.n_emergency,
        row.medical_specialty,
        row.diag_1,
        row.diag_2,
        row.diag_3,
        row.glucose_test,
        row.A1Ctest,
        row.change,
        row.diabetes_med,
        row.readmitted
    )

conn.commit()
cursor.close()

