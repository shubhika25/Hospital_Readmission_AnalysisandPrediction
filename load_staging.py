import pandas as pd
import pyodbc

df = pd.read_csv("cleaned_readmissions.csv")

# Replace NaN with None for SQL
df = df.where(pd.notnull(df), None)

conn = pyodbc.connect(
    "Driver={ODBC Driver 17 for SQL Server};"
    "Server=localhost\\SQLEXPRESS;"
    "Database=hospital_readmission_db;"
    "Trusted_Connection=yes;"
)

cursor = conn.cursor()

sql = """
INSERT INTO staging.readmissions (
    encounter_id,
    patient_nbr,
    race,
    gender,
    age,
    time_in_hospital,
    medical_specialty,
    num_lab_procedures,
    num_procedures,
    num_medications,
    number_outpatient,
    number_emergency,
    number_inpatient,
    diag_1,
    diag_2,
    diag_3,
    max_glu_serum,
    A1Cresult,
    change,
    diabetesMed,
    readmitted
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

cursor.executemany(sql, df.values.tolist())
conn.commit()
cursor.close()

print("Successfully loaded", len(df), "rows into staging.")
