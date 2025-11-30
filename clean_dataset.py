import pandas as pd
import pyodbc

df = pd.read_csv("sample_data/readmission.csv")

# Keep required columns only


# Columns we want to keep
cols = [
    "encounter_id",
    "patient_nbr",
    "race",
    "gender",
    "age",
    "time_in_hospital",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "diag_1",
    "diag_2",
    "diag_3",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed",
    "readmitted"
]

clean_df = df[cols]

# Replace ? with NULL for SQL inserts
clean_df = clean_df.replace("?", None)

# Save cleaned file
clean_df.to_csv("cleaned_readmissions.csv", index=False)

print("Saved cleaned_readmissions.csv with shape:", clean_df.shape)


