

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("Training Started")
print("=" * 70)

# Load cleaned dataset
df = pd.read_csv("cleaned_readmissions.csv")
print(f"Loaded dataset: {df.shape}")

#Target variable
df["readmitted_flag"] = (df["readmitted"] == "<30").astype(int)
y = df["readmitted_flag"]


drop_cols = ["encounter_id", "patient_nbr", "readmitted", "readmitted_flag"]
X = df.drop(columns=[col for col in drop_cols if col in df.columns])


def age_to_numeric(age_str):
    if pd.isna(age_str):
        return 50
    age_map = {
        '[0-10)': 5,'[10-20)': 15,'[20-30)': 25,'[30-40)': 35,'[40-50)': 45,
        '[50-60)': 55,'[60-70)': 65,'[70-80)': 75,'[80-90)': 85,'[90-100)': 95
    }
    for key, val in age_map.items():
        if key in str(age_str):
            return val
    return 50

def engineer_features(df):
    df = df.copy()
    if "age" in df: df["age_num"] = df["age"].apply(age_to_numeric)
    if "num_medications" in df: df["polypharmacy"] = (df["num_medications"] > 10).astype(int)
    if "max_glu_serum" in df: df["glucose_high"] = df["max_glu_serum"].isin([">200", ">300"]).astype(int)
    if "A1Cresult" in df: df["a1c_high"] = df["A1Cresult"].isin([">7", ">8"]).astype(int)
    if "number_inpatient" in df: df["inpatient_multi"] = (df["number_inpatient"] >= 2).astype(int)
    if all(c in df for c in ["number_outpatient","number_emergency","number_inpatient"]):
        df["total_visits"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    diag_cols = ["diag_1","diag_2","diag_3"]
    df["diag_count"] = df[diag_cols].notna().sum(axis=1)
    if "num_procedures" in df: df["high_procedures"] = (df["num_procedures"] > 0).astype(int)
    return df

X = engineer_features(X)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# ------------------------------------------------------------
# Final model
# ------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", model)
])


print("\nTraining model...")
pipe.fit(X_train, y_train)


pred = pipe.predict(X_test)
proba = pipe.predict_proba(X_test)[:, 1]

print("\nAUC:", roc_auc_score(y_test, proba))
print(classification_report(y_test, pred))
# SAve model and preprocessor
joblib.dump(pipe, "models/best_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")

# Save final feature names
feature_names = preprocessor.get_feature_names_out()
joblib.dump(feature_names, "models/feature_names.pkl")

print("\nSaved:")
print(f"- best_model.pkl")
print(f"- preprocessor.pkl")
print(f"- feature_names.pkl ({len(feature_names)} features)")

print("\nTraining Completed Successfully")
