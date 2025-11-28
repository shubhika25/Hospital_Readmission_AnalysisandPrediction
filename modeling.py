# --- modeling.py ---
"""
Training and loading RandomForest model for Readmission Prediction.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

MODEL_PATH = os.path.join("models", "rf_model.pkl")


# -----------------------------------------------------------
# DATA PREPARATION
# -----------------------------------------------------------
def prepare_data(df, target_col="readmitted"):
    df = df.copy()
    df = df.dropna(subset=[target_col])

    y = df[target_col]
    X = df.drop(columns=[target_col])

    if y.dtype == "object":
        y = y.map(lambda v: 1 if str(v).strip().lower() in
                  ['yes','1','y','true','readmitted','readmit']
                  else 0)

    return X, y


def simple_impute_and_encode(X):
    X = X.copy()

    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(include=["object","category"]).columns

    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())

    for c in cat_cols:
        X[c] = X[c].fillna("missing")

    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X


# -----------------------------------------------------------
# TRAINING PIPELINE
# -----------------------------------------------------------
def train_and_save_model(df, target_col="readmitted"):

    os.makedirs("models", exist_ok=True)

    X, y = prepare_data(df)
    X = simple_impute_and_encode(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_scaled, y_train)

    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_res, y_res)

    save_obj = {
        "model": rf,
        "scaler": scaler,
        "columns": list(X.columns)
    }

    joblib.dump(save_obj, MODEL_PATH)

    # Evaluation
    proba = rf.predict_proba(X_test_scaled)[:, 1]
    pred = rf.predict(X_test_scaled)
    print("ROC-AUC:", roc_auc_score(y_test, proba))
    print(classification_report(y_test, pred))
    print(confusion_matrix(y_test, pred))

    return save_obj


# -----------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None
