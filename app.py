import streamlit as st
import pandas as pd
import numpy as np
import joblib
from eda import show_eda

st.set_page_config(
    page_title="Hospital Readmission Risk Predictor",
    layout="wide"
)

# Cleanig options for selectboxes
def clean_options(series):
    return sorted(series.dropna().astype(str).unique())



def age_to_numeric(age_str):
    """Convert age range to numeric midpoint"""
    if pd.isna(age_str):
        return 50
    age_str = str(age_str)
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    for key, val in age_map.items():
        if key in age_str:
            return val
    return 50


# Feature engineering function
def engineer_features(df):
    df = df.copy()
    

    if 'age' in df.columns:
        df['age_num'] = df['age'].apply(age_to_numeric)
    
    # Polypharmacy flag
    if 'num_medications' in df.columns:
        df['polypharmacy'] = (df['num_medications'] > 10).astype(int)
    
    # High glucose flag
    if 'max_glu_serum' in df.columns:
        df['glucose_high'] = (df['max_glu_serum'].isin(['>200', '>300'])).astype(int)
    
    # High A1C flag
    if 'A1Cresult' in df.columns:
        df['a1c_high'] = (df['A1Cresult'].isin(['>7', '>8'])).astype(int)
    
    # Multiple inpatient visits
    if 'number_inpatient' in df.columns:
        df['inpatient_multi'] = (df['number_inpatient'] >= 2).astype(int)
    
    # Total visits
    if all(col in df.columns for col in ['number_outpatient', 'number_emergency', 'number_inpatient']):
        df['total_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
    
    # Diagnosis count
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    existing_diag = [c for c in diag_cols if c in df.columns]
    if existing_diag:
        df['diag_count'] = df[existing_diag].notna().sum(axis=1)
    else:
        df['diag_count'] = 0
    
    # High procedure count
    if 'num_procedures' in df.columns:
        df['high_procedures'] = (df['num_procedures'] > 0).astype(int)
    
    return df

# Loading the sample data
@st.cache_data
def load_sample_data():
    return pd.read_csv("cleaned_readmissions.csv")

sample_df = load_sample_data()

#Loading the model and preprocessor
@st.cache_resource
def load_model():
  
    model_obj = None
    preprocessor_obj = None
    feature_names = None

    try:
        model_obj = joblib.load("models/best_model.pkl")
        
    except Exception:
        model_obj = None

    # Try separate saved preprocessor
    try:
        preprocessor_obj = joblib.load("preprocessor.pkl")
    except Exception:
        preprocessor_obj = None

    if model_obj is None:
      
        try:
            model_obj = joblib.load("models/final_pipeline.pkl")
        except Exception:
            pass

    # If we have no pipeline but have separate model artifact file name
    if model_obj is None:
  
        try:
            model_obj = joblib.load("models/best_model.pkl")
        except Exception:
            pass

    # Load feature names
    try:
        feature_names = joblib.load("models/feature_names.pkl")
    except Exception:
        feature_names = None


    try:
        if model_obj is not None and hasattr(model_obj, "named_steps"):
            if "preprocessor" in model_obj.named_steps:
                preprocessor_obj = model_obj.named_steps["preprocessor"]
    except Exception:
        pass

    return model_obj, preprocessor_obj, feature_names

model, preprocessor, feature_names = load_model()

# def show_final_features(preprocessor, feature_names=None):
#     st.subheader(" Columns Used by Final Model")
#     try:
#         if feature_names is not None:
#             st.write(f"Total transformed features: {len(feature_names)}")
#             st.dataframe(pd.DataFrame(feature_names, columns=["Feature"]), use_container_width=True)
#             return

#         if preprocessor is None:
#             st.info("Preprocessor not available. Retrain or recreate 'preprocessor.pkl' or 'models/best_model.pkl'.")
#             return

#         # show raw numeric/categorical inputs expected by the preprocessor
#         transformers = preprocessor.transformers_
#         rows = []
#         for name, transformer, cols in transformers:
#             # cols may be a list of column names or slice object
#             rows.append({"step": name, "num_input_columns": len(cols) if hasattr(cols, "__len__") else "unknown", "columns": cols})
#         df_show = pd.DataFrame(rows)
#         st.write("Transformer details (raw inputs before encoding):")
#         st.dataframe(df_show, use_container_width=True)

#         # If we can compute final feature count (safe attempt)
#         try:
#             # create a single-row zero-filled DataFrame with columns equal to concatenation of cols
#             raw_cols = []
#             for _, _, cols in transformers:
#                 try:
#                     raw_cols.extend(list(cols))
#                 except Exception:
#                     pass
#             if raw_cols:
#                 dummy = pd.DataFrame([ [np.nan]*len(raw_cols) ], columns=raw_cols)
#                 transformed = preprocessor.transform(dummy)
#                 st.info(f"Total features after transformation: {transformed.shape[1]}")
#         except Exception:
#             pass

#     except Exception as e:
#         st.error(f"Error inspecting preprocessor: {e}")

# #
# Main APP

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA", "Predict Readmission"])

# EDA Page 
if page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    
    st.write("### Choose Dataset")
    data_choice = st.radio("Select data source:", ["Use Sample Dataset", "Upload Your Own CSV"])

    if data_choice == "Use Sample Dataset":
        df = sample_df.copy()
    else:
        user_file = st.file_uploader("Upload CSV", type=['csv'])
        if user_file is not None:
            df = pd.read_csv(user_file)
        else:
            st.info("Please upload a CSV file to continue.")
            st.stop()

    show_eda(df)  # your existing EDA function

    # # show feature names used by model (optional, non-intrusive)
    # with st.expander("View Columns Used in the Model"):
    #     if feature_names is None and preprocessor is None:
    #         st.info("Feature names / preprocessor not found. Train the model to generate them.")
    #     else:
    #         show_final_features(preprocessor, feature_names)

# Prediction Page
elif page == "Predict Readmission":

    st.title("🔮 Readmission Risk Prediction")

    st.write("### Choose Dataset for Prediction")
    pred_source = st.radio("Select prediction input:", ["Enter Manually", "Use Sample Row", "Upload CSV for Bulk Prediction"])

    # Helper to run prediction in a robust way
    def robust_predict(predict_df):
   
        if model is None:
            raise ValueError("Model not loaded. Please train first.")
        
        # If model is a pipeline that contains preprocessor, we can pass raw df directly
        try:
            if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
                preds = model.predict(predict_df)
                prob = model.predict_proba(predict_df)[:, 1]
                return preds, prob
        except Exception:
            # fallback to separate logic
            pass

        # If we have separate preprocessor and raw classifier
        if preprocessor is not None and not hasattr(model, "named_steps"):
            X_trans = preprocessor.transform(predict_df)
            preds = model.predict(X_trans)
            prob = model.predict_proba(X_trans)[:, 1]
            return preds, prob

        # If model is a classifier but preprocessor is missing, try to use model directly (assume saved pipeline)
        if not hasattr(model, "named_steps"):
            try:
                preds = model.predict(predict_df)
                prob = model.predict_proba(predict_df)[:, 1]
                return preds, prob
            except Exception as e:
                raise ValueError(f"Prediction failed. Preprocessor missing or model expects transformed input. Error: {e}")

        raise ValueError("Unable to run prediction with current model/preprocessor configuration.")

 # 1. The manual entry form
    if pred_source == "Enter Manually":

        df = sample_df.copy()  # for dropdown options only

        st.subheader("Enter Patient Details")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Demographics**")
            race = st.selectbox("Race", clean_options(df["race"]), key="race")
            gender = st.selectbox("Gender", clean_options(df["gender"]), key="gender")
            age = st.selectbox("Age Group", clean_options(df["age"]), key="age")
            
            st.write("**Visit Information**")
            time_in_hospital = st.number_input("Time in Hospital (days)", min_value=1, max_value=30, step=1, key="time")
            num_lab = st.number_input("Number of Lab Procedures", min_value=0, max_value=100, step=1, key="lab")
            num_proc = st.number_input("Number of Procedures", min_value=0, max_value=20, step=1, key="proc")
            num_med = st.number_input("Number of Medications", min_value=0, max_value=50, step=1, key="med")
            
            st.write("**Previous Visits**")
            n_out = st.number_input("Outpatient Visits", min_value=0, max_value=20, step=1, key="out")
            n_em = st.number_input("Emergency Visits", min_value=0, max_value=20, step=1, key="em")
            n_in = st.number_input("Inpatient Visits", min_value=0, max_value=20, step=1, key="in")

        with col2:
            st.write("**Diagnosis**")
            diag_1 = st.selectbox("Primary Diagnosis Code (diag_1)", clean_options(df["diag_1"]), key="diag1")
            diag_2_opts = clean_options(df["diag_2"]) if "diag_2" in df.columns else [""]
            diag_2 = st.selectbox("Secondary Diagnosis Code (diag_2)", ["None"] + diag_2_opts, key="diag2")
            if diag_2 == "None":
                diag_2 = None
            diag_3_opts = clean_options(df["diag_3"]) if "diag_3" in df.columns else [""]
            diag_3 = st.selectbox("Tertiary Diagnosis Code (diag_3)", ["None"] + diag_3_opts, key="diag3")
            if diag_3 == "None":
                diag_3 = None
            
            medical_specialty_opts = clean_options(df["medical_specialty"]) if "medical_specialty" in df.columns else [""]
            medical_specialty = st.selectbox("Medical Specialty", ["None"] + medical_specialty_opts, key="specialty")
            if medical_specialty == "None":
                medical_specialty = None
            
            st.write("**Diabetes Related**")
            max_glu_serum_opts = clean_options(df["max_glu_serum"]) if "max_glu_serum" in df.columns else [""]
            max_glu_serum = st.selectbox("Max Glucose Serum", ["None"] + max_glu_serum_opts, key="glu")
            if max_glu_serum == "None":
                max_glu_serum = None
            A1Cresult_opts = clean_options(df["A1Cresult"]) if "A1Cresult" in df.columns else [""]
            A1Cresult = st.selectbox("A1C Result", ["None"] + A1Cresult_opts, key="a1c")
            if A1Cresult == "None":
                A1Cresult = None
            change = st.selectbox("Medication Change", clean_options(df["change"]), key="change")
            diabetesMed = st.selectbox("Diabetes Medication", clean_options(df["diabetesMed"]), key="diabmed")

        # Prepare row
        user_row = pd.DataFrame([{
            'race': race,
            'gender': gender,
            'age': age,
            'time_in_hospital': time_in_hospital,
            'medical_specialty': medical_specialty,
            'num_lab_procedures': num_lab,
            'num_procedures': num_proc,
            'num_medications': num_med,
            'number_outpatient': n_out,
            'number_emergency': n_em,
            'number_inpatient': n_in,
            'diag_1': diag_1,
            'diag_2': diag_2,
            'diag_3': diag_3,
            'max_glu_serum': max_glu_serum,
            'A1Cresult': A1Cresult,
            'change': change,
            'diabetesMed': diabetesMed
        }])

        # Add engineered features (same as training)
        user_row = engineer_features(user_row)

        if st.button("Predict Readmission", type="primary"):

            if model is None:
                st.error("⚠️ Model not loaded. Please train first (run train_model.py).")
            else:
                try:
                    preds, probas = robust_predict(user_row)
                    pred = int(preds[0])
                    proba = float(probas[0])

                    st.success(f"**Prediction: {'Readmitted in <30 days' if pred == 1 else 'Not Readmitted in <30 days'}**")
                    st.metric("Probability of Readmission", f"{proba:.1%}")
                    
                    # Show risk level
                    if proba >= 0.7:
                        st.warning(" High Risk: Patient has high probability of readmission")
                    elif proba >= 0.4:
                        st.info("Moderate Risk: Patient has moderate probability of readmission")
                    else:
                        st.success("Low Risk: Patient has low probability of readmission")
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.info("Please ensure all required fields are filled correctly. If problem persists, retrain model or check model files.")

    # -----------------------------------------------------------------
    # 🟩 B. SAMPLE ROW
    # -----------------------------------------------------------------
    elif pred_source == "Use Sample Row":
        if model is None:
            st.error(" Model not loaded. Please run train_model.py first.")
        else:
            sample = sample_df.sample(1).copy()
            
            # Show sample data
            st.subheader("Sample Patient Data")
            display_cols = [col for col in sample.columns if col not in ['encounter_id', 'patient_nbr', 'readmitted']]
            st.dataframe(sample[display_cols], use_container_width=True)
            
            # Show actual readmission status if available
            if 'readmitted' in sample.columns:
                actual = sample['readmitted'].iloc[0]
                st.info(f"**Actual Readmission Status:** {actual}")

            if st.button("Predict for Sample Row", type="primary"):
                try:
                    # Prepare data for prediction
                    pred_data = sample.drop(columns=['encounter_id', 'patient_nbr', 'readmitted'], errors='ignore')
                    
                    # Add engineered features
                    pred_data = engineer_features(pred_data)
                    
                    preds, probas = robust_predict(pred_data)
                    pred = int(preds[0])
                    proba = float(probas[0])

                    st.success(f"**Prediction: {'Readmitted in <30 days' if pred == 1 else 'Not Readmitted in <30 days'}**")
                    st.metric("Probability of Readmission", f"{proba:.1%}")
                    
                    # Compare with actual if available
                    if 'readmitted' in sample.columns:
                        actual_flag = 1 if sample['readmitted'].iloc[0] == '<30' else 0
                        if pred == actual_flag:
                            st.success("✓ Prediction matches actual outcome!")
                        else:
                            st.warning("⚠️ Prediction differs from actual outcome")
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")


 # 3. Bulk CSV upload for prediction
    elif pred_source == "Upload CSV for Bulk Prediction":
        if model is None:
            st.error("Model not loaded. Please run train_model.py first.")
        else:
            uploaded = st.file_uploader("Upload CSV", type=['csv'], help="Upload a CSV file with patient data for bulk prediction")

            if uploaded:
                try:
                    data = pd.read_csv(uploaded)
                    st.subheader("Uploaded Data Preview")
                    st.dataframe(data.head(10), use_container_width=True)
                    st.info(f"Total records: {len(data)}")

                    if st.button("Run Bulk Prediction", type="primary"):
                        with st.spinner("Processing predictions..."):
                            try:
                                # Drop ID and target columns if present
                                cols_to_drop = ['encounter_id', 'patient_nbr', 'readmitted', 'readmitted_flag']
                                pred_data = data.drop(columns=[col for col in cols_to_drop if col in data.columns], errors='ignore')
                                
                                # Add engineered features
                                pred_data = engineer_features(pred_data)
                                
                                # Transform and predict
                                preds, probas = robust_predict(pred_data)
                                preds = preds.astype(int)
                                probas = probas.astype(float)

                                # Add predictions to original data
                                result_data = data.copy().reset_index(drop=True)
                                result_data["prediction"] = preds
                                result_data["prediction_label"] = result_data["prediction"].map({1: "Readmitted <30 days", 0: "Not Readmitted <30 days"})
                                result_data["probability"] = probas

                                st.success(f"Prediction completed for {len(result_data)} records!")
                                
                                # Show summary statistics
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Total Predictions", len(result_data))
                                col2.metric("Predicted Readmissions", int(result_data["prediction"].sum()))
                                col3.metric("Average Probability", f"{result_data['probability'].mean():.1%}")
                                
                                # Show results preview
                                st.subheader("Prediction Results Preview")
                                display_cols = ['prediction_label', 'probability'] + [col for col in result_data.columns if col not in ['prediction', 'prediction_label', 'probability']]
                                st.dataframe(result_data[display_cols].head(20), use_container_width=True)
                                
                                # Download button
                                csv = result_data.to_csv(index=False)
                                st.download_button(
                                    label=" Download Predictions CSV",
                                    data=csv,
                                    file_name="bulk_predictions.csv",
                                    mime="text/csv"
                                )
                                
                            except Exception as e:
                                st.error(f"Error during bulk prediction: {str(e)}")
                                st.info("Please ensure your CSV file has the required columns matching the training data.")
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
