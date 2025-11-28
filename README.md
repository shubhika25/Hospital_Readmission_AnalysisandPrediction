# Hospital_Readmission_AnalysisandPrediction

A Healthcare Analytics + Machine Learning project to predict patient readmissions, generate insights, detect anomalies, and store data in a SQL Server warehouse using a Streamlit app

<img width="1825" height="817" alt="image" src="https://github.com/user-attachments/assets/431a77a8-384a-425b-b039-16391f412c66" />


# Key Features
-----------------------------------------------------
**1. SQL Data Warehouse**

CSV → staging.readmissions

ETL → dim tables & fact.visits

Star schema for analytics

**2. Streamlit App**

Automated EDA (YData Profiling)

Custom EDA dashboards

Readmission prediction

Anomaly detection

Downloadable insight reports (PDF)

Optional SQL storage for predictions

**3. ML Pipeline**

Evaluated Logistic Regression, Random Forest, XGBoost, Naive Bayes, SVM, KNN, Decision Tree.

Final selected model: Random Forest Classifier

**Model Comparison**
| Model               | Accuracy   | Precision  | Recall     | F1         | ROC-AUC    |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 0.6120     | 0.6329     | 0.4086     | 0.4966     | 0.6445     |
| **Random Forest**   | **0.6068** | **0.5946** | **0.5047** | **0.5459** | **0.6388** |
| XGBoost             | 0.6008     | 0.5844     | 0.5111     | 0.5453     | 0.6336     |
| Naive Bayes         | 0.5980     | 0.6007     | 0.4227     | 0.4962     | 0.6214     |
| SVM                 | 0.5854     | 0.7245     | 0.1853     | 0.2951     | 0.6522     |
| KNN                 | 0.5456     | 0.5169     | 0.4560     | 0.4846     | 0.5488     |
| Decision Tree       | 0.5452     | 0.5145     | 0.5149     | 0.5147     | 0.5434     |


**Why Random Forest Was Selected??**

Random Forest offered the best balance between accuracy, precision, recall, and F1 score.
Higher recall than Logistic Regression → better at catching true readmissions
This makes it clinically safer and statistically consistent.

# Insights Gained (EDA + SQL + Advanced Analysis)

1. Elderly patients show significantly higher readmission rates

Patients aged 70–79, 80–89, and 90+ return to the hospital more frequently.

2. Multi-diagnosis cases have higher risk

Patients with 2 or 3 diagnoses per visit show:

higher medical complexity

higher readmission percentages

3. Top high-risk diagnosis categories

1. Diabetes
2. Circulatory diseases (Heart Failure, CAD, Hypertension)
3. Respiratory illnesses (COPD, Pneumonia)

4. Length of Stay (LOS) patterns

LOS 6–10 days had a noticeable spike in readmission risk — suggesting:

borderline discharge decisions

incomplete recovery

underlying comorbidities

5. Frequent hospital visitors are extremely high-risk
Total Visits	Risk Category
0–1	Low
2–3	Moderate
4–10	High
10+	Very High

Patients with repeated hospitalization cycles need proactive case management.

6. Diabetes medication change is a strong predictor

Patients with:

change = Yes

diabetes_med = Yes

showed significantly higher readmission rates.

This indicates medication instability or poor glycemic control.

7. Emergency visits highly correlate with readmission

Higher n_emergency values consistently increase:

clinical severity

likelihood of returning within 30 days

8. Treatment intensity is a risk marker

High usage of:
lab tests
procedures
medications

9. High-risk diagnosis pairs

Certain combinations like:

Diabetes + Respiratory

Circulatory + Diabetes

Respiratory + COPD

were strongly associated with readmissions.

10. Specialties with highest readmissions

Cardiology

Pulmonology

Geriatrics

These require targeted discharge and follow-up protocols.

# Architecture Overview

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/e18d5348-9fee-46f5-a836-4914f9690f13" />

# Tech Stack

Python, Pandas, NumPy
Scikit-learn, XGBoost
Streamlit, YData Profiling
SQL Server
Isolation Forest (Anomaly Detection)


