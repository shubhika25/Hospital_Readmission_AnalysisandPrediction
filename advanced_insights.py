# advanced_insights.py
"""
Advanced insight generation for Readmission project, with plotting utilities.

Contains:
 - existing analytic functions (overview_metrics, age_group_analysis, ...)
 - plotting helpers returning matplotlib.Figure objects:
    plot_age_group_bar, plot_top_diag_bar, plot_visitor_pie,
    plot_los_bar, plot_treatment_cluster_bar, plot_correlation_heatmap
 - text-only PDF generator (generate_insights_pdf)
"""

import pandas as pd
import numpy as np
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from textwrap import shorten
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ---------------------------
# ICD9 -> Category mapping
# ---------------------------
def icd9_to_category(code):
    if pd.isna(code):
        return "Other"
    try:
        s = str(code).strip()
        low = s.lower()
        if any(k in low for k in ["diabet", "diabetes"]):
            return "Diabetes"
        if any(k in low for k in ["respir", "pneum", "lung"]):
            return "Respiratory"
        if any(k in low for k in ["card", "circulat", "heart", "myocard"]):
            return "Circulatory"
        if any(k in low for k in ["digest", "gastro", "abd", "stomach", "intestin"]):
            return "Digestive"
        if any(k in low for k in ["injury", "fract", "trauma"]):
            return "Injury"
        if any(k in low for k in ["musculo", "arthritis", "joint", "back"]):
            return "Musculoskeletal"

        numpart = s.split('.')[0]
        num = int(''.join([c for c in numpart if c.isdigit()]) or 0)
    except Exception:
        return "Other"

    if num == 250:
        return "Diabetes"
    if 390 <= num <= 459:
        return "Circulatory"
    if 460 <= num <= 519:
        return "Respiratory"
    if 520 <= num <= 579:
        return "Digestive"
    if 710 <= num <= 739:
        return "Musculoskeletal"
    if 800 <= num <= 999:
        return "Injury"
    return "Other"


# ---------------------------
# BASIC OVERVIEW
# ---------------------------
def overview_metrics(df, target_col="readmitted"):
    df = df.copy()
    total = len(df)
    y = df[target_col]
    if y.dtype == 'object':
        y = y.map(lambda v: 1 if str(v).strip().lower() in ['yes','y','1','true','readmitted','readmit'] else 0)
    readmit_rate = round(100 * y.sum() / max(1, total), 2)
    avg_los = None
    if 'time_in_hospital' in df.columns:
        avg_los = round(df['time_in_hospital'].dropna().mean(), 2)
    return {
        "total_records": int(total),
        "total_columns": int(df.shape[1]),
        "readmission_rate_pct": readmit_rate,
        "avg_length_of_stay": avg_los
    }


# ---------------------------
# AGE-GROUP ANALYSIS
# ---------------------------
def age_group_analysis(df, age_col='age', target_col='readmitted'):
    df2 = df.copy()
    if age_col not in df2.columns:
        return pd.DataFrame()
    y = df2[target_col]
    if y.dtype == 'object':
        y = y.map(lambda v: 1 if str(v).strip().lower() in ['yes','y','1','true','readmitted','readmit'] else 0)
    df2['__y'] = y
    if df2[age_col].dtype == object:
        grp = df2.groupby(age_col)['__y'].agg(['count','sum']).rename(columns={'sum':'readmissions'}).reset_index()
        grp['rate'] = (grp['readmissions'] / grp['count']).round(4)
        grp['readmission_rate_pct'] = (grp['rate']*100).round(2)
        return grp.sort_values(by='readmission_rate_pct', ascending=False)
    else:
        bins = [0,40,50,60,70,80,90,120]
        labels = ['0-39','40-49','50-59','60-69','70-79','80-89','90+']
        df2['age_bucket'] = pd.cut(df2[age_col], bins=bins, labels=labels, right=False)
        grp = df2.groupby('age_bucket')['__y'].agg(['count','sum']).reset_index().rename(columns={'sum':'readmissions'})
        grp['rate'] = (grp['readmissions'] / grp['count']).round(4)
        grp['readmission_rate_pct'] = (grp['rate']*100).round(2)
        return grp.sort_values(by='readmission_rate_pct', ascending=False)


# ---------------------------
# MULTI vs SINGLE DIAGNOSIS
# ---------------------------
def multi_vs_single_diag(df, diag_cols=['diag_1','diag_2','diag_3'], target_col='readmitted'):
    df2 = df.copy()
    df2['diag_count'] = df2[diag_cols].notnull().sum(axis=1)
    y = df2[target_col]
    if y.dtype == 'object':
        y = y.map(lambda v: 1 if str(v).strip().lower() in ['yes','y','1','true','readmitted','readmit'] else 0)
    df2['__y'] = y
    grp = df2.groupby('diag_count')['__y'].agg(['count','sum']).reset_index().rename(columns={'sum':'readmissions'})
    grp['rate'] = (grp['readmissions'] / grp['count']).round(4)
    grp['readmission_rate_pct'] = (grp['rate']*100).round(2)
    return grp.sort_values(by='diag_count')


# ---------------------------
# FREQUENT VISITOR BUCKETS (fixed)
# ---------------------------
def frequent_visitor_buckets(df, cols=['n_outpatient','n_inpatient','n_emergency'], target_col='readmitted'):
    df2 = df.copy()
    for c in cols:
        if c not in df2.columns:
            df2[c] = 0
    df2['total_visits'] = df2[cols].sum(axis=1)
    df2['visitor_bucket'] = pd.cut(
        df2['total_visits'],
        bins=[-1, 1, 3, 10, 9999],
        labels=['1', '2-3', '4-10', '10+']
    )
    # convert to strings to avoid categorical setitem issues
    df2['visitor_bucket'] = df2['visitor_bucket'].astype(str)
    y = df2[target_col]
    if y.dtype == 'object':
        y = y.map(lambda v: 1 if str(v).strip().lower() in ['yes','y','1','true','readmitted','readmit'] else 0)
    df2['__y'] = y
    grp = df2.groupby('visitor_bucket')['__y'].agg(['count','sum']).reset_index().fillna(0)
    grp = grp.rename(columns={'sum':'readmissions'})
    grp['readmission_rate_pct'] = ((grp['readmissions'] / grp['count']).replace([np.inf, np.nan], 0) * 100).round(2)
    return grp


# ---------------------------
# TOP DIAGNOSIS BY READMISSION (diag_1)
# ---------------------------
def top_diag_by_readmission(df, diag_col='diag_1', target_col='readmitted', top_n=10):
    df2 = df.copy()
    df2['diag_cat'] = df2[diag_col].apply(icd9_to_category)
    y = df2[target_col]
    if y.dtype == 'object':
        y = y.map(lambda v: 1 if str(v).strip().lower() in ['yes','y','1','true','readmitted','readmit'] else 0)
    df2['__y'] = y
    grp = df2.groupby('diag_cat')['__y'].agg(['count','sum']).reset_index().rename(columns={'sum':'readmissions'})
    grp['rate'] = (grp['readmissions'] / grp['count']).round(4)
    grp['readmission_rate_pct'] = (grp['rate']*100).round(2)
    return grp.sort_values(by='readmission_rate_pct', ascending=False).head(top_n)


# ---------------------------
# DIAGNOSIS PAIRS (high risk)
# ---------------------------
def diag_pairs_risk(df, diag_a='diag_1', diag_b='diag_2', target_col='readmitted', min_cases=10, top_n=20):
    df2 = df.copy()
    df2['a_cat'] = df2[diag_a].apply(icd9_to_category)
    df2['b_cat'] = df2[diag_b].apply(icd9_to_category)
    y = df2[target_col]
    if y.dtype == 'object':
        y = y.map(lambda v: 1 if str(v).strip().lower() in ['yes','y','1','true','readmitted','readmit'] else 0)
    df2['__y'] = y
    grp = df2.groupby(['a_cat','b_cat'])['__y'].agg(['count','sum']).reset_index().rename(columns={'sum':'readmissions'})
    grp = grp[grp['count'] >= min_cases]
    grp['rate'] = (grp['readmissions'] / grp['count']).round(4)
    grp['readmission_rate_pct'] = (grp['rate']*100).round(2)
    return grp.sort_values(by='readmission_rate_pct', ascending=False).head(top_n)


# ---------------------------
# DIABETES SPECIFIC BREAKDOWN
# ---------------------------
def diabetes_breakdown(df, target_col='readmitted'):
    df2 = df.copy()
    res = {}
    key = 'diabetes_med'
    if key in df2.columns:
        grp = df2.groupby(key)[target_col].apply(
            lambda s: pd.Series({
                'cases': len(s),
                'readmit_rate_pct': round(100 * (s.map(lambda v: 1 if str(v).strip().lower() in ['yes','y','1','true','readmitted','readmit'] else 0).sum() / max(1, len(s))), 2)
            })
        ).unstack()
        res['diabetes_med'] = grp.reset_index()
    else:
        res['diabetes_med'] = pd.DataFrame()
    key2 = 'glucose_test'
    if key2 in df2.columns:
        grp2 = df2.groupby(key2)[target_col].apply(
            lambda s: pd.Series({
                'cases': len(s),
                'readmit_rate_pct': round(100 * (s.map(lambda v: 1 if str(v).strip().lower() in ['yes','y','1','true','readmitted','readmit'] else 0).sum() / max(1, len(s))), 2)
            })
        ).unstack()
        res['glucose_test'] = grp2.reset_index()
    else:
        res['glucose_test'] = pd.DataFrame()
    return res


# ---------------------------
# LOS BUCKETS
# ---------------------------
def los_buckets(df, los_col='time_in_hospital', target_col='readmitted'):
    df2 = df.copy()
    if los_col not in df2.columns:
        return pd.DataFrame()
    bins = [-1, 2, 5, 10, 9999]
    labels = ['0-2', '3-5', '6-10', '10+']
    df2['los_bucket'] = pd.cut(df2[los_col], bins=bins, labels=labels)
    df2['los_bucket'] = df2['los_bucket'].astype(str)
    y = df2[target_col]
    if y.dtype == 'object':
        y = y.map(lambda v: 1 if str(v).strip().lower() in ['yes','y','1','true','readmitted','readmit'] else 0)
    df2['__y'] = y
    grp = df2.groupby('los_bucket')['__y'].agg(['count','sum']).reset_index().rename(columns={'sum':'readmissions'})
    grp['readmission_rate_pct'] = ((grp['readmissions'] / grp['count']).replace([np.inf, np.nan], 0) * 100).round(2)
    return grp


# ---------------------------
# TREATMENT CLUSTER SUMMARY
# ---------------------------
def treatment_cluster_summary(df, cols=['n_lab_procedures','n_procedures','n_medications']):
    df2 = df.copy()
    for c in cols:
        if c not in df2.columns:
            df2[c] = 0
    df2['treatment_score'] = df2[cols].sum(axis=1)
    # guard against constant values for qcut
    try:
        df2['treatment_cluster'] = pd.qcut(df2['treatment_score'].rank(method='first'), q=4, labels=['low','med','high','very_high'])
    except Exception:
        df2['treatment_cluster'] = pd.cut(df2['treatment_score'], bins=4, labels=['low','med','high','very_high'])
    grp = df2.groupby('treatment_cluster')['treatment_score'].agg(['count','mean']).reset_index()
    return grp


# ---------------------------
# RECOMMENDATIONS
# ---------------------------
def generate_recommendations():
    recs = [
        "Flag elderly patients with multi-diagnoses for post-discharge follow-up.",
        "Prioritise patients with >2 emergency visits for intensive case management.",
        "Review medication changes for diabetic patients — higher readmission observed.",
        "Investigate LOS ranges (6-10 days) where readmission spikes and refine discharge criteria.",
        "Consider targeted interventions for top diagnosis categories (Diabetes, Respiratory, Circulatory)."
    ]
    return recs


# ---------------------------
# PLOTTING HELPERS (return matplotlib.Figure)
# ---------------------------
def plot_age_group_bar(df, age_col='age', target_col='readmitted', top_n=8):
    ag = age_group_analysis(df, age_col=age_col, target_col=target_col)
    if ag.empty:
        fig, ax = plt.subplots()
        ax.text(0.5,0.5,"No age data", ha='center')
        return fig
    # prepare plotting order by rate desc
    ag_plot = ag.sort_values(by='readmission_rate_pct', ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(ag_plot.iloc[:,0].astype(str), ag_plot['readmission_rate_pct'])
    ax.set_ylabel("Readmission Rate (%)")
    ax.set_title("Readmission Rate by Age Group")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_top_diag_bar(df, diag_col='diag_1', target_col='readmitted', top_n=8):
    td = top_diag_by_readmission(df, diag_col=diag_col, target_col=target_col, top_n=top_n)
    if td.empty:
        fig, ax = plt.subplots()
        ax.text(0.5,0.5,"No diagnosis data", ha='center')
        return fig
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(td['diag_cat'].astype(str), td['readmission_rate_pct'])
    ax.set_ylabel("Readmission Rate (%)")
    ax.set_title("Top Diagnosis Categories by Readmission Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_visitor_pie(df, cols=['n_outpatient','n_inpatient','n_emergency'], target_col='readmitted'):
    grp = frequent_visitor_buckets(df, cols=cols, target_col=target_col)
    if grp.empty:
        fig, ax = plt.subplots()
        ax.text(0.5,0.5,"No visitor data", ha='center')
        return fig
    labels = grp['visitor_bucket'].astype(str)
    sizes = grp['count']
    fig, ax = plt.subplots(figsize=(6,4))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.set_title("Visitor Bucket Distribution")
    ax.axis('equal')
    return fig


def plot_los_bar(df, los_col='time_in_hospital', target_col='readmitted'):
    lb = los_buckets(df, los_col=los_col, target_col=target_col)
    if lb.empty:
        fig, ax = plt.subplots()
        ax.text(0.5,0.5,"No LOS data", ha='center')
        return fig
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(lb['los_bucket'].astype(str), lb['readmission_rate_pct'])
    ax.set_ylabel("Readmission Rate (%)")
    ax.set_title("Readmission Rate by Length of Stay Bucket")
    plt.tight_layout()
    return fig


def plot_treatment_cluster_bar(df, cols=['n_lab_procedures','n_procedures','n_medications']):
    tc = treatment_cluster_summary(df, cols=cols)
    if tc.empty:
        fig, ax = plt.subplots()
        ax.text(0.5,0.5,"No treatment data", ha='center')
        return fig
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(tc['treatment_cluster'].astype(str), tc['count'])
    ax.set_ylabel("Number of Patients")
    ax.set_title("Treatment Complexity Clusters")
    plt.tight_layout()
    return fig


def safe_numeric_df_for_corr(df):
    # select numeric columns only and drop constant columns
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.loc[:, num.nunique() > 1]
    return num


def plot_correlation_heatmap(df):
    num = safe_numeric_df_for_corr(df)
    if num.shape[1] == 0:
        fig, ax = plt.subplots()
        ax.text(0.5,0.5,"No numeric data for correlation", ha='center')
        return fig
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    return fig


# ---------------------------
# COMPOSE FULL REPORT DICT
# ---------------------------
def generate_insights_report(df, target_col='readmitted'):
    report = {}
    report['overview'] = overview_metrics(df, target_col=target_col)
    report['age_group'] = age_group_analysis(df, age_col='age', target_col=target_col)
    report['multi_vs_single'] = multi_vs_single_diag(df, diag_cols=['diag_1','diag_2','diag_3'], target_col=target_col)
    report['visitor_buckets'] = frequent_visitor_buckets(df, cols=['n_outpatient','n_inpatient','n_emergency'], target_col=target_col)
    report['top_diag'] = top_diag_by_readmission(df, diag_col='diag_1', target_col=target_col, top_n=12)
    report['diag_pairs'] = diag_pairs_risk(df, diag_a='diag_1', diag_b='diag_2', target_col=target_col, min_cases=5, top_n=20)
    report['diabetes'] = diabetes_breakdown(df, target_col=target_col)
    report['treatment_clusters'] = treatment_cluster_summary(df)
    report['los_buckets'] = los_buckets(df, los_col='time_in_hospital', target_col=target_col)
    report['recommendations'] = generate_recommendations()
    return report


# ---------------------------
# PDF (text-only) GENERATOR (unchanged)
# ---------------------------
def generate_insights_pdf(report_dict):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 50
    y = height - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Hospital Readmission — Executive Report")
    y -= 30

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Overview")
    y -= 18
    c.setFont("Helvetica", 10)
    for k,v in report_dict['overview'].items():
        line = f"{k.replace('_',' ').title()}: {v}"
        c.drawString(margin, y, line)
        y -= 14
    y -= 8

    # Age group (top 6)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Readmission by Age Group (Top)")
    y -= 16
    c.setFont("Helvetica", 9)
    ag = report_dict['age_group'].head(8)
    if not ag.empty:
        for _, row in ag.iterrows():
            lab = row.get('age_bucket') if 'age_bucket' in row.index else row.get('age')
            lab = str(lab)
            line = f"{shorten(lab, width=18)} — n:{int(row['count'])} readmit:{int(row.get('readmissions',0))} rate:{row.get('readmission_rate_pct',0)}%"
            c.drawString(margin, y, line)
            y -= 12
            if y < 120:
                c.showPage(); y = height - margin
    else:
        c.drawString(margin, y, "No age data")
        y -= 14

    y -= 6
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Top Diagnosis Categories by Readmission Rate")
    y -= 16
    c.setFont("Helvetica", 9)
    td = report_dict['top_diag']
    for _, r in td.head(10).iterrows():
        line = f"{r['diag_cat'] if 'diag_cat' in r.index else r.get('diag_cat','')}: n:{int(r['count'])} rate:{r['readmission_rate_pct']}%"
        c.drawString(margin, y, line)
        y -= 12
        if y < 120:
            c.showPage(); y = height - margin

    # Diagnosis pairs
    y -= 6
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "High-risk Diagnosis Pairs (Top)")
    y -= 16
    c.setFont("Helvetica", 9)
    dp = report_dict['diag_pairs']
    for _, r in dp.head(10).iterrows():
        line = f"{r['a_cat']} + {r['b_cat']}: n:{int(r['count'])} rate:{r['readmission_rate_pct']}%"
        c.drawString(margin, y, line)
        y -= 12
        if y < 120:
            c.showPage(); y = height - margin

    # Diabetes
    y -= 6
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Diabetes-specific Breakdown")
    y -= 16
    c.setFont("Helvetica", 9)
    dm = report_dict.get('diabetes', {})
    if 'diabetes_med' in dm and not dm['diabetes_med'].empty:
        for _, r in dm['diabetes_med'].iterrows():
            key = r.get(0) if 0 in r.index else r.name
            cases = int(r.get('cases', r.get('cases',0)))
            rate = r.get('readmit_rate_pct', 0)
            line = f"diabetes_med={key} — cases:{cases} rate:{rate}%"
            c.drawString(margin, y, line)
            y -= 12
            if y < 120:
                c.showPage(); y = height - margin
    else:
        c.drawString(margin, y, "No diabetes_med data")
        y -= 14

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Length of Stay Buckets")
    y -= 16
    c.setFont("Helvetica", 9)
    lb = report_dict.get('los_buckets', pd.DataFrame())
    if not lb.empty:
        for _, r in lb.iterrows():
            line = f"{r.get('los_bucket')} — n:{int(r['count'])} rate:{r['readmission_rate_pct']}%"
            c.drawString(margin, y, line)
            y -= 12
            if y < 120:
                c.showPage(); y = height - margin
    else:
        c.drawString(margin, y, "No LOS data")
        y -= 12

    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Recommendations")
    y -= 16
    c.setFont("Helvetica", 10)
    for rec in report_dict.get('recommendations', []):
        c.drawString(margin, y, f"- {rec}")
        y -= 12
        if y < 120:
            c.showPage(); y = height - margin

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# ---------------------------
# BASIC KPI FUNCTION
# ---------------------------
def kpis(df, target_col="readmitted"):
    df2 = df.copy()
    y = df2[target_col]
    if y.dtype == 'object':
        y = y.map(lambda v: 1 if str(v).strip().lower() in ['yes','y','1','true','readmitted','readmit'] else 0)
    total = len(df2)
    pos = int(y.sum())
    neg = int(total - pos)
    rate = round((pos / max(1, total)) * 100, 2)
    return {
        "total_samples": total,
        "readmission_rate": rate,
        "pos_count": pos,
        "neg_count": neg
    }
