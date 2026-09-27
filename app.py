
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Job Acceptance Prediction System",
    page_icon="🎯",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("job_acceptance_cleaned.csv")

st.title("🎯 Job Acceptance Prediction System")
st.caption("HR Analytics & Machine Learning Dashboard")

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("🔎 Filters")

selected_tier = st.sidebar.multiselect(
    "Company Tier",
    options=sorted(df["company_tier"].dropna().unique()),
    default=sorted(df["company_tier"].dropna().unique())
)

selected_experience = st.sidebar.multiselect(
    "Experience Category",
    options=sorted(df["experience_category"].dropna().unique()),
    default=sorted(df["experience_category"].dropna().unique())
)

selected_skills = st.sidebar.multiselect(
    "Skills Match Level",
    options=sorted(df["skills_match_level"].dropna().unique()),
    default=sorted(df["skills_match_level"].dropna().unique())
)

filtered_df = df[
    (df["company_tier"].isin(selected_tier)) &
    (df["experience_category"].isin(selected_experience)) &
    (df["skills_match_level"].isin(selected_skills))
]

# -----------------------------
# KPI
# -----------------------------
total_candidates = len(filtered_df)
placed = (filtered_df["status"] == "placed").sum()

placement_rate = (
    placed / total_candidates * 100
    if total_candidates > 0 else 0
)

avg_skills = (
    filtered_df["skills_match_percentage"].mean()
    if total_candidates > 0 else 0
)

avg_technical = (
    filtered_df["technical_score"].mean()
    if total_candidates > 0 else 0
)

k1, k2, k3, k4 = st.columns(4)

k1.metric("👥 Total Candidates", f"{total_candidates:,}")
k2.metric("✅ Placement Rate", f"{placement_rate:.2f}%")
k3.metric("🎯 Avg Skills Match", f"{avg_skills:.2f}%")
k4.metric("📝 Avg Technical Score", f"{avg_technical:.2f}")

st.divider()

# -----------------------------
# PLACEMENT STATUS
# -----------------------------
st.subheader("📊 Placement Status")

status_data = filtered_df["status"].value_counts()
st.bar_chart(status_data)

# -----------------------------
# COMPANY TIER ANALYSIS
# -----------------------------
st.subheader("🏢 Company Tier vs Placement Rate")

tier_analysis = (
    filtered_df
    .groupby("company_tier")["target"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

st.bar_chart(tier_analysis)

# -----------------------------
# EXPERIENCE ANALYSIS
# -----------------------------
st.subheader("💼 Experience Category vs Placement Rate")

experience_analysis = (
    filtered_df
    .groupby("experience_category")["target"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

st.bar_chart(experience_analysis)

# -----------------------------
# SKILLS ANALYSIS
# -----------------------------
st.subheader("🎯 Skills Match Level vs Placement Rate")

skills_analysis = (
    filtered_df
    .groupby("skills_match_level")["target"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

st.bar_chart(skills_analysis)

# -----------------------------
# EDUCATION ANALYSIS
# -----------------------------
st.subheader("🎓 Degree Specialization Analysis")

degree_analysis = (
    filtered_df
    .groupby("degree_specialization")["target"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

st.bar_chart(degree_analysis)

# -----------------------------
# DATA PREVIEW
# -----------------------------
st.subheader("📋 Candidate Data Preview")

st.dataframe(
    filtered_df.head(100),
    use_container_width=True
)

st.caption(
    "Job Acceptance Prediction System | HR Analytics Dashboard"
)




# ==========================================
# FINAL JOB ACCEPTANCE PREDICTION
# ==========================================

import joblib

st.divider()
st.header("🎯 Job Acceptance Prediction")
st.write("Enter candidate details to predict job acceptance.")

model_pred = joblib.load("job_acceptance_model_compressed.pkl")
model_cols = joblib.load("model_columns.pkl")

# Use dataset values as a safe base row
base_row = df.drop(columns=["target"]).iloc[0].copy()

with st.form("prediction_form"):

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", 18, 60, 25)
        ssc = st.slider("SSC %", 0, 100, 70)
        hsc = st.slider("HSC %", 0, 100, 70)
        degree = st.slider("Degree %", 0, 100, 70)
        technical = st.slider("Technical Score", 0, 100, 70)

    with c2:
        aptitude = st.slider("Aptitude Score", 0, 100, 70)
        communication = st.slider("Communication Score", 0, 100, 70)
        skills = st.slider("Skills Match %", 0, 100, 70)
        experience_years = st.number_input(
            "Years of Experience", 0, 30, 1
        )
        certifications = st.number_input(
            "Certifications Count", 0, 20, 1
        )

    with c3:
        gender = st.selectbox(
            "Gender", ["male", "female"]
        )

        specialization = st.selectbox(
            "Degree Specialization",
            [
                "computer science",
                "electronics",
                "information technology",
                "mechanical",
                "others"
            ]
        )

        company = st.selectbox(
            "Company Tier",
            ["tier 1", "tier 2", "tier 3"]
        )

        experience_category = st.selectbox(
            "Experience Category",
            ["fresher", "junior", "senior"]
        )

        skills_level = st.selectbox(
            "Skills Match Level",
            ["high", "medium", "low"]
        )

    submit = st.form_submit_button(
        "🔍 Predict Job Acceptance",
        use_container_width=True
    )

if submit:

    # Update base candidate with entered values
    base_row["age_years"] = age
    base_row["gender"] = gender
    base_row["ssc_percentage"] = ssc
    base_row["hsc_percentage"] = hsc
    base_row["degree_percentage"] = degree
    base_row["degree_specialization"] = specialization
    base_row["technical_score"] = technical
    base_row["aptitude_score"] = aptitude
    base_row["communication_score"] = communication
    base_row["skills_match_percentage"] = skills
    base_row["years_of_experience"] = experience_years
    base_row["certifications_count"] = certifications
    base_row["company_tier"] = company
    base_row["experience_category"] = experience_category
    base_row["skills_match_level"] = skills_level

    candidate = pd.DataFrame([base_row])

    candidate_encoded = pd.get_dummies(
        candidate,
        drop_first=True
    )

    candidate_encoded = candidate_encoded.reindex(
        columns=model_cols,
        fill_value=0
    )

    prediction = model_pred.predict(candidate_encoded)[0]

    if hasattr(model_pred, "predict_proba"):
        probability = model_pred.predict_proba(
            candidate_encoded
        )[0][1]

        st.metric(
            "Acceptance Probability",
            f"{probability * 100:.2f}%"
        )

    if prediction == 1:
        st.success("✅ Prediction: Job Accepted / Placed")
    else:
        st.error("❌ Prediction: Job Not Accepted / Not Placed")

