import streamlit as st
import pandas as pd
import pickle

def pipe_feature_engineer(input_df):
    df = input_df.copy()

    def get_age_group(age):
        if age <= 18:
            return "Young"
        elif age <= 45:
            return "Adult"
        elif age <= 60:
            return "Middle-aged"
        else:
            return "Senior"

    df["age_grp"] = df["age"].apply(get_age_group)

    def get_bmi_grp(num):
        if pd.isna(num): return "Normal"
        if num < 18.5:
            return "Underweight"
        elif num < 25:
            return "Normal"
        else:
            return "Overweight"

    df["bmi_grp"] = df["bmi"].apply(get_bmi_grp)

    def get_glucose_level(num):
        if num < 140:
            return "Normal"
        elif num < 200:
            return "Pre-diabetic"
        else:
            return "Diabetic"

    df["glucose_level"] = df["avg_glucose_level"].apply(get_glucose_level)

    df["age-glucose"] = df["age"] * df["avg_glucose_level"]

    def calculate_risk(row):
        risk_score = 0
        if row["age_grp"] == "Senior":
            risk_score += 3
        elif row["age_grp"] == "Middle-aged":
            risk_score += 2
        elif row["age_grp"] == "Adult":
            risk_score += 1

        if row["bmi_grp"] in ["Overweight", "Underweight"]: risk_score += 1
        if row["glucose_level"] == "Diabetic":
            risk_score += 3
        elif row["glucose_level"] == "Pre-diabetic":
            risk_score += 2

        if row["hypertension"] == 1: risk_score += 3
        if row["heart_disease"] == 1: risk_score += 3

        if row["smoking_status"] == "smokes":
            risk_score += 2
        elif row["smoking_status"] == "formerly smoked":
            risk_score += 1

        if risk_score >= 8:
            return "High"
        elif risk_score >= 4:
            return "Medium"
        else:
            return "Low"

    df["risk_factor"] = df.apply(calculate_risk, axis=1)
    return df


#Page Configuration
st.set_page_config(page_title="Stroke Risk Analyzer", page_icon="🫀", layout="wide")

#Sidebar
with st.sidebar:
    st.title("STROKE PREDICTOR")
    st.write("### Model Performance")
    st.write("- **Model:** Logistic Regression")
    st.write("- **Recall:** 79% (Catching Strokes)")
    st.write("- **Strategy:** SMOTE + Feature Engineering")


#Pipeline
@st.cache_resource
def load_model():
    with open('model/stroke_model_v1.pkl', 'rb') as f:
        return pickle.load(f)

try:
    pipeline = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure the pickle file is in the 'model/' folder.")

#Header
st.title("🫀 Stroke Risk Prediction System")
st.markdown("Enter patient parameters to assess statistical risk based on medical data patterns.")
st.divider()

#Input
with st.form("input_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", 0, 120, 45)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with c2:
        glucose = st.number_input("Avg Glucose Level", 50.0, 300.0, 100.0)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with c3:
        smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
        married = st.selectbox("Ever Married", ["Yes", "No"])
        work = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence = st.selectbox("Residence Type", ["Urban", "Rural"])

    submitted = st.form_submit_button("Generate Risk Assessment")

#Prediction
if submitted:
    raw_input = pd.DataFrame([{
        "gender": gender, "age": age, "hypertension": hypertension,
        "heart_disease": heart_disease, "ever_married": married,
        "work_type": work, "Residence_type": residence,
        "avg_glucose_level": glucose, "bmi": bmi, "smoking_status": smoking
    }])

    prob = pipeline.predict_proba(raw_input)[0][1]

    processed_input = pipe_feature_engineer(raw_input)
    calculated_risk = processed_input['risk_factor'].values[0]

    st.subheader("Statistical Results")
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric("Stroke Probability", f"{prob:.1%}")
        if prob > 0.4:
            st.error("Status: High Risk Detected")
        elif prob > 0.2:
            st.warning("Status: Elevated Risk")
        else:
            st.success("Status: Low Risk")

    with res_col2:
        st.write(f"**Custom Risk Category:** {calculated_risk}")
        st.progress(prob)
        st.write("---")
        st.markdown(f"""
        **Analysis Summary:**
        - Patient is categorized as **{calculated_risk} Risk** based on health history.
        - Age group categorized as: **{processed_input['age_grp'].values[0]}**.
        - Glucose level status: **{processed_input['glucose_level'].values[0]}**.
        """)
