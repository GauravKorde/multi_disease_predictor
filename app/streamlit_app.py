import streamlit as st
import pandas as pd
import joblib
import os

# Automatically detect BASE PATH no matter where streamlit is executed
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = "models"


# Load models safely with absolute paths
diabetes_model = joblib.load(f"{MODEL_DIR}/diabetes_model.pkl")
heart_model = joblib.load(f"{MODEL_DIR}/heart_model.pkl")
liver_model = joblib.load(f"{MODEL_DIR}/liver_model.pkl")




st.set_page_config(page_title="Multi Disease Predictor", layout="wide")
st.title("🧠 Multi-Disease Prediction System")
st.write("Predict **Diabetes**, **Heart Disease**, and **Liver Disease** using trained ML models.")

tabs = st.tabs(["Diabetes", "Heart Disease", "Liver Disease"])


# ----------------------------- DIABETES -----------------------------------
with tabs[0]:
    st.header("🩸 Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies", 0, 20, 2, key="d_preg")
    glucose = st.number_input("Glucose", 0, 300, 120, key="d_glu")
    bp = st.number_input("Blood Pressure", 0, 150, 70, key="d_bp")
    skin = st.number_input("Skin Thickness", 0, 100, 20, key="d_skin")
    insulin = st.number_input("Insulin", 0, 900, 80, key="d_ins")
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0, key="d_bmi")
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5, key="d_dpf")
    age = st.number_input("Age", 1, 120, 45, key="d_age")

    if st.button("Predict Diabetes"):
        data = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bp,
            "SkinThickness": skin,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }

        df = pd.DataFrame([data])
        pred = diabetes_model.predict(df)[0]
        prob = diabetes_model.predict_proba(df)[0][1]

        st.success(f"Prediction: {'Diabetic' if pred==1 else 'Not Diabetic'}")
        st.info(f"Probability: {prob:.2f}")


# ------------------------------- HEART -------------------------------------
with tabs[1]:
    st.header("❤️ Heart Disease Prediction")

    age = st.number_input("Age", 1, 120, 50, key="h_age")
    sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1], key="h_sex")
    cp = st.number_input("Chest Pain Type (0–3)", 0, 3, 1, key="h_cp")
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120, key="h_trestbps")
    chol = st.number_input("Cholesterol", 100, 600, 250, key="h_chol")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1], key="h_fbs")
    restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2], key="h_restecg")
    thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150, key="h_thalach")
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1], key="h_exang")
    oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, key="h_oldpeak")
    slope = st.selectbox("Slope (0–2)", [0, 1, 2], key="h_slope")
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3], key="h_ca")
    thal = st.selectbox("Thal (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)", [0, 1, 2], key="h_thal")

    if st.button("Predict Heart Disease"):
        data = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }

        df = pd.DataFrame([data])
        pred = heart_model.predict(df)[0]
        prob = heart_model.predict_proba(df)[0][1]

        st.success(f"Prediction: {'Heart Disease' if pred==1 else 'No Heart Disease'}")
        st.info(f"Probability: {prob:.2f}")


# ------------------------------- LIVER -------------------------------------
with tabs[2]:
    st.header("🫁 Liver Disease Prediction")

    gender = st.selectbox("Gender", ["Male", "Female"], key="l_gender")
    age = st.number_input("Age", 1, 120, 45, key="l_age")
    tb = st.number_input("Total Bilirubin", 0.0, 10.0, 1.0, key="l_tb")
    db = st.number_input("Direct Bilirubin", 0.0, 5.0, 0.5, key="l_db")
    alk = st.number_input("Alkaline Phosphotase", 50, 500, 200, key="l_alk")
    alt = st.number_input("Alamine Aminotransferase", 0, 500, 30, key="l_alt")
    ast = st.number_input("Aspartate Aminotransferase", 0, 500, 30, key="l_ast")
    tp = st.number_input("Total Proteins", 0.0, 10.0, 6.0, key="l_tp")
    alb = st.number_input("Albumin", 0.0, 6.0, 3.0, key="l_alb")
    agr = st.number_input("Albumin and Globulin Ratio", 0.0, 3.0, 1.0, key="l_agr")

    if st.button("Predict Liver Disease"):
        data = {
            "Gender": gender,
            "Age": age,
            "Total_Bilirubin": tb,
            "Direct_Bilirubin": db,
            "Alkaline_Phosphotase": alk,
            "Alamine_Aminotransferase": alt,
            "Aspartate_Aminotransferase": ast,
            "Total_Protiens": tp,
            "Albumin": alb,
            "Albumin_and_Globulin_Ratio": agr
        }

        df = pd.DataFrame([data])
        pred = liver_model.predict(df)[0]
        prob = liver_model.predict_proba(df)[0][1]

        st.success(f"Prediction: {'Liver Disease' if pred==1 else 'No Liver Disease'}")
        st.info(f"Probability: {prob:.2f}")
