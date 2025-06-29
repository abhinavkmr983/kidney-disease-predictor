import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load('kidney_disease_optimized_pipeline.joblib')

st.title("🩺 Kidney Disease Prediction App")
st.markdown("Enter patient details below and click **Predict** to check for CKD.")

# Input fields matching your model features order
age = st.number_input("Age", 1, 120, 45)
bp = st.number_input("Blood Pressure (mm/Hg)", 50, 200, 80)
sg = st.selectbox("Specific Gravity", [1.005, 1.01, 1.015, 1.02, 1.025])
al = st.slider("Albumin", 0, 5, 1)
su = st.slider("Sugar", 0, 5, 0)
rbc = st.selectbox("Red Blood Cells (0: normal, 1: abnormal)", [0, 1])
pc = st.selectbox("Pus Cell (0: normal, 1: abnormal)", [0, 1])
pcc = st.selectbox("Pus Cell Clumps (0: not present, 1: present)", [0, 1])
ba = st.selectbox("Bacteria (0: not present, 1: present)", [0, 1])
bgr = st.number_input("Blood Glucose Random (mg/dl)", 50, 500, 120)
bu = st.number_input("Blood Urea (mg/dl)", 5, 150, 40)
sc = st.number_input("Serum Creatinine (mg/dl)", 0.1, 15.0, 1.2)
sod = st.number_input("Sodium (mEq/L)", 100, 150, 135)
pot = st.number_input("Potassium (mEq/L)", 2.0, 8.0, 4.5)
hemo = st.number_input("Hemoglobin (gms)", 5.0, 17.0, 13.0)
pcv = st.number_input("Packed Cell Volume", 20, 55, 40)
wc = st.number_input("White Blood Cell Count", 4000, 18000, 7500)
rc = st.number_input("Red Blood Cell Count", 3.0, 6.5, 4.5)
htn = st.selectbox("Hypertension (0: no, 1: yes)", [0,1])
dm = st.selectbox("Diabetes Mellitus (0: no, 1: yes)", [0,1])
cad = st.selectbox("Coronary Artery Disease (0: no, 1: yes)", [0,1])
appet = st.selectbox("Appetite (0: good, 1: poor)", [0,1])
pe = st.selectbox("Pedal Edema (0: no, 1: yes)", [0,1])
ane = st.selectbox("Anemia (0: no, 1: yes)", [0,1])

features = np.array([[age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot,
                      hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane,0]])

if st.button("Predict"):
    proba = model.predict_proba(features)[0]
    pred = np.argmax(proba)

    st.write(f"🔍 Probability of No CKD: {proba[0]*100:.2f}%")
    st.write(f"⚠️ Probability of CKD: {proba[1]*100:.2f}%")

    if pred == 1:
        st.error("🚨 Likely Chronic Kidney Disease Detected")
    else:
        st.success("✅ Likely No Kidney Disease")


