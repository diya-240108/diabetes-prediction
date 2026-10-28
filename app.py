from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).parent / "diabetes.csv"
FAKE_ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

st.set_page_config(page_title="Diabetes Risk Screening", page_icon="🩺", layout="wide")


@st.cache_resource
def train_model():
    df = pd.read_csv(DATA)
    # Zeros in these columns are impossible values, i.e. missing data -> median
    df[FAKE_ZERO_COLS] = df[FAKE_ZERO_COLS].replace(0, np.nan)
    df[FAKE_ZERO_COLS] = df[FAKE_ZERO_COLS].fillna(df[FAKE_ZERO_COLS].median())

    X = df.drop(columns="Outcome")
    y = df["Outcome"]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(pipe, X, y, cv=cv).mean()
    pipe.fit(X, y)
    return pipe, cv_acc, list(X.columns)


def risk_level(p):
    if p < 0.35:
        return "Low", "success"
    if p < 0.60:
        return "Moderate", "warning"
    return "High", "error"


st.title("🩺 Diabetes Risk Screening")
st.caption("A machine learning screening aid built on the Pima Indians Diabetes dataset. "
           "It is not a medical diagnosis.")

if not DATA.exists():
    st.error("diabetes.csv not found. Put diabetes.csv in the same folder as app.py and rerun.")
    st.stop()

model, cv_acc, columns = train_model()

with st.sidebar:
    st.header("About the model")
    st.write("Logistic Regression with feature scaling.")
    st.metric("5-fold cross-validation accuracy", f"{cv_acc * 100:.1f}%")
    st.write("Zero values in glucose, blood pressure, skin thickness, insulin and BMI "
             "were treated as missing and replaced with the median.")

st.subheader("Enter patient details")
c1, c2, c3, c4 = st.columns(4)
with c1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 2)
    glucose = st.number_input("Glucose (mg/dL)", 40, 300, 120)
with c2:
    bp = st.number_input("Blood pressure (mm Hg)", 30, 140, 70)
    skin = st.number_input("Skin thickness (mm)", 5, 100, 25)
with c3:
    insulin = st.number_input("Insulin (mu U/ml)", 10, 900, 80)
    bmi = st.number_input("BMI", 10.0, 70.0, 28.0, step=0.1)
with c4:
    dpf = st.number_input("Diabetes pedigree function", 0.05, 2.5, 0.45, step=0.01)
    age = st.number_input("Age", 15, 100, 33)

if st.button("Predict", type="primary"):
    row = pd.DataFrame(
        [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]], columns=columns
    )
    proba = float(model.predict_proba(row)[0][1])
    level, kind = risk_level(proba)

    left, right = st.columns([1, 2])
    with left:
        st.metric("Predicted probability of diabetes", f"{proba * 100:.1f}%")
        getattr(st, kind)(f"Risk level: {level}")
        st.caption("Risk bands (low < 35%, moderate 35-60%, high > 60%) are illustrative.")
    with right:
        scaled = model.named_steps["scaler"].transform(row)[0]
        coefs = model.named_steps["clf"].coef_[0]
        contrib = pd.Series(coefs * scaled, index=columns).sort_values()
        st.write("What pushed this prediction (positive = raises risk, negative = lowers it)")
        st.bar_chart(contrib)

st.divider()
st.caption("This tool is for learning and screening support only. "
           "Please consult a doctor for any medical decision.")
