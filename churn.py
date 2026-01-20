import streamlit as st
import joblib
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

body {
    background: linear-gradient(to right, #f8f9fa, #eef2f7);
}

.main {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
}

h1 {
    text-align: center;
    color: #FF4B4B;
    font-weight: 700;
}

.stButton > button {
    background: linear-gradient(to right, #FF4B4B, #FF0000);
    color: white;
    font-size: 18px;
    font-weight: bold;
    padding: 12px;
    border-radius: 12px;
    border: none;
    width: 100%;
    transition: 0.3s ease-in-out;
}

.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(to right, #FF0000, #CC0000);
}

.result-box {
    padding: 18px;
    border-radius: 14px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}

.churn {
    background-color: #ffe5e5;
    color: #cc0000;
}

.no-churn {
    background-color: #e6fff2;
    color: #008000;
}

.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
scaler = joblib.load("scaler.pkl")
model = joblib.load("best_model.pkl")

# ---------------- UI ----------------
st.title("📉 Customer Churn Prediction")

st.image("churn.jpeg", use_container_width=True)

st.markdown("### 🧾 Enter Customer Details")

with st.container():
    age = st.number_input("👤 Age", min_value=10, max_value=100, value=30)
    tenure = st.number_input("📆 Tenure (Months)", min_value=0, max_value=130, value=10)
    monthly_charges = st.number_input("💳 Monthly Charges", min_value=0, max_value=200, value=70)
    gender = st.selectbox("⚥ Gender", ["Male", "Female"])

st.markdown("---")

predict_button = st.button("🔍 Predict Churn")

# ---------------- PREDICTION ----------------
if predict_button:
    gender_val = 1 if gender == "Female" else 0
    input_data = np.array([[age, tenure, monthly_charges, gender_val]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.markdown(
            "<div class='result-box churn'>⚠️ Customer is Likely to CHURN</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='result-box no-churn'>✅ Customer is NOT Likely to Churn</div>",
            unsafe_allow_html=True
        )

    st.balloons()

else:
    st.info("👆 Enter details and click **Predict Churn**")

# ---------------- FOOTER ----------------
st.markdown(
    "<div class='footer'>Built with ❤️ using Machine Learning & Streamlit</div>",
    unsafe_allow_html=True
)

