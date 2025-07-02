import streamlit as st
import numpy as np
import joblib


model = joblib.load('model.pkl')
st.title("Mobile Price Range Predictor")
st.sidebar.header("Enter mobile specs")

battery_power = st.sidebar.slider("Battery power (mAh)", 500, 2000, 1000)
ram = st.sidebar.slider("RAM (MB)", 256, 4000, 2000)
px_height = st.sidebar.slider("Pixel height", 0, 1960, 1000)
px_width = st.sidebar.slider("Pixel width", 0, 2000, 1000)
mobile_wt = st.sidebar.slider("Weight (grams)", 80, 250, 150)
clock_speed = st.sidebar.slider("Clock speed (GHz)", 0.5, 3.0, 1.5)
fc = st.sidebar.slider("Front camera (MP)", 0, 20, 5)
pc = st.sidebar.slider("Primary camera (MP)", 0, 20, 10)

input_data = np.array([[battery_power, blue := 1, clock_speed, dual_sim := 1,
                        fc, four_g := 1, mobile_wt, px_height, px_width,
                        ram, talk_time := 10, three_g := 1, touch_screen := 1,
                        wifi := 1, pc, n_cores := 4, int_memory := 32,
                        m_dep := 0.2, sc_h := 15, sc_w := 7, 
                        talk_time, 
                        1 if four_g else 0]]).astype(np.float64)

if st.button("Predict Price Range"):
    prediction = model.predict(input_data)[0]
    price_map = {
        0: "Low cost",
        1: "Medium cost",
        2: "High cost",
        3: "Very high cost 🔴"
    }
    st.success(f"Predicted price range: {price_map[prediction]}")
    st.info(f"Class (0–3): {prediction}")

st.subheader("Feature Importance ")
st.bar_chart(model.feature_importances_)