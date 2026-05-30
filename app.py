import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Electricity Forecasting System",
    page_icon="⚡",
    layout="centered"
)

# -------------------------------
# LOAD MODEL + SCALER
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("electricity_rnn_model.keras")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("PJME_hourly.csv")
df = df[['PJME_MW']]

data = scaler.transform(df)

# last 60 hours
last_60 = data[-60:].reshape(1, 60, 1)

# -------------------------------
# TITLE
# -------------------------------
st.title("⚡ Electricity Forecasting System (SimpleRNN)")

st.markdown("""
This application uses a **SimpleRNN deep learning model** to predict
electricity consumption for the next 24 hours based on historical usage patterns.

It helps in understanding:
- Energy demand trends
- Peak load hours
- Seasonal consumption behavior
""")

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict Next 24 Hours"):

    input_seq = last_60.copy()
    future = []

    for i in range(24):
        pred = model.predict(input_seq, verbose=0)[0][0]
        future.append(pred)

        input_seq = np.append(
            input_seq[:, 1:, :],
            [[[pred]]],
            axis=1
        )

    # -------------------------------
    # INVERSE TRANSFORM (ONLY ONCE)
    # -------------------------------
    future = scaler.inverse_transform(np.array(future).reshape(-1, 1))

    # -------------------------------
    # TABLE OUTPUT
    # -------------------------------
    st.subheader("📊 Predicted Values (Next 24 Hours)")

    df_result = pd.DataFrame({
        "Hour": list(range(1, 25)),
        "Predicted Consumption (kWh)": future.flatten()
    })

    st.dataframe(df_result)

    st.success(f"Hour 1 Prediction: {future[0][0]:.2f} kWh")

    # -------------------------------
    # CHART OUTPUT
    # -------------------------------
    st.subheader("📈 Forecast Graph")

    st.line_chart(df_result.set_index("Hour"))

    st.success("Prediction completed successfully ⚡")

# -------------------------------
# FOOTER
# -------------------------------
st.caption("Deep Learning - Electricity Forecasting")