import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model + scaler + columns
model = joblib.load("models/cyber_attack_rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/columns.pkl")
le = joblib.load("models/label_encoder.pkl")

# Page settings
st.set_page_config(
    page_title="Cyber Attack Detection",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.title {
    font-size:40px;
    font-weight:bold;
    text-align:center;
    color:#38bdf8;
}

.subtitle {
    font-size:18px;
    text-align:center;
    color:gray;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="title">🛡️ Cyber Attack Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Explainable AI based Intrusion Detection using Random Forest & CNN-LSTM</p>', unsafe_allow_html=True)

st.write("")

# Upload CSV
uploaded_file = st.file_uploader("Upload Network Traffic CSV", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    data.columns = data.columns.str.strip()

    # Remove Label column if exists
    if "Label" in data.columns:
        data = data.drop("Label", axis=1)

   

    # Replace infinity values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop missing values
    data.dropna(inplace=True)

    # Limit rows for fast demo
    data = data.sample(min(5000, len(data)))

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(data.head())

    if st.button("Detect Attack"):

        try:
            # Match training column order
            for col in columns:
                  if col not in data.columns:
                        data[col] = 0

            # Keep only required columns in correct order
            data = data[columns]

            # Scale the data
            data_scaled = scaler.transform(data)

            results = []

            # counters
            counts = {}
            normal_count = 0

            # UI placeholders
            metric_box = st.empty()
            chart_box = st.empty()
            live_box = st.empty()
            
            for i in range(len(data_scaled)):
                 row = data_scaled[i].reshape(1, -1)
                 pred = model.predict(row)
                 label = le.inverse_transform(pred)[0]
                 results.append(label)
                 

                 # COUNT UPDATE
                 if label == "BENIGN":
                      normal_count += 1
                 else:
                      counts[label] = counts.get(label, 0) + 1

                     
                 total_attacks = sum(counts.values())

                 # STEP 4 — LIVE METRICS
                 with metric_box.container():
                      col1, col2 = st.columns(2)
                      with col1:
                           st.metric("Normal Traffic", normal_count)
                      with col2:
                           st.metric("Detected Attacks", total_attacks)

                 # STEP 5 — LIVE CHART
                 chart_data = {"BENIGN": normal_count}
                 chart_data.update(counts)
                 
                 chart_box.bar_chart(pd.Series(chart_data))

                 # LIVE STATUS
                 live_box.markdown(f"### 🔍 Row {i+1}: **{label}**")
                 import time
                 time.sleep(0.01)
            data["Prediction"] = results
            st.success("Detection Completed")
            st.subheader("Prediction Results")
            st.dataframe(data)

            # Download results
            csv = data.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="Download Results",
                data=csv,
                file_name="attack_detection_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error("Prediction failed. Please check dataset format.")
            st.write(e)  