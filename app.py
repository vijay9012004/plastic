import streamlit as st
import pickle
import os

st.set_page_config(page_title="Plastic Waste Prediction", layout="centered")
st.title("🗑️ Plastic Waste Prediction App")

# Base directory
BASE_DIR = os.path.dirname(__file__)

# Paths to model and encoder
model_path = os.path.join(BASE_DIR, "plastic.pkl")
encoder_path = os.path.join(BASE_DIR, "entity_encoder.pkl")

# Check if files exist
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
elif not os.path.exists(encoder_path):
    st.error(f"Encoder file not found: {encoder_path}")
else:
    # Load model and encoder
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        le_entity = pickle.load(f)

    # User inputs
    st.subheader("📊 Enter Details")
    entity = st.selectbox("Select Country/Entity", le_entity.classes_)
    year = st.number_input("Enter Year", min_value=1950, max_value=2100, value=2020)

    # Encode input
    entity_enc = le_entity.transform([entity])[0]

    # Predict button
    if st.button("Predict Plastic Waste"):
        prediction = model.predict([[entity_enc, year]])
        st.success(f"Predicted Plastic Waste: {prediction[0]:,.2f} tonnes")

st.markdown("---")
st.caption("Model: Logistic Regression | Dataset: Plastic Waste Generation")
