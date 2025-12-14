import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import os

st.set_page_config(page_title="Plastic Waste Prediction", layout="centered")
st.title("🗑️ Plastic Waste Prediction App")

# Base directory
BASE_DIR = os.path.dirname(__file__)

# Load dataset to get Entity names
df = pd.read_csv(os.path.join(BASE_DIR, "plastic-waste-generation2.csv"))

# Recreate LabelEncoder from dataset
le_entity = LabelEncoder()
le_entity.fit(df['Entity'])

# Load trained model
with open(os.path.join(BASE_DIR, "plastic.pkl"), "rb") as f:
    model = pickle.load(f)

# User inputs
st.subheader("📊 Enter Details")
entity = st.selectbox("Select Country/Entity", le_entity.classes_)
year = st.number_input("Enter Year", min_value=1950, max_value=2100, value=2020)

# Encode input
entity_enc = le_entity.transform([entity])[0]

# Predict
if st.button("Predict Plastic Waste"):
    prediction = model.predict([[entity_enc, year]])
    st.success(f"Predicted Plastic Waste: {prediction[0]:,.2f} tonnes")

st.markdown("---")
st.caption("Model: Logistic Regression | Dataset: Plastic Waste Generation")
