import streamlit as st
import pickle
import os
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Plastic Waste Prediction", layout="centered")
st.title("🗑️ Plastic Waste Prediction App")

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "plastic.pkl")

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
else:
    # Load model + classes
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    
    model = data['model']
    le_entity = LabelEncoder()
    le_entity.classes_ = data['classes']

    # User input
    entity = st.selectbox("Select Country/Entity", le_entity.classes_)
    year = st.number_input("Enter Year", min_value=1950, max_value=2100, value=2020)
    entity_enc = le_entity.transform([entity])[0]

    if st.button("Predict Plastic Waste"):
        prediction = model.predict([[entity_enc, year]])
        st.success(f"Predicted Plastic Waste: {prediction[0]:,.2f} tonnes")

st.markdown("---")
st.caption("Model: Logistic Regression | Dataset: Plastic Waste Generation")
