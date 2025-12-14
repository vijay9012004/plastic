import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle
import os

st.title("Plastic Waste Category Prediction")

def load_model():
    """Try to load existing pickle file, return None if invalid."""
    if os.path.exists("plastic.pkl"):
        try:
            with open("plastic.pkl", "rb") as f:
                LR, le_Entity, salf = pickle.load(f)
            st.success("Loaded existing trained model (plastic.pkl)")
            return LR, le_Entity, salf
        except Exception as e:
            st.warning(f"Failed to load pickle: {e}. Re-training required.")
            os.remove("plastic.pkl")  # Remove invalid pickle
    return None, None, None

# Try to load model
LR, le_Entity, salf = load_model()

# If no valid model, upload CSV and train
if LR is None:
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        salf = pd.read_csv(uploaded_file)
        st.write("Dataset loaded successfully!")
        st.dataframe(salf.head())

        # Encode countries
        le_Entity = LabelEncoder()
        salf['Entity_encoded'] = le_Entity.fit_transform(salf['Entity'])

        # Convert target into categories
        salf['Plastic_category'] = pd.qcut(
            salf['Plastic waste generation (tonnes, total)'],
            q=3, labels=[0,1,2]
        )

        # Features and target
        X = salf[['Entity_encoded', 'Year']]
        y = salf['Plastic_category']

        # Train Logistic Regression
        LR = LogisticRegression(max_iter=1000)
        LR.fit(X, y)

        # Save model, encoder, and dataset together
        with open("plastic.pkl", "wb") as f:
            pickle.dump((LR, le_Entity, salf), f)
        st.success("Model trained and saved as plastic.pkl")
    else:
        st.warning("Please upload a CSV file to continue.")
        st.stop()

# Select country and year for prediction
country = st.selectbox("Select Country", salf['Entity'].unique())

# Fix for StreamlitValueAboveMaxError
min_year = int(salf['Year'].min())
max_year = int(salf['Year'].max())
default_year = max_year if max_year < 2023 else 2023

year = st.number_input(
    "Enter Year",
    min_value=min_year,
    max_value=max_year,
    value=default_year
)

# Encode country
country_encoded = le_Entity.transform([country])[0]

# Make prediction
pred_cat = LR.predict([[country_encoded, year]])[0]
category_name = {0:"Low", 1:"Medium", 2:"High"}[pred_cat]

st.success(f"Predicted Plastic Waste Category: {category_name}")
