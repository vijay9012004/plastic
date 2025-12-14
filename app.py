# plastic_app.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle
import os

st.title("Plastic Waste Category Prediction")

# Check if the model already exists
if os.path.exists("plastic.pkl"):
    # Load existing model
    with open("plastic.pkl", "rb") as f:
        LR, le_Entity, salf = pickle.load(f)
    st.success("Loaded existing trained model (plastic.pkl)")
else:
    # Upload CSV
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file is not None:
        salf = pd.read_csv(uploaded_file)
        st.write("Dataset loaded successfully!")
        st.dataframe(salf.head())

        # Encode countries
        le_Entity = LabelEncoder()
        salf['Entity_encoded'] = le_Entity.fit_transform(salf['Entity'])

        # Convert continuous target into categories
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

        # Save model for future use
        with open("plastic.pkl", "wb") as f:
            pickle.dump((LR, le_Entity, salf), f)
        st.success("Model trained and saved as plastic.pkl")
    else:
        st.warning("Please upload a CSV file to continue.")
        st.stop()

# Select country and year for prediction
country = st.selectbox("Select Country", salf['Entity'].unique())
year = st.number_input(
    "Enter Year", 
    min_value=int(salf['Year'].min()), 
    max_value=int(salf['Year'].max()), 
    value=2023
)

# Encode country
country_encoded = le_Entity.transform([country])[0]

# Make prediction
pred_cat = LR.predict([[country_encoded, year]])[0]
category_name = {0:"Low", 1:"Medium", 2:"High"}[pred_cat]

st.success(f"Predicted Plastic Waste Category: {category_name}")
