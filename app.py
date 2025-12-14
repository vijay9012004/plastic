# plastic_logistic.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
salf = pd.read_csv("plastic-waste-generation2.csv")

# Encode country names
le_Entity = LabelEncoder()
salf['Entity_encoded'] = le_Entity.fit_transform(salf['Entity'])

# Convert continuous plastic waste into 3 categories
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

# Streamlit UI
st.title("Plastic Waste Category Prediction")

country = st.selectbox("Select Country", salf['Entity'].unique())
year = st.number_input("Enter Year", min_value=int(salf['Year'].min()), max_value=int(salf['Year'].max()), value=2023)

# Encode country
country_encoded = le_Entity.transform([country])[0]

# Predict category
pred_cat = LR.predict([[country_encoded, year]])[0]
category_name = {0:"Low", 1:"Medium", 2:"High"}[pred_cat]

st.success(f"Predicted Plastic Waste Category: {category_name}")

# Save model
with open("plastic_logistic_model.pkl", "wb") as f:
    pickle.dump(LR, f)
