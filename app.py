import streamlit as st
import pandas as pd

st.title("Plastic Waste Prediction")

# Upload CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    salf = pd.read_csv(uploaded_file)
    st.write("Dataset loaded successfully!")
    st.write(salf.head())
else:
    st.warning("Please upload the CSV file to continue.")
