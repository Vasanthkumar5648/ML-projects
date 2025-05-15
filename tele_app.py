import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Title and description
st.title("Telco Customer Churn Prediction")
st.write("""
This app predicts whether a customer will churn based on their demographics and service usage.
""")

@st.cache_resource
def load_model():
    # GitHub raw content URLs
    model_url = "https://github.com/Vasanthkumar5648/ML-projects/raw/main/churn_model.pkl"
    scaler_url = "https://github.com/Vasanthkumar5648/ML-projects/raw/main/scaler.pkl"
    
    try:
        # Download model
        model_response = requests.get(model_url)
        model = joblib.load(BytesIO(model_response.content))
        
        # Download scaler
        scaler_response = requests.get(scaler_url)
        scaler = joblib.load(BytesIO(scaler_response.content))
        
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None

model, scaler = load_model()

if model is None or scaler is None:
    st.stop()

# Rest of your Streamlit app code...
def user_input_features():
    # Collect user inputs (same as before)
    # ... [rest of your input collection code]

input_df = user_input_features()

# Display and prediction code
if st.button('Predict Churn'):
    try:
        # Preprocess input
        processed_input = preprocess_input(input_df.copy())
        prediction = model.predict(processed_input)
        prediction_proba = model.predict_proba(processed_input)
        
        # Display results
        st.subheader('Prediction')
        churn_status = "Churn" if prediction[0] == 1 else "Not Churn"
        st.write(f"Prediction: **{churn_status}**")
        
        st.subheader('Prediction Probability')
        st.write(f"Probability of Churn: {prediction_proba[0][1]:.2f}")
        st.write(f"Probability of Not Churn: {prediction_proba[0][0]:.2f}")
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
