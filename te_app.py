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

@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_model():
    # GitHub raw content URLs (make sure these are correct)
    model_url = "https://raw.githubusercontent.com/Vasanthkumar5648/ML-projects/main/churn_model.pkl"
    scaler_url = "https://raw.githubusercontent.com/Vasanthkumar5648/ML-projects/main/scaler.pkl"
    
    try:
        # Download model with timeout and verification
        model_response = requests.get(model_url, timeout=10)
        model_response.raise_for_status()  # Raise error for bad status
        
        # Download scaler
        scaler_response = requests.get(scaler_url, timeout=10)
        scaler_response.raise_for_status()
        
        # Load from memory
        model = joblib.load(BytesIO(model_response.content))
        scaler = joblib.load(BytesIO(scaler_response.content))
        
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.error("Please check:")
        st.error(f"1. Model URL exists: {model_url}")
        st.error(f"2. Scaler URL exists: {scaler_url}")
        st.error("3. Files are not corrupted")
        return None, None

model, scaler = load_model()

if model is None or scaler is None:
    st.stop()

# Input collection function
def user_input_features():
  # Collect user inputs
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    SeniorCitizen = st.sidebar.selectbox('Senior Citizen', ['No', 'Yes'])
    Partner = st.sidebar.selectbox('Partner', ['No', 'Yes'])
    Dependents = st.sidebar.selectbox('Dependents', ['No', 'Yes'])
    tenure = st.sidebar.slider('Tenure (months)', 1, 72, 12)
    PhoneService = st.sidebar.selectbox('Phone Service', ['No', 'Yes'])
    MultipleLines = st.sidebar.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
    InternetService = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    OnlineSecurity = st.sidebar.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    OnlineBackup = st.sidebar.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
    DeviceProtection = st.sidebar.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    TechSupport = st.sidebar.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    StreamingTV = st.sidebar.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    StreamingMovies = st.sidebar.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
    Contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    PaperlessBilling = st.sidebar.selectbox('Paperless Billing', ['No', 'Yes'])
    PaymentMethod = st.sidebar.selectbox('Payment Method', 
                                       ['Electronic check', 'Mailed check', 
                                        'Bank transfer (automatic)', 'Credit card (automatic)'])
    MonthlyCharges = st.sidebar.slider('Monthly Charges ($)', 18.0, 120.0, 50.0)
    TotalCharges = st.sidebar.slider('Total Charges ($)', 18.0, 9000.0, 1000.0)
    
    # Create a dictionary of the inputs
    data = {
        'gender': gender,
        'SeniorCitizen': 1 if SeniorCitizen == 'Yes' else 0,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    
    # Convert to DataFrame
    features = pd.DataFrame(data, index=[0])
  return pd.DataFrame(data, index=[0])
input_df = user_input_features()

# Preprocessing function
def preprocess_input(data):
  processed_input = preprocess_input(input_df.copy())
  prediction = model.predict(processed_input)
  prediction_proba = model.predict_proba(processed_input)
  return processed_data

# Main app logic
input_df = user_input_features()

if st.button('Predict Churn'):
    try:
        processed_input = preprocess_input(input_df.copy())
        prediction = model.predict(processed_input)
        prediction_proba = model.predict_proba(processed_input)
        
        st.success(f"Prediction: {'Churn' if prediction[0] == 1 else 'Not Churn'}")
        st.write(f"Churn Probability: {prediction_proba[0][1]:.2%}")
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
