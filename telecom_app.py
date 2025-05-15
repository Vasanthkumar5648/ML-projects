import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the saved model and scaler
model = joblib.load('https://github.com/Vasanthkumar5648/ML-projects/raw/main/churn_model.pkl')
scaler = joblib.load('https://github.com/Vasanthkumar5648/ML-projects/raw/main/scaler.pkl')

# Title and description
st.title("Telco Customer Churn Prediction")
st.write("""
This app predicts whether a customer will churn based on their demographics and service usage.
""")

# Sidebar for user input features
st.sidebar.header('Customer Details')

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
    PaymentMethod = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    MonthlyCharges = st.sidebar.slider('Monthly Charges ($)', 18.0, 120.0, 50.0)
    TotalCharges = st.sidebar.slider('Total Charges ($)', 18.0, 9000.0, 1000.0)
    
    # Create a dictionary of the inputs
    data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
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
    return features

input_df = user_input_features()

# Display user inputs
st.subheader('Customer Details')
st.write(input_df)

# Preprocess the input data
def preprocess_input(data):
    # Encode categorical variables (same as training)
    categorical_cols = data.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    
    # Scale numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    data[numerical_cols] = scaler.transform(data[numerical_cols])
    
    return data

# Preprocess and predict
if st.button('Predict Churn'):
    processed_input = preprocess_input(input_df.copy())
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)
    
    st.subheader('Prediction')
    churn_status = "Churn" if prediction[0] == 1 else "Not Churn"
    st.write(f"Prediction: **{churn_status}**")
    
    st.subheader('Prediction Probability')
    st.write(f"Probability of Churn: {prediction_proba[0][1]:.2f}")
    st.write(f"Probability of Not Churn: {prediction_proba[0][0]:.2f}")
    
    # Visualize the probability
    proba_df = pd.DataFrame({
        'Status': ['Not Churn', 'Churn'],
        'Probability': [prediction_proba[0][0], prediction_proba[0][1]]
    })
    st.bar_chart(proba_df.set_index('Status'))
