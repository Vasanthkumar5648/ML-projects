import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
telecom_cust = pd.read_csv('https://raw.github.com/Vasanthkumar5648/ML-projects/main/Telecom_Customer_Churn/Telco_Customer_Churn.csv')

# Data preprocessing
# Fill missing values in 'TotalCharges' and convert to numeric
telecom_cust['TotalCharges'] = pd.to_numeric(telecom_cust['TotalCharges'], errors='coerce')
telecom_cust['TotalCharges'].fillna(0, inplace=True)

# Convert 'Churn' to binary labels
label_encoder = LabelEncoder()
telecom_cust['Churn'] = label_encoder.fit_transform(telecom_cust['Churn'])

# Use Label Encoding for 'InternetService' and 'Contract'
telecom_cust['InternetService'] = label_encoder.fit_transform(telecom_cust['InternetService'])
telecom_cust['Contract'] = label_encoder.fit_transform(telecom_cust['Contract'])

# Select features
selected_features = ['tenure', 'InternetService', 'Contract', 'MonthlyCharges', 'TotalCharges']
X = telecom_cust[selected_features]
y = telecom_cust['Churn']

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=101)
model.fit(X, y)

#creating a background

import streamlit as st

# Custom CSS styling
st.markdown(
    """
    <style>
        html, body, .stApp {
            font-family: 'Arial', sans-serif;
            background-color: #FFFFFF;
        }

        .title-box {
            font-family: 'Helvetica', sans-serif;
            font-size: 28px;
            font-weight: bold;
            text-align: center;
            color: black;
            margin-bottom: 30px;
        }

        .section-header {
            font-family: 'Helvetica', sans-serif;
            font-size: 28px;
            font-weight: bold;
            color: black;
            margin-bottom: 20px;
        }

        .prediction-box {
            background-color: #D3D3D3;
            font-family: 'Courier New', monospace;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            color: black;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
        }

        .custom-text {
            font-family: 'Verdana', sans-serif;
            font-size: 16px;
            color: black;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .input-value {
            font-family: 'Courier New', monospace;
            font-size: 16px;
            color: black;
            font-weight: bold;
            margin-left: 15px;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Main title with custom styling
st.markdown('<div class="title-box">📊 Customer Churn Prediction App</div>', unsafe_allow_html=True)

# Input fields in sidebar
st.sidebar.header("Enter Customer Information")
tenure = st.sidebar.selectbox("Tenure (in months)", list(range(0, 110, 10)))
internet_service = st.sidebar.selectbox("Internet Service", ('DSL', 'Fiber optic', 'No'))
contract = st.sidebar.selectbox("Contract", ('Month-to-month', 'One year', 'Two year'))
monthly_charges = st.sidebar.selectbox("Monthly Charges", list(range(0, 210, 10)))
total_charges = st.sidebar.selectbox("Total Charges", list(range(0, 10100, 10)))

if st.sidebar.button("Predict Churn"):
    # Selected Input Values section
    st.markdown('<div class="section-header">🔍 Selected Input Values</div>', unsafe_allow_html=True)
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="custom-text">Customer Details:</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="input-value">• Tenure: {tenure} months</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="input-value">• Internet Service: {internet_service}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="input-value">• Contract: {contract}</p>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<p class="custom-text">Financial Details:</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="input-value">• Monthly Charges: ${monthly_charges}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="input-value">• Total Charges: ${total_charges}</p>', unsafe_allow_html=True)
    
    # Prediction Result section
    st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)
    
    # Mock prediction (replace with your actual model prediction)
    prediction = [0]  # 0 = stay, 1 = churn
    
    if prediction[0] == 0:
        st.markdown(
            '<div class="prediction-box">'
            '✅ This customer is likely to stay.'
            '</div>', 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="prediction-box">'
            '⚠️ This customer is likely to churn.'
            '</div>', 
            unsafe_allow_html=True
        )
