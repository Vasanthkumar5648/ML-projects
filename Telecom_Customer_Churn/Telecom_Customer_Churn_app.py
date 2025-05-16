import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
telecom_cust = pd.read_csv('https://raw.githubusercontent.com/Vasanthkumar5648/ML-projects/main/Telco_Customer_Churn%20(1).csv')

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
st.markdown(
    """
    <style>
        html, body, .stApp {
            font-family: 'Arial', sans-serif; /* Change font here */
            background-color: #FFFFFF;
        }

        .title-box {
            font-family: 'Helvetica', sans-serif;
            font-size: 28px;
            font-weight: bold;
            text-align: center;
            color: black;
        }

        .prediction-box {
            background-color:#D3D3D3;
            font-family: 'Courier New', monospace;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            color: black;
        }

        .custom-text {
            font-family: 'Verdana', sans-serif;
            font-size: 16px;
            color: black;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Create a Streamlit app
st.title("📊 Customer Churn Prediction App")
# Input fields for feature values on the main screen
st.sidebar.header("Enter Customer Information")
tenure = st.sidebar.selectbox("Tenure (in months)", list(range(0, 110, 10)))
internet_service = st.sidebar.selectbox("Internet Service", ('DSL', 'Fiber optic', 'No'))
contract = st.sidebar.selectbox("Contract", ('Month-to-month', 'One year', 'Two year'))
monthly_charges = st.sidebar.selectbox("Monthly Charges", list(range(0, 210, 10)))
total_charges = st.sidebar.selectbox("Total Charges", list(range(0, 10100, 10)))

# Predict Button
if st.sidebar.button("Predict Churn"):
    st.subheader("🔍 Selected Input Values")
    st.write(f"**Tenure:** {tenure} months")
    st.write(f"**Internet Service:** {internet_service}")
    st.write(f"**Contract:** {contract}")
    st.write(f"**Monthly Charges:** ${monthly_charges}")
    st.write(f"**Total Charges:** ${total_charges}")


# Map input values to numeric using the label mapping
label_mapping = {
    'DSL': 0,
    'Fiber optic': 1,
    'No': 2,
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2,
}
internet_service = label_mapping[internet_service]
contract = label_mapping[contract]

# Make a prediction using the model
prediction = model.predict([[tenure, internet_service, contract, monthly_charges, total_charges]])

# Display the prediction result on the main screen
st.header("Prediction Result")
if prediction[0] == 0:
    st.markdown('<div class="prediction-box" style="color: Blue;">✅ This customer is likely to stay.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="prediction-box" style="color: Red;">⚠️ This customer is likely to churn.</div>', unsafe_allow_html=True)
