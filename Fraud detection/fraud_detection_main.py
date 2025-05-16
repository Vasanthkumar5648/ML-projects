import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, classification_report, auc
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from sklearn.decomposition import PCA
import joblib
import os
import streamlit as st

# Load dataset
df = pd.read_csv('https://raw.github.com/Vasanthkumar5648/ML-projects/main/Fraud%20detection/Fraud_Analysis_Dataset.csv')

# Column Descriptions:
# step: 1 step = 1 hour of time
# type: CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER
# amount: transaction amount in local currency
# nameOrig: originator of the transaction
# oldbalanceOrg: balance before the transaction
# newbalanceOrig: balance after the transaction
# nameDest: recipient of the transaction
# oldbalanceDest: recipient balance before the transaction
# newbalanceDest: recipient balance after the transaction
# isFraud: 1 if fraudulent, else 0

# Preprocessing
df_model = df.copy()

# Extract basic NLP features from nameOrig
df_model['nameOrig_len'] = df_model['nameOrig'].apply(len)
df_model['nameOrig_digit_count'] = df_model['nameOrig'].str.count(r'\d')
df_model['nameOrig_alpha_count'] = df_model['nameOrig'].str.count(r'[A-Za-z]')

# Drop high-cardinality columns
df_model.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)

# Encode 'type'
le = LabelEncoder()
df_model['type'] = le.fit_transform(df_model['type'])

# Define features and target
X = df_model.drop('isFraud', axis=1)
y = df_model['isFraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Deep Learning Model
model = Sequential([
    Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train model
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=20, batch_size=256, callbacks=[es], verbose=1)

# Predictions
y_pred_dl = (model.predict(X_test_scaled) > 0.5).astype(int).flatten()
y_proba_dl = model.predict(X_test_scaled).flatten()

# Evaluation function
def evaluate_model(y_test, y_pred, y_proba, model_name):
    print(f"\n📊 {model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}\n")

# Evaluate deep learning model
evaluate_model(y_test, y_pred_dl, y_proba_dl, "Deep Learning")

# ROC Curve Comparison
fpr_dl, tpr_dl, _ = roc_curve(y_test, y_proba_dl)
auc_dl = auc(fpr_dl, tpr_dl)

plt.figure(figsize=(10, 6))
plt.plot(fpr_dl, tpr_dl, label=f'Deep Learning (AUC = {auc_dl:.2f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('📉 ROC Curve - Deep Learning Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 💰 Financial Impact Analysis
avg_fraud_amount = df[df['isFraud'] == 1]['amount'].mean()
num_frauds = df['isFraud'].sum()

conf_matrix = confusion_matrix(y_test, y_pred_dl)
tn, fp, fn, tp = conf_matrix.ravel()

saved_loss = tp * avg_fraud_amount
missed_fraud = fn * avg_fraud_amount
false_positive_cost = fp * avg_fraud_amount * 0.1

total_profit = saved_loss - missed_fraud - false_positive_cost

print("\n💰 Financial Impact Analysis (Deep Learning):")
print(f"Average Fraud Amount: ${avg_fraud_amount:,.2f}")
print(f"Detected Frauds (TP): {tp} → Saved Loss: ${saved_loss:,.2f}")
print(f"Missed Frauds (FN): {fn} → Potential Loss: ${missed_fraud:,.2f}")
print(f"False Alarms (FP): {fp} → Estimated Cost: ${false_positive_cost:,.2f}")
print(f"Total Estimated Profit: ${total_profit:,.2f}")

# Streamlit app code
st.set_page_config(page_title="🚨 Fraud Detection App", layout="wide")
st.title("🚨 Real-Time Fraud Detection App")
st.markdown("Upload a transaction dataset to analyze and predict fraudulent activities.")
