import streamlit as st
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import (accuracy_score, confusion_matrix, 
                            classification_report, roc_auc_score, roc_curve)
#import pickle

# Set page config
st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")

# Title
st.title("Breast Cancer Classification App")
st.write("""
This app uses a Logistic Regression model to classify breast cancer tumors as malignant or benign.
""")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
This application was developed to demonstrate a machine learning model for breast cancer classification.
The model was trained on the Wisconsin Breast Cancer Dataset.
""")
