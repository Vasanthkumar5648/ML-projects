# Fraud Detection Using Machine Learning

This project presents a comprehensive machine learning pipeline to detect fraudulent financial transactions. It includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, hyperparameter tuning, deep learning, and a real-time simulation setup.

## Table of Contents
- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Key Features](#key-features)
- [EDA Highlights](#eda-highlights)
- [Modeling and Performance](#modeling-and-performance)
- [Deep Learning Models](#deep-learning-models)
- [Business Impact Analysis](#business-impact-analysis)
- [How to Run](#how-to-run)
- [Results](#results)
- [Conclusion](#conclusion)

---

## Overview

The project addresses the problem of **financial fraud detection**, where a small proportion of transactions are fraudulent. We tackle challenges like:
- Class imbalance
- Real-time prediction
- High precision-recall tradeoffs

## Technologies Used

- Python
- pandas, numpy, seaborn, matplotlib
- scikit-learn
- imbalanced-learn (SMOTE)
- XGBoost, LightGBM
- Optuna (for hyperparameter optimization)
- TensorFlow / Keras (for LSTM and CNN models)

---

## Key Features

- **Data Preprocessing**: Label encoding, feature scaling, SMOTE for class balancing
- **Dimensionality Reduction**: PCA to retain 95% variance
- **Model Training**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Hyperparameter Tuning**: RandomizedSearchCV & Optuna
- **Deep Learning**: LSTM and CNN architectures
- **Real-Time Simulation**: Predict fraud on live-like samples
- **Threshold Tuning**: Custom thresholds to optimize F1-Score
- **Business Value Estimation**: Savings vs costs analysis

---

## EDA Highlights

- Visualized transaction types, fraud distribution, and time-based fraud patterns
- Detected significant imbalance (only ~10% fraud)
- Engineered new features like balance errors for better signal extraction

---

## Modeling and Performance

| Model              | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------------------|----------|-----------|--------|----------|---------|
| LogisticRegression| 98.9%    | 97.8%     | 91.2%  | 94.4%    | 0.97    |
| Random Forest      | 99.1%    | 95.1%     | 96.5%  | 95.8%    | 0.99    |
| XGBoost            | 99.0%    | 94.8%     | 96.2%  | 95.5%    | 0.99    |
| LightGBM           | 99.1%    | 94.3%     | 97.3%  | 95.8%    | 0.99    |

---

## Deep Learning Models

**LSTM:**  
Achieved 94.6% validation accuracy with strong sequence modeling on transaction patterns.

**CNN:**  
Achieved 92.5%+ validation accuracy using 1D convolutions over tabular inputs.

---

## Business Impact Analysis

We calculated:
- Potential savings from catching fraud (True Positives)
- Costs of false alerts (False Positives)
- Net profit per model

**Best ROI model:** LightGBM with highest net savings and least investigation overhead.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fraud-detection-ml.git
   cd fraud-detection-ml
   
