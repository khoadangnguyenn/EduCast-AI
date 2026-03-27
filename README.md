# EduCast AI: Predictive Academic Modeling

EduCast AI is a machine learning pipeline for predicting student pass probability based on behavioral logs and lab submission patterns.

The system leverages ensemble learning with gradient boosting models to optimize **AUPRC (Area Under Precision-Recall Curve)** — a key metric for imbalanced classification.

---

## Key Features

- End-to-end ML pipeline: data processing → feature engineering → model training → evaluation  
- Ensemble of **LightGBM, XGBoost, CatBoost**  
- Advanced **feature engineering (50+ features)**  
- Robust validation with **Stratified 7-Fold Cross Validation**  
- Optimized for **imbalanced classification (AUPRC-focused)**  

---

## Pipeline Overview

### 1. Data Processing
- Extract features from raw student activity logs  
- Handle missing values and normalize temporal signals  

### 2. Feature Engineering
- Behavioral features: submission frequency, time gaps  
- Temporal patterns: weekday/weekend, time-of-day activity  
- Performance dynamics:
  - grade trends (slope)
  - rolling statistics (mean/std)
  - momentum & consistency scores  

### 3. Model Training
- Train 3 gradient boosting models:
  - LightGBM  
  - XGBoost  
  - CatBoost  
- Use **Stratified 7-Fold CV** to ensure stable evaluation  
- Generate **Out-of-Fold (OOF) predictions**  

### 4. Ensemble Learning
- Combine models using:
  - Stacking (Logistic Regression meta-learner)  
  - Weighted averaging  
- Optimize final predictions based on **AUPRC**

---

## Future Improvements

- Add deep learning models (LSTM for sequence modeling)  
- Feature selection & dimensionality reduction  
- Real-time prediction system  
