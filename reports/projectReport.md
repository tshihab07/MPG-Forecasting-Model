# MPG Forecasting Project Report

**Author:** Md. Tushar Shihab  
**Date:** July 2025 
**Project Goal:** Build a production-ready model to predict vehicle fuel efficiency (MPG) using machine learning.

---

## Project Overview

This project aims to develop a **robust and accurate machine learning model** to predict **Miles Per Gallon (MPG)** based on vehicle characteristics such as engine size, weight, horsepower, origin, and age.


Dataset: `seaborn.load_dataset('mpg')`  
Target Variable: `mpg`  
Features: `cylinders`, `horsepower`, `weight`, `car_age`, `origin_japan`, `origin_usa`

---

## Exploratory Data Analysis (EDA)

### Key Insights from EDA

- **MPG decreases** as the number of cylinders increases.
- **Weight** and **horsepower** show strong negative correlation with MPG.
- **Model Year** shows a positive trend — newer cars are more fuel-efficient.
- Cars from **Japan** tend to have higher MPG than those from the USA or Europe.

---

## Feature Engineering

### Features Used

| Feature | Type | Description |
|--------|------|-------------|
| `cylinders` | Numerical | Number of engine cylinders |
| `horsepower` | Numerical | Engine power |
| `weight` | Numerical | Vehicle weight (lbs) |
| `car_age` | Derived | `1982 - model_year` |
| `origin_japan` | One-hot encoded | Binary flag for Japanese cars |
| `origin_usa` | One-hot encoded | Binary flag for American cars |

> Note: `origin_europe` was dropped to avoid multicollinearity.

---

## Model Training & Evaluation

### Baseline Model: Linear Regression

| Metric | Train | Test |
|--------|-------|------|
| R² Score | 0.83 | 0.78 |
| MSE | 10.99 | 11.24 |
| RMSE | 3.31 | 3.35 |
| MAE | 2.53 | 2.52 |

> Solid baseline with mild overfitting.

---

## Advanced Modeling

Five models were trained and optimized:

1. Linear Regression  
2. Ridge Regression  
3. Random Forest Regressor  
4. Gradient Boosting Regressor  
5. XGBoost Regressor  

Each model was tuned using:
- Grid Search
- Random Search
- Bayesian Optimization

Final selection based on **cross-validation and test performance**.

---

## Model Comparison

### Best Performing Model

| Model | Tuning Method | Test R² | Test RMSE |
|-------|----------------|---------|-----------|
| **Random Forest** | **Bayesian Search** | **0.88** | **2.68** |

> **Random Forest** tuned via **Bayesian Optimization** achieved the highest accuracy across all models and search methods.

### Model Comparison Chart

![Model Comparison Chart](images\modelsComparison.png)

> The chart shows Test R² scores across all models and tuning methods. **Random Forest with Bayesian Search** outperforms others, indicating its superior generalization and robustness.

---

## Prediction Error Visualization

### Prediction Error Plot

![Prediction Error Plot](images\predictionError-advanceModel.png)

> Shows actual vs predicted MPG values. Points close to the diagonal line indicate good predictions.

### Residuals Plot

![Residuals Plot](images\residualsPlot-advanceModel.png)

> Residuals are randomly scattered around zero, indicating no systematic bias in predictions.

---

## Model Interpretation (SHAP)

### SHAP Summary Plot

![SHAP Summary Plot](images\shapSummary-advanceModel.png)

### Key Insights

- **`weight`** and **`horsepower`** are the **top drivers** of lower MPG.
- **`car_age`** has a **positive impact** — newer cars are more efficient.
- **Japanese origin** contributes positively to fuel economy.
- **Heavier, high-horsepower vehicles** (common in the USA) reduce MPG significantly.

---

## Best Model Performance Summary

| Metric | Value | Test |
|--------|-------|------|
| Best Model | Random Forest (Bayesian Search) |
| Test R² | 0.88 |
| Test MSE | 5.90|
| Test RMSE | 2.43 |
| Test MAE | 1.82 |
| Cross-Validation R² | 0.85 |

## Conclusion
This project successfully developed a high-performance, production-ready MPG forecasting model using advanced machine learning techniques. Key achievements include:

- Comprehensive EDA and feature engineering
- Rigorous hyperparameter tuning with multiple search strategies
- Selection of Random Forest (Bayesian Search) as the optimal model
- Full model interpretation using SHAP
- Visual validation of predictions and residuals
- Final model saved for deployment

Random Forest’s ability to capture non-linear interactions between features like `weight`, `horsepower`, and `car_age` makes it particularly effective for this task.