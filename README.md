# MPG Forecasting System

## A Production-Ready Machine Learning Pipeline for Fuel Efficiency Prediction

This repository contains a complete machine learning pipeline to predict **Miles Per Gallon (MPG)** for passenger vehicles using historical automotive data. The system includes **end-to-end modeling**, **hyperparameter optimization**, **model interpretation**, and **deployment-ready artifacts**.

**Primary Goal:** Build a robust, interpretable, and high-performance regression model suitable for real-world deployment in automotive analytics or fleet management systems.

**Status:** Trained, evaluated, and validated  
**Output:** Pickled model file ready for integration  

---


## 📌 Table of Contents

- [Project Overview](#project-overview)
    - [Key Engineering Decision](#key-engineering-decisions)
- [Dataset & Features](#dataset-features)
- [Modeling Pipeline](#modeling-pipeline)
- [Best Performing Model](#best-performing-model)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
    - [Clone the Repository](#clone-the-repository)
    - [Create a virtual environment](#create-a-virtual-environment)
    - [Install Dependencies](#install-dependencies)
- [Model Inference](#model-inference)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [Contact](#contact)
- [License](#license)

---


## Project Overview

The project leverages the built-in `mpg` dataset from Seaborn, which contains observations of 398 vehicles from 1970–1982. The target variable is continuous (`mpg`), making this a **regression problem**.

### Key Engineering Decisions
- Feature derivation: `car_age = 1982 - model_year`
- One-hot encoding of `origin` with `origin_europe` dropped to avoid multicollinearity
- Rigorous hyperparameter tuning across five models using three strategies
- Final model selected based on cross-validation and test performance

**All visualizations, metrics, and interpretations are documented in [`reports\projectReport.md`](reports\projectReport.md).**

---


## Dataset & Features

| Attribute | Type | Description |
|---------|------|-------------|
| `cylinders` | Numerical | Number of engine cylinders |
| `horsepower` | Numerical | Engine power output |
| `weight` | Numerical | Vehicle weight (lbs) |
| `car_age` | Derived | Age of vehicle in years (from 1982) |
| `origin_japan` | Binary | 1 if car is Japanese-made |
| `origin_usa` | Binary | 1 if car is American-made |

> Note: `origin_europe` was excluded as reference category.

**Sample Size:** 392 records (after dropping missing values)  
**Target Distribution:** Skewed right; mean = 23.45 MPG, std = 7.81

---


## Modeling Pipeline

1. **Preprocessing**
    - Train/Test split: 80/20 (`random_state=42`)
    - No scaling required for tree-based models
    - Categorical variables one-hot encoded

2. **Models Evaluated**
    - Linear Regression
    - Ridge Regression
    - Random Forest Regressor
    - Gradient Boosting Regressor
    - XGBoost Regressor

3. **Hyperparameter Optimization**
    Each model tuned using:
    - **Grid Search CV** – Exhaustive search over small spaces
    - **Random Search CV** – Efficient sampling over larger spaces
    - **Bayesian Optimization (skopt)** – Sequential model-based optimization

4. **Validation Strategy**
    - **5-fold Cross-Validation** on training set
    - Final evaluation on held-out **test set**
    - Metrics tracked: MSE, MAE, RMSE, R²

5. **Model Selection Criteria**
    - Highest Test R²
    - Lowest RMSE
    - Minimal overfitting (Train vs Test gap < 0.05)
    - Interpretability and stability

---

## Best Performing Model

| Metric | Value |
|-------|-------|
| **Model** | Random Forest Regressor |
| **Tuning Method** | Bayesian Optimization |
| **Test R² Score** | 0.88 |
| **Test RMSE** | 2.43 |
| **Test MAE** | 1.82 |
| **CV R²** | 0.85 |

Outperformed all other models due to its ability to capture non-linear interactions without overfitting.

Key drivers identified via SHAP:

- `weight` (strong negative impact)
- `horsepower`, `cylinders`
- `car_age` (positive effect)
- `origin_japan` (efficiency advantage)

For full analysis including plots and SHAP visuals, see: [`reports/final_report.md`](reports/final_report.md)

---


## File Structure

```bash
mpg-forecasting-project/
│
├── README.md
├── advancedModeling.ipynb
├── baselineModeling.ipynb
├── LICENSE
├── modelPresumption.py
├── preprocessing.ipynb
├── requirements.txt
├── data/
│    └──mpg.csv
├── graphs/
│    ├── boxplot for mpg-cylinders.png
│    ├── boxplot for mpg-horsepower.png
│    ├── boxplot for mpg-modelYear.png
│    ├── countplot for cylinders.png
│    ├── countplot for model year.png
│    ├── countplot for origin.png
│    ├── distplot for displacement.png
│    ├── distplot for horsepower.png
│    ├── heatmap for correlation between the variables.png
│    ├── lmplot for mpg-acceleration in origins.png
│    ├── lmplot for mpg-displacement in origins.png
│    ├── lmplot for mpg-horsepower in origins.png
│    └── lmplot for mpg-weight in origins.png
├── images/
│   ├── modelsComparison.png
│   ├── predictionError-advanceModel.png
│   ├── predictionError-baselineModel.png
│   ├── residualsPlot-advanceModel.png
│   ├── residualsPlot-baselineModel.png
│   └── shapSummary-advanceModel.png
├── reports/
│   ├── forecastingModel.pkl
│   ├── model_comparisons.csv
│   └── projectReport.md
```

---


## Dependencies

```python
# Core Data Science Libraries
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Machine Learning (Scikit-learn & Extensions)
scikit-learn>=1.3.0
scikit-optimize>=0.9.0

# Gradient Boosting Models
xgboost>=1.7.0

# Model Persistence
joblib>=1.3.0

# Model Interpretability
shap>=0.41.0

# Optional: Enhanced Visualization
yellowbrick>=1.5.0
```

---


## Installation

### Clone the Repository

```bash
git clone https://github.com/tshihab07/MPG-Forecasting-Model.git
```

```bash
cd MPG-Forecasting-Model
```

### Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\activate   # Windows
```

### Install Dependencies

```python
pip install -r requirements.txt
```

---


## Model Inference

**Run Python Script**
```bash
python modelPresumption.py
```

---


## Future Improvements
- Build a dashboard for interactive predictions
- Deploy as a REST API using Flask/FastAPI
- Add model monitoring for performance tracking
- Expand dataset with modern vehicle data

---


## Contributing

Contributions are welcome! Please feel free to submit a pull request.
- Fork the project.
- Create your feature branch
- Commit changes
- Push
- Open a Pull Request

---


## Contact

E-mail: tushar.shihab13@gmail.com <br>
More Projects: 👉🏿 [Projects](https://github.com/tshihab07?tab=repositories)<br>
LinkedIn: [Tushar Shihab](https://www.linkedin.com/in/tshihab07/)

---


## License

This project is licensed under the [MIT License](LICENSE).
