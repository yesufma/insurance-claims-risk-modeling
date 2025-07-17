# Insurance Claims Risk Modeling

This repository contains the implementation of a machine learning project for predicting insurance claim severity (`TotalClaims`) as part of the 10Academy training program. The project spans Tasks 1–4 and focuses on data preprocessing, exploratory data analysis (EDA), feature engineering, and predictive modeling to support risk assessment and pricing strategies.

> **Tools Used**: Python · Jupyter Notebook · DVC · scikit-learn · XGBoost · Git

---

##  Project Overview

The goal is to predict the **monetary value of insurance claims (`TotalClaims`)** for policies with non-zero claims, using a dataset located at `data/insurance_claims_cleaned.csv`.

###  Tasks Completed:
- **Task 1**: Data collection and preprocessing
- **Task 2**: Exploratory data analysis (EDA)
- **Task 3**: Feature engineering and selection (41 features)
- **Task 4**: Predictive modeling and SHAP interpretability



##  Tasks and Methodology

###  Task 1: Data Collection and Preprocessing
- Loaded `insurance_claims_cleaned.csv` using pandas.
- Filtered for `TotalClaims > 0` and dropped rows with missing values.
- Tracked dataset with DVC: `data/insurance_claims_cleaned.csv.dvc`
<img width="1440" height="600" alt="categorical_distributions" src="https://github.com/user-attachments/assets/41f0081e-11b5-43c0-bf57-09f99fbfddfd" />
<img width="1440" height="720" alt="claims_temporal_trend" src="https://github.com/user-attachments/assets/57ae01ec-63ba-4db1-b701-e99ccfaed10e" />
<img width="1440" height="600" alt="outlier_boxplots" src="https://github.com/user-attachments/assets/4b57afc2-c603-4263-ba6f-3da95a0072d0" />


###  Task 2: Exploratory Data Analysis (EDA)
- Analyzed distributions of numeric (`SumInsured`, `kilowatts`) and categorical (`Province`, `CoverType`) features.
- Generated:
 <img width="960" height="720" alt="correlation_matrix" src="https://github.com/user-attachments/assets/d80270f5-84af-4b4f-84e7-5761804a11e3" />

- Found **positive correlation** between `SumInsured` and `TotalClaims`.

###  Task 3: Feature Engineering and Selection
- Selected:
  - 6 numerical features: `SumInsured`, `RegistrationYear`, `Cylinders`, `cubiccapacity`, `kilowatts`, `NumberOfDoors`
  - 4 categorical features: `Province`, `Gender`, `VehicleType`, `CoverType`
- Applied **one-hot encoding**, producing **41 features**.
- Feature list saved to: `reports/feature_list_task4.txt`

###  Task 4: Predictive Modeling
- Trained models in `notebooks/predictive_modeling_task4.ipynb`:
  - **Linear Regression**:  
    - RMSE = `33,630.26`  
    - R² = `0.297`
  - **Random Forest**:  
    - RMSE = 36854.7037  
    - R² = 0.15543
  - **XGBoost**:  
    - RMSE = 39704.19416 
    - R² = 0.019788
- Visualizations:
 <img width="1000" height="600" alt="actual_vs_predicted_lr_task4" src="https://github.com/user-attachments/assets/bed619c8-f244-422a-986e-586b1efe2682" />
<img width="1000" height="600" alt="actual_vs_predicted_rf_task4" src="https://github.com/user-attachments/assets/5a4d9460-8091-4c5a-88e3-9f06cf0c4bd9" />
<img width="1000" height="600" alt="actual_vs_predicted_xgb_task4" src="https://github.com/user-attachments/assets/74df0f59-52c1-4efc-b3a2-629d0813760d" />

  <img width="1000" height="600" alt="model_comparison_task4" src="https://github.com/user-attachments/assets/bccfaa8a-06ea-427d-adb5-e7740db931b6" />


- Interpretability with SHAP:
<img width="1086" height="698" alt="shap_importance_task4" src="https://github.com/user-attachments/assets/9005b284-535c-4c46-a59b-36564862928a" />


---
🔗 **Project URL**: [github.com/yesufma/insurance-claims-risk-modeling](https://github.com/yesufma/insurance-claims-risk-modeling)

---

## 📁 Repository Structure
```bash
insurance-claims-risk-modeling/
├── .dvc/ # DVC configuration
├── data/ # Datasets
│ ├── insurance_claims_cleaned.csv.dvc
│ └── MachineLearningRating_v3.txt
├── notebooks/ # Jupyter notebooks
│ ├── eda_task2.ipynb
│ └── predictive_modeling_task4.ipynb
├── reports/ # Outputs and reports
│ ├── feature_list_task4.txt
│ ├── eda_histograms_task2.png
│ ├── eda_correlation_task2.png
│ ├── actual_vs_predicted_lr_task4.png
│ ├── actual_vs_predicted_rf_task4.png
│ ├── actual_vs_predicted_xgb_task4.png
│ ├── model_comparison_task4.png
│ ├── shap_importance_task4.png
│ ├── project_report.md
│ └── project_report.docx
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation and Setup

###  Clone the Repository
```bash
git clone https://github.com/yesufma/insurance-claims-risk-modeling.git
cd insurance-claims-risk-modeling
## Create Environment & Install Requirements
conda env create -f environment.yml
conda activate insurance-risk
pip install -r requirements.txt
# Pull Data Using DVC
dvc pull



