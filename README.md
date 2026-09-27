# Loan_data_approval1
# Loan Status Prediction Model 🏦

Yeh project ek **Machine Learning pipeline** generate karta hai jo applicants ki personal aur financial details ke mutabiq unka **Loan Approval Status (Y/N)** predict karta hai. Isme data preprocessing aur model training ke liye scikit-learn pipelines ka istemal kiya gaya hai.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset Details](#dataset-details)
- [Data Preprocessing & Cleaning](#data-preprocessing--cleaning)
- [Model Architecture](#model-architecture)
- [Installation & Requirements](#installation--requirements)
- [Results & Performance](#results--performance)

## 🔍 Project Overview
Project ka main maqsad classification model build karna hai jo yeh handle kar sake:
1. Missing values ko automate tareeqe se impute karna.
2. Categorical features ko properly encode karna.
3. Continuous features ko scale karna.
4. **Logistic Regression** classifier ke zariye status predict karna.

## 📊 Dataset Details
Dataset (`mEzNPN.csv`) mein **381 rows** aur **13 columns** hain. Columns details niche darj hain:
* `Loan_ID`: Unique Identifier (Dropped during training) [1]
* `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed` (Categorical/Demographic features) [1]
* `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History` (Financial features) [1]
* `Property_Area`: Urban/Rural classification [1]
* `Loan_Status`: Target variable (`1` for Approved, `0` for Rejected) [1]

## 🛠️ Data Preprocessing & Cleaning
Notebook mein raw data par yeh steps apply kiye gye hain:
* **Manual Mapping:** `Gender`, `Education`, `Self_Employed`, `Property_Area`, aur `Married` columns ko numerical values mein map kiya gaya.
* **Target Encoding:** `Loan_Status` ke `Y`/`N` ko `1`/`0` mein badla gaya.
* **Feature Dropping:** Unique identifier column (`Loan_ID`) ko remove kiya gaya.

### Sklearn Robust Pipelines
Data leakage se bachne ke liye `ColumnTransformer` ka istemal kiya gaya hai:
* **Numerical Features:** `SimpleImputer(strategy="median")` -> `StandardScaler()` [1]
* **Categorical Features:** `SimpleImputer(strategy="most_frequent")` -> `OneHotEncoder(handle_unknown="ignore")` [1]

## 🧠 Model Architecture
Model training ke liye scikit-learn ki end-to-end `Pipeline` bani hui hai:
1. **Preprocessing Layer:** `ColumnTransformer` (scaling aur encoding ke liye).
2. **Estimator:** `LogisticRegression()` classifier.

Data ko **80% Train** aur **20% Test** split mein divide kiya gaya (`random_state=36`).

## 💻 Installation & Requirements
Is project ko run karne ke liye aapke system mein Python aur niche di gayi libraries honi chahiye:

```bash
pip install pandas numpy scikit-learn
```

### Run Kaise Karein?
Apna Jupyter Notebook open karein aur saare cells sequential order mein run karein.

## 📈 Results & Performance
Model ne test set par kafi acchi accuracy show ki hai:
* **Test Accuracy Score:** `~83.12%` (`0.8311688311688312`) [1]
