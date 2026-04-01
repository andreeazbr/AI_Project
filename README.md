## 📌 Project Type
Academic project – Artficial Intelligence & Machine Learning

# Adult Income Dataset – Exploratory Data Analysis (EDA)

This project focuses on the exploratory data analysis and preprocessing of the **Adult dataset** from the UCI Machine Learning Repository.  
The goal is to understand the structure of the data and prepare it for a binary classification task.

---

## 📊 Problem Description

The objective is to build a model that predicts whether a person's income exceeds **$50K/year**, based on demographic and socio-economic features.

Formally, the problem can be defined as:

f: ℝⁿ → {0,1}

where:
- n = number of features
- output = income class (0: <=50K, 1: >50K)

---

## 📁 Dataset

- Source: UCI Machine Learning Repository
- Number of instances: 48,842
- Number of features: 14 + target
- Mixed data types: numerical and categorical

---

## ⚙️ Data Preprocessing

The following preprocessing steps were applied:

### 1. Data Cleaning
- Removed trailing characters (e.g. "." in target labels)
- Trimmed whitespace from categorical values
- Converted missing values represented by `"?"` into `NaN`

### 2. Missing Values Handling
- Numerical features → imputed with **median**
- Categorical features → imputed with **most frequent category**

### 3. Duplicate Removal
- Identified and removed duplicate instances
- Dataset index reset after cleaning

---

## 📊 Exploratory Data Analysis

### 🔹 Target Distribution
- Significant class imbalance:
  - at least 50K → ~76%
  - less than 50K → ~24%

### 🔹 Numerical Features
- Descriptive statistics computed (mean, std, quartiles)
- Histograms used to analyze distributions
- Strong right-skew observed in:
  - capital-gain
  - capital-loss

### 🔹 Outlier Detection
- Method: Interquartile Range (IQR)
- High number of outliers detected in:
  - hours-per-week
  - capital-gain

Note: Outliers were not removed as they represent valid real-world values.

---

## 🔹 Categorical Features Analysis

- Cardinality varies significantly:
  - Low: sex (2 categories)
  - High: native-country (41 categories)

- Rare categories identified (<1%)

Example:
- Holand-Netherlands → extremely rare

---

## 🔧 Encoding Strategy (Planned)

The encoding is **not applied yet**, but the following strategy is proposed:

- Binary encoding → for dichotomous variables (e.g. sex)
- One-Hot Encoding → for low/moderate cardinality features
- Grouping rare categories → for high-cardinality features

---

## 🔗 Feature–Target Relationship

- Pearson correlation used
- Equivalent to **point-biserial correlation** for binary target

### Most informative features:
- education-num (≈ 0.33)
- age (≈ 0.23)
- hours-per-week (≈ 0.23)

### Low relevance:
- fnlwgt (≈ 0)

---

## 📚 References

1. Becker, B., & Kohavi, R. (1996). Adult Dataset. UCI Repository  
2. Bishop, C. M. – Pattern Recognition and Machine Learning  
3. Géron, A. – Hands-On Machine Learning  
4. Tukey, J. W. – Exploratory Data Analysis  

---

## 🚀 Future Work

- Feature encoding implementation
- Model training and evaluation
- Handling class imbalance
- Feature selection

---

## 🧠 Key Takeaways

- The dataset is imbalanced → evaluation metrics must be chosen carefully  
- Numerical features show different scales → normalization required  
- Categorical features contain rare values → need special handling  
- Some features show meaningful correlation with income  

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib / Seaborn

---

## 👩‍💻 Author

Andreea Zbranca
