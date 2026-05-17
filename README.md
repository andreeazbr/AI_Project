# Adult Income Classification Project

This project focuses on the analysis, preprocessing, modeling, and evaluation of the Adult Income dataset from the UCI Machine Learning Repository.

The goal is to predict whether a person's annual income is less than or equal to 50K USD or greater than 50K USD, based on demographic and socio-economic attributes.

The project was developed as part of an Artificial Intelligence coursework assignment and follows three main stages:

1. Exploratory Data Analysis
2. Data Preprocessing
3. Modeling and Evaluation with XGBoost

---

## Dataset

The dataset used in this project is the Adult dataset, available from the UCI Machine Learning Repository.

The prediction target is:

- `<=50K`
- `>50K`

This is a binary classification problem.

The dataset contains:

- 48,842 initial instances
- 14 input features
- 1 target variable: `income`

The features include both numerical and categorical variables.

### Numerical Features

- `age`
- `fnlwgt`
- `education-num`
- `capital-gain`
- `capital-loss`
- `hours-per-week`

### Categorical Features

- `workclass`
- `education`
- `marital-status`
- `occupation`
- `relationship`
- `race`
- `sex`
- `native-country`

---

## Project Structure

```text
.
├── main.py
├── 02_preprocessing.py
├── preprocessing_utils.py
├── 03_modeling_xgboost.py
├── requirements.txt
├── reports/
│   └── Raport_proiect_IA.pdf
├── outputs/
│   ├── confusion_matrix_experiment_0_baseline_xgboost.png
│   ├── classification_report_experiment_0_baseline_xgboost.xlsx
│   ├── experiment_results_summary.xlsx
│   ├── experiment_1_max_depth_results.xlsx
│   ├── experiment_2_n_estimators_results.xlsx
│   ├── experiment_3_learning_rate_results.xlsx
│   ├── feature_importance_xgboost.xlsx
│   ├── feature_importance_xgboost.png
│   ├── feature_importance_xgboost_aggregated.xlsx
│   └── feature_importance_xgboost_aggregated.png
└── README.md
```

---

## Stage 1: Exploratory Data Analysis

The first stage focuses on understanding the dataset before applying machine learning models.

The analysis includes:

- general dataset description;
- target variable analysis;
- class distribution;
- missing value detection;
- duplicate instance detection;
- descriptive statistics for numerical features;
- histogram visualization;
- outlier detection using the IQR method;
- categorical feature analysis;
- rare category detection;
- correlation analysis between numerical features and the target variable.

### Main Observations

The target variable is imbalanced:

| Class | Percentage |
|---|---:|
| `<=50K` | approximately 76% |
| `>50K` | approximately 24% |

This means that accuracy alone is not sufficient for model evaluation.

The dataset also contains missing values represented by `"?"`, mainly in:

- `workclass`
- `occupation`
- `native-country`

Some categorical variables, especially `native-country`, contain many rare categories.

---

## Stage 2: Data Preprocessing

The preprocessing stage prepares the dataset for machine learning.

The main preprocessing steps are:

1. Cleaning target labels
2. Removing extra whitespace from categorical values
3. Replacing `"?"` with missing values
4. Removing duplicate instances
5. Splitting the data into train and test sets
6. Applying imputation
7. Grouping rare categories
8. Applying One-Hot Encoding

The train/test split uses an 80/20 ratio and stratification in order to preserve the class distribution.

```text
Train set: 39,032 instances
Test set: 9,758 instances
```

After preprocessing, the number of features increases from 14 to 63 due to One-Hot Encoding.

```text
X_train_processed: 39,032 × 63
X_test_processed: 9,758 × 63
```

The preprocessing transformations are fitted only on the training set and then applied to the test set in order to avoid data leakage.

### Rare Category Handling

Rare categories are grouped into a common category named `Other`.

A category is considered rare if it appears in less than 1% of the training instances.

This helps reduce dimensionality after One-Hot Encoding and avoids creating many almost-empty binary columns.

---

## Stage 3: Modeling and Evaluation

The model used in this project is XGBoost.

XGBoost is treated as a black-box model. The focus is not on the internal algorithmic details, but on:

- correct preprocessing;
- evaluation metrics;
- model interpretation;
- hyperparameter influence;
- feature importance.

The baseline model is trained using:

```python
XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)
```

---

## Evaluation Metrics

Because the dataset is imbalanced, the model is evaluated using multiple metrics:

- accuracy;
- precision;
- recall;
- F1-score;
- confusion matrix;
- macro average;
- weighted average.

The analysis focuses especially on the minority class `>50K`.

---

## Baseline Model Results

The baseline XGBoost model achieved the following results on the test set:

| Metric | Value |
|---|---:|
| Accuracy | 0.8775 |
| Macro F1-score | 0.8238 |
| Weighted F1-score | 0.8745 |
| F1-score (`<=50K`) | 0.9211 |
| F1-score (`>50K`) | 0.7265 |

The model performs better on the majority class `<=50K` than on the minority class `>50K`.

This confirms that the class imbalance affects model performance.

---

## Confusion Matrix

The baseline model produced the following confusion matrix:

| | Predicted `<=50K` | Predicted `>50K` |
|---|---:|---:|
| Actual `<=50K` | 6976 | 446 |
| Actual `>50K` | 749 | 1587 |

The model correctly identifies most instances from the majority class, but it has more difficulty identifying the minority class `>50K`.

The number of false negatives for the `>50K` class is higher than the number of false positives.

---

## Hyperparameter Experiments

Three groups of experiments were performed.

### Experiment 1: `max_depth`

Tested values:

```text
3, 6, 10
```

Best result:

```text
max_depth = 6
```

This value corresponds to the baseline configuration and provides the best balance among the tested values.

### Experiment 2: `n_estimators`

Tested values:

```text
50, 100, 200
```

Best result:

```text
n_estimators = 100
```

The value `100` corresponds to the default baseline configuration.

### Experiment 3: `learning_rate`

Tested values:

```text
0.01, 0.1, 0.3, 0.5
```

Best result:

```text
learning_rate = 0.3
```

A very small learning rate caused the model to learn too slowly with the default number of estimators, while a larger value reduced generalization performance.

---

## Final Model Choice

The baseline XGBoost configuration was kept as the final selected model.

The final configuration is approximately:

```text
max_depth = 6
n_estimators = 100
learning_rate = 0.3
```

The experiments showed that changing these hyperparameters individually did not improve performance on the test set.

This does not mean that the default values are always optimal, but for the tested values and this dataset, they provided the best balance between global performance and minority class performance.

---

## Feature Importance

Feature importance was analyzed using the final XGBoost model.

Because categorical variables were transformed using One-Hot Encoding, feature importance was also aggregated at the original feature level.

The most important original features were:

1. `marital-status`
2. `occupation`
3. `relationship`
4. `education-num`
5. `capital-gain`

These results are consistent with the exploratory analysis, where `education-num` and `capital-gain` were found to be informative numerical features.

The model also highlights the importance of categorical features such as `marital-status`, `occupation`, and `relationship`, which were not directly captured by the Pearson correlation analysis performed only on numerical variables.

Feature importance should be interpreted as a predictive association identified by the model, not as a causal relationship.

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

Replace the repository URL with your actual GitHub repository link.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it.

On Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Exploratory Analysis

```bash
python main.py
```

### 5. Run the Preprocessing Stage

```bash
python 02_preprocessing.py
```

### 6. Run the Modeling and Evaluation Stage

```bash
python 03_modeling_xgboost.py
```

The generated charts and Excel files will be saved in the `outputs/` directory.

---

## Main Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- ucimlrepo
- openpyxl

---

## Notes

The project follows a structured machine learning workflow:

```text
Exploratory Data Analysis
        ↓
Data Preprocessing
        ↓
Model Training
        ↓
Evaluation
        ↓
Feature Importance Analysis
```

The main focus is not only training a model, but also understanding the data, applying correct preprocessing, evaluating the results critically, and interpreting the behavior of the model.

---

## Author

Zbranca Andreea  
Group 432Aa  
Artificial Intelligence Project
