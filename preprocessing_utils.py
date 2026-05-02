from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class RareCategoryGrouper(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=0.01, other_label="Other"):
        self.threshold = threshold
        self.other_label = other_label
        self.frequent_categories_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        for col in X.columns:
            freq = X[col].value_counts(normalize=True, dropna=False)
            self.frequent_categories_[col] = freq[freq >= self.threshold].index.tolist()

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for col in X.columns:
            frequent_categories = self.frequent_categories_.get(col, [])
            X[col] = X[col].where(X[col].isin(frequent_categories), self.other_label)

        return X


def load_and_preprocess_data(
    test_size=0.20,
    random_state=42,
    rare_threshold=0.01
):
    """
    Încarcă datasetul Adult, aplică pașii de curățare și preprocesare,
    apoi returnează seturile train/test pregătite pentru modelare.

    Returnează:
    - X_train_processed
    - X_test_processed
    - y_train
    - y_test
    - preprocessor
    - feature_names
    """

    adult = fetch_ucirepo(id=2)

    X = adult.data.features.copy()
    y = adult.data.targets.copy()

    y["income"] = (
        y["income"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.strip()
    )

    X = X.apply(
        lambda col: col.str.strip()
        if col.dtype in ["object", "string"]
        else col
    )

    X = X.replace("?", np.nan)

    y["income"] = y["income"].map({"<=50K": 0, ">50K": 1})
    y = y["income"]

    df_full = pd.concat(
        [X.reset_index(drop=True), y.reset_index(drop=True)],
        axis=1
    )

    df_full = df_full.drop_duplicates().reset_index(drop=True)

    X = df_full.drop(columns=["income"])
    y = df_full["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    numeric_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("rare_grouper", RareCategoryGrouper(
            threshold=rare_threshold,
            other_label="Other"
        )),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    encoded_cat_features = (
        preprocessor
        .named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(categorical_features)
    )

    feature_names = numeric_features + list(encoded_cat_features)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
        feature_names
    )