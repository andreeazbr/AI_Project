from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Incarcarea setului de date
adult = fetch_ucirepo(id=2)

X = adult.data.features.copy()
y = adult.data.targets.copy()

# Curatare initiala

y["income"] = y["income"].astype(str).str.replace(".", "", regex=False).str.strip() # curatare etichete
X = X.apply(lambda col: col.str.strip() if col.dtype in ["object", "string"] else col) # curatare spatii in var categoriale
X = X.replace("?", np.nan) # inlocuire valori lipsa cu NaN

y["income"] = y["income"].map({"<=50K": 0, ">50K": 1}) # codificare
y = y["income"] # series

print("Dimensiune inițială după curățare:")
print("X:", X.shape)
print("y:", y.shape)

print("\nDistribuție target după codificare:")
print(y.value_counts())

# eliminare duplicate

df_full = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
duplicates_count = df_full.duplicated().sum()
duplicates_percent = duplicates_count / len(df_full) * 100

print("\nDuplicate înainte de eliminare:")
print(f"Număr duplicate: {duplicates_count}")
print(f"Procent duplicate: {duplicates_percent:.2f}%")

df_full = df_full.drop_duplicates().reset_index(drop=True)

X = df_full.drop(columns=["income"])
y = df_full["income"]

print("\nDimensiune după eliminarea duplicatelor:")
print("X:", X.shape)
print("y:", y.shape)

# Impartire Train/Test

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDimensiuni seturi:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\nDistribuție clase în train (%):")
print((y_train.value_counts(normalize=True) * 100).round(2))

print("\nDistribuție clase în test (%):")
print((y_test.value_counts(normalize=True) * 100).round(2))