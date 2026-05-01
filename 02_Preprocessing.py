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

