from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

# metadata
print(adult.metadata)

# variable information
print(adult.variables)

# aici incepe codul meu, respectiv partea 1

print("X shape (instante, trasaturi):", X.shape)
print("y shape:", y.shape)

print("Feature columns:", list(X.columns))
print("Target columns:", list(y.columns) if hasattr(y, "columns") else "target")

print(X.dtypes)

# am observat ca datele contin spatii si puncte care vor genera erori
print("Etichete income (înainte de curățare):", y['income'].unique())

y['income'] = y['income'].astype(str).str.replace('.', '', regex=False).str.strip() # aici corectez

print("Etichete income (după curățare):", y['income'].unique()) # afisez dupa corectie

y['income'] = y['income'].str.replace('.', '', regex=False)
y['income'] = y['income'].map({'<=50K': 0, '>50K': 1})

print("Etichete income (după codificare):", y['income'].unique())

# am spatii si in variabilele categoriale, acestea trebuie eliminate
X = X.apply(lambda col: col.str.strip() if col.dtype in ["object", "string"] else col)
