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

# partea 2

# nr de instante pe clasa + procente
counts = y['income'].value_counts()
percents = y['income'].value_counts(normalize=True) * 100

# tabel
dist_table = pd.DataFrame({
    "Număr instanțe": counts,
    "Procent (%)": percents.round(2)
})
print("\nDistribuția claselor (income):")
print(dist_table)

# grafic
plt.figure()
ax = sns.countplot(x=y['income'])

plt.title("Distribuția claselor pentru variabila țintă (income)")
plt.xlabel("Clasă")
plt.ylabel("Număr de instanțe")
plt.xticks(rotation=0)
plt.tight_layout()

for p in ax.patches:
    height = p.get_height()
    height = int (height)
    ax.annotate(f'{height}',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom')

plt.show()

# concluzie
minority_pct = percents.min()
majority_pct = percents.max()

# nu exista un prag universal, acesta este cel pe care l-am ales eu

if majority_pct >= 70:
    concluzie = "Dataset dezechilibrat (clasa majoritară ≥ 70%)."
elif majority_pct >= 60:
    concluzie = "Dataset moderat dezechilibrat (clasa majoritară între 60% și 70%)."
else:
    concluzie = "Dataset relativ echilibrat (clase apropiate ca proporție)."

print("\nConcluzie:", concluzie)

# partea 3

X = X.replace("?", np.nan) # inlocuire

# tabel
missing_count = X.isna().sum()
missing_percent = (missing_count / len(X)) * 100

missing_table = pd.DataFrame({
    "Valori lipsă (număr)": missing_count,
    "Valori lipsă (%)": missing_percent.round(2)
}).sort_values("Valori lipsă (număr)", ascending=False)

print("\n--- Valori lipsă pe trăsături (număr și procent) ---")
print(missing_table)

num_features_with_missing = (missing_count > 0).sum()
print(f"\nNumăr trăsături cu valori lipsă: {num_features_with_missing} din {X.shape[1]}")

# verificare duplicate
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
df_full = pd.concat([X, y], axis=1)

dup_count = df_full.duplicated().sum()
dup_percent = (dup_count / len(df_full)) * 100

print("\n--- Duplicate ---")
print(f"Instanțe duplicate (număr): {dup_count}")
print(f"Instanțe duplicate (%): {dup_percent:.2f}")

# eliminare duplicate
df_full = df_full.drop_duplicates().reset_index(drop=True)

# Separăm înapoi X și y
X = df_full.drop(columns=y.columns)
y = df_full[y.columns]

# Imputare valori lipsă:
# - numeric: mediană
# - categorial:  (cel mai frecvent)
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "string"]).columns

# imputare numerică cu mediana
for col in num_cols:
    med = X[col].median()
    X[col] = pd.to_numeric(X[col], errors="coerce")
    X[col] = X[col].fillna(med)

# imputare categorială cu cea mai frecvenat categorie
for col in cat_cols:
    mode = X[col].mode(dropna=True)
    fill_value = mode.iloc[0] if len(mode) > 0 else "Unknown"
    X[col] = X[col].fillna(fill_value)

# verificare finală
print("\n--- Verificare după curățare/imputare ---")
print("Total valori lipsă rămase în X:", int(X.isna().sum().sum()))
print("Dimensiune după eliminare duplicate:", X.shape, y.shape)
