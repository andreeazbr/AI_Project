from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

# metadata
print(adult.metadata)

# variable information
print(adult.variables)

# aici incepe codul meu, respectiv partea 1 - Descrierea setului de date si a variabilei tinta

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

# partea 2 - Distributia claselor

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

# partea 3 - Identificarea valorilor lipsa si a problemelor de calitate

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

# partea 4a - Analiza trasaturilor numerice

# selectare coloane numerice
num_cols = X.select_dtypes(include=["int64", "float64"]).columns

print("Trăsături numerice:", list(num_cols))

desc_stats = X[num_cols].describe()

# eliminare notație științifică
pd.set_option("display.float_format", "{:.2f}".format)

# afișare toate coloanele
pd.set_option("display.max_columns", None)

# statistici rotunjite
desc_stats = X[num_cols].describe().round(2)

print("\n--- Statistici descriptive (formatate) ---")
print(desc_stats)
desc_stats = X[num_cols].describe().round(2)
desc_stats.to_excel("statistici_numerice.xlsx") #asta este ca sa am tabelul deja facut, se comenteaza sau se sterge fisierul cand rulez iar

std_values = desc_stats.loc["std"].sort_values(ascending=False)

top_15 = std_values.head(15).index

print("\nPrimele trăsături cu deviația standard cea mai mare:")
print(list(top_15))

for col in top_15:
    plt.figure()
    plt.hist(X[col], bins=30)
    plt.title(f"Histogramă - {col}")
    plt.xlabel(col)
    plt.ylabel("Frecvență")
    plt.tight_layout()
    plt.show()

print("\n--- Detectare outlieri (IQR) ---")

outlier_dict = {}

for col in num_cols:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = X[(X[col] < lower_bound) | (X[col] > upper_bound)][col]
    outlier_count = len(outliers)

    outlier_dict[col] = outlier_count

# transformare în DataFrame
outlier_df = pd.DataFrame(
    outlier_dict.items(),
    columns=["Trăsătură", "Număr outliere"]
)

# sortare descrescător
outlier_df = outlier_df.sort_values(by="Număr outliere", ascending=False)
outlier_df["Procent (%)"] = (outlier_df["Număr outliere"] / len(X) * 100).round(2)

print("\n--- Outliere sortate descrescător ---")
print(outlier_df)
outlier_df.to_excel("outliere.xlsx") # tot pt afisare

# partea 4b - Analiza trasaturilor categoriale

# selectare trăsături categoriale
cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

print("\nTrăsături categoriale:", cat_cols)

summary_rows = []

for col in cat_cols:
    n_categories = X[col].nunique()
    summary_rows.append({
        "Trăsătură": col,
        "Număr categorii": n_categories
    })

cat_summary_df = pd.DataFrame(summary_rows)

# sortare descrescător după număr categorii
cat_summary_df = cat_summary_df.sort_values(
    by="Număr categorii",
    ascending=False
)

print("\n--- Număr categorii pentru fiecare trăsătură categorială ---")
print(cat_summary_df)
cat_summary_df.to_excel("categorii.xlsx")

for col in cat_cols:
    print(f"\n=== Distribuția categoriilor pentru '{col}' ===")
    counts = X[col].value_counts()
    percents = X[col].value_counts(normalize=True) * 100
    dist_table = pd.DataFrame({
        "Număr instanțe": counts,
        "Procent (%)": percents.round(2)
    })

    # sortare descrescător după număr instanțe
    dist_table = dist_table.sort_values(
        by="Număr instanțe",
        ascending=False
    )
print(dist_table)

# partea 5 - Relatia dintre trasaturi si target

# selectare trăsături numerice
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
print("\nTrăsături numerice:", list(num_cols))

# combinăm X și y
df_corr = pd.concat([X[num_cols], y], axis=1)

# matrice de corelație
corr_matrix = df_corr.corr()

print("\nMatricea de corelație:")
print(corr_matrix)

plt.figure(figsize=(8, 6))

sns.heatmap(
     corr_matrix,
     annot=True,
     cmap="coolwarm",
     center=0,
     fmt=".2f",
     linewidths=0.5,
     vmin=-1,
     vmax=1
)

plt.title("Matricea de corelație între trăsăturile numerice și income")

plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()

plt.show()

target_corr = corr_matrix["income"].drop("income")

# sortare după valoarea absolută
target_corr = target_corr.reindex(
target_corr.abs().sort_values(ascending=False).index
)

print("\nCorelația trăsăturilor numerice cu income:")
print(target_corr)

top_features = target_corr.head(3).index

print("\nCele mai informative trăsături:", list(top_features))

for feature in top_features:
   plt.figure(figsize=(6, 4))
   sns.histplot(
        data=df_corr,
        x=feature,
        hue="income",
        bins=30,
        kde=False,
        multiple="dodge"
   )

plt.title(f"Distribuția variabilei {feature} în funcție de income")
plt.xlabel(feature)
plt.ylabel("Număr instanțe")

plt.show()