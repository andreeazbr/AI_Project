import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from preprocessing_utils import load_and_preprocess_data
from xgboost import XGBClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

def evaluate_model(model, X_test, y_test, experiment_name, save_outputs=True):
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["<=50K", ">50K"],
        yticklabels=["<=50K", ">50K"]
    )

    plt.title(f"Confusion Matrix - {experiment_name}")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.tight_layout()

    if save_outputs:
        file_name = experiment_name.lower().replace(" ", "_").replace("-", "_")
        plt.savefig(
            f"outputs/confusion_matrix_{file_name}.png",
            dpi=300,
            bbox_inches="tight"
        )
    plt.show()

    report = classification_report(
        y_test,
        y_pred,
        target_names=["<=50K", ">50K"],
        output_dict=True
    )
    print(f"\nRaport de clasificare pentru {experiment_name}:")
    report_df = pd.DataFrame(report).transpose()
    print(report_df.round(4))

    if save_outputs:
        file_name = experiment_name.lower().replace(" ", "_").replace("-", "_")
        report_df.round(4).to_excel(
            f"outputs/classification_report_{file_name}.xlsx"
        )

    return {
        "experiment": experiment_name,
        "accuracy": report["accuracy"],

        "precision_<=50K": report["<=50K"]["precision"],
        "recall_<=50K": report["<=50K"]["recall"],
        "f1_<=50K": report["<=50K"]["f1-score"],

        "precision_>50K": report[">50K"]["precision"],
        "recall_>50K": report[">50K"]["recall"],
        "f1_>50K": report[">50K"]["f1-score"],

        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"]
    }

os.makedirs("outputs", exist_ok=True)

X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names = (
    load_and_preprocess_data()
)

print("Dimensiuni date preprocesate:")
print("X_train_processed:", X_train_processed.shape)
print("X_test_processed:", X_test_processed.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

all_results = []

# experimentul 0

baseline_model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

baseline_model.fit(X_train_processed, y_train)

baseline_results = evaluate_model(
    model=baseline_model,
    X_test=X_test_processed,
    y_test=y_test,
    experiment_name="Experiment 0 Baseline XGBoost"
)

all_results.append(baseline_results)

# experimentul 1

max_depth_values = [3, 6, 10]

for depth in max_depth_values:
    model = XGBClassifier(
        max_depth=depth,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train_processed, y_train)

    result = evaluate_model(
        model=model,
        X_test=X_test_processed,
        y_test=y_test,
        experiment_name=f"Experiment 1 max_depth {depth}",
        save_outputs=False
    )

    result["max_depth"] = depth
    result["n_estimators"] = "default"
    result["learning_rate"] = "default"

    all_results.append(result)

# experiment 2

n_estimators_values = [50, 100, 200]

for n in n_estimators_values:
    model = XGBClassifier(
        n_estimators=n,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train_processed, y_train)

    result = evaluate_model(
        model=model,
        X_test=X_test_processed,
        y_test=y_test,
        experiment_name=f"Experiment 2 n_estimators {n}",
        save_outputs=False
    )

    result["max_depth"] = "default"
    result["n_estimators"] = n
    result["learning_rate"] = "default"

    all_results.append(result)

# ============================
# Experiment 3 - Variația learning_rate
# ============================

learning_rate_values = [0.01, 0.1, 0.3, 0.5]

for lr in learning_rate_values:
    model = XGBClassifier(
        learning_rate=lr,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train_processed, y_train)

    result = evaluate_model(
        model=model,
        X_test=X_test_processed,
        y_test=y_test,
        experiment_name=f"Experiment 3 learning_rate {lr}",
        save_outputs=False
    )

    result["max_depth"] = "default"
    result["n_estimators"] = "default"
    result["learning_rate"] = lr

    all_results.append(result)

results_df = pd.DataFrame(all_results)

print("\nRezultate comparative:")
print(results_df.round(4))

results_df.round(4).to_excel(
    "outputs/experiment_results_summary.xlsx",
    index=False
)

experiment_1_df = results_df[
    results_df["experiment"].str.contains("Experiment 1")
]

selected_columns = [
    "experiment",
    "accuracy",
    "f1_<=50K",
    "f1_>50K",
    "macro_f1",
    "weighted_f1",
    "max_depth"
]

experiment_1_df[selected_columns].round(4).to_excel(
    "outputs/experiment_1_max_depth_results.xlsx",
    index=False
)

experiment_2_df = results_df[
    results_df["experiment"].str.contains("Experiment 2")
]

selected_columns = [
    "experiment",
    "accuracy",
    "f1_<=50K",
    "f1_>50K",
    "macro_f1",
    "weighted_f1",
    "n_estimators"
]

experiment_2_df[selected_columns].round(4).to_excel(
    "outputs/experiment_2_n_estimators_results.xlsx",
    index=False
)

experiment_3_df = results_df[
    results_df["experiment"].str.contains("Experiment 3")
]

selected_columns = [
    "experiment",
    "accuracy",
    "f1_<=50K",
    "f1_>50K",
    "macro_f1",
    "weighted_f1",
    "learning_rate"
]

experiment_3_df[selected_columns].round(4).to_excel(
    "outputs/experiment_3_learning_rate_results.xlsx",
    index=False
)

