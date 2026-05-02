import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from preprocessing_utils import load_and_preprocess_data
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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
    print("\nRaport de clasificare:")
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

results_df = pd.DataFrame([baseline_results])
print("\nRezultate sumar:")
print(results_df.round(4))

results_df.round(4).to_excel(
    "outputs/experiment_results_summary.xlsx",
    index=False
)