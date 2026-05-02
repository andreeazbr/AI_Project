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

X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names = (
    load_and_preprocess_data()
)

print("Dimensiuni date preprocesate:")
print("X_train_processed:", X_train_processed.shape)
print("X_test_processed:", X_test_processed.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# primul model

baseline_model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

baseline_model.fit(X_train_processed, y_train)

y_pred = baseline_model.predict(X_test_processed)

# evaluare

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Experiment 0: Baseline XGBoost ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nMatrice de confuzie:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nRaport de clasificare:")
print(classification_report(y_test, y_pred))