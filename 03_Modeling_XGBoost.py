from preprocessing_utils import load_and_preprocess_data

X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names = (
    load_and_preprocess_data()
)

print("X_train_processed:", X_train_processed.shape)
print("X_test_processed:", X_test_processed.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
print("Număr feature names:", len(feature_names))