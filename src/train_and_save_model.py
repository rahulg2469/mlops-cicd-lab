import os
import sys
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')          # Google Cloud Storage bucket name
VERSION_FILE_NAME = os.getenv('VERSION_FILE_NAME')  # File that stores the model version
ACCURACY_THRESHOLD = float(os.getenv('ACCURACY_THRESHOLD', '0.85'))  # Quality gate for CI/CD

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from google.cloud import storage
import joblib
from datetime import datetime


# MODIFIED: Wine dataset (178 samples, 13 features, 3 classes) instead of Iris
def download_data():
    from sklearn.datasets import load_wine
    wine = load_wine()
    features = pd.DataFrame(wine.data, columns=wine.feature_names)
    target = pd.Series(wine.target)
    return features, target


# MODIFIED: stratified split so all 3 wine classes are represented in train and test
def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


# MODIFIED: Gradient Boosting instead of Random Forest
def train_model(X_train, y_train):
    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
    model.fit(X_train, y_train)
    return model


# NEW: compute a fuller set of evaluation metrics
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
    }


# NEW: quality gate - returns True if model meets the accuracy threshold
def passes_quality_gate(metrics, threshold=ACCURACY_THRESHOLD):
    return metrics["accuracy"] >= threshold


# Retrieve the current model version from GCS
def get_model_version(bucket_name, version_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(version_file_name)
    if blob.exists():
        version = int(blob.download_as_text())
    else:
        version = 0
    return version


# Update the model version in GCS
def update_model_version(bucket_name, version_file_name, version):
    if not isinstance(version, int):
        raise ValueError("Version must be an integer")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(version_file_name)
        blob.upload_from_string(str(version))
        return True
    except Exception as e:
        print(f"Error updating model version: {e}")
        return False


# Ensure a folder exists in the bucket
def ensure_folder_exists(bucket, folder_name):
    blob = bucket.blob(f"{folder_name}/")
    if not blob.exists():
        blob.upload_from_string('')
        print(f"Created folder: {folder_name}")


# Save the model locally and to GCS
def save_model_to_gcs(model, bucket_name, blob_name):
    joblib.dump(model, "model.joblib")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    ensure_folder_exists(bucket, "trained_models")
    blob = bucket.blob(blob_name)
    blob.upload_from_filename('model.joblib')


def main():
    current_version = get_model_version(BUCKET_NAME, VERSION_FILE_NAME)
    new_version = current_version + 1

    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    for name, value in metrics.items():
        print(f"{name.capitalize()}: {value:.4f}")

    # NEW: block deployment if model quality is too low
    if not passes_quality_gate(metrics):
        print(f"Accuracy {metrics['accuracy']:.4f} is below threshold {ACCURACY_THRESHOLD}. "
              f"Model will NOT be saved or deployed.")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    blob_name = f"trained_models/model_v{new_version}_{timestamp}.joblib"
    save_model_to_gcs(model, BUCKET_NAME, blob_name)
    print(f"Model saved to gs://{BUCKET_NAME}/{blob_name}")

    if update_model_version(BUCKET_NAME, VERSION_FILE_NAME, new_version):
        print(f"Model version updated to {new_version}")
        print(f"MODEL_VERSION_OUTPUT: {new_version}")
    else:
        print("Failed to update model version")
        sys.exit(1)


if __name__ == "__main__":
    main()
