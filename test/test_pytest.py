import pytest
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from unittest.mock import patch, MagicMock
from src.train_and_save_model import download_data, preprocess_data, train_model
from src.train_and_save_model import evaluate_model, passes_quality_gate
from src.train_and_save_model import get_model_version, update_model_version
from src.train_and_save_model import ensure_folder_exists, save_model_to_gcs
from google.cloud import storage


# ----------------- Test Download ----------------- #
# MODIFIED: also checks Wine-specific shape (178 samples, 13 features, 3 classes)
def test_download_data():
    X, y = download_data()

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert not X.empty
    assert not y.empty
    assert X.shape[0] == y.shape[0]
    assert X.shape == (178, 13)         # Wine dataset dimensions
    assert y.nunique() == 3             # Three wine cultivars

# ----------------- Test Preprocess ----------------- #
# MODIFIED: also checks the stratified split keeps all classes in the test set
def test_preprocess_data():
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == y.shape[0]
    assert X_train.shape[1] == X.shape[1]
    assert set(y_test.unique()) == set(y.unique())  # Stratification keeps every class

# ----------------- Test Train model ----------------- #
# MODIFIED: uses a small slice of real Wine data (GradientBoosting needs 2+ classes)
def test_train_model():
    X, y = download_data()
    X_train, _, y_train, _ = preprocess_data(X, y)

    model = train_model(X_train, y_train)

    assert isinstance(model, GradientBoostingClassifier)
    assert hasattr(model, 'predict')

# ----------------- Test Evaluate model (NEW) ----------------- #
def test_evaluate_model():
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0

# ----------------- Test Quality gate (NEW) ----------------- #
def test_passes_quality_gate():
    assert passes_quality_gate({"accuracy": 0.90}, threshold=0.85) is True
    assert passes_quality_gate({"accuracy": 0.85}, threshold=0.85) is True
    assert passes_quality_gate({"accuracy": 0.70}, threshold=0.85) is False

# ----------------- Test Model meets threshold (NEW) ----------------- #
# Guards against a code change silently degrading model quality
def test_model_meets_accuracy_threshold():
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    assert passes_quality_gate(metrics)

# ----------------- Test Model versioning ----------------- #
def test_get_model_version():
    with patch('google.cloud.storage.Client') as mock_storage_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        bucket_name = "bucket-test"
        version_file_name = "version.txt"

        # Version file exists (FIXED: set return value BEFORE calling the function)
        mock_blob.exists.return_value = True
        mock_blob.download_as_text.return_value = '1'
        version = get_model_version(bucket_name, version_file_name)

        assert version == 1
        mock_storage_client.return_value.bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(version_file_name)
        mock_blob.download_as_text.assert_called_once()

        mock_storage_client.reset_mock()
        mock_bucket.reset_mock()
        mock_blob.reset_mock()

        # Version file does not exist
        mock_blob.exists.return_value = False
        version = get_model_version(bucket_name, version_file_name)

        assert version == 0
        mock_storage_client.return_value.bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(version_file_name)
        mock_blob.download_as_text.assert_not_called()

# ----------------- Test Update Model version ----------------- #
def test_update_model_version():
    with patch('google.cloud.storage.Client') as mock_storage_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        bucket_name = 'bucket-test'
        version_file_name = 'version.txt'
        new_version = 2

        result = update_model_version(bucket_name, version_file_name, new_version)
        assert result == True
        mock_storage_client.return_value.bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(version_file_name)
        mock_blob.upload_from_string.assert_called_once_with(str(new_version))

        mock_storage_client.reset_mock()
        mock_bucket.reset_mock()
        mock_blob.reset_mock()

        with pytest.raises(ValueError):
            update_model_version(bucket_name, version_file_name, 'invalid_version')

        mock_blob.upload_from_string.side_effect = Exception("Upload failed")
        result = update_model_version(bucket_name, version_file_name, new_version)
        assert result == False
        mock_storage_client.return_value.bucket.assert_called_once_with(bucket_name)
        mock_bucket.blob.assert_called_once_with(version_file_name)
        mock_blob.upload_from_string.assert_called_once_with(str(new_version))

# ----------------- Test Ensure Folder Exists ----------------- #
def test_ensure_folder_exists():
    with patch('google.cloud.storage.Client') as mock_storage_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        folder_name = "trained_models"

        mock_blob.exists.return_value = False
        ensure_folder_exists(mock_bucket, folder_name)
        mock_bucket.blob.assert_called_with(f"{folder_name}/")
        mock_blob.upload_from_string.assert_called_once_with('')

        mock_blob.reset_mock()

        mock_blob.exists.return_value = True
        ensure_folder_exists(mock_bucket, folder_name)
        mock_bucket.blob.assert_called_with(f"{folder_name}/")
        mock_blob.upload_from_string.assert_not_called()

# ----------------- Test Save model to GCS ----------------- #
# MODIFIED: uses GradientBoostingClassifier
def test_save_model_to_gcs():
    model = GradientBoostingClassifier()

    with patch('google.cloud.storage.Client') as mock_storage_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = False

        save_model_to_gcs(model, 'bucket-test', 'blob-test')

        mock_storage_client.assert_called_once()
        mock_storage_client.return_value.bucket.assert_called_once_with('bucket-test')
        assert mock_bucket.blob.call_count == 2
        mock_bucket.blob.assert_any_call('trained_models/')
        mock_bucket.blob.assert_any_call('blob-test')
        mock_blob.upload_from_filename.assert_called_once_with('model.joblib')
