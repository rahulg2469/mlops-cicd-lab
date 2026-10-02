import pytest
from unittest.mock import MagicMock, patch
from sklearn.ensemble import RandomForestClassifier

from src.train_and_save_model import (
    download_data,
    preprocess_data,
    train_model,
    get_model_version,
    update_model_version,
    save_model_to_gcs,
)

MODULE = "src.train_and_save_model"


# ---------- Data + model tests (no GCP needed) ----------

def test_download_data():
    X, y = download_data()
    assert X.shape == (150, 4)
    assert len(y) == 150


def test_preprocess_data():
    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    assert len(X_train) == 120
    assert len(X_test) == 30
    assert len(y_train) == 120
    assert len(y_test) == 30


def test_train_model():
    X, y = download_data()
    X_train, X_test, y_train, _ = preprocess_data(X, y)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    assert len(model.predict(X_test)) == len(X_test)


# ---------- GCP tests (mocked with MagicMock + patch) ----------

@patch(f"{MODULE}.storage.Client")
def test_get_model_version_exists(mock_client):
    mock_blob = MagicMock()
    mock_blob.exists.return_value = True
    mock_blob.download_as_text.return_value = "3"
    mock_client.return_value.bucket.return_value.blob.return_value = mock_blob

    assert get_model_version("test-bucket", "model_version.txt") == 3


@patch(f"{MODULE}.storage.Client")
def test_get_model_version_not_exists(mock_client):
    mock_blob = MagicMock()
    mock_blob.exists.return_value = False
    mock_client.return_value.bucket.return_value.blob.return_value = mock_blob

    assert get_model_version("test-bucket", "model_version.txt") == 0


@patch(f"{MODULE}.storage.Client")
def test_update_model_version_success(mock_client):
    mock_blob = MagicMock()
    mock_client.return_value.bucket.return_value.blob.return_value = mock_blob

    assert update_model_version("test-bucket", "model_version.txt", 5) is True
    mock_blob.upload_from_string.assert_called_once_with("5")


def test_update_model_version_invalid():
    with pytest.raises(ValueError):
        update_model_version("test-bucket", "model_version.txt", "five")


@patch(f"{MODULE}.storage.Client")
def test_update_model_version_failure(mock_client):
    mock_client.side_effect = Exception("GCP down")
    assert update_model_version("test-bucket", "model_version.txt", 1) is False


@patch(f"{MODULE}.joblib.dump")
@patch(f"{MODULE}.storage.Client")
def test_save_model_to_gcs(mock_client, mock_dump):
    mock_blob = MagicMock()
    mock_client.return_value.bucket.return_value.blob.return_value = mock_blob

    save_model_to_gcs(MagicMock(), "test-bucket", "trained_models/model_v1.joblib")

    mock_dump.assert_called_once()
    mock_blob.upload_from_filename.assert_called_once_with("model.joblib")
