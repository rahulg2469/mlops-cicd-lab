# CI/CD for ML with GitHub Actions and Google Cloud Platform

**Author:** Rahul Gudivada
**Course:** MLOps, Northeastern University (Lab Assignment 1)
**Based on:** [raminmohammadi/MLOps - Labs/Github_Labs/Lab4](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab4)

This project is an automated CI/CD pipeline for a machine learning model. On every push to `main`, GitHub Actions tests the code, trains a model, checks that it meets a quality bar, versions it in Google Cloud Storage, and packages it as a Docker image in Google Artifact Registry.

---

## What I Changed Compared to the Original Lab

### Summary

| Area | Original Lab | This Version |
|---|---|---|
| Dataset | Iris (150 samples, 4 features) | **Wine** (178 samples, 13 features, 3 classes) |
| Model | `RandomForestClassifier` | **`GradientBoostingClassifier`** |
| Train/test split | Random 80/20 split | **Stratified** 80/20 split |
| Evaluation | Accuracy only | **Accuracy, precision, recall, F1** |
| Deployment safety | Every trained model is deployed | **Quality gate** blocks models below 85% accuracy |
| Tests | 7 tests | **10 tests** (3 new, 4 updated) |
| CI/CD workflow | Not included in the lab folder | **`.github/workflows/cicd.yml` written from scratch** |

### 1. Dataset: Iris → Wine
`download_data()` now loads the sklearn **Wine** dataset instead of Iris. It has 178 samples, 13 chemical features (alcohol, malic acid, flavanoids, color intensity, etc.), and 3 classes (wine cultivars). It's a harder, higher-dimensional problem than Iris, which most models classify almost perfectly.

### 2. Model: Random Forest → Gradient Boosting
`train_model()` now uses `GradientBoostingClassifier` (100 estimators, learning rate 0.1, max depth 3, random state 42). Gradient boosting builds trees sequentially, each one correcting the errors of the previous ones, whereas a random forest builds independent trees in parallel.

### 3. Stratified Train/Test Split
`preprocess_data()` now passes `stratify=y` to `train_test_split`, so the class proportions in the test set match the full dataset. This matters more with Wine, whose classes are slightly imbalanced (59 / 71 / 48 samples).

### 4. New: Full Evaluation Metrics
I added an `evaluate_model()` function that returns **accuracy, precision, recall, and F1** (weighted averages for the 3-class problem) instead of printing accuracy alone. All four are printed during every pipeline run.

### 5. New: Accuracy Quality Gate
I added a `passes_quality_gate()` function. If test accuracy falls below a threshold (default **0.85**, set with the `ACCURACY_THRESHOLD` environment variable), the script:
- prints a warning,
- exits with an error code,
- does **not** save the model or update the version.

Since the script fails, the GitHub Actions run fails too, and **no Docker image is built or pushed**. A bad model can't reach Artifact Registry. In the original lab, any trained model was deployed regardless of quality.

### 6. Tests: Updated and Expanded

**New tests:**
- `test_evaluate_model`: all four metrics are returned and fall between 0 and 1
- `test_passes_quality_gate`: the gate passes at or above the threshold and fails below it
- `test_model_meets_accuracy_threshold`: regression check that the real model still clears the gate, so a code change that degrades quality fails CI

**Updated tests:**
- `test_download_data`: now checks the Wine shape `(178, 13)` and 3 classes
- `test_preprocess_data`: now checks that stratification keeps every class in the test set
- `test_train_model`: the original used 5 rows of fake Iris data with a **single class**, which `GradientBoostingClassifier` rejects. It now trains on real Wine data and checks for the correct model type.
- `test_save_model_to_gcs`: uses `GradientBoostingClassifier`

### 7. CI/CD Workflow Added
The original Lab4 folder had no workflow file, so I wrote `.github/workflows/cicd.yml` myself. It runs tests, authenticates to GCP, trains and evaluates the model, versions it in GCS, and builds and pushes a Docker image tagged with the model version and `latest`.

---

## Results

| Metric | Score |
|---|---|
| Accuracy | **0.9444** |
| Precision (weighted) | 0.9466 |
| Recall (weighted) | 0.9444 |
| F1 (weighted) | 0.9443 |

- Passes the 0.85 quality gate
- All **10/10 tests pass**
- Each successful run publishes a new versioned Docker image (`model-image:<version>`) plus a `latest` tag to Artifact Registry
