# AutoML Bank Marketing Prediction Platform

An end-to-end machine learning platform for predicting whether a bank customer is likely to subscribe to a term deposit, built using the UCI Bank Marketing dataset.

This V1 project focuses on demonstrating the progression from classical machine learning fundamentals to a deployable ML application: preprocessing, class balancing, model comparison, hyperparameter tuning, experiment tracking, model selection, API inference, containerization, and a user-facing interface.

## Project Overview

The training pipeline evaluates multiple classification algorithms and automatically selects the best-performing model using F1-score on an untouched test set.

The selected champion model is exported and served through a FastAPI inference service, while a Streamlit application provides a simple interface for making predictions.

## Architecture

```text
UCI Bank Marketing Dataset
          |
          v
   Data Preparation
          |
          v
   Train / Test Split
          |
          +--------------------+
          |                    |
          v                    v
   Training Data          Untouched Test Data
          |
          v
    StandardScaler
          |
          v
        SMOTE
          |
          v
 RandomizedSearchCV
          |
          v
 +-------------------------+
 | Model Comparison        |
 | - Random Forest         |
 | - XGBoost               |
 | - Logistic Regression   |
 | - Decision Tree         |
 +-------------------------+
          |
          v
     Best Model
          |
          +---------> MLflow Tracking
          |
          v
   Champion Model
          |
          v
    best_model.pkl
          |
          v
      FastAPI
          |
          v
     Streamlit UI
```

## Machine Learning Workflow

1. Fetch the UCI Bank Marketing dataset.
2. Convert categorical variables using one-hot encoding.
3. Convert the target to binary labels (`yes = 1`, `no = 0`).
4. Split the data into training and test sets.
5. Standardize features using a scaler fitted only on the training data.
6. Apply SMOTE to the training data to address class imbalance.
7. Compare multiple classification algorithms.
8. Tune model hyperparameters using `RandomizedSearchCV` with 3-fold cross-validation.
9. Use F1-score as the model-selection metric.
10. Evaluate the tuned models on the untouched test set.
11. Track experiments, parameters, metrics, and models with MLflow.
12. Select the highest-F1 model as the champion.
13. Save the champion model, scaler, and feature-column information as a single artifact.
14. Serve predictions through FastAPI.
15. Provide a Streamlit frontend for interactive predictions.

## Models Evaluated

| Model | Tuning / Regularization |
|---|---|
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split` |
| XGBoost | `learning_rate`, `max_depth`, L1/L2 regularization |
| Logistic Regression | Elastic Net, `C`, `l1_ratio` |
| Decision Tree | `max_depth`, `min_samples_split`, `min_samples_leaf` |

SVM was initially considered but removed from V1 to reduce computational cost during model search.

## Why F1-Score?

The bank marketing target is imbalanced, so accuracy alone can be misleading. F1-score balances precision and recall and is therefore used as the primary metric for selecting the champion model.

The test set remains untouched during training and hyperparameter search and is used only for final model evaluation.

## Deployment

The application is split into two main services:

```text
Streamlit Frontend
       |
       v
FastAPI Inference API
       |
       v
Champion ML Model
```

### FastAPI endpoints

- `GET /health` — checks API/model availability.
- `POST /predict` — accepts customer information and returns a subscription prediction and probability when supported by the model.

### Docker

The project includes Docker configurations for containerized execution of the ML/API components and the Streamlit application.

## Currency Handling

The training dataset uses its original balance scale, while the Streamlit interface is designed for Indian users and accepts balance in INR.

Before inference, the application converts the INR input to the approximate scale used by the training data using a configurable conversion assumption. This conversion is an application-layer input transformation and is not part of the model training process.

## Project Structure

```text
bank-marketing-ml-platform-v1/
├── app/
│   └── streamlit_app.py
├── api/
│   └── main.py
├── src/
│   └── train.py
├── models/
│   └── best_model.pkl
├── Dockerfile.api
├── Dockerfile.app
├── requirements.txt
├── .gitignore
└── README.md
```

## Tech Stack

- Python
- pandas
- scikit-learn
- XGBoost
- imbalanced-learn / SMOTE
- MLflow
- FastAPI
- Pydantic
- Streamlit
- Docker
- Joblib

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/VENKATRAM027/bank-marketing-ml-platform-v1.git
cd bank-marketing-ml-platform-v1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the models

```bash
python src/train.py
```

This downloads the dataset, trains and tunes the configured models, logs experiments to MLflow, and creates:

```text
models/best_model.pkl
```

### 4. Start the FastAPI service

```bash
uvicorn api.main:app --reload
```

The API will be available locally on port `8000`.

### 5. Start the Streamlit application

In another terminal:

```bash
streamlit run app/streamlit_app.py
```

If the API is hosted somewhere else, set the `API_URL` environment variable accordingly.

## MLflow

MLflow is used to track:

- Model name
- Hyperparameters
- F1-score
- Trained model artifacts

Local MLflow tracking data is intentionally excluded from Git using `.gitignore`.

## V1 Design Decisions

This version intentionally prioritizes clarity and demonstration of core ML concepts over production-level complexity.

Key V1 decisions include:

- Classical ML models rather than deep learning.
- F1-score instead of accuracy as the primary metric.
- SMOTE for training-set class balancing.
- Randomized hyperparameter search to control compute requirements.
- MLflow for experiment tracking.
- FastAPI for model serving.
- Streamlit for a lightweight user interface.

## Future V2 Improvements

Planned improvements for V2 include:

- Modular ML pipeline and cleaner package structure.
- `ColumnTransformer` and unified preprocessing.
- SMOTE inside the cross-validation pipeline to avoid preprocessing leakage between CV folds.
- A single serialized preprocessing + model pipeline for inference.
- Automated tests.
- CI/CD with GitHub Actions.
- Better model/version management.
- Monitoring and retraining workflows.
- More production-oriented container and deployment architecture.

## Disclaimer

This project is an educational machine learning application built using the UCI Bank Marketing dataset. Predictions are for demonstration purposes and should not be used as real financial advice or as a production banking decision system.
