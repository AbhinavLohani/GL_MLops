"""
Model Training script for the "Visit with Us" Wellness Tourism Package project.

Loads the train/test splits produced by the data-prep job, builds a
preprocessing + XGBoost pipeline, tunes it with GridSearchCV, logs
parameters and metrics to MLflow, evaluates the tuned model, and saves the
best pipeline so the workflow can commit it to the repository.
"""

import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

MODEL_DIR = os.path.join("tourism_project", "deployment")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")

PARAM_GRID = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}


def load_splits():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")
    return Xtrain, Xtest, ytrain, ytest


def build_pipeline(Xtrain: pd.DataFrame) -> "make_pipeline":
    categorical_cols = Xtrain.select_dtypes(include="object").columns.tolist()
    numeric_cols = [col for col in Xtrain.columns if col not in categorical_cols]

    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        (StandardScaler(), numeric_cols),
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
    )

    return make_pipeline(preprocessor, model)


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    Xtrain, Xtest, ytrain, ytest = load_splits()
    pipeline = build_pipeline(Xtrain)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("tourism-wellness-package")

    with mlflow.start_run():
        grid_search = GridSearchCV(
            pipeline,
            param_grid=PARAM_GRID,
            scoring="f1",
            cv=3,
            n_jobs=-1,
        )
        grid_search.fit(Xtrain, ytrain)

        best_model = grid_search.best_estimator_

        # Log every tuned hyperparameter.
        mlflow.log_params(grid_search.best_params_)

        ypred = best_model.predict(Xtest)

        metrics = {
            "accuracy": accuracy_score(ytest, ypred),
            "precision": precision_score(ytest, ypred),
            "recall": recall_score(ytest, ypred),
            "f1_score": f1_score(ytest, ypred),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        print("Best parameters:", grid_search.best_params_)
        print("-" * 60)
        print("Test set performance:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
        print("-" * 60)
        print(classification_report(ytest, ypred))

    joblib.dump(best_model, MODEL_PATH)
    print(f"Best model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
