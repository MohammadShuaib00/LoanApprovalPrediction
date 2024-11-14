import os
import sys
import numpy as np
import pandas as pd
from LoanPrediction.exception.exception import LoanException
from LoanPrediction.logger.logging import logging
from LoanPrediction.entity.config_entity import ModelTrainerConfig
from LoanPrediction.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact,
)
from LoanPrediction.utils.metric.get_classification_metric import (
    get_classification_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from LoanPrediction.utils.common import evaluate_models, save_object


class ModelTrainer:
    def __init__(
        self,
        model_training_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_training_config = model_training_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise LoanException(e, sys.exc_info())

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise LoanException(e, sys.exc_info())

    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            # Define models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Support Vector Machine": SVC(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
            }

            # Define hyperparameters for each model
            param_grid = {
                "Logistic Regression": {
                    "C": [0.1, 1, 10],
                    "penalty": ["l2"],
                    "solver": ["liblinear"],
                },
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10],
                },
                "AdaBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1, 1],
                },
                "Support Vector Machine": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"],
                },
                "K-Nearest Neighbors": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                },
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10],
                },
            }

            # Evaluate models using GridSearchCV
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=param_grid,
            )

            # Select the best model based on evaluation
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            print("Best Model Name is : ",best_model_name)

            # Train the best model
            best_model.fit(X_train, y_train)

            # Evaluate on train data
            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_score(
                y_true=y_train, y_pred=y_train_pred
            )

            # Evaluate on test data
            y_test_pred = best_model.predict(X_test)
            classification_test_metric = get_classification_score(
                y_true=y_test, y_pred=y_test_pred
            )

            # Log the metrics
            logging.info(f"Training Metrics: {classification_train_metric}")
            logging.info(f"Test Metrics: {classification_test_metric}")

            # Saving the model
            model_dir = self.model_training_config.model_trainer_model_dir
            os.makedirs(model_dir, exist_ok=True)  # Make sure the directory exists

            # Save the trained model
            save_object(
                self.model_training_config.model_trainer_file_path, obj=best_model
            )

            # Model trainer Artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_training_config.model_trainer_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric,
            )

            return model_trainer_artifact

        except Exception as e:
            raise LoanException(e, sys.exc_info())

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Creating the folder model trainer")
            # Ensure the folder for storing the model exists
            dir_path = self.model_training_config.model_trainer_dir
            os.makedirs(dir_path, exist_ok=True)

            # Load the transformed train and test data
            train_file_path = (
                self.data_transformation_artifact.data_transformed_train_file_path
            )
            test_file_path = (
                self.data_transformation_artifact.data_transformed_test_file_path
            )

            logging.info("Loading the training data and test data")
            train_data = ModelTrainer.read_data(train_file_path)
            test_data = ModelTrainer.read_data(test_file_path)

            # Prepare features and labels
            X_train = train_data.drop(columns=["Loan_Status"])
            y_train = train_data["Loan_Status"]
            X_test = test_data.drop(columns=["Loan_Status"])
            y_test = test_data["Loan_Status"]

            logging.info(
                f"X_train {X_train.shape} X_test {X_test.shape}, y_train {y_train.shape} and y_test {y_test.shape}"
            )

            # Train the model and get the best model and metrics
            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)

            return model_trainer_artifact

        except Exception as e:
            raise LoanException(e, sys.exc_info())
