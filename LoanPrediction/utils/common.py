import os
import sys
from typing import List
import numpy as np
import pandas as pd
import pyaml
import pickle
import yaml
from LoanPrediction.exception.exception import LoanException
from LoanPrediction.logger.logging import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, accuracy_score


def read_yaml(file_path: str) -> dict:
    """Reads a yaml file"""
    try:
        with open(file_path, "rb") as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise LoanException(e, sys.exc_info())


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise LoanException(e, sys.exc_info())


def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise LoanException(e, sys.exc_info())


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        # Loop through all models and their corresponding hyperparameters
        for model_name, model in models.items():
            # Fetch hyperparameters specific to the model
            model_param = param.get(model_name, {})

            # Apply GridSearchCV to tune hyperparameters
            gs = GridSearchCV(model, model_param, cv=3, scoring="accuracy")
            gs.fit(X_train, y_train)

            # Get the best estimator (already fitted with best params)
            best_model = gs.best_estimator_

            # Predict on training and test data
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Calculate the accuracy score
            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)

            # Store the test score in the report
            report[model_name] = test_model_score

            # Log the scores for each model
            logging.info(
                f"{model_name} - Train Accuracy: {train_model_score}, Test Accuracy: {test_model_score}"
            )

        return report

    except Exception as e:
        raise LoanException(e, sys.exc_info())
