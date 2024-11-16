import os
import sys
from LoanPrediction.logger.logging import logging
from LoanPrediction.exception.exception import LoanException
from LoanPrediction.constant.constants import SAVED_MODEL_DIR, MODEL_FILE_NAME


class LoanModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise LoanException(e, sys.exc_info())

    def predict(self, x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise LoanException(e, sys.exc_info())


class ModelResolver:
    def __init__(self, model_dir=SAVED_MODEL_DIR):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise LoanException(e, sys.exc_info())

    def get_best_model_path(self) -> str:
        try:
            timestamps = [
                int(timestamp)
                for timestamp in os.listdir(self.model_dir)
                if timestamp.isdigit()
            ]

            if not timestamps:
                raise ValueError(
                    "No valid timestamp directories found in the model directory."
                )

            latest_timestamp = max(timestamps)
            latest_model_path = os.path.join(
                self.model_dir, f"{latest_timestamp}", MODEL_FILE_NAME
            )

            return latest_model_path
        except Exception as e:
            raise LoanException(e, sys.exc_info())

    def is_model_exists(self) -> bool:
        try:
            if not os.path.exists(self.model_dir):
                return False

            timestamps = [
                timestamp
                for timestamp in os.listdir(self.model_dir)
                if timestamp.isdigit()
            ]
            if len(timestamps) == 0:
                return False

            latest_model_path = self.get_best_model_path()

            if not os.path.exists(latest_model_path):
                return False

            return True
        except Exception as e:
            raise LoanException(e, sys.exc_info())
