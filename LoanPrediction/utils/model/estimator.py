import os,sys
from LoanPrediction.logger.logging import logging
from LoanPrediction.exception.exception import LoanException


class NetworkModel:
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