import os
import sys
import pandas as pd
from LoanPrediction.logger.logging import logging
from LoanPrediction.exception.exception import LoanException
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
import joblib


# Custom transformer for label encoding
class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}

    def fit(self, X):
        try:
            for col in X.select_dtypes(include=["object"]).columns:
                le = LabelEncoder()
                le.fit(X[col])
                self.encoders[col] = le
            return self
        except Exception as e:
            raise LoanException(e, sys.exc_info())

    def transform(self, X):
        X = X.copy()
        for col, le in self.encoders.items():
            X[col] = le.transform(X[col])
        return X


# Custom transformer for normalization
class NormalizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X.select_dtypes(include=["float64", "int64"]))
        return self

    def transform(self, X):
        X = X.copy()
        numeric_columns = X.select_dtypes(include=["float64", "int64"]).columns
        X[numeric_columns] = self.scaler.transform(X[numeric_columns])
        return X


# Create the pipeline
preprocessing_pipeline = Pipeline(
    [
        ("label_encoder", LabelEncoderTransformer()),
        ("normalizer", NormalizeTransformer()),
    ]
)
