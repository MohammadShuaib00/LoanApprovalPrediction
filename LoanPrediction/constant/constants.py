import os, sys
from typing import List
from datetime import datetime

"""
Defining common constant variable for training pipeline
"""
ARTIFACT_DIR: str = "artifact"
TRAIN_FILE_PATH: str = "train.csv"
TEST_FILE_PATH: str = "test.csv"
RESULT: str = "Loan_Status"
DATABASE_NAME: str = "LoanDatabase"
COLLECTION_NAME: str = "loan_data"
SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")
SAVE_MODEL_DIR:str = os.path.join("save_models")

"""
Data Ingestion related constant start with 
Data_INGESTION VAR NAME
"""
DATA_INGESTION_DIR: str = "data_ingestion"
DATA_FEATURE_STORE_DIR: str = "feature_store"
DATA_FEATURE_STORE_FILE_PATH: str = "raw.csv"
DATA_INGESTED_STORE_DIR: str = "ingested"
TRAIN_TEST_SPLIT_RATIO: float = 0.2

"""
Data Validation related constant start with Data_INGESTION VAR NAME
"""
DATA_VALIDATION_DIR: str = "data_validation"
DATA_VALID_DIR: str = "validated"
DATA_INVALID_DIR: str = "invalided"
DATA_DRIFT_REPORT_DIR: str = "validated_report"
DRIFT_REPORT_FILE_PATH: bool = "report.yaml"

"""
Data transformation related constant
"""
DATA_TRANSFORMATION_DIR: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_FILE_PATH: str = "preprocessor.pkl"

"""
Model Trainer related constant
"""
MODEL_TRAINER_DIR: str = "model_trainer"
MODEL_TRAINER_MODEL_DIR: str = "trained_mode"
MODEL_TRAINER_FILE_PATH: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float = 0.05
