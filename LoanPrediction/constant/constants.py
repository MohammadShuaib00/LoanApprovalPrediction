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

"""
Data Ingestion related constant start with 
Data_INGESTION VAR NAME
"""
DATA_INGESTION_DIR: str = "data_ingestion"
DATA_FEATURE_STORE_DIR: str = "feature_store"
DATA_FEATURE_STORE_FILE_PATH: str = "raw.csv"
DATA_INGESTED_STORE_DIR: str = "ingested"
TRAIN_TEST_SPLIT_RATIO: float = 0.2
