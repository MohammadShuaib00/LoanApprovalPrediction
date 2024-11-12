import os, sys
from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str
    train_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    validation_status: str
    drift_report_file_path: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str


@dataclass
class DataTransformationArtifact:
    data_transformed_train_file_path: str
    data_transformed_test_file_path: str
