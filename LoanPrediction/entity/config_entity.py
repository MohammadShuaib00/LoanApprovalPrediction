import os, sys
from datetime import datetime
from dataclasses import dataclass
from LoanPrediction.constant import constants
from LoanPrediction.logger.logging import logging
from LoanPrediction.exception.exception import LoanException
from LoanPrediction.constant import constants

print(f"Artifact dir pipeline: {constants.ARTIFACT_DIR}")


class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        try:
            timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
            self.artifact_name = constants.ARTIFACT_DIR
            self.artifact_dir: str = os.path.join(self.artifact_name, timestamp)
            self.model_dir: str = os.path.join("final_model")
        except Exception as e:
            raise LoanException(e, sys.exc_info())


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.training_pipeline_config = training_pipeline_config
        self.data_ingestion_dir: str = os.path.join(
            self.training_pipeline_config.artifact_dir, constants.DATA_INGESTION_DIR
        )

        self.data_feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            constants.DATA_FEATURE_STORE_DIR,
            constants.DATA_FEATURE_STORE_FILE_PATH,
        )

        self.data_train_file_path: str = os.path.join(
            self.data_ingestion_dir,
            constants.DATA_INGESTED_STORE_DIR,
            constants.TRAIN_FILE_PATH,
        )
        self.data_test_file_path: str = os.path.join(
            self.data_ingestion_dir,
            constants.DATA_INGESTED_STORE_DIR,
            constants.TEST_FILE_PATH,
        )

        self.split_train_test_split_ratio: float = constants.TRAIN_TEST_SPLIT_RATIO
        self.collection_name: str = constants.COLLECTION_NAME
        self.database_name: str = constants.DATABASE_NAME


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.training_pipeline_config = training_pipeline_config
            self.validation_dir = os.path.join(
                training_pipeline_config.artifact_dir, constants.DATA_VALIDATION_DIR
            )

            self.data_valid_train_file_path: str = os.path.join(
                self.validation_dir, constants.DATA_VALID_DIR, constants.TRAIN_FILE_PATH
            )
            self.data_valid_test_file_path: str = os.path.join(
                self.validation_dir, constants.DATA_VALID_DIR, constants.TEST_FILE_PATH
            )

            self.data_invalid_train_file_path: str = os.path.join(
                self.validation_dir,
                constants.DATA_INVALID_DIR,
                constants.TRAIN_FILE_PATH,
            )
            self.data_invalid_test_file_path: str = os.path.join(
                self.validation_dir,
                constants.DATA_INVALID_DIR,
                constants.TEST_FILE_PATH,
            )
            self.data_report_file_path: bool = os.path.join(
                self.validation_dir,
                constants.DATA_DRIFT_REPORT_DIR,
                constants.DRIFT_REPORT_FILE_PATH,
            )
        except Exception as e:
            raise LoanException(e, sys.exc_info())


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.training_pipeline_config: str = training_pipeline_config

            self.data_transformation_dir: str = os.path.join(
                self.training_pipeline_config.artifact_dir,
                constants.DATA_TRANSFORMATION_DIR,
            )
            self.data_transformed_train_file_path: str = os.path.join(
                self.data_transformation_dir,
                constants.DATA_TRANSFORMATION_TRANSFORMED_DIR,
                constants.TRAIN_FILE_PATH,
            )
            self.data_transformation_test_file_path: str = os.path.join(
                self.data_transformation_dir,
                constants.DATA_TRANSFORMATION_TRANSFORMED_DIR,
                constants.TEST_FILE_PATH,
            )

        except Exception as e:
            raise LoanException(e, sys.exc_info())


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.training_pipeline_config: str = training_pipeline_config
            self.model_trainer_dir: str = os.path.join(
                self.training_pipeline_config.artifact_dir, constants.MODEL_TRAINER_DIR
            )
            self.model_trainer_model_dir: str = os.path.join(
                self.model_trainer_dir, constants.MODEL_TRAINER_MODEL_DIR
            )

            self.model_trainer_file_path: str = os.path.join(
                self.model_trainer_model_dir, constants.MODEL_TRAINER_FILE_PATH
            )
            self.model_trainer_expexted_score: float = (
                constants.MODEL_TRAINER_EXPECTED_SCORE
            )
            self.model_trainer_over_fitting_under_fitting_threshold: float = (
                constants.MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD
            )
        except Exception as e:
            raise LoanException(e, sys.exc_info())
