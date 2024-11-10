import os, sys
from datetime import datetime
from dataclasses import dataclass
from LoanPrediction.constant import constants
from LoanPrediction.logger.logging import logging
from LoanPrediction.exception.exception import LoanException

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
