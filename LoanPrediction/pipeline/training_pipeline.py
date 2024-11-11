import os
import sys
import numpy as np
import pandas as pd
from LoanPrediction.logger.logging import logging
from LoanPrediction.exception.exception import LoanException
from LoanPrediction.entity.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig,
    DataValidationConfig
)
from LoanPrediction.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from LoanPrediction.components.data_ingestion import DataIngestion
from LoanPrediction.components.data_validation import DataValidation


class TrainingPipeline:
    def __init__(self):
        try:
            self.training_pipeline_config = TrainingPipelineConfig()
        except Exception as e:
            raise LoanException(e, sys.exc_info())

    def started_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting Data Ingestion Pipeline")
            data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Completed Data Ingestion Pipeline")
            return data_ingestion_artifact

        except Exception as e:
            raise LoanException(e, sys.exc_info())

    def started_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            logging.info("Starting Data Validation Pipeline")
            data_validation_config = DataValidationConfig(self.training_pipeline_config)
            data_validation = DataValidation(
                data_validation_config, data_ingestion_artifact
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Completed Data Validation Pipeline")
            return data_validation_artifact

        except Exception as e:
            raise LoanException(e, sys.exc_info())
