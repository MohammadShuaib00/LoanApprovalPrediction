import os
import sys
import numpy as np
import pandas as pd
from LoanPrediction.logger.logging import logging
from LoanPrediction.exception.exception import LoanException
from LoanPrediction.entity.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)
from LoanPrediction.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from LoanPrediction.components.data_ingestion import DataIngestion
from LoanPrediction.components.data_validation import DataValidation
from LoanPrediction.components.data_transformation import DataTransformation
from LoanPrediction.components.model_trainer import ModelTrainer


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
            print(data_validation_artifact)
            return data_validation_artifact

        except Exception as e:
            raise LoanException(e, sys.exc_info())

    def started_data_transformation(
        self, data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        try:
            logging.info("Started Data Transformation Pipeline")
            data_transformation_config = DataTransformationConfig(
                self.training_pipeline_config
            )
            data_tansformation = DataTransformation(
                data_transformation_config, data_validation_artifact
            )
            data_transformation_artifact = (
                data_tansformation.initiate_data_transformation()
            )
            logging.info("Completed Data Transformation Pipeline")
            return data_transformation_artifact
        except Exception as e:
            raise LoanException(e, sys.exc_info())

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ):
        try:
            logging.info("Started Model Trainer Pipeline")
            model_trainer_config = ModelTrainerConfig(
                self.training_pipeline_config
            )
            model_trainer = ModelTrainer(
                model_trainer_config,data_transformation_artifact
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Model Trainer Pipeline Completed")
            return model_trainer_artifact
        except Exception as e:
            raise LoanException(e, sys.exc_info())
