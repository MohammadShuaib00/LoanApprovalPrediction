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
    ModelTrainerConfig,
    ModelEvaluationConfig,
)
from LoanPrediction.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)
from LoanPrediction.components.data_ingestion import DataIngestion
from LoanPrediction.components.data_validation import DataValidation
from LoanPrediction.components.data_transformation import DataTransformation
from LoanPrediction.components.model_trainer import ModelTrainer
from LoanPrediction.components.model_evaluation import ModelEvaluation


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
            logging.info(f"Data Validation Artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise LoanException(e, sys.exc_info())

    def started_data_transformation(
        self, data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        try:
            logging.info("Starting Data Transformation Pipeline")
            data_transformation_config = DataTransformationConfig(
                self.training_pipeline_config
            )
            data_transformation = DataTransformation(
                data_transformation_config, data_validation_artifact
            )
            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )
            logging.info("Completed Data Transformation Pipeline")
            return data_transformation_artifact
        except Exception as e:
            raise LoanException(e, sys.exc_info())

    def started_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            logging.info("Starting Model Trainer Pipeline")
            model_trainer_config = ModelTrainerConfig(self.training_pipeline_config)
            model_trainer = ModelTrainer(
                model_trainer_config, data_transformation_artifact
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Completed Model Trainer Pipeline")
            return model_trainer_artifact
        except Exception as e:
            raise LoanException(e, sys.exc_info())

    def started_model_evaluation(
        self,
        data_validation_artifact: DataValidationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting Model Evaluation Pipeline")
            model_evaluation_config = ModelEvaluationConfig(
                self.training_pipeline_config
            )
            model_evaluation = ModelEvaluation(
                model_evaluation_config,
                data_validation_artifact,
                model_trainer_artifact,
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            logging.info("Completed Model Evaluation Pipeline")
            return model_evaluation_artifact
        except Exception as e:
            raise LoanException(e, sys.exc_info())

    def run_pipeline(self):
        try:
            logging.info("Starting the Loan Prediction Training Pipeline")

            # Data Ingestion
            data_ingestion_artifact = self.started_data_ingestion()
            logging.info("Data Ingestion Pipeline completed successfully.")

            # Data Validation
            data_validation_artifact = self.started_data_validation(
                data_ingestion_artifact
            )
            logging.info("Data Validation Pipeline completed successfully.")

            # Data Transformation
            data_transformation_artifact = self.started_data_transformation(
                data_validation_artifact
            )
            logging.info("Data Transformation Pipeline completed successfully.")

            # Model Training
            model_trainer_artifact = self.started_model_trainer(
                data_transformation_artifact
            )
            logging.info("Model Trainer Pipeline completed successfully.")

            # Model Evaluation
            model_evaluation_artifact = self.started_model_evaluation(
                data_validation_artifact, model_trainer_artifact
            )
            logging.info(f"Model Evaluation Result: {model_evaluation_artifact}")

            logging.info("Loan Prediction Training Pipeline completed successfully.")

        except Exception as e:
            logging.error(
                f"Error occurred in Loan Prediction Training Pipeline: {str(e)}"
            )
            raise LoanException(e, sys.exc_info())
