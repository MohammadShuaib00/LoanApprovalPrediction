import os
import sys
from LoanPrediction.logger.logging import logging
from LoanPrediction.exception.exception import LoanException
from LoanPrediction.components.data_ingestion import DataIngestion
from LoanPrediction.components.data_validation import DataValidation
from LoanPrediction.components.data_transformation import DataTransformation
from LoanPrediction.pipeline.training_pipeline import TrainingPipeline


def run_pipeline():
    try:
        logging.info("Starting the Loan Prediction Training Pipeline")

        training_pipeline = TrainingPipeline()

        # Data Ingestion
        data_ingestion_artifact = training_pipeline.started_data_ingestion()
        logging.info("Data Ingestion Pipeline completed successfully.")

        # Data Validation
        data_validation_artifact = training_pipeline.started_data_validation(
            data_ingestion_artifact
        )
        logging.info("Data Validation Pipeline completed successfully.")

        data_transformation_artifact = training_pipeline.started_data_transformation(
            data_validation_artifact
        )
        logging.info("Data Transformation Pipeline Completed Successfully ")

        model_trainer_artifact = training_pipeline.started_model_trainer(
            data_transformation_artifact
        )

        logging.info("Model Trainer Pipeline Completed")

        training_pipeline.stated_model_evaluation(
            data_validation_artifact, model_trainer_artifact
        )

        logging.info("Loan Prediction Training Pipeline completed successfully.")

    except Exception as e:
        logging.error(f"Error occurred in Loan Prediction Training Pipeline: {str(e)}")
        raise LoanException(e, sys.exc_info())
