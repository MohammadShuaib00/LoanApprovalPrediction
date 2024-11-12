import os
import sys
from LoanPrediction.logger.logging import logging
from LoanPrediction.exception.exception import LoanException
from LoanPrediction.components.data_ingestion import DataIngestion
from LoanPrediction.components.data_validation import DataValidation
from LoanPrediction.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
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

        training_pipeline.started_data_transformation(data_validation_artifact)
        logging.info("Data Transformation Pipeline Completed Successfully ")

        logging.info("Loan Prediction Training Pipeline completed successfully.")

    except Exception as e:
        logging.error(f"Error occurred in Loan Prediction Training Pipeline: {str(e)}")
        raise LoanException(e, sys.exc_info())
