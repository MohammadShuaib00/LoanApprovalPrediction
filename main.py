import os, sys
from LoanPrediction.logger.logging import logging
from LoanPrediction.exception.exception import LoanException
from LoanPrediction.components.data_ingestion import DataIngestion
from LoanPrediction.pipeline.training_pipeline import TrainingPipeline



if __name__ == "__main__":
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.started_data_ingestion()
    except Exception as e:
        raise LoanException(e,sys.exc_info())