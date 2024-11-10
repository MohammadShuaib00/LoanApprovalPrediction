import os, sys
import numpy as np
import pandas as pd
from LoanPrediction.logger.logging import logging
from LoanPrediction.exception.exception import LoanException
from LoanPrediction.entity.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig,
)
from LoanPrediction.entity.artifact_entity import DataIngestionArtifact
from LoanPrediction.components.data_ingestion import DataIngestion


class TrainingPipeline:
    def __init__(self):
        try:
            self.training_pipeline_config = TrainingPipelineConfig()
        except Exception as e:
            raise LoanException(e, sys.exc_info())

    def started_data_ingestion(self):
        try:
            print("Started Data Ingestion Pipeline")
            data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            print("Completed Data Ingstion Pipeline")
            return data_ingestion_artifact

        except Exception as e:
            raise LoanException(e, sys.exc_info())
