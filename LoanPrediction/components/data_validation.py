import os
import sys
import numpy as np
import pandas as pd
from LoanPrediction.entity.config_entity import (
    DataValidationConfig,
    TrainingPipelineConfig,
)
from LoanPrediction.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from LoanPrediction.constant import constants
from LoanPrediction.logger.logging import logging
from LoanPrediction.exception.exception import LoanException
from scipy.stats import ks_2samp
from LoanPrediction.utils.common import read_yaml, write_yaml_file


class DataValidation:
    def __init__(
        self,
        data_validation_config: DataValidationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self._schema_config = read_yaml(constants.SCHEMA_FILE_PATH)
        except Exception as e:
            raise LoanException(e, sys.exc_info())

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise LoanException(e, sys.exc_info())

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            required_columns = len(self._schema_config["COLUMNS"])
            actual_columns = len(dataframe.columns)
            logging.info(f"Required number of columns: {required_columns}")
            logging.info(f"Data frame has columns: {actual_columns}")
            return actual_columns == required_columns
        except Exception as e:
            raise LoanException(e, sys.exc_info())

    def detect_data_drift(
        self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05
    ) -> bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                ks_stat = ks_2samp(d1, d2)
                drift_detected = ks_stat.pvalue < threshold
                report[column] = {
                    "p_value": float(ks_stat.pvalue),
                    "drift_status": drift_detected,
                }
                if drift_detected:
                    status = False

            drift_report_file_path = self.data_validation_config.data_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)

            return status

        except Exception as e:
            raise LoanException(e, sys.exc_info())

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Reading data files
            logging.info("Reading the dataframes...")
            train_dataframe = self.read_data(train_file_path)
            test_dataframe = self.read_data(test_file_path)
            logging.info("Data read successfully.")

            # Column validation
            column_validation_status = True
            error_messages = []

            if not self.validate_number_of_columns(train_dataframe):
                error_messages.append(
                    "Train dataframe does not contain all required columns."
                )
                column_validation_status = False
            if not self.validate_number_of_columns(test_dataframe):
                error_messages.append(
                    "Test dataframe does not contain all required columns."
                )
                column_validation_status = False

            # Data drift detection
            data_drift_status = self.detect_data_drift(
                base_df=train_dataframe, current_df=test_dataframe
            )

            # Saving validated data files
            dir_path = os.path.dirname(
                self.data_validation_config.data_valid_train_file_path
            )
            os.makedirs(dir_path, exist_ok=True)

            train_dataframe.to_csv(
                self.data_validation_config.data_valid_train_file_path,
                index=False,
                header=True,
            )
            test_dataframe.to_csv(
                self.data_validation_config.data_valid_test_file_path,
                index=False,
                header=True,
            )
            logging.info("Data validation completed successfully.")

            # Consolidate validation status
            validation_status = column_validation_status and data_drift_status
            if error_messages:
                for msg in error_messages:
                    logging.error(msg)

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.data_valid_train_file_path,
                valid_test_file_path=self.data_validation_config.data_valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.data_report_file_path,
            )

            return data_validation_artifact

        except Exception as e:
            raise LoanException(e, sys.exc_info())
