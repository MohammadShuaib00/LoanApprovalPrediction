import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from LoanPrediction.exception.exception import LoanException
from LoanPrediction.logger.logging import logging
from LoanPrediction.entity.artifact_entity import *
from LoanPrediction.entity.config_entity import *
from scipy import stats
from LoanPrediction.pipeline.preprocessing import preprocessing_pipeline
from LoanPrediction.utils.common import *


class DataTransformation:
    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            logging.info(
                f"DataTransformation initialized with config: {data_transformation_config}"
            )
        except Exception as e:
            raise LoanException(e, sys.exc_info())

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            logging.info(f"Reading data from {file_path}")
            data = pd.read_csv(file_path)
            logging.info(f"Data read successfully with shape {data.shape}")
            return data
        except Exception as e:
            raise LoanException(
                f"Error reading data from {file_path}: {e}", sys.exc_info()
            )

    def handle_outlier(
        self,
        df: pd.DataFrame,
        continuous_columns: list,
        remove_outliers=True,
        cap_outliers=False,
        z_threshold=3,
        iqr_factor=1.5,
    ) -> pd.DataFrame:
        try:
            logging.info("Handling outliers")
            outliers_summary = []

            # Z-score method for outlier detection
            for col in continuous_columns:
                logging.info(f"Checking outliers for column {col} using Z-score method")
                z_scores = stats.zscore(df[col])
                outliers_col_z = df[abs(z_scores) > z_threshold]
                outliers_summary.append(
                    {"column": col, "z_outliers": outliers_col_z.shape[0]}
                )

            # IQR Method for outlier detection
            for col in continuous_columns:
                logging.info(f"Checking outliers for column {col} using IQR method")
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                upper_bound = Q3 + iqr_factor * IQR
                lower_bound = Q1 - iqr_factor * IQR
                if remove_outliers:
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                elif cap_outliers:
                    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
                    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])

            # Log outlier summary
            if outliers_summary:
                logging.info(f"Outliers detected in columns: {outliers_summary}")

            return df

        except Exception as e:
            raise LoanException(f"Error handling outliers: {e}", sys.exc_info())

    @staticmethod
    def label_encode_target(series: pd.Series) -> pd.Series:
        try:
            if series.empty:
                raise ValueError("The input series is empty.")

            logging.info("Label encoding the target variable.")

            # Encode the target variable
            le = LabelEncoder()
            encoded_series = le.fit_transform(series)

            logging.info(f"Classes in the target variable: {list(le.classes_)}")
            return pd.Series(encoded_series, name=series.name)
        except Exception as e:
            raise LoanException(e, sys.exc_info())

    def initiate_data_transformation(self):
        try:
            logging.info("Starting data transformation process")

            # Create the data transformation directory if it doesn't exist
            os.makedirs(
                self.data_transformation_config.data_transformation_dir, exist_ok=True
            )

            # Read train and test data
            train_file_path = self.data_validation_artifact.valid_train_file_path
            test_file_path = self.data_validation_artifact.valid_test_file_path
            train_data = DataTransformation.read_data(train_file_path)
            test_data = DataTransformation.read_data(test_file_path)

            # Validate train and test data have the same columns
            assert list(train_data.columns) == list(
                test_data.columns
            ), "Mismatch in train and test columns."

            logging.info(f"Shape of train data: {train_data.shape}")
            logging.info(f"Shape of test data: {test_data.shape}")

            # Automatically detect continuous columns
            continuous_columns = train_data.select_dtypes(
                include=["float64", "int64"]
            ).columns.tolist()
            logging.info(f"Continuous columns identified: {continuous_columns}")

            # Handle outliers
            train_data = self.handle_outlier(
                train_data, continuous_columns, remove_outliers=True, cap_outliers=False
            )
            logging.info(
                f"Outliers removed from train_data, resulting shape: {train_data.shape}"
            )

            test_data = self.handle_outlier(
                test_data, continuous_columns, remove_outliers=True, cap_outliers=False
            )
            logging.info(
                f"Outliers removed from test_data, resulting shape: {test_data.shape}"
            )

            # Splitting features and target
            input_feature_train_df = train_data.drop(columns=["Loan_Status"], axis=1)
            target_feature_train_df = train_data["Loan_Status"]
            target_feature_train_df = DataTransformation.label_encode_target(
                target_feature_train_df
            )

            # Splitting the test data into features and target
            input_feature_test_df = test_data.drop(columns=["Loan_Status"], axis=1)
            target_feature_test_df = test_data["Loan_Status"]
            target_feature_test_df = DataTransformation.label_encode_target(
                target_feature_test_df
            )

            # Preprocessing and transformation
            preprocessor_object = preprocessing_pipeline.fit(input_feature_train_df)
            transformed_input_train_df = preprocessor_object.transform(
                input_feature_train_df
            )
            transformed_input_test_df = preprocessor_object.transform(
                input_feature_test_df
            )

            # Prepare numpy arrays
            train_arr = np.c_[
                transformed_input_train_df, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                transformed_input_test_df, np.array(target_feature_test_df)
            ]

            save_numpy_array_data(
                self.data_transformation_config.data_transformed_train_file_path,
                train_arr,
            )

            save_numpy_array_data(
                self.data_transformation_config.data_transformation_test_file_path,
                test_arr,
            )

            save_object(
                self.data_transformation_config.data_transformed_object_file_path,
                preprocessor_object,
            )

            logging.info("Storing the file into artifact dir")
            data_transformation_artifact = DataTransformationArtifact(
                data_transformed_object_file_path=self.data_transformation_config.data_transformed_object_file_path,
                data_transformed_train_file_path=self.data_transformation_config.data_transformed_train_file_path,
                data_transformed_test_file_path=self.data_transformation_config.data_transformation_test_file_path,
            )
            return data_transformation_artifact

        except Exception as e:
            raise LoanException(
                f"Error during data transformation: {e}", sys.exc_info()
            )
