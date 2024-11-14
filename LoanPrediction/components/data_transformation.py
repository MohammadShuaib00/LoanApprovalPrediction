import os, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from LoanPrediction.exception.exception import LoanException
from LoanPrediction.logger.logging import logging
from LoanPrediction.entity.artifact_entity import *
from LoanPrediction.entity.config_entity import *
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


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
            outliers_all = pd.DataFrame()

            # Z-score method for outlier detection
            for col in continuous_columns:
                logging.info(f"Checking outliers for column {col} using Z-score method")
                z_scores = stats.zscore(df[col])
                outliers_col_z = df[abs(z_scores) > z_threshold]
                outliers_all = pd.concat([outliers_all, outliers_col_z])

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

            # Log outliers detected
            if not outliers_all.empty:
                logging.info(f"Outliers detected: {outliers_all.shape[0]} rows")
                outliers_all.to_csv("outliers_log.csv", index=False)

            return df

        except Exception as e:
            raise LoanException(f"Error handling outliers: {e}", sys.exc_info())

    @staticmethod
    def label_encode_features(df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Automatically identify categorical features
            categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
            logging.info(f"Identified categorical features: {categorical_features}")

            # Encode categorical features in place
            for col in categorical_features:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])

            return df
        except Exception as e:
            raise LoanException(e, sys.exc_info())

    # Function to normalize data
    @staticmethod
    def normalize_data(df):
        # Initialize MinMaxScaler to scale features to the range [0, 1]
        scaler = MinMaxScaler()

        # Select only numeric columns for normalization
        numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns

        # Apply normalization on the numeric columns
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

        return df

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
            logging.info(f"Shape of train data: {train_data.shape}")
            logging.info(f"Shape of test data: {test_data.shape}")

            continuous_columns = [
                "Loan_Amount",
                "Income_Annum",
                "Residential_Assets_Value",
                "Commercial_Assets_Value",
                "Luxury_Assets_Value",
                "Bank_Asset_Value",
            ]

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

            # Label encode categorical features automatically detected in each dataset
            train_data = self.label_encode_features(train_data)
            test_data = self.label_encode_features(test_data)
            print(train_data.head())
            logging.info("Label encoding completed for train and test data.")

            # Splitting features and target
            input_feature_train_df = train_data.drop(columns=["Loan_Status"], axis=1)
            target_feature_train_df = train_data["Loan_Status"]
            input_feature_test_df = test_data.drop(columns=["Loan_Status"], axis=1)
            target_feature_test_df = test_data["Loan_Status"]

            # Normalize the features
            X_train = DataTransformation.normalize_data(input_feature_train_df)
            X_test = DataTransformation.normalize_data(input_feature_test_df)
            logging.info("Normalization Completed")

            # Concatenate the normalized features with the target column
            train_data_final = pd.concat([X_train, target_feature_train_df], axis=1)
            test_data_final = pd.concat([X_test, target_feature_test_df], axis=1)

            transformed_dir = os.path.dirname(
                self.data_transformation_config.data_transformed_train_file_path
            )
            os.makedirs(transformed_dir, exist_ok=True)
            train_data_final.to_csv(
                self.data_transformation_config.data_transformed_train_file_path,
                index=False,
                header=True,
            )
            test_data_final.to_csv(
                self.data_transformation_config.data_transformation_test_file_path,
                index=False,
                header=True,
            )

            logging.info("Storing the file into artifact dir")
            data_transformation_artifact = DataTransformationArtifact(
                data_transformed_train_file_path=self.data_transformation_config.data_transformed_train_file_path,
                data_transformed_test_file_path=self.data_transformation_config.data_transformation_test_file_path,
            )
            return data_transformation_artifact

        except Exception as e:
            raise LoanException(
                f"Error during data transformation: {e}", sys.exc_info()
            )
