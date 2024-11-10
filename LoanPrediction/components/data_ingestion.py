import os
import sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
import pymongo
import certifi
from LoanPrediction.exception.exception import LoanException
from LoanPrediction.logger.logging import logging
from LoanPrediction.entity.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig,
)
from LoanPrediction.entity.artifact_entity import DataIngestionArtifact
from LoanPrediction.utils.common import read_yaml
from LoanPrediction.constant import constants
from sklearn.model_selection import train_test_split

# Load environment variables
load_dotenv()

# Verify MongoDB URL from environment variables
MONGODB_URL = os.environ.get("MONGO_DB_URL")
if not MONGODB_URL:
    raise LoanException("MongoDB URL not found in environment variables", sys.exc_info())
logging.info(f"MONGODB_URL: {MONGODB_URL[:15]}...")  # Log first few characters for verification


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            # Initialize data ingestion configuration and MongoDB client
            self.data_ingestion_config = data_ingestion_config
            self.mongo_client = pymongo.MongoClient(MONGODB_URL)
            logging.info("MongoDB connection established.")
        except Exception as e:
            logging.error(f"Error in DataIngestion __init__: {e}")
            raise LoanException(e, sys.exc_info())

    def export_collection_as_dataframe(self):
        try:
            # Fetch database and collection names from config
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            collections = self.mongo_client[database_name][collection_name]

            # Log the database and collection being accessed
            logging.info(f"Using database: {database_name}, collection: {collection_name}")

            # Fetch records from MongoDB collection
            records = list(collections.find())
            logging.info(f"Number of records retrieved: {len(records)}")

            # Convert records to DataFrame
            if records:
                df = pd.DataFrame(records)
                logging.info(f"Data Shape from MongoDB: {df.shape}")
            else:
                logging.warning("No records found in the MongoDB collection.")
                raise LoanException("MongoDB collection is empty", sys.exc_info())

            # Drop "_id" and "Loan_ID" columns if they exist
            if "_id" in df.columns:
                df = df.drop(columns=["_id"], axis=1)
                logging.info("_id column dropped.")
            if "Loan_ID" in df.columns:
                df = df.drop(columns=["Loan_ID"], axis=1)
                logging.info("Loan_ID column dropped.")

            # Replace 'na' values with NaN
            df.replace({"na": np.nan}, inplace=True)

            # Return the cleaned DataFrame
            return df

        except Exception as e:
            logging.error(f"Error in export_collection_as_dataframe: {e}")
            raise LoanException(e, sys.exc_info())

    def export_data_feature_store(self, dataframe: pd.DataFrame):
        try:
            # Ensure the directory for storing feature data exists
            dir_path = os.path.dirname(self.data_ingestion_config.data_feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Export DataFrame to CSV file
            dataframe.to_csv(
                self.data_ingestion_config.data_feature_store_file_path,
                index=False,
                header=True,
            )
            logging.info(f"Data stored at: {self.data_ingestion_config.data_feature_store_file_path}")
            return dataframe
        except Exception as e:
            logging.error(f"Error in export_data_feature_store: {e}")
            raise LoanException(e, sys.exc_info())

    def splitting_data_into_train_test_file(self, dataframe: pd.DataFrame):
        try:
            dir_path = os.path.dirname(self.data_ingestion_config.data_train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            # Split data into training and test sets
            train_data, test_data = train_test_split(dataframe, test_size=0.3)
            train_data.to_csv(
                self.data_ingestion_config.data_train_file_path,
                index=False,
                header=True,
            )
            test_data.to_csv(
                self.data_ingestion_config.data_test_file_path,
                index=False,
                header=True,
            )
            logging.info("Exported train and test data to CSV files")
        except Exception as e:
            logging.error(f"Error in splitting_data_into_train_test_file: {e}")
            raise LoanException(e, sys.exc_info())

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Reading data from MongoDB...")
            dataframe = self.export_collection_as_dataframe()

            logging.info("Data ingestion completed successfully.")
            logging.info("Storing raw data into the feature store folder...")
            dataframe = self.export_data_feature_store(dataframe=dataframe)

            logging.info("Raw file stored successfully.")
            logging.info("Splitting the raw data into train and test file")
            self.splitting_data_into_train_test_file(dataframe)

            logging.info("Save into data ingestion artifact directory.")
            data_ingestion_artifact = DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.data_feature_store_file_path,
                train_file_path=self.data_ingestion_config.data_train_file_path,
                test_file_path=self.data_ingestion_config.data_test_file_path
            )
            return data_ingestion_artifact
        except Exception as e:
            logging.error(f"Error in initiate_data_ingestion: {e}")
            raise LoanException(e, sys.exc_info())


if __name__ == "__main__":
    try:
        # Initialize pipeline and configuration entities
        data_training_pipeline = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(data_training_pipeline)

        # Create DataIngestion object and initiate ingestion
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion.initiate_data_ingestion()

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise LoanException(e, sys.exc_info())
