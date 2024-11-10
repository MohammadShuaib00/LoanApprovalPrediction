import os
import sys
import pymongo
import certifi
import requests
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()

MONGO_URL = os.environ.get("MONGO_DB_URL")  # Corrected this line

if not MONGO_URL:
    print("Error: MONGO_DB_URL environment variable not set.")
    sys.exit(1)

ca = certifi.where()

class MongoDB:
    def __init__(self):
        try:
            self.mongodb_client = pymongo.MongoClient(MONGO_URL, tlsCAFile=ca)
            print("MongoDB connection established.")
        except Exception as e:
            print(f"Error while connecting to MongoDB: {e}")
            sys.exit(1)

    def convert_csv_into_json(self, file_path):
        try:
            # Read CSV file into DataFrame
            data = pd.read_csv(file_path)
            # Reset the index and convert the dataframe to JSON-like format
            data.reset_index(drop=True, inplace=True)
            # Convert to list of dictionaries
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            print(f"Error while converting CSV to JSON: {e}")
            return []

    def insert_data_into_mongodb(self, records, database, collection):
        try:
            # Connect to the database and collection
            db = self.mongodb_client[database]
            col = db[collection]
            # Insert records into MongoDB
            if records:
                col.insert_many(records)
                print(f"Inserted {len(records)} records into {collection} collection.")
            else:
                print("No records to insert.")
        except Exception as e:
            print(f"Error while inserting data into MongoDB: {e}")

if __name__ == "__main__":
    mong = MongoDB()

    # Path to your CSV file
    file_path = "data\\loan_data.csv"
    # MongoDB database and collection
    database = "LoanDatabase"  # Make sure database name is correct
    collection = "loan_data"  # Ensure collection name is correct

    # Convert CSV to JSON-like records
    records = mong.convert_csv_into_json(file_path)
    print(f"Converted {len(records)} records.")

    # Insert records into MongoDB
    mong.insert_data_into_mongodb(records, database, collection)
