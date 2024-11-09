import os
import sys
import pymongo
import certifi
import requests
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# load the environment
load_dotenv()

MONGO_DB_URL = os.environ("MONGO_DB_URL")

ca = certifi.where()


class MongoDB:
    def __init__(self):
        try:
            self.mongodb_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
        except Exception as e:
            print(e)

    def convert_csv_into_json(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(data, drop=False, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            print(e)

    def insert_data_into_mongodb(self, records, database, collection):
        try:
            db = self.mongodb_client[database]
            col = db[collection]
            col.insert_many(records)
            return len(records)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    mong = MongoDB()
    file_path = "data\\loan_data.csv"
    database = "LoadDatabase"
    collection = "loandata"
    records = mong.convert_csv_into_json(file_path)
    print(records)
    mong.insert_data_into_mongodb(records,database,collection)