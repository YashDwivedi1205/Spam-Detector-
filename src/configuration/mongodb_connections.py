import os
import sys
import certifi
import pymongo
from dotenv import load_dotenv
from src.constant.training_pipeline import DATABASE_NAME
from src.exception import SpamDetectorException
from src.logger import logging

load_dotenv()

class MongoDBClient:
    client = None
    try:

        def __init__(self, database_name = DATABASE_NAME):
            if MongoDBClient.client is  None:
                mongo_url = os.getenv("MONGO_URL")
                if mongo_url is None:
                    raise Exception("Environment variable MONGO_URL is not set")
                MongoDBClient.client = pymongo.MongoClient(mongo_url, tlsCAFile = certifi.where())
                logging.info("MongoDB new client creation successful")
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("MongoDB connection successful")
    except Exception as e:
        raise SpamDetectorException(e, sys)
if __name__ == '__main__':
    mongo_client = MongoDBClient()
    print("Connection successful")
    print(mongo_client.client.list_database_names())
