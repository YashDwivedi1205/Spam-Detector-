import sys
import pandas as pd
from src.logger import logging
from src.exception import SpamDetectorException
from src.constant.training_pipeline import DATABASE_NAME
from src.configuration.mongodb_connections import MongoDBClient

class SpamData:
    def __init__(self):
        try:
            self.mongo_client = MongoDBClient()
        except Exception as e:
            raise SpamDetectorException(e, sys)

    def export_collection_as_dataframe(self, collection_name, database_name = None) -> pd.DataFrame:
        try:
            if database_name is None:
                database_name = DATABASE_NAME
            collect = self.mongo_client.client[database_name][collection_name]
            documents = collect.find({})
            df = pd.DataFrame(list(documents))
            if '_id' in df.columns:
                df.drop(columns=["_id"], axis= 1, inplace= True)
            return df
        except Exception as e:
            raise SpamDetectorException(e, sys)

if __name__ == '__main__':
    from src.constant.training_pipeline import COLLECTION_NAME
    spam_data = SpamData()
    df = spam_data.export_collection_as_dataframe(collection_name=COLLECTION_NAME)
    print(df.shape)
    print(df.head())