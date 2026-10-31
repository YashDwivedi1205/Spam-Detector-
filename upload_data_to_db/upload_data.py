import pandas as pd
import json
import sys
import os
from src.logger import logging
from src.constant.training_pipeline import DATABASE_NAME, COLLECTION_NAME
from src.configuration.mongodb_connections import MongoDBClient
from src.exception import SpamDetectorException

def read_csv_data(file_path:str) -> pd.DataFrame:
    """Read CSV and return the dataframe with shape"""
    df = pd.read_csv(file_path)
    logging.info(f"Rows and columns: {df.shape}")
    return df

def clear_collection(collection) -> None:
    """Remove the data present in the collection before inserting"""
    Deleted_Result = collection.delete_many({})
    logging.info(f"Deleted {Deleted_Result.deleted_count} documents from '{collection.name}'")

def push_to_mongodb(df:pd.DataFrame, database_name: str, collection_name: str) -> int:
    """Inserting the data in to the MONGODB database"""
    mongo_client = MongoDBClient(database_name)
    collection = mongo_client.database[collection_name]

    clear_collection(collection)

    records = df.to_dict(orient="records")
    InsertedResult = collection.insert_many(records)
    inserted_count = len(InsertedResult.inserted_ids)
    logging.info(f"Inserted {inserted_count} documents into '{collection_name}'")
    return inserted_count

def verify_upload(df:pd.DataFrame, collection) -> None:
    """Compare CSV counts with what is actually stored in MongoDB"""
    db_count = collection.count_documents({})
    if db_count == len(df):
        logging.info(f"Upload verified: {db_count} documents")
    else:
        logging.warning(f"Count mismatch: CSV = {len(df)}, MongoDB = {db_count}")

def main():
    try:
        #CSV file path
        FILE_PATH = os.path.join("data", "spamham_final.csv")
        df = read_csv_data(FILE_PATH)
        push_to_mongodb(df, DATABASE_NAME, COLLECTION_NAME)
        collection = MongoDBClient(DATABASE_NAME).database[COLLECTION_NAME]
        verify_upload(df, collection)
    except Exception as e:
        raise SpamDetectorException(e, sys)

if __name__ == '__main__':
    main()
