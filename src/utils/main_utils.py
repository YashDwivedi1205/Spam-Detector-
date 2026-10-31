import sys 
import pickle
import numpy as np
import os
import yaml
from src.exception import SpamDetectorException
from src.logger import logging
from scipy import sparse

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, 'rb') as yaml_data:
            return yaml.safe_load(yaml_data)

    except Exception as e:
        raise SpamDetectorException(e, sys)

def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok= True)
        with open(file_path,  "wb") as newfile:
            pickle.dump(obj, newfile)
    except Exception as e:
        raise SpamDetectorException(e, sys)

def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not Found: {file_path}.")
        with open(file_path, 'rb') as newfile:
            return pickle.load(newfile)

    except Exception as e:
        raise SpamDetectorException(e, sys)

def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok= True)
        with open(file_path, "wb") as newfile:
            np.save(newfile, array)
    except Exception as e:
        raise SpamDetectorException(e, sys)

def load_numpy_array_data(file_path: str) -> np.ndarray:
    try:
        with open(file_path, "rb") as newfile:
            return np.load(newfile)

    except Exception as e:
        raise SpamDetectorException(e, sys)

def save_sparse_matrix(file_path: str, matrix) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok= True)
        sparse.save_npz(file_path, matrix)
    except Exception as e:
        raise SpamDetectorException(e, sys)

def load_sparse_matrix(file_path: str):
    try:
        matrix = sparse.load_npz(file_path)
        return matrix
    except Exception as e:
        raise SpamDetectorException(e, sys)
    

# if __name__ == "__main__":
#     m = sparse.csr_matrix(np.array([[1, 0], [0, 2]]))
#     save_sparse_matrix("test_temp/m.npz", m)
#     print(load_sparse_matrix("test_temp/m.npz").toarray())