import os
import sys
from typing import List
import numpy as np
import pandas as pd
import pyaml
import pickle
import yaml
from LoanPrediction.exception.exception import LoanException
from LoanPrediction.logger.logging import logging


def read_yaml(file_path: str) -> dict:
    """Reads a yaml file"""
    try:
        with open(file_path, "rb") as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise LoanException(e, sys.exc_info())


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise LoanException(e, sys.exc_info())
    
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise LoanException(e, sys.exc_info())

