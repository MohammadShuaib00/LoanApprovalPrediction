import os
import sys
from typing import List
import numpy as np
import pandas as pd
import pyaml
import pickle
import yaml
from LoanPrediction.exception.exception import LoanException


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
