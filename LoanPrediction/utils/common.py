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
