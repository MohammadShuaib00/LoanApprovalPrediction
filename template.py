import os, sys
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")

project_name = "LoanPrediction"

list_of_files = [
    ".github/workflows/main.yml",
    "data/data.csv",
    # Cloud-related operations
    f"{project_name}/cloud/__init__.py",
    f"{project_name}/cloud/s3_operations.py",
    f"{project_name}/cloud/gcp_operations.py",
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_pusher.py",
    # Exception folder and file
    f"{project_name}/exception/__init__.py",
    f"{project_name}/exception/exception.py",
    # Logger folder and logging
    f"{project_name}/logger/__init__.py",
    f"{project_name}/logger/logging.py",
    # Pipeline scripts
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/pipeline/training_pipeline.py",
    f"{project_name}/pipeline/prediction_pipeline.py",
    # Configuration
    f"{project_name}/config/config.yaml",
    # Utility files
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/common.py",
    # Constant folder and file
    f"{project_name}/constant/__init__.py",
    f"{project_name}/constant/constants.py",
    f"templates/index.html",
    f"static/style.css",
    # Docker-related files
    "Dockerfile",
    # Environment Variables
    ".env",
    # Setup files
    "setup.py",
    # Git ignore
    ".gitignore",
    # ReadMe
    "README.md",
    # Notebooks
    "notebooks/experiment.ipynb",
    "notebooks/prediction_pipeline.ipynb",
]


for file in list_of_files:
    filepath = Path(file)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Directory created: {filedir}")
    if (not filepath.exists()) or (filepath.stat().st_size == 0):
        with open(filepath, "w") as file:
            pass
            logging.info(f"File created: {filepath}")
    else:
        logging.info(f"File is already exits{filepath}")
