import os, sys
import logging
from datetime import datetime

LOG_FILE_PATH = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_path = os.path.join(os.getcwd(), "logs")

os.makedirs(log_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_path, LOG_FILE_PATH)

# SETUP UP LOGGING CONFIGURATION
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s - %(module)s - Line: %(lineno)d",
)
