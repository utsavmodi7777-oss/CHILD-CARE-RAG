
import logging
import warnings
import os

def suppress_warnings():
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

def setup_clean_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
