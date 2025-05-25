# error_handler.py

import logging
import traceback
import os

LOG_FILE = os.path.join(os.path.dirname(__file__), "app_errors.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_error(context, exception):
    """
    Logs detailed error information to app_errors.log

    Args:
        context (str): A string indicating where the error occurred.
        exception (Exception): The exception that was raised.
    """
    error_msg = f"[{context}] Error: {str(exception)}\n{traceback.format_exc()}"
    logging.error(error_msg)
