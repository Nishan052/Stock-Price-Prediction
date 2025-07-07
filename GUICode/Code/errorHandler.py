##
# @file error_handler.py
# @brief Handles application-level error logging with contextual messages.
#
# @details
# This module configures a logging system to record errors into `app_errors.log`.
# It also loads structured error context messages from `messages.json` (if available),
# allowing the use of predefined human-readable logging templates.
#
# @section usage Usage
# Call `logError("context_key", exception, col="Open")` to log an error with context.
#
# @author Nishan Chandrashekar Poojary
# @author Sandesh Nonavinakere Sunil
# @date June 2025
##

import logging
import traceback
import os
import json

# Setup base paths for logging and messages file
baseDir = os.path.dirname(__file__)
logFile = os.path.join(baseDir, "app_errors.log")
messageFile = os.path.join(baseDir, "messages.json")

##
# @brief Configures Python's logging module to output to `app_errors.log`.
#
# @details
# The log level is set to ERROR. Each log entry includes timestamp, level, and message.
##
logging.basicConfig(
    filename=logFile,
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

##
# @brief Loads predefined error context templates from messages.json.
#
# @details
# These messages are expected under the key `log_contexts` in the JSON structure.
# If the file is missing or malformed, a warning is logged, and a fallback empty dictionary is used.
##
try:
    with open(messageFile, "r", encoding="utf-8") as f:
        MESSAGES = json.load(f)
except Exception as e:
    MESSAGES = {}
    logging.warning(f"Failed to load messages.json: {e}")

##
# @brief Logs detailed error information to a log file, using contextual messages.
#
# @details
# Fetches human-readable error context from `messages.json` and logs a formatted traceback.
#
# @param context_key str: Key to retrieve a formatted context string from `log_contexts` in `messages.json`.
# @param exception Exception: The exception object raised during code execution.
# @param kwargs dict: Optional arguments to format into the retrieved context string (e.g., col="Close").
#
# @return None
##
def logError(context_key, exception, **kwargs):
    # Use fallback if messages.json is missing or key is not found
    contextTemplate = MESSAGES.get("log_contexts", {}).get(context_key, context_key)
    context = contextTemplate.format(**kwargs) if kwargs else contextTemplate

    error_msg = f"[{context}] Error: {str(exception)}\n{traceback.format_exc()}"
    logging.error(error_msg)