##
# @file errorHandler.py
# @brief Handles centralized logging of runtime exceptions.
#
# @details
# This utility module provides a single function to log errors, along with a contextual tag.
# It captures both a user-friendly message (via the MessageHandler) and a detailed stack trace.
#
# @date June 2025

##

import logging
import traceback
from utils.messageHandler import MessageHandler

##
# @var msg
# @brief Global message handler instance for retrieving error message templates.
##
msg = MessageHandler()

##
# @brief Logs an error with context and traceback to `log.txt`.
#
# @param error The exception instance that was caught.
# @param context A string describing where the error occurred (e.g., "Data Loading", "Model Training").
#
# @details
# This function performs two logging actions:
# - Logs a high-level error summary with context and a user-readable message
# - Logs the full traceback for debugging purposes in DEBUG level
#
# The error message uses a localized template from `messages.json` via the MessageHandler.
#
# @return None
##
def logError(error: Exception, context: str = "Unhandled"):
    logging.error(f"[{context}] {msg.get('error_generic')}: {error}")
    logging.debug(traceback.format_exc())