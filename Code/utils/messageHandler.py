##
# @file messageHandler.py
# @brief Provides a wrapper to load and access localized messages from a JSON file.
#
# @details
# This utility module is designed to support multi-language or message-based logging
# and UI strings throughout the forecasting app. Messages are stored in a centralized
# `messages.json` file for consistency and maintainability.
#
# @date June 2025

##

import json
import os

##
# @class MessageHandler
# @brief Loads and retrieves message strings from a JSON message dictionary.
#
# @details
# The class is useful for managing UI text, logging templates, or status messages.
# It supports default English messages and can be extended for multilingual support.
##
class MessageHandler:
    ##
    # @brief Initialize the message handler by loading the JSON file.
    #
    # @param lang Language code (default: "en"). Reserved for future use.
    # @param base_path Optional path override to locate the `messages.json` file.
    #
    # @details
    # If `base_path` is not provided, it defaults to the directory containing this script.
    # The JSON file is expected to be in the parent directory (`../messages.json`).
    ##
    def __init__(self, lang="en", base_path=None):
        base_path = base_path or os.path.dirname(__file__)
        messages_file = os.path.join(base_path, "..", "messages.json")
        with open(messages_file, "r", encoding="utf-8") as f:
            self.messages = json.load(f)

    ##
    # @brief Retrieve a message string by key.
    #
    # @param key The key corresponding to the message in the JSON file.
    # @return str The message string if found, otherwise a fallback string in [key] format.
    #
    # @details
    # This method ensures that missing keys are flagged clearly to the developer.
    ##
    def get(self, key):
        return self.messages.get(key, f"[{key}]")