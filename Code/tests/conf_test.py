##
# @file conftest.py
# @brief Pytest configuration file to set up environment and path resolution.
#
# @details
# This file adds the `Code` directory to `sys.path`, allowing test modules
# to import application logic without relative import errors.
# It should be placed in the root `tests/` folder and is automatically discovered by pytest.
#
# @date June 2025

##

import os
import sys

##
# @brief Add the 'Code' directory to the Python module path.
#
# @details
# This ensures that test scripts can import modules such as `main`, `data_handler`, etc.
# without needing to adjust their own relative paths manually.
##
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Code"))
)