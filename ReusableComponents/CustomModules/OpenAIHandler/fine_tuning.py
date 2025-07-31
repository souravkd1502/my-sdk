"""
text_handler.py
---------------
This module contains functions that handle chat completion using OpenAI Chat Completion API. It includes the following:
- chat_completion: This function takes a prompt and returns a completion of the prompt using OpenAI's Chat Completion API.
- Handles Conversational history: This function takes a prompt and a list of previous messages and returns a completion of the prompt using OpenAI's Chat Completion API.
- Handles Function calling: This function takes a prompt and a list of previous messages and returns a completion of the prompt using OpenAI's Chat Completion API.
- Handles Structured responses: This function takes a prompt and a list of previous messages and returns a completion of the prompt using OpenAI's Chat Completion API.

Requirements:
--------------
- openai==1.59.3
- python-dotenv==1.0.1

Author:
-------
Sourav Das

Date:
-----
2025-10-01
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Importing necessary libraries and modules
import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

from typing import List, Dict, Any, Optional

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Load environment variables
load_dotenv(override=True)