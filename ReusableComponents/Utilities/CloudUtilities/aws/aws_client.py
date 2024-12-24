"""
aws_client.py
----------------

This module contains the AWSClient class which is used to interact with AWS services.



Author:
----------
Sourav Das

Date:
----------
2024-12-24
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Import dependencies
import boto3
import logging

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)