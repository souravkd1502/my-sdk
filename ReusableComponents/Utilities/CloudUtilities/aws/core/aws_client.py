"""
aws_client.py
----------------
This module contains the AWSClient class which is used to interact with AWS services.



Author:
----------
Sourav Das

Date:
----------
24-01-2025
"""

# Adding directories to system path to allow importing custom modules
import sys

import boto3.session

sys.path.append("./")
sys.path.append("../")

# Import dependencies
import boto3
import logging
from dotenv import load_dotenv

from botocore.exceptions import NoCredentialsError, PartialCredentialsError

from typing import Any, Dict

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Load environment variables
load_dotenv(override=True)


class AWSClient:
    def __init__(
        self,
        service_name: str = None,
        region_name: str = None,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        aws_session_token: str = None,
    ) -> None:
        """
        Initialize the AWSClient.

        Args:
        ----------
        service_name (str): The name of the AWS service.
        region_name (str): The name of the AWS region.
        aws_access_key_id (str): The AWS access key ID.
        aws_secret_access_key (str): The AWS secret access key.
        aws_session_token (str): The AWS session token.

        Raises:
        -------
        ValueError: If invalid AWS credentials are provided.
        """
        _logger.info("Initializing AWSClient...")

        self.service_name = service_name
        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.client = None

        try:
            # Validate credentials and initialize boto3 session
            self.session = boto3.Session(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_session_token=self.aws_session_token,
                region_name=self.region_name,
            )
            _logger.info("AWS session successfully created.")
        except (NoCredentialsError, PartialCredentialsError) as e:
            # Log and raise an exception if invalid credentials are provided
            _logger.error("AWS credentials are invalid or incomplete.")
            raise ValueError("Invalid AWS credentials provided.") from e

        # Initialize service client if service_name is provided
        if self.service_name:
            self.initialize_client()

    def initialize_client(self) -> None:
        """
        Initialize the boto3 client for the specified AWS service.

        Raises:
            Exception: If the client initialization fails.
        """
        try:
            # Initialize the boto3 client using the provided service name
            self.client = self.session.client(self.service_name)
            _logger.info(f"{self.service_name} client successfully initialized.")
        except Exception as e:
            # Log and raise an exception if the client initialization fails
            _logger.error(f"Failed to initialize {self.service_name} client: {str(e)}")
            raise

    def get_client(self):
        """
        Retrieve the boto3 client for the AWS service.

        This method returns the initialized boto3 client for the specified AWS service.
        If the client is not initialized, it raises a ValueError.

        Returns
        -------
        boto3.client
            The initialized AWS service client.

        Raises
        ------
        ValueError
            If the AWS client is not initialized.
        """
        # Check if the client is initialized
        if not self.client:
            # Raise an exception if the client is not initialized
            raise ValueError("AWS client is not initialized. Provide a service name during initialization.")
        
        # Return the initialized client
        return self.client
