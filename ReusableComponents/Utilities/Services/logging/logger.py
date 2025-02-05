"""
logger.py
---------

Description:
------------
This module provides a class for logging and metric tracking. It supports logging to console and file, 
exporting logs to CSV, uploading logs to Azure Blob Storage or AWS S3, recording metrics, 
and visualizing metrics using matplotlib.

Usage:
------
1. Create an instance of EnhancedLogger with a name and an optional log file path.
2. Log messages using the log method with a log level ('info', 'warning', 'error').
3. Export logs to CSV using the export_logs_to_csv method.
4. Upload logs to Azure Blob Storage using the upload_to_azure_blob method.
5. Upload logs to AWS S3 using the upload_to_s3 method.

Requirements:
-------------
- boto3==1.36.2
- matplotlib==3.10.0
- azure-core==1.32.0
- azure-storage-blob==12.24.0

Author:
-------
Sourav Das

Version:
--------
1.0

Date Created:
-------------
21-01-2025

Updated on:
-----------
05-02-2025
"""

# Adding directories to system path to allow importing custom modules

import sys

sys.path.append("./")
sys.path.append("../")

# Import dependencies
import os
import re
import csv
import json
import boto3
import logging
import logging.handlers
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient

from typing import Optional, Literal, Dict, List


class EnhancedLogger:
    """
    A logger with support for console and file logging, log file management,
    CSV export, and cloud storage uploads (Azure Blob Storage and AWS S3).
    """

    FILE_NOT_FOUND_ERROR_MSG = "Log file not found or not specified."

    def __init__(
        self, name: str, log_file: Optional[str] = None, service: Optional[str] = None
    ):
        """
        Initialize the EnhancedLogger.

        Args:
            name (str): Name of the logger.
            log_file (Optional[str]): Path to the log file (if logging to a file is needed).
            service (Optional[str]): Service name for contextual logging.

        Example:
        --------
        ```python
            logger = EnhancedLogger("MyLogger", "app.log", service="MyService")
            logger.log("info", "Application started.") # Logs an info message
            logger.log("error", "An error occurred.") # Logs an error message
        ```
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.log_file = log_file
        if log_file:
            self._setup_file_handler(log_file, formatter)

        self.service = service or "DefaultService"

    def _setup_file_handler(self, log_file: str, formatter: logging.Formatter):
        """
        Sets up the file handler, ensuring the log directory exists.

        Args:
            log_file (str): Path to the log file.
            formatter (logging.Formatter): Log message formatter.
        """
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log(
        self,
        level: str,
        message: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        organization_id: Optional[str] = None,
    ):
        """
        Logs a message at the specified logging level.

        Args:
            level (str): Logging level ("debug", "info", "warning", "error", "critical", "exception").
            message (str): Message to log.
            user_id (Optional[str]): User ID associated with the log entry.
            session_id (Optional[str]): Session ID associated with the log entry.
            organization_id (Optional[str]): Organization ID associated with the log entry.

        Example:
        --------
        ```python
            logger.log("info", "Application started.") # Logs an info message
            logger.log("error", "An error occurred.") # Logs an error message
            logger.log("warning", "Warning message.", user_id="user123", session_id="session456", organization_id="org789") # Logs a warning message
        ```
        """
        valid_levels = {"debug", "info", "warning", "error", "critical", "exception"}
        if level.lower() not in valid_levels:
            raise ValueError(
                f"Invalid log level: {level}. Valid levels are {valid_levels}."
            )

        log_entry = {
            "service": self.service,
            "user": user_id,
            "session": session_id,
            "organization": organization_id,
            "message": message,
            "level": level,
        }
        getattr(self.logger, level.lower())(json.dumps(log_entry))

    def export_logs_to_csv(self, csv_file: str):
        """
        Exports logs to a CSV file.

        Args:
            csv_file (str): Path to the CSV file.

        Raises:
            FileNotFoundError: If the log file does not exist.

        Example:
        --------
        ```python
            logger.export_logs_to_csv("logs.csv") # Exports logs to a CSV file
        ```
        """
        if not self.log_file or not os.path.exists(self.log_file):
            raise FileNotFoundError(self.FILE_NOT_FOUND_ERROR_MSG)

        with open(self.log_file, "r") as log_file, open(
            csv_file, "w", newline=""
        ) as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Timestamp", "Level", "Message"])

            for line in log_file:
                parts = line.strip().split(" - ", 2)
                if len(parts) == 3:
                    csv_writer.writerow(parts)

    @staticmethod
    def clear_log_file(log_file: str):
        """
        Clears all content in the specified log file.

        Args:
            log_file (str): Path to the log file.

        Example:
        --------
        ```python
            EnhancedLogger.clear_log_file("app.log") # Clears the contents of the log file
        ```
        """
        try:
            with open(log_file, "w"):
                pass  # Clears the file contents
        except Exception as e:
            logging.error(f"Failed to clear log file: {e}")

    def _generate_cloud_log_path(self) -> str:
        """
        Generates a structured cloud storage path for log files.

        Returns:
            str: Generated cloud storage path.
        """
        current_time = datetime.now()
        return (
            f"{self.service}/"
            f"{current_time.year}/{current_time.month:02}/{current_time.day:02}/"
            f"{current_time.timestamp()}.log"
        )

    def upload_to_azure_blob(self, connection_string: str, container_name: str):
        """
        Uploads the log file to Azure Blob Storage with a structured path.

        Args:
            connection_string (str): Azure Blob Storage connection string.
            container_name (str): Target blob container.
            user_id (str): User identifier.
            service_id (str): Service identifier.

        Raises:
            FileNotFoundError: If the log file is not found.

        Example:
        --------
        ```python
            logger.upload_to_azure_blob("<connection-string>", "test-logs") # Uploads logs to Azure Blob Storage
        ```
        """
        if not self.log_file or not os.path.exists(self.log_file):
            raise FileNotFoundError(self.FILE_NOT_FOUND_ERROR_MSG)

        blob_path = self._generate_cloud_log_path()
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=blob_path
        )

        with open(self.log_file, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

    def upload_to_s3(
        self,
        bucket_name: str,
        aws_access_key: str,
        aws_secret_key: str,
        region_name: str,
    ):
        """
        Uploads the log file to AWS S3 with a structured path.

        Args:
            bucket_name (str): Target S3 bucket.
            user_id (str): User identifier.
            service_id (str): Service identifier.
            aws_access_key (str): AWS access key ID.
            aws_secret_key (str): AWS secret access key.
            region_name (str): AWS region.

        Raises:
            FileNotFoundError: If the log file is not found.

        Example:
        --------
        ```python
            logger.upload_to_s3("<bucket_name>", "<aws_access_key>", "<aws_secret_key>", "<region_name>") # Uploads logs to AWS S3
        ```
        """
        if not self.log_file or not os.path.exists(self.log_file):
            raise FileNotFoundError(self.FILE_NOT_FOUND_ERROR_MSG)

        object_name = self._generate_cloud_log_path()
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name,
        )
        s3_client.upload_file(self.log_file, bucket_name, object_name)

    def view_logs(
        self,
        platform: Literal["azure", "aws"],
        platform_config: Dict[str, str],
        service: Optional[str] | Optional[List[str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        organization: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        level: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Fetches logs from Azure Blob Storage or AWS S3 with optional filters.
        """
        self._validate_platform_config(platform, platform_config)

        if platform == "azure":
            return self._fetch_logs_azure(
                platform_config,
                service,
                user_id,
                session_id,
                organization,
                start_date,
                end_date,
                level,
                limit,
                offset,
            )
        elif platform == "aws":
            raise NotImplementedError("AWS S3 log fetching is not implemented yet.")

    def _fetch_logs_azure(
        self,
        platform_config: Dict[str, str],
        services: Optional[List[str]] | Optional[str],
        user_id: Optional[str],
        session_id: Optional[str],
        organization: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        level: Optional[str],
        limit: Optional[int],
        offset: Optional[int],
    ) -> List[Dict[str, str]]:
        """Fetch logs from Azure Blob Storage."""
        blob_service_client = BlobServiceClient.from_connection_string(
            platform_config["connection_string"]
        )
        container_client = blob_service_client.get_container_client(
            platform_config["container_name"]
        )
        logs = []

        if services and isinstance(services, str):
            services = [services]

        if not services or len(services) == 0 or services == "":
            raise ValueError("Service name is required for log filtering.")

        if start_date:
            blob_patterns = self._create_blob_key_filter(start_date, end_date, services)

        for blob in container_client.list_blobs():
            if start_date and not any(
                pattern in blob.name for pattern in blob_patterns
            ):
                continue  # Skip blobs that don't match any of the expected date patterns

            blob_client = container_client.get_blob_client(blob.name)
            log_data = blob_client.download_blob().readall().decode(encoding="cp1252")

            for line in log_data.split("\n")[:-1]:
                try:
                    log_entry = self._parse_log_line(line)
                    logs.append(log_entry)
                except json.JSONDecodeError:
                    continue  # Ignore corrupted log lines
        logs = self._apply_filters(
            logs, services, user_id, session_id, level, organization
        )

        return logs[offset : offset + limit] if offset is not None and limit else logs

    def _parse_log_line(self, log_line: str) -> Optional[Dict[str, str]]:
        """
        Parses a structured log line containing a timestamp, log level, and JSON payload.

        Args:
            log_line (str): A log line in the format 'YYYY-MM-DD HH:MM:SS,MS - LEVEL - {JSON}'

        Returns:
            Optional[Dict[str, str]]: A dictionary with extracted log details or None if parsing fails.
        """
        log_pattern = re.compile(
            r"^\s*(?P<timestamp>[\d-]+\s[\d:,]+)\s*-\s*(?P<level>[A-Z]+)\s*-\s*(?P<json>{.*})\s*$"
        )

        match = log_pattern.match(log_line)
        if match:
            try:
                log_data = json.loads(match.group("json"))
                log_data["timestamp"] = match.group("timestamp")
                log_data["level"] = match.group(
                    "level"
                ).lower()  # Normalize level to lowercase
                return log_data
            except json.JSONDecodeError:
                return None  # Handle malformed JSON

        return None  # Handle invalid log formats

    def _apply_filters(
        self,
        logs: List[Dict[str, str]],
        service: Optional[str] | Optional[List[str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        level: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Filters a list of log entries based on optional keyword filters.

        Args:
            logs (List[Dict[str, str]]): List of log entries (each entry is a dictionary).
            service (Optional[str | list[str]]): Filter by service name, either a single string or a list of strings.
            user_id (Optional[str]): Filter by user.
            session_id (Optional[str]): Filter by session.
            level (Optional[str]): Filter by log level.
            organization (Optional[str]): Filter by organization.

        Returns:
            List[Dict[str, str]]: Filtered list of log entries.
        """
        # Ensure 'service' is always a list (if it's a string, make it a list)
        if isinstance(service, str):
            service = [service]
        
        return [
            log
            for log in logs
            if (service is None or log.get("service") in service)
            and (user_id is None or log.get("user") == user_id)
            and (session_id is None or log.get("session") == session_id)
            and (level is None or log.get("level") == level)
            and (organization is None or log.get("organization") == organization)
        ]

    def _create_blob_key_filter(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        services: Optional[List[str]] = None,
    ) -> list[str]:
        """
        Creates a list of blob key prefixes for Azure Blob Storage based on an optional date range and service.
        If the dates are provided in an incorrect format, it attempts to correct them to 'YYYY/MM/DD'.

        Args:
            start_date (Optional[str]): The start date in various formats; it will be converted to "YYYY/MM/DD".
            end_date (Optional[str]): The end date in various formats; it will be converted to "YYYY/MM/DD".
            service (Optional[str]): The service name to filter logs. If None, all services are included.

        Returns:
            list[str]: A list of blob key prefixes to filter the logs.
        """
        final_date_format = "%Y/%m/%d"

        def _normalize_date(date_str: str) -> datetime:
            """Attempts to normalize different date formats into a datetime object."""
            for fmt in (
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%Y.%m.%d",
                "%d-%m-%Y",
                "%d/%m/%Y",
                "%d.%m.%Y",
            ):
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            raise ValueError(
                f"Invalid date format: {date_str}. Expected formats: YYYY-MM-DD, DD-MM-YYYY, etc."
            )

        # Normalize start_date
        if start_date:
            try:
                start_date = _normalize_date(start_date)
            except ValueError as e:
                raise ValueError(f"Invalid start_date input: {e}")

            # If end_date is not provided, default to today's date
            if not end_date:
                end_date = datetime.today().strftime(
                    "%Y-%m-%d"
                )  # Keep format consistent

        # Normalize end_date
        if end_date:
            try:
                end_date = _normalize_date(end_date)
            except ValueError as e:
                raise ValueError(f"Invalid end_date input: {e}")

        # Ensure start_date is before or equal to end_date
        if start_date and end_date and start_date > end_date:
            raise ValueError("end_date cannot be earlier than start_date.")

        # Generate list of blob key prefixes for the date range
        blob_keys = []
        current_date = start_date

        while current_date <= end_date:
            for service in services:
                date_str = current_date.strftime(final_date_format)
                blob_keys.append(
                    f"{service}/{date_str}" if service else f"*/{date_str}"
                )  # Wildcard for all services
                current_date += timedelta(days=1)

        return blob_keys

    def _return_html(self, logs: List[Dict[str, str]]) -> str:
        """Return logs as an HTML table."""
        html_table = "<table><tr><th>Timestamp</th><th>Level</th><th>Message</th></tr>"
        for log in logs:
            html_table += f"<tr><td>{log['timestamp']}</td><td>{log['level']}</td><td>{log['message']}</td></tr>"
        return html_table + "</table>"

    def _validate_platform_config(
        self, platform: Literal["azure", "aws"], platform_config: Dict[str, str]
    ):
        """
        Validates the platform configuration for Azure Blob Storage or AWS S3.

        Args:
            platform (Literal['azure', 'aws']): Cloud platform to validate.
            platform_config (Dict[str, str]): Platform configuration parameters.

        Raises:
            ValueError: If the platform configuration is invalid.
            ValueError: If the platform is not supported.
        """
        if platform == "azure":
            required_keys = {"connection_string", "container_name"}
        elif platform == "aws":
            required_keys = {
                "bucket_name",
                "aws_access_key",
                "aws_secret_key",
                "region_name",
            }
        else:
            raise ValueError(
                "Invalid platform. Supported platforms are 'azure' and 'aws'."
            )

        missing_keys = required_keys - set(platform_config.keys())
        if missing_keys:
            raise ValueError(
                f"Missing required keys in platform configuration: {missing_keys}."
            )
