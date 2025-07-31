"""
logger.py

Summary:
--------
A robust, multi-destination logging and log management module for Python applications.
Supports console, rotating file, and asynchronous SQLite logging, as well as exporting and uploading logs to cloud storage (AWS S3, Azure Blob).
Designed for scalable, multi-tenant, and cloud-native environments.

This module provides:
- EnhancedLogger: Flexible logger with contextual metadata (user, session, organization).
- LogExporter: Export logs from files or SQLite DB to CSV/JSON with filtering.
- CloudStorageConfig: Unified configuration for Azure, AWS, or filesystem log storage.
- CloudLogUploader: Upload log files or database logs to cloud storage.

------------------------------------

Usage:
------

File and SQLite path constants
-------------------------------------------------
LOG_FILE = "logs/example_app.log"
DB_PATH = "logs/example_app.db"
CSV_EXPORT = "logs/exported_logs.csv"
JSON_EXPORT = "logs/exported_logs.json"

# 1. Initialize EnhancedLogger
-------------------------------------------------
logger = EnhancedLogger(
    name="example_app",
    log_file=LOG_FILE,
    db_path=DB_PATH,
    service="ExampleService"
)

# 2. Log messages at various levels with/without metadata
-------------------------------------------------
logger.log("info", "Application started")
logger.log("debug", "Debugging details", user_id="user42", session_id="sess1")
logger.log("warning", "Low disk space", organization="OrgA")
logger.log("error", "An error occurred", user_id="user42", extra={"error_code": 123})
logger.log("critical", "Critical failure!", session_id="sess1", organization="OrgA")
logger.log("info", "User login", user_id="user99", session_id="sess2", organization="OrgB", extra={"ip": "127.0.0.1"})

# 3. Export logs to CSV and JSON (from DB)
-------------------------------------------------
exporter = LogExporter(
    source=DB_PATH,
    output_path=CSV_EXPORT,
    use_db=True
)
exporter.export_to_csv(filters={})  # Export all logs
exporter = LogExporter(
    source=DB_PATH,
    output_path=JSON_EXPORT,
    use_db=True
)
exporter.export_to_json(filters={"level": "error"})  # Export only error logs
print(f"Logs exported to {CSV_EXPORT} (all logs) and {JSON_EXPORT} (error logs)")

# 4. Display logs in CLI using LogDashboardViewer
-------------------------------------------------
print("\nDisplaying first page of logs from DB in CLI:")
viewer = LogDashboardViewer(
    source=DB_PATH,
    source_type="db",
    page_size=10
)
viewer.display_cli(page=1)
print("\nExample complete. Check the logs/ directory for output files.")

# 5. Display logs in Web page using LogDashboardViewer
--------------------------------------------------
viewer.start_html_dashboard(
    port=5000,
)


------------------------------------

Dependencies:
-------------
- Python Standard Library
    - `os`, `csv`, `json`, `sqlite3`, `logging`, `threading`, `datetime`, `concurrent.futures`
- Third-party
    - `rich` (for CLI log display)
    - `flask` (for web dashboard)
    - `boto3` (for AWS S3)
    - `azure-storage-blob` (for Azure Blob Storage)
    - `pyyaml` (for YAML config support)

To install:
    pip install boto3 azure-storage-blob pyyaml rich flask
    # Note: SQLite3 is included with Python 3.x by default, no separate installation
    # Ensure SQLite3 is available in your Python installation

------------------------------------

Features:
---------
- Console, rotating file, and async SQLite logging with contextual metadata
- Log exporting to CSV/JSON from file or SQLite, with advanced filtering
- Cloud upload support for AWS S3 and Azure Blob Storage
- Thread-safe, non-blocking logging and exporting
- Designed for distributed, multi-tenant, and production environments

------------------------------------

Author:
-------
Sourav Das (2025)
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Import dependencies
import os
import csv
import json
import yaml
import boto3
import sqlite3
import logging
import threading
from rich.table import Table
from rich.console import Console
from datetime import datetime, timezone
from azure.storage.blob import BlobServiceClient
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, render_template_string

from typing import Optional, Literal, Dict, List, Any, Union


class EnhancedLogger:
    """
    EnhancedLogger provides a logging utility that supports:
    - Console logging
    - Rotating file logging
    - Asynchronous logging to SQLite
    - Optional contextual metadata (user, session, organization)

    Attributes:
        FILE_NOT_FOUND_ERROR_MSG: Error message for missing log file
        DB_INIT_LOCK: Class-level lock for thread-safe DB initialization
    """

    FILE_NOT_FOUND_ERROR_MSG = "Log file not found or not specified."
    DB_INIT_LOCK = threading.Lock()

    def __init__(
        self,
        name: str,
        log_file: Optional[str] = None,
        db_path: Optional[str] = None,
        service: Optional[str] = None,
        max_bytes: int = 1024 * 1024,
        backup_count: int = 5,
    ):
        """
        Initialize the logger with console, optional file, and optional SQLite support.

        Args:
            name: Logger name.
            log_file: Path to a rotating log file (optional).
            db_path: SQLite DB file path for storing logs (optional).
            service: Name of the logging service/application.
            max_bytes: Max file size for rotating logs.
            backup_count: Number of rotated logs to keep.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        self.log_file = log_file
        self.db_path = db_path
        self.service = service or "DefaultService"

        self.executor = ThreadPoolExecutor(max_workers=2)

        # Console handler setup
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Optional file handler
        if self.log_file:
            self._setup_file_handler(formatter, max_bytes, backup_count)

        # Optional DB initialization
        if self.db_path:
            self._init_db()

        if not self.db_path and not self.log_file:
            raise ValueError(
                "At least one of 'log_file' or 'db_path' must be specified for EnhancedLogger."
            )

    def _setup_file_handler(
        self,
        formatter: logging.Formatter,
        max_bytes: int,
        backup_count: int,
    ) -> None:
        """
        Set up a rotating file handler for log file output.

        Args:
            formatter: Log formatter instance.
            max_bytes: Maximum size of a single log file.
            backup_count: Number of backup files to retain.
        """
        try:
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            self.logger.error(f"Failed to setup file handler: {e}")

    def _init_db(self) -> None:
        """
        Initialize the SQLite database and create a `logs` table if it doesn't exist.
        Uses a thread-safe lock to avoid race conditions.
        """
        try:
            with self.DB_INIT_LOCK:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS logs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            level TEXT,
                            message TEXT,
                            service TEXT,
                            session_id TEXT,
                            user_id TEXT,
                            organization TEXT,
                            extra TEXT
                        );
                        """
                    )
                    conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")

    def log(
        self,
        level: str,
        message: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        organization: Optional[str] = None,
        extra: Optional[Dict] = None,
    ) -> None:
        """
        Log a message to the console, file, and optionally to a SQLite DB.

        Args:
            level: Logging level (e.g., info, error, debug, warning).
            message: Log message string.
            user_id: ID of the user (optional).
            session_id: Session identifier (optional).
            organization: Organization name or ID (optional).
            extra: Additional structured data to log (optional).

        Raises:
            ValueError: If an invalid log level is provided.
        """
        valid_levels = {"debug", "info", "warning", "error", "critical", "exception"}
        level = level.lower()

        if level not in valid_levels:
            raise ValueError(
                f"Invalid log level: {level}. Valid levels are {valid_levels}."
            )

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
            "service": self.service,
            "session_id": session_id,
            "user_id": user_id,
            "organization": organization,
            "extra": json.dumps(extra or {}),
        }

        # File or console logging
        try:
            if self.log_file:
                formatted_log = f"{log_entry['timestamp']} - {level.upper()} - {json.dumps(log_entry)}"
                getattr(self.logger, level)(formatted_log)
            else:
                # Console fallback
                getattr(self.logger, level)(message)
        except Exception as e:
            self.logger.error(f"Failed to log to file/console: {e}")

        # Asynchronous DB logging
        if self.db_path:
            self.executor.submit(self._log_to_db, log_entry)

    def _log_to_db(self, entry: Dict[str, str]) -> None:
        """
        Write a log entry to the SQLite database asynchronously.

        Args:
            entry: A dictionary containing log information.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO logs (
                        timestamp, level, message, service,
                        session_id, user_id, organization, extra
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry["timestamp"],
                        entry["level"],
                        entry["message"],
                        entry["service"],
                        entry["session_id"],
                        entry["user_id"],
                        entry["organization"],
                        entry["extra"],
                    ),
                )
                conn.commit()
        except Exception as e:
            # Avoid infinite loop if DB logging also fails
            self.logger.error(f"Failed to write log to DB: {e}")


class LogExporter:
    """
    A class responsible for exporting logs from either a log file or SQLite database
    into structured formats like CSV or JSON, with optional filtering.
    """

    def __init__(
        self,
        source: str,
        output_path: str,
        use_db: bool = False,
        db_table: str = "logs",
    ):
        """
        Initialize the LogExporter.

        Args:
            source (str): Path to the log file or SQLite database.
            output_path (str): Path to export the filtered logs.
            use_db (bool): If True, source is treated as a SQLite database.
            db_table (str): Table name to query logs from (SQLite only).
        """
        self.source = source
        self.output_path = output_path
        self.use_db = use_db
        self.db_table = db_table

    def _parse_timestamp(self, ts_str: str) -> Optional[datetime]:
        """
        Parse an ISO 8601 timestamp string into a datetime object.

        Args:
            ts_str (str): Timestamp string.

        Returns:
            Optional[datetime]: Parsed datetime object, or None if invalid.
        """
        try:
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _matches_filters(
        self,
        record: Dict[str, Any],
        filters: Dict[str, Optional[Union[str, List[str]]]],
    ) -> bool:
        """
        Check if a record matches the given filters.

        Args:
            record (Dict[str, Any]): A log entry as a dictionary.
            filters (Dict[str, Any]): Dictionary of filtering parameters.

        Returns:
            bool: True if record matches all provided filters.
        """
        # Direct field matches
        for key in ["level", "organization", "user_id", "service", "session_id"]:
            val = filters.get(key)
            if val and record.get(key) not in (val if isinstance(val, list) else [val]):
                return False

        # Timestamp filtering
        ts_str = record.get("timestamp")
        if not ts_str:
            return False

        ts = self._parse_timestamp(ts_str)
        if not ts:
            return False

        start_date = filters.get("start_date")
        end_date = filters.get("end_date")

        if start_date and ts.date() < datetime.strptime(start_date, "%Y-%m-%d").date():
            return False
        return (
            not end_date or ts.date() <= datetime.strptime(end_date, "%Y-%m-%d").date()
        )

    def _load_logs_from_file(
        self, filters: Dict[str, Optional[Union[str, List[str]]]]
    ) -> List[Dict[str, Any]]:
        """
        Load and filter logs from a .log file.

        Args:
            filters (dict): Filter options.

        Returns:
            List[Dict[str, Any]]: Filtered list of logs.
        """
        if not os.path.exists(self.source):
            raise FileNotFoundError(f"Log file '{self.source}' not found.")

        results = []
        with open(self.source, "r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, 1):
                try:
                    record = json.loads(line.strip())
                    if self._matches_filters(record, filters):
                        results.append(record)
                except json.JSONDecodeError:
                    print(f"[Warning] Skipping malformed JSON on line {line_num}")
        return results

    def _load_logs_from_db(
        self, filters: Dict[str, Optional[Union[str, List[str]]]]
    ) -> List[Dict[str, Any]]:
        """
        Load and filter logs from a SQLite database using SQL WHERE clause.

        Args:
            filters (dict): Filter conditions where keys are column names
                            and values are either str or list of str.

        Returns:
            List[Dict[str, Any]]: Filtered list of log records.

        Raises:
            FileNotFoundError: If the SQLite database file is not found.
            RuntimeError: If there's an error querying the database.
        """
        if not os.path.exists(self.source):
            raise FileNotFoundError(f"Database file '{self.source}' not found.")

        query = f"SELECT * FROM {self.db_table}"
        conditions = []
        params = []

        for key, value in filters.items():
            if value is None:
                continue
            if isinstance(value, list):
                placeholders = ",".join(["?"] * len(value))
                conditions.append(f"{key} IN ({placeholders})")
                params.extend(value)
            else:
                conditions.append(f"{key} = ?")
                params.append(value)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        try:
            with sqlite3.connect(self.source) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.DatabaseError as e:
            raise RuntimeError(f"SQLite error while querying logs: {e}") from e

    def _load_filtered_logs(
        self, filters: Dict[str, Optional[Union[str, List[str]]]]
    ) -> List[Dict[str, Any]]:
        """
        Load logs from the configured source and apply filters.

        Args:
            filters (dict): Filter parameters.

        Returns:
            List[Dict[str, Any]]: Filtered logs.
        """
        if self.use_db:
            return self._load_logs_from_db(filters)
        else:
            return self._load_logs_from_file(filters)

    def export_to_csv(
        self, filters: Dict[str, Optional[Union[str, List[str]]]]
    ) -> None:
        """
        Export logs to a CSV file after filtering.

        Args:
            filters (dict): Filter parameters.
        """
        logs = self._load_filtered_logs(filters)
        if not logs:
            raise ValueError("No logs matched the given filters.")

        headers = sorted({key for entry in logs for key in entry})
        try:
            with open(self.output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for log in logs:
                    writer.writerow(log)
        except IOError as e:
            raise RuntimeError(f"Failed to write CSV file: {e}") from e

    def export_to_json(
        self, filters: Dict[str, Optional[Union[str, List[str]]]]
    ) -> None:
        """
        Export logs to a JSON file after filtering.

        Args:
            filters (dict): Filter parameters.
        """
        logs = self._load_filtered_logs(filters)
        if not logs:
            raise ValueError("No logs matched the given filters.")

        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2)
        except IOError as e:
            raise RuntimeError(f"Failed to write JSON file: {e}") from e


class CloudStorageConfig:
    """
    Configuration class for cloud storage operations.

    Attributes:
        cloud_storage_type (Literal['azure', 'aws']): Type of cloud storage ('azure' or 'aws'). Required.
        connection_string (Optional[str]): Connection string for Azure Blob Storage. Optional.
        container_name (Optional[str]): Name of the Azure Blob Storage container. Optional.
        bucket_name (Optional[str]): Name of the AWS S3 bucket. Optional.
        aws_access_key (Optional[str]): AWS access key ID. Optional.
        aws_secret_key (Optional[str]): AWS secret access key. Optional.
        region_name (Optional[str]): AWS region name. Optional.
    """

    SUPPORTED_TYPES = ("azure", "aws", "filesystem")

    def __init__(
        self,
        cloud_storage_type: Literal["azure", "aws", "filesystem"],
        log_file: Optional[str] = None,
        connection_string: Optional[str] = None,
        container_name: Optional[str] = None,
        bucket_name: Optional[str] = None,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        region_name: Optional[str] = None,
    ):
        # Validate cloud_storage_type
        cst = cloud_storage_type or os.getenv("CLOUD_STORAGE_TYPE")
        if cst not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Invalid cloud storage type '{cst}'. Supported types: {self.SUPPORTED_TYPES}"
            )

        self.cloud_storage_type = cst
        self.connection_string = connection_string or os.getenv(
            "AZURE_CONNECTION_STRING"
        )
        self.container_name = container_name or os.getenv("AZURE_CONTAINER_NAME")
        self.bucket_name = bucket_name or os.getenv("AWS_BUCKET_NAME")
        self.aws_access_key = aws_access_key or os.getenv("AWS_ACCESS_KEY")
        self.aws_secret_key = aws_secret_key or os.getenv("AWS_SECRET_KEY")
        self.region_name = region_name or os.getenv("AWS_REGION_NAME")

        # Validate required fields for Azure
        if self.cloud_storage_type == "azure":
            if not self.connection_string:
                raise ValueError("Azure Blob Storage requires 'connection_string'.")
            if not self.container_name:
                raise ValueError("Azure Blob Storage requires 'container_name'.")
        # Validate required fields for AWS
        elif self.cloud_storage_type == "aws":
            missing = []
            if not self.bucket_name:
                missing.append("bucket_name")
            if not self.aws_access_key:
                missing.append("aws_access_key")
            if not self.aws_secret_key:
                missing.append("aws_secret_key")
            if not self.region_name:
                missing.append("region_name")
            if missing:
                raise ValueError(f"AWS S3 requires: {', '.join(missing)}.")
        # Validate log file for filesystem storage
        elif self.cloud_storage_type == "filesystem":
            if not log_file:
                raise ValueError("Filesystem storage requires 'log_file' path.")
            if not os.path.exists(os.path.dirname(log_file)):
                os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def __repr__(self):
        # Mask sensitive info
        return (
            f"CloudStorageConfig(cloud_storage_type={self.cloud_storage_type}, "
            f"connection_string={'***' if self.connection_string else None}, "
            f"container_name={self.container_name}, "
            f"bucket_name={self.bucket_name}, "
            f"aws_access_key={'***' if self.aws_access_key else None}, "
            f"aws_secret_key={'***' if self.aws_secret_key else None}, "
            f"region_name={self.region_name})"
            f", log_file={self.log_file if hasattr(self, 'log_file') else None}"
        )

    @classmethod
    def from_json(cls, json_file: str):
        """
        Create a CloudStorageConfig instance from a JSON file.

        Args:
            json_file (str): Path to the JSON configuration file.

        Returns:
            CloudStorageConfig: Config instance with loaded values.
        """
        with open(json_file, "r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_yaml(cls, yaml_file: str):
        """
        Create a CloudStorageConfig instance from a YAML file.

        Args:
            yaml_file (str): Path to the YAML configuration file.

        Returns:
            CloudStorageConfig: Config instance with loaded values.
        """
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


class CloudLogUploader:
    """
    A utility class to upload log files from the local file system or SQLite database
    to cloud storage platforms like AWS S3 or Azure Blob Storage.

    Attributes:
        config (CloudStorageConfig): Configuration object containing cloud credentials and options.
    """

    def __init__(self, config: "CloudStorageConfig") -> None:
        """
        Initializes the CloudLogUploader with given cloud storage configuration.

        Args:
            config (CloudStorageConfig): Cloud storage configuration object.
        """
        self.config = config

        # Initialize appropriate cloud storage client
        if self.config.cloud_storage_type == "aws":
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=self.config.aws_access_key,
                aws_secret_access_key=self.config.aws_secret_key,
                region_name=self.config.region_name,
            )
        elif self.config.cloud_storage_type == "azure":
            self.blob_service_client = BlobServiceClient.from_connection_string(
                self.config.connection_string
            )
        else:
            raise NotImplementedError(
                f"Unsupported cloud storage type: {self.config.cloud_storage_type}"
            )

    def _upload_to_azure(self, blob_name: str, data: bytes) -> None:
        """
        Uploads binary data to Azure Blob Storage.

        Args:
            blob_name (str): Name of the blob in Azure.
            data (bytes): Binary content to be uploaded.
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.config.container_name, blob=blob_name
            )
            blob_client.upload_blob(data, overwrite=True)
        except Exception as e:
            raise RuntimeError(f"Azure upload failed: {e}") from e

    def _upload_to_s3(self, file_path: str, object_key: str) -> None:
        """
        Uploads a file to AWS S3.

        Args:
            file_path (str): Local path to the file.
            object_key (str): Key under which file will be stored in S3.
        """
        try:
            self.s3_client.upload_file(file_path, self.config.bucket_name, object_key)
        except Exception as e:
            raise RuntimeError(f"S3 upload failed: {e}") from e

    def upload_log_file(self, file_path: str) -> None:
        """
        Uploads a log file from the local file system to the configured cloud storage.

        Args:
            file_path (str): Path to the log file to be uploaded.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Log file '{file_path}' not found.")

        filename = os.path.basename(file_path)
        timestamped_key = (
            f"logs/{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{filename}"
        )

        try:
            if self.config.cloud_storage_type == "aws":
                self._upload_to_s3(file_path, timestamped_key)
            elif self.config.cloud_storage_type == "azure":
                with open(file_path, "rb") as f:
                    self._upload_to_azure(timestamped_key, f.read())
        except Exception as e:
            raise RuntimeError(f"Error during log file upload: {e}") from e

    def upload_from_sqlite(
        self,
        db_path: str,
        table: str = "logs",
        log_column: str = "log_data",
        limit: Optional[int] = None,
    ) -> None:
        """
        Extracts log data from a SQLite database and uploads it as a log file.

        Args:
            db_path (str): Path to the SQLite database.
            table (str): Table containing the logs.
            log_column (str): Column in the table containing log strings.
            limit (Optional[int]): Number of rows to fetch; if None, fetches all.
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database '{db_path}' not found.")

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            query = f"SELECT {log_column} FROM {table}"
            if limit:
                query += f" LIMIT {limit}"

            cursor.execute(query)
            rows = cursor.fetchall()
        except sqlite3.Error as e:
            raise RuntimeError(f"SQLite error: {e}") from e
        finally:
            conn.close()

        if not rows:
            print("No log entries found in database.")
            return

        # Create temporary file from fetched logs
        temp_filename = (
            f"temp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
        )
        try:
            with open(temp_filename, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(f"{row[0]}\n")

            self.upload_log_file(temp_filename)
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    def upload_logs_cron(self, source: str, **kwargs) -> None:
        """
        High-level method for automated log upload via scheduled jobs (e.g., cron).

        Args:
            source (str): Source of logs, either 'file' or 'sqlite'.
            **kwargs: Additional arguments like 'file_path', 'db_path', 'table', 'log_column', etc.
        """
        try:
            if source == "file":
                if file_path := kwargs.get("file_path"):
                    self.upload_log_file(file_path)

                else:
                    raise ValueError("Missing required argument: 'file_path'")
            elif source == "sqlite":
                if db_path := kwargs.get("db_path"):
                    self.upload_from_sqlite(
                        db_path=db_path,
                        table=kwargs.get("table", "logs"),
                        log_column=kwargs.get("log_column", "log_data"),
                        limit=kwargs.get("limit"),
                    )

                else:
                    raise ValueError("Missing required argument: 'db_path'")
            else:
                raise ValueError(
                    f"Invalid source: '{source}'. Expected 'file' or 'sqlite'."
                )
        except Exception as e:
            raise RuntimeError(f"Failed to upload logs: {e}") from e


class LogDashboardViewer:
    """
    Lightweight dashboard class to view logs from log files or SQLite database.

    Core Features:
    - Filtering: Supports filtering by timestamp, level, message, service, session_id, user_id, and organization.
    - Pagination: Supports configurable page size and navigation.
    - Performance: Reads only necessary lines or rows, minimizes memory usage.

    Performance Notes:
    - Avoids loading entire file/db content in memory.
    - Uses generators and efficient cursor pagination.
    - TTL caching can be implemented externally if needed.

    Security:
    - Read-only operations only.
    """

    def __init__(
        self,
        source: str,
        source_type: str = "file",
        db_table: str = "logs",
        page_size: int = 50
    ) -> None:
        """
        Initialize the log viewer.

        Args:
            source (str): Path to log file or SQLite DB.
            source_type (str): Either 'file' or 'db'.
            db_table (str): Name of DB table if using SQLite.
            page_size (int): Number of log entries per page.
        """
        self.source = source
        self.source_type = source_type
        self.db_table = db_table
        self.page_size = page_size

    def _filter_entry(self, entry: Dict[str, Any], filters: Dict[str, Optional[str]]) -> bool:
        """
        Check if a log entry matches the given filters.

        Args:
            entry (dict): A single log entry.
            filters (dict): Filter conditions.

        Returns:
            bool: True if the entry matches all filters.
        """
        for key, value in filters.items():
            if value:
                if key == "message":
                    if value.lower() not in str(entry.get(key, "")).lower():
                        return False
                else:
                    if str(entry.get(key, "")).lower() != value.lower():
                        return False
        return True

    def _read_logs_from_file(self, page: int, filters: Dict[str, Optional[str]]) -> List[Dict[str, Any]]:
        if not os.path.exists(self.source):
            raise FileNotFoundError(f"Log file '{self.source}' not found.")

        start = (page - 1) * self.page_size
        end = start + self.page_size
        result: List[Dict[str, Any]] = []
        matched = 0

        try:
            with open(self.source, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if self._filter_entry(entry, filters):
                            if matched >= start and matched < end:
                                result.append(entry)
                            matched += 1
                            if matched >= end:
                                break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            raise RuntimeError(f"Error reading from file: {e}") from e

        return result

    def _read_logs_from_db(self, page: int, filters: Dict[str, Optional[str]]) -> List[Dict[str, Any]]:
        if not os.path.exists(self.source):
            raise FileNotFoundError(f"Database file '{self.source}' not found.")

        conditions = []
        values = []

        for key, value in filters.items():
            if value:
                if key == "message":
                    conditions.append(f"LOWER({key}) LIKE ?")
                    values.append(f"%{value.lower()}%")
                else:
                    conditions.append(f"LOWER({key}) = ?")
                    values.append(value.lower())

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        offset = (page - 1) * self.page_size

        query = f"""
            SELECT * FROM {self.db_table} 
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """

        result: List[Dict[str, Any]] = []
        try:
            with sqlite3.connect(self.source) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, (*values, self.page_size, offset))
                rows = cursor.fetchall()
                result = [dict(row) for row in rows]
        except sqlite3.DatabaseError as e:
            raise RuntimeError(f"SQLite error: {e}") from e

        return result

    def get_logs(self, page: int = 1, filters: Optional[Dict[str, Optional[str]]] = None) -> List[Dict[str, Any]]:
        if filters is None:
            filters = {}

        if self.source_type == "file":
            return self._read_logs_from_file(page, filters)
        elif self.source_type == "db":
            return self._read_logs_from_db(page, filters)
        else:
            raise ValueError(f"Unsupported source_type: {self.source_type}")

    def display_cli(self, page: int = 1, filters: Optional[Dict[str, Optional[str]]] = None) -> None:
        """
        Display logs in CLI using rich tables.
        """
        console = Console()
        logs = self.get_logs(page=page, filters=filters)

        if not logs:
            console.print("[yellow]No logs found.[/yellow]")
            return

        table = Table(title=f"Logs - Page {page}")
        for key in logs[0].keys():
            table.add_column(str(key))

        for entry in logs:
            table.add_row(*[str(entry.get(k, "")) for k in logs[0].keys()])

        console.print(table)
        
    def _parse_filters(self, args: dict) -> Dict[str, Optional[str]]:
        """
        Extract filters from query parameters.

        Args:
            args (dict): Request.args

        Returns:
            Dict[str, Optional[str]]: Cleaned filter dictionary
        """
        keys = [
            "timestamp",
            "level",
            "message",
            "service",
            "session_id",
            "user_id",
            "organization",
        ]
        return {k: args.get(k) for k in keys if args.get(k)}
        
    def start_html_dashboard(self, port: int = 8080) -> None:
        """
        Launch Flask dashboard for log viewing.

        Args:
            port (int): Port to run on.
        """
        app = Flask(__name__)

        @app.route("/logs")
        def logs():
            try:
                filters = self._parse_filters(request.args)
                page = int(request.args.get("page", 1))
                logs = self.get_logs(page=page, filters=filters)
                return render_template_string(self._html_template(), logs=logs, page=page)
            except Exception as e:
                return f"Error: {e}", 500

        app.run(port=port)

    def _html_template(self) -> str:
        """
        Provides the improved HTML template for the log viewer dashboard.

        Returns:
            str: A formatted HTML string using Jinja2 placeholders.
        """
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Log Viewer</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 40px;
                    background-color: #f9f9f9;
                    color: #333;
                }

                h2 {
                    margin-bottom: 20px;
                    color: #444;
                }

                form.filter-form {
                    margin-bottom: 20px;
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    align-items: center;
                }

                form.filter-form label {
                    margin-right: 5px;
                }

                form.filter-form input[type="text"] {
                    padding: 5px 10px;
                    border-radius: 4px;
                    border: 1px solid #ccc;
                }

                form.filter-form button {
                    padding: 6px 14px;
                    border: none;
                    border-radius: 4px;
                    background-color: #007bff;
                    color: white;
                    cursor: pointer;
                }

                form.filter-form button:hover {
                    background-color: #0056b3;
                }

                table {
                    width: 100%;
                    border-collapse: collapse;
                    background-color: white;
                    box-shadow: 0 0 10px rgba(0,0,0,0.05);
                }

                th, td {
                    padding: 10px 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }

                th {
                    background-color: #f1f1f1;
                    font-weight: bold;
                }

                tr:hover {
                    background-color: #f5f5f5;
                }

                .pagination-buttons {
                    margin-top: 20px;
                }

                .pagination-buttons form {
                    display: inline-block;
                    margin-right: 10px;
                }

                .pagination-buttons button {
                    padding: 6px 14px;
                    border: none;
                    border-radius: 4px;
                    background-color: #28a745;
                    color: white;
                    cursor: pointer;
                }

                .pagination-buttons button:hover {
                    background-color: #1e7e34;
                }

                .clear-button {
                    background-color: #dc3545 !important;
                }

                .clear-button:hover {
                    background-color: #a71d2a !important;
                }
            </style>
        </head>
        <body>
            <h2>Log Viewer</h2>

            <form method="get" class="filter-form">
                {% for key in ['timestamp', 'level', 'message', 'service', 'session_id', 'user_id', 'organization'] %}
                    <label for="{{ key }}">{{ key }}:</label>
                    <input type="text" name="{{ key }}" id="{{ key }}" value="{{ request.args.get(key, '') }}">
                {% endfor %}
                <button type="submit">Apply Filter</button>
                <button type="submit" name="clear" value="1" class="clear-button">Clear Filters</button>
            </form>

            <p>Page {{ page }}</p>

            <table>
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Level</th>
                        <th>Message</th>
                        <th>Service</th>
                        <th>Session</th>
                        <th>User</th>
                        <th>Org</th>
                    </tr>
                </thead>
                <tbody>
                    {% for log in logs %}
                        <tr>
                            <td>{{ log.get('timestamp') }}</td>
                            <td>{{ log.get('level') }}</td>
                            <td>{{ log.get('message') }}</td>
                            <td>{{ log.get('service') }}</td>
                            <td>{{ log.get('session_id') }}</td>
                            <td>{{ log.get('user_id') }}</td>
                            <td>{{ log.get('organization') }}</td>
                        </tr>
                    {% endfor %}
                </tbody>
            </table>

            <div class="pagination-buttons">
                <form method="get">
                    {% for k, v in request.args.items() %}
                        {% if k != 'page' and k != 'clear' %}
                            <input type="hidden" name="{{ k }}" value="{{ v }}">
                        {% endif %}
                    {% endfor %}
                    <input type="hidden" name="page" value="{{ page - 1 }}">
                    <button type="submit" {% if page <= 1 %}disabled{% endif %}>Previous Page</button>
                </form>

                <form method="get">
                    {% for k, v in request.args.items() %}
                        {% if k != 'page' and k != 'clear' %}
                            <input type="hidden" name="{{ k }}" value="{{ v }}">
                        {% endif %}
                    {% endfor %}
                    <input type="hidden" name="page" value="{{ page + 1 }}">
                    <button type="submit">Next Page</button>
                </form>
            </div>
        </body>
        </html>
        """
