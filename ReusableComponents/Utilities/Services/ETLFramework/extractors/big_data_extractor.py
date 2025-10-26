""" """

# Import required libraries
import os
import logging
import pandas as pd
import pyarrow as pa
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv
from dataclasses import dataclass
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator, Optional, List, Iterator, Dict, Any
from azure.storage.filedatalake import DataLakeServiceClient, FileSystemClient

# Import ETL framework modules with proper error handling
try:
    from core.errors import (
        DataSourceConnectionError,
        DataSourceAuthenticationError,
        CloudExtractorError,
    )
    from extractors.base import BaseExtractor
    from core.checkpoint import CheckpointManager
except ImportError:
    # Fallback for standalone usage
    import sys
    from pathlib import Path

    etl_framework_path = Path(__file__).parent.parent
    sys.path.insert(0, str(etl_framework_path))

    try:
        from core.errors import (
            DataSourceConnectionError,
            DataSourceAuthenticationError,
            CloudExtractorError,
        )
        from extractors.base import BaseExtractor
        from core.checkpoint import CheckpointManager
    except ImportError as e:
        raise ImportError(
            f"Could not import ETL framework dependencies: {e}. "
            "Ensure the ETL framework is properly installed."
        )

# Module logger - use proper initialization pattern
logger = logging.getLogger(__name__)

# Configure logging if not already configured
if not logger.handlers:
    # Set log level
    logger.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    # Prevent propagation to avoid duplicate logs if parent loggers are configured
    logger.propagate = False

# Load environment variables from .env file if it exists
load_dotenv(override=True)


@dataclass
class AuthConfig:
    """
    Configuration for Authentication.

    Args:
    ------
    source_type (str): Type of the big data source (e.g., 'hadoop', 'spark', 'bigquery').
    connection_string (str): Connection string or endpoint to connect to the data source.
    method (str): Authentication method (e.g., 'oauth', 'service_account', 'key_file').
    credentials (dict): Credentials required for authentication.
    port (int, optional): Port number for the connection. Defaults to None.
    host (str, optional): Host address for the connection. Defaults to None.
    """

    source_type: str
    connection_string: str | None = None
    method: str | None = None
    credentials: dict | None = None
    port: int | None = None
    host: str | None = None


class BigDataExtractor(BaseExtractor, ABC):
    """
    Abstract base class for big data extractors.

    This class provides a unified interface and base functionality for extracting
    data from large-scale sources such as HDFS, Spark, Hive, or other distributed
    storage systems. It supports metadata creation, checkpointing, and robust
    exception handling.

    Subclasses must implement:
        - _connect_to_source(): establish the connection to the data source
        - extract_data(): extract data based on query and optional date ranges
        - stream_extract_data(): stream large datasets efficiently

    Args:
    ------
    auth_config (AuthConfig): Configuration for authentication and source details.
    checkpoint_path (str, optional): Path to store checkpoint files. Defaults to None.
    """

    def __init__(
        self, auth_config: "AuthConfig", checkpoint_path: Optional[str] = None
    ):
        super().__init__(checkpoint_path=checkpoint_path)
        self.auth_config = auth_config
        self.connection = None

        logger.debug("Initializing BigDataExtractor with config: %s", auth_config)
        self._initialize_connection()

    def _initialize_connection(self):
        """
        Initialize connection to the big data source.

        Handles authentication and connection errors robustly and logs relevant details.
        Raises:
            - DataSourceAuthenticationError
            - DataSourceConnectionError
            - CloudExtractorError for unexpected errors
        """
        try:
            logger.info(
                "Attempting to connect to the big data source: %s",
                self.auth_config.source_type,
            )
            self.connection = self._connect_to_source()
            logger.info(
                "Successfully connected to the big data source: %s",
                self.auth_config.source_type,
            )
        except DataSourceAuthenticationError as e:
            logger.error(
                "Authentication failed for source %s: %s",
                self.auth_config.source_type,
                str(e),
            )
            raise
        except DataSourceConnectionError as e:
            logger.error(
                "Connection failed for source %s: %s",
                self.auth_config.source_type,
                str(e),
            )
            raise
        except Exception as e:
            logger.exception("Unexpected error during connection initialization")
            raise CloudExtractorError(f"Failed to initialize connection: {e}") from e

    @abstractmethod
    def _connect_to_source(self):
        """
        Establish connection to the big data source.

        Must be implemented by subclasses for specific backends (HDFS, Hive, Spark, etc.)

        Returns:
            Connection object specific to the source.
        """
        pass

    @abstractmethod
    def extract(
        self,
        query: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Iterator[dict]:
        """
        Extract data from the big data source based on query and optional date filters.

        Args:
        ------
        query (str): Query or command to fetch data.
        start_date (datetime, optional): Start date for filtering data. Defaults to None.
        end_date (datetime, optional): End date for filtering data. Defaults to None.

        Yields:
        -------
        Iterator[dict]: Iterator over extracted records.

        Raises:
        -------
        DataSourceQueryError: If the query fails.
        """
        pass

    @abstractmethod
    def stream_extract(
        self,
        query: str,
        chunk_size: int = 100000,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Iterator[pd.DataFrame]:
        """
        Stream large datasets from the source in chunks.

        This is essential for memory-efficient extraction of big files.

        Args:
        ------
        query (str): Query or command to fetch data.
        chunk_size (int): Number of records per chunk. Defaults to 100,000.
        start_date (datetime, optional): Start date filter.
        end_date (datetime, optional): End date filter.

        Yields:
        -------
        Iterator[pd.DataFrame]: Iterator over data chunks as pandas DataFrames.

        Raises:
        -------
        DataSourceQueryError: If the query fails.
        """
        pass

    def _create_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate metadata for an extracted DataFrame.

        Metadata includes source, extraction time, record count, column info,
        and a statistical summary.

        Args:
        -----
        df (pd.DataFrame): Extracted data.

        Returns:
        --------
        Dict[str, Any]: Metadata dictionary.
        """
        try:
            metadata: Dict[str, Any] = {
                "source": self.auth_config.source_type,
                "extraction_time": pd.Timestamp.now().isoformat(),
                "num_records": len(df),
                "num_columns": len(df.columns),
                "columns": df.columns.tolist(),
                "summary": df.describe(include="all").to_dict() if not df.empty else {},
            }
            logger.debug(
                "Metadata created for source %s: %s",
                self.auth_config.source_type,
                metadata,
            )
            return metadata
        except Exception as e:
            logger.warning("Metadata creation failed: %s", str(e))
            return {
                "source": getattr(self.auth_config, "source_type", "unknown"),
                "extraction_time": pd.Timestamp.now().isoformat(),
                "num_records": len(df) if hasattr(df, "__len__") else 0,
                "num_columns": len(df.columns) if hasattr(df, "columns") else 0,
                "columns": df.columns.tolist() if hasattr(df, "columns") else [],
                "summary": {},
            }


class AzureDataLakeExtractor(BigDataExtractor):
    """
    Extractor for Azure Data Lake using BigDataExtractor as base.

    Supports reading multiple files with optional multi-threaded extraction
    for large datasets.

    Args:
    -----
    auth_config (AuthConfig): Authentication and source configuration.
    checkpoint_path (str, optional): Path to store checkpoint files.
    """

    def __init__(
        self, auth_config, checkpoint_path: Optional[str] = None, max_threads: int = 4
    ):
        super().__init__(auth_config, checkpoint_path)
        self.max_threads = max_threads
        self._validate_auth_config()
        self.file_system_client: Optional[FileSystemClient] = None

    def _validate_auth_config(self):
        """
        Validate the authentication configuration for Azure Data Lake.

        Ensures all required fields are present and correctly formatted.

        Raises:
            ValueError: If any required field is missing or invalid.
        """
        required_fields = [
            "storage_account_name",
            "file_system_name",
            "connection_string",
        ]
        credentials = getattr(self.auth_config, "credentials", {})

        if not credentials:
            raise ValueError("Credentials cannot be None or empty")

        if not set(required_fields).issubset(credentials.keys()):
            raise ValueError(
                f"Credentials must include {required_fields}. Provided keys: {list(credentials.keys())}"
            )

    def _connect_to_source(self):
        """
        Establish connection to Azure Data Lake.

        Returns:
            DataLakeServiceClient: Azure Data Lake connection client.

        Raises:
            DataSourceConnectionError: If connection fails.
        """
        try:
            if getattr(self.auth_config, "method", None) == "connection_string":
                credentials = self.auth_config.credentials
                connection_string = credentials.get("connection_string")

                if not connection_string:
                    raise ValueError(
                        "Connection string is required for Azure Data Lake"
                    )

                service_client = DataLakeServiceClient.from_connection_string(
                    conn_str=connection_string
                )
                self.file_system_client = service_client.get_file_system_client(
                    file_system=credentials.get("file_system_name")
                )
                logger.info(
                    "Connected to Azure Data Lake file system: %s",
                    credentials.get("file_system_name"),
                )
                return service_client

            else:
                raise ValueError(
                    f"Unsupported authentication method: {self.auth_config.method}"
                )

        except Exception as e:
            logger.error("Failed to connect to Azure Data Lake: %s", str(e))
            raise DataSourceConnectionError(
                f"Failed to connect to Azure Data Lake: {e}"
            ) from e

    def _read_single_file(self, file_path: str) -> pd.DataFrame:
        """
        Read a single file from Azure Data Lake into a DataFrame.

        Args:
            file_path (str): Path to the file within the file system.

        Returns:
            pd.DataFrame: Extracted data.
        """
        try:
            file_client = self.file_system_client.get_file_client(file_path)
            download = file_client.download_file()
            file_content = download.readall()

            # Assuming parquet format; can extend to csv/json based on self.auth_config
            df = pd.read_parquet(pd.io.common.BytesIO(file_content))
            logger.debug("Read file: %s, rows: %d", file_path, len(df))
            return df

        except Exception as e:
            logger.error("Failed to read file %s: %s", file_path, str(e))
            return (
                pd.DataFrame()
            )  # Return empty DataFrame on failure to avoid stopping extraction

    def extract_data(
        self, file_paths: List[str], use_multithreading: bool = True
    ) -> Iterator[pd.DataFrame]:
        """
        Extract data from multiple files, optionally using multi-threading.

        Args:
            file_paths (List[str]): List of file paths to extract.
            use_multithreading (bool): Whether to use ThreadPoolExecutor. Defaults to True.

        Yields:
            Iterator[pd.DataFrame]: Iterator of DataFrames from each file.
        """
        logger.info(
            "Starting extraction of %d files. Multi-threading: %s",
            len(file_paths),
            use_multithreading,
        )

        if use_multithreading:
            with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                future_to_file = {
                    executor.submit(self._read_single_file, fp): fp for fp in file_paths
                }
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        df = future.result()
                        yield df
                    except Exception as e:
                        logger.error("Error extracting file %s: %s", file_path, str(e))
        else:
            for fp in file_paths:
                yield self._read_single_file(fp)

    def stream_extract_data(
        self,
        file_paths: List[str],
        chunk_size: int = 100000,
        use_multithreading: bool = True,
    ) -> Iterator[pd.DataFrame]:
        """
        Stream large files from Azure Data Lake in chunks.

        Args:
            file_paths (List[str]): List of file paths.
            chunk_size (int): Number of rows per chunk. Defaults to 100,000.
            use_multithreading (bool): Enable multi-threaded reading.

        Yields:
            Iterator[pd.DataFrame]: DataFrames in chunks.
        """
        for df in self.extract_data(file_paths, use_multithreading):
            if df.empty:
                continue
            # Yield in chunks to avoid memory overload
            for start in range(0, len(df), chunk_size):
                end = min(start + chunk_size, len(df))
                yield df.iloc[start:end]
