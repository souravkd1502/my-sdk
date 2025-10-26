"""
api_extractor.py
================
Optimized API data extraction classes for the ETL framework with checkpoint support.

This module provides robust, memory-efficient extractors for API data sources
with proper interface compliance, unified request handling, and comprehensive
error management. Additionally, it supports **incremental extraction** using
checkpoints stored in pluggable backends (File, Redis, SQLite, etc.), allowing
resumable and reliable ETL operations.

Key Features
------------
- Interface compliant with BaseExtractor (returns Dict from extract())
- Unified request handling with proper session management
- Memory-efficient streaming with chunked data creation
- Incremental extraction support using CheckpointManager
    - Stores last processed record or timestamp per endpoint
    - Resumable extraction after failures
    - Pluggable storage backends: file system, Redis, SQLite, PostgreSQL
- Comprehensive error handling using ETL framework exceptions
- Connection pooling and retry mechanisms
- Proper resource cleanup and context manager support
- Structured logging and metadata collection with extraction timing

Classes
-------
- BaseAPIExtractor: Abstract base for all API extractors
- RESTAPIExtractor: Production-ready REST API extractor with pagination and checkpoint support

Usage Example
-------------
>>> from utilities.checkpoint import CheckpointManager, FileCheckpointBackend
>>> checkpoint_mgr = CheckpointManager(FileCheckpointBackend("checkpoints.json"))
>>>
>>> with RESTAPIExtractor(
...     base_url="https://api.example.com",
...     pagination_type="page",
...     page_size=100,
...     checkpoint_manager=checkpoint_mgr,
...     checkpoint_field="updated_at"
... ) as extractor:
...     result = extractor.extract_endpoint("/users", incremental_field="updated_at")
...     metadata = extractor.get_metadata()
...     print(f"Extracted {result.get('count', 0)} records")
...
>>> # The last checkpoint is automatically updated after each batch/page.
"""

import time
import logging
import requests
import pandas as pd
from textwrap import dedent
from urllib.parse import urljoin
from abc import ABC, abstractmethod
from requests.adapters import HTTPAdapter, Retry
from typing import Any, Dict, List, Optional, Union, Generator

# Import ETL framework modules with proper error handling
try:
    from ..core.errors import (
        ExtractorError,
        DataSourceConnectionError,
        DataSourceAuthenticationError,
        DataReadError,
    )
    from .base import BaseExtractor
    from ..core.checkpoint import CheckpointManager
except ImportError:
    # Fallback for standalone usage
    import sys
    from pathlib import Path

    etl_framework_path = Path(__file__).parent.parent
    sys.path.insert(0, str(etl_framework_path))

    try:
        from core.errors import (
            ExtractorError,
            DataSourceConnectionError,
            DataSourceAuthenticationError,
            DataReadError,
        )
        from .base import BaseExtractor
        from ..core.checkpoint import CheckpointManager
    except ImportError as e:
        raise ImportError(
            f"Could not import ETL framework dependencies: {e}. "
            "Ensure the ETL framework is properly installed."
        )

# Module logger - use proper initialization pattern
logger = logging.getLogger(__name__)


class APISessionManager:
    """
    Manages HTTP sessions with connection pooling and retry logic.

    Provides a reusable session configuration for API extractors with
    optimized connection pooling, retry strategies, and proper cleanup.
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        timeout: int = 30,
        pool_connections: int = 20,
        pool_maxsize: int = 20,
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self.pool_connections = pool_connections
        self.pool_maxsize = pool_maxsize
        self._session = None

    @property
    def session(self) -> requests.Session:
        """Get or create the configured session."""
        if self._session is None:
            self._session = self._create_session()
        return self._session

    def _create_session(self) -> requests.Session:
        """Create a new session with optimized configuration."""
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE"],
            raise_on_status=False,
        )

        # Configure adapter with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.pool_connections,
            pool_maxsize=self.pool_maxsize,
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)

        logger.debug(
            f"Created session with retries={self.max_retries}, "
            f"pool_size={self.pool_maxsize}, timeout={self.timeout}"
        )

        return session

    def close(self):
        """Close the session and cleanup resources."""
        if self._session:
            self._session.close()
            self._session = None
            logger.debug("Session closed and resources cleaned up")


class BaseAPIExtractor(BaseExtractor, ABC):
    """
    Abstract base class for API data extraction with interface compliance.

    This class provides a foundation for building extractors that interact with
    APIs while ensuring compliance with the BaseExtractor interface. Key features:

    - Returns Dict from extract() method (interface compliant)
    - Provides separate metadata access via get_metadata()
    - Unified request handling with proper error mapping
    - Session management with context manager support
    - Request logging and performance tracking

    Subclasses must implement the _extract_data() method to define specific
    extraction logic while the extract() method handles data conversion.
    """

    def __init__(
        self,
        base_url: str,
        auth: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ) -> None:
        """
        Initialize the API extractor.

        Parameters
        ----------
        base_url : str
            The root URL of the API (e.g., "https://api.example.com")
        auth : Any, optional
            Authentication tuple or requests.auth.AuthBase instance
        headers : Dict[str, str], optional
            HTTP headers to send with requests
        params : Dict[str, Any], optional
            Query parameters to send with requests
        timeout : int, default 30
            Request timeout in seconds
        max_retries : int, default 3
            Maximum number of retries for failed requests
        backoff_factor : float, default 0.5
            Backoff factor for retry delays

        Raises
        ------
        ValueError
            If base_url is empty or invalid
        """
        # Validate parameters
        if not base_url or not isinstance(base_url, str):
            raise ValueError("base_url must be a non-empty string")

        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.headers = headers or {}
        self.params = params or {}

        # Initialize session manager
        self._session_manager = APISessionManager(
            max_retries=max_retries, backoff_factor=backoff_factor, timeout=timeout
        )

        # Internal state
        self._request_log: List[Dict[str, Any]] = []
        self._last_metadata: Optional[Dict[str, Any]] = None
        self._is_closed = False
        self._extraction_start_time: Optional[float] = None
        self._extraction_end_time: Optional[float] = None

        logger.debug(
            f"Initialized {self.__class__.__name__} with base_url={self.base_url}"
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()

    def close(self):
        """Close the extractor and cleanup resources."""
        if not self._is_closed:
            self._session_manager.close()
            self._is_closed = True
            logger.debug("API extractor closed")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()

    # ─────────────────────────────
    # Interface Compliance Methods
    # ─────────────────────────────

    def extract(self) -> Dict[str, Any]:
        """
        Extract data from the API and return as Dict.

        This method ensures interface compliance with BaseExtractor by always
        returning a dictionary. It delegates to _extract_data() for the
        actual extraction logic and handles data conversion.

        Returns
        -------
        Dict[str, Any]
            Extracted data as a dictionary

        Raises
        ------
        ExtractorError
            If extraction fails
        DataReadError
            If data cannot be converted to dictionary
        """
        try:
            # Track extraction timing
            self._extraction_start_time = time.time()

            data = self._extract_data()
            result_dict = self._convert_to_dict(data)

            self._extraction_end_time = time.time()

            # Create and store metadata
            self._last_metadata = self._create_metadata(result_dict)

            logger.info(
                f"Successfully extracted data with {len(result_dict)} top-level keys"
            )
            return result_dict

        except Exception as e:
            self._extraction_end_time = time.time()
            if isinstance(
                e,
                (
                    ExtractorError,
                    DataSourceConnectionError,
                    DataSourceAuthenticationError,
                    DataReadError,
                ),
            ):
                raise
            logger.error(f"Unexpected error during extraction: {e}")
            raise ExtractorError(f"Extraction failed: {e}") from e

    @abstractmethod
    def _extract_data(self) -> Union[List[Dict], Dict, pd.DataFrame]:
        """
        Abstract method to extract raw data from the API.

        Subclasses must implement this method to define their specific
        extraction logic. The data will be converted to Dict by extract().

        Returns
        -------
        Union[List[Dict], Dict, pd.DataFrame]
            Raw extracted data
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata from the last extraction.

        Returns
        -------
        Dict[str, Any]
            Metadata dictionary containing extraction details
        """
        if self._last_metadata is None:
            return {"message": "No extraction performed yet"}
        return self._last_metadata.copy()

    def get_request_log(self) -> List[Dict[str, Any]]:
        """
        Get the request log for debugging purposes.

        Returns
        -------
        List[Dict[str, Any]]
            List of request details
        """
        return self._request_log.copy()

    # ─────────────────────────────
    # Request Handling
    # ─────────────────────────────

    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> requests.Response:
        """
        Make a unified HTTP request with proper error handling.

        Parameters
        ----------
        endpoint : str
            API endpoint relative to base_url
        method : str, default "GET"
            HTTP method
        data : Dict[str, Any], optional
            Request body data
        **kwargs
            Additional arguments for requests

        Returns
        -------
        requests.Response
            HTTP response object

        Raises
        ------
        DataSourceConnectionError
            For connection or HTTP errors
        DataSourceAuthenticationError
            For authentication failures
        ExtractorError
            For unexpected errors
        """
        if self._is_closed:
            raise ExtractorError("Cannot make request: extractor is closed")

        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))
        start_time = time.time()

        try:
            logger.debug(f"Making {method} request to {url}")

            # Prepare request parameters
            request_kwargs = {
                "method": method,
                "url": url,
                "auth": self.auth,
                "headers": self.headers,
                "params": self.params,
                "timeout": self._session_manager.timeout,
                **kwargs,
            }

            if data is not None:
                request_kwargs["json"] = data

            # Make request using session
            response = self._session_manager.session.request(**request_kwargs)
            duration = time.time() - start_time

            # Log request details
            self._log_request(response, duration, request_kwargs)

            # Handle HTTP errors
            self._handle_response_errors(response)

            logger.debug(
                f"Request successful: {response.status_code} in {duration:.3f}s"
            )
            return response

        except requests.exceptions.Timeout as e:
            logger.error(f"Request timeout for {url}: {e}")
            raise DataSourceConnectionError(f"Request timeout: {url}") from e

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error for {url}: {e}")
            raise DataSourceConnectionError(f"Connection failed: {url}") from e

        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception for {url}: {e}")
            raise DataSourceConnectionError(f"Request failed: {e}") from e

        except Exception as e:
            logger.exception(f"Unexpected error during request to {url}")
            raise ExtractorError(f"Unexpected request error: {e}") from e

    def _handle_response_errors(self, response: requests.Response):
        """Handle HTTP response errors with proper exception mapping."""
        if response.status_code == 401:
            raise DataSourceAuthenticationError(
                f"Authentication failed: {response.status_code}"
            )
        elif response.status_code == 403:
            raise DataSourceAuthenticationError(
                f"Access forbidden: {response.status_code}"
            )
        elif 400 <= response.status_code < 500:
            raise DataSourceConnectionError(
                f"Client error {response.status_code}: {response.text}"
            )
        elif response.status_code >= 500:
            raise DataSourceConnectionError(
                f"Server error {response.status_code}: {response.text}"
            )

    def _log_request(
        self,
        response: requests.Response,
        duration: float,
        request_kwargs: Dict[str, Any],
    ):
        """Log request details for metadata and debugging."""
        request_info = {
            "url": str(response.url),
            "method": request_kwargs.get("method", "GET"),
            "status_code": response.status_code,
            "duration_sec": round(duration, 3),
            "response_size_bytes": len(response.content),
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        self._request_log.append(request_info)

        logger.info(
            f"Request logged: {request_info['method']} {request_info['url']} "
            f"-> {request_info['status_code']} ({request_info['duration_sec']}s)"
        )

    # ─────────────────────────────
    # Data Conversion and Metadata
    # ─────────────────────────────

    def _convert_to_dict(
        self, data: Union[List[Dict], Dict, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Convert various data formats to Dictionary.

        Parameters
        ----------
        data : Union[List[Dict], Dict, pd.DataFrame]
            Raw data to convert

        Returns
        -------
        Dict[str, Any]
            Converted Dictionary

        Raises
        ------
        DataReadError
            If conversion fails
        """
        try:
            if isinstance(data, dict):
                return data
            elif isinstance(data, list):
                if not data:
                    return {}
                return {"data": data, "count": len(data)}
            elif isinstance(data, pd.DataFrame):
                return {
                    "data": data.to_dict("records"),
                    "count": len(data),
                    "columns": data.columns.tolist(),
                }
            else:
                raise DataReadError(f"Cannot convert {type(data)} to Dictionary")

        except Exception as e:
            logger.error(f"Dictionary conversion failed: {e}")
            raise DataReadError(f"Failed to convert data to Dictionary: {e}") from e

    def _create_metadata(self, result_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive metadata for the extracted data."""
        try:
            # Calculate request statistics
            total_requests = len(self._request_log)
            if total_requests > 0:
                avg_duration = (
                    sum(r["duration_sec"] for r in self._request_log) / total_requests
                )
                total_bytes = sum(r["response_size_bytes"] for r in self._request_log)

                # Calculate response code statistics
                response_codes = {}
                success_count = 0
                for request in self._request_log:
                    status_code = request.get("status_code", 0)
                    response_codes[status_code] = response_codes.get(status_code, 0) + 1
                    if 200 <= status_code < 300:
                        success_count += 1

                success_rate = (
                    round((success_count / total_requests) * 100, 1)
                    if total_requests > 0
                    else 0
                )
            else:
                avg_duration = 0
                total_bytes = 0
                response_codes = {}
                success_rate = 0

            # Calculate total extraction time
            extraction_duration = 0
            if self._extraction_start_time and self._extraction_end_time:
                extraction_duration = round(
                    self._extraction_end_time - self._extraction_start_time, 3
                )

            # Determine data count - count actual records, not response objects
            data_count = 0
            if "data" in result_dict and isinstance(result_dict["data"], list):
                # Count actual records within the data structure
                for item in result_dict["data"]:
                    if isinstance(item, dict):
                        # Look for common record array field names
                        for key in [
                            "characters",
                            "data",
                            "results",
                            "items",
                            "records",
                            "entries",
                        ]:
                            if key in item and isinstance(item[key], list):
                                data_count += len(item[key])
                                break
                        else:
                            # If no array field found, count the item itself as one record
                            data_count += 1
                    else:
                        # If item is not a dict, count it as one record
                        data_count += 1
            elif "count" in result_dict:
                data_count = result_dict["count"]
            elif isinstance(result_dict, dict):
                data_count = len(result_dict)

            metadata = {
                "extraction_time": pd.Timestamp.now().isoformat(),
                "source": self.base_url,
                "total_extraction_duration_sec": extraction_duration,
                "data_info": {
                    "record_count": data_count,
                    "top_level_keys": list(result_dict.keys()),
                },
                "request_statistics": {
                    "total_requests": total_requests,
                    "avg_duration_sec": round(avg_duration, 3),
                    "total_response_bytes": total_bytes,
                    "total_extraction_time_sec": extraction_duration,
                    "response_codes": response_codes,
                    "success_rate_percent": success_rate,
                },
            }

            logger.debug(f"Created metadata: {metadata['data_info']}")
            return metadata

        except Exception as e:
            logger.warning(f"Metadata creation failed: {e}")
            return {
                "extraction_time": pd.Timestamp.now().isoformat(),
                "source": self.base_url,
                "error": f"Metadata creation failed: {e}",
            }


class RESTAPIExtractor(BaseAPIExtractor):
    """
    Production-ready REST API extractor with advanced features.

    Features
    --------
    - Multiple pagination strategies (page, cursor, offset)
    - Memory-efficient streaming with configurable batch sizes
    - Incremental extraction support
    - Automatic response format detection
    - Rate limiting and request optimization
    - Comprehensive error handling and recovery

    Usage Example
    -------------
    >>> with RESTAPIExtractor(
    ...     base_url="https://api.example.com",
    ...     pagination_type="page",
    ...     page_size=100,
    ...     max_pages=10
    ... ) as extractor:
    ...     result = extractor.extract("/users")
    ...     print(f"Extracted {result.get('count', 0)} users")
    ...
    ...     # Stream large datasets
    ...     for batch_dict in extractor.stream_extract("/transactions"):
    ...         process_batch(batch_dict)
    """

    def __init__(
        self,
        base_url: str,
        auth: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        pagination_type: str = "page",
        pagination_key: str = "page",
        page_size: int = 100,
        max_pages: Optional[int] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        checkpoint_key_template: str = "{base_url}:{endpoint}",
        checkpoint_field: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize REST API extractor.

        Parameters
        ----------
        base_url : str
            API base URL
        auth : Any, optional
            Authentication configuration
        headers : Dict[str, str], optional
            HTTP headers
        params : Dict[str, Any], optional
            Query parameters
        pagination_type : str, default "page"
            Pagination strategy: "page", "cursor", "offset", or "none"
        pagination_key : str, default "page"
            Parameter name for pagination
        page_size : int, default 100
            Number of records per page
        max_pages : int, optional
            Maximum pages to fetch (None for unlimited)
        checkpoint_manager : CheckpointManager, optional
            CheckpointManager instance for tracking extraction state
        checkpoint_key_template : str, default "{base_url}:{endpoint}"
            Template for generating checkpoint keys. Available variables: base_url, endpoint
        checkpoint_field : str, optional
            Field name in the response data to use as checkpoint value (e.g., 'timestamp', 'id')
        **kwargs
            Additional arguments for BaseAPIExtractor
        """
        super().__init__(base_url, auth, headers, params, **kwargs)

        # Validate pagination configuration
        valid_pagination_types = {"page", "cursor", "offset", "none"}
        if pagination_type not in valid_pagination_types:
            raise ValueError(f"pagination_type must be one of {valid_pagination_types}")

        if page_size <= 0:
            raise ValueError("page_size must be positive")

        self.pagination_type = pagination_type
        self.pagination_key = pagination_key
        self.page_size = page_size
        self.max_pages = max_pages

        # Checkpoint management
        self.checkpoint_manager = checkpoint_manager
        self.checkpoint_key_template = checkpoint_key_template
        self.checkpoint_field = checkpoint_field

        # State for pagination
        self._current_page = 1
        self._next_token = None
        self._current_offset = 0

        # Checkpoint state
        self._current_checkpoint_key = None
        self._last_checkpoint_value = None
        self._latest_checkpoint_value = None

        logger.debug(
            f"Initialized RESTAPIExtractor with pagination_type={pagination_type}, "
            f"page_size={page_size}, checkpoint_enabled={checkpoint_manager is not None}"
        )

    def _generate_checkpoint_key(self, endpoint: str) -> str:
        """
        Generate a checkpoint key for the given endpoint.

        Parameters
        ----------
        endpoint : str
            API endpoint

        Returns
        -------
        str
            Generated checkpoint key
        """
        # Clean the base URL and endpoint for key generation
        clean_base_url = self.base_url.replace("://", "_").replace("/", "_")
        clean_endpoint = endpoint.replace("/", "_").strip("_")

        # Use the template to generate the key
        checkpoint_key = self.checkpoint_key_template.format(
            base_url=clean_base_url, endpoint=clean_endpoint
        )

        logger.debug(f"Generated checkpoint key: {checkpoint_key}")
        return checkpoint_key

    def _load_checkpoint(self, endpoint: str) -> Optional[str]:
        """
        Load the last checkpoint value for the given endpoint.

        Parameters
        ----------
        endpoint : str
            API endpoint

        Returns
        -------
        Optional[str]
            Last checkpoint value or None if not found
        """
        if not self.checkpoint_manager:
            return None

        checkpoint_key = self._generate_checkpoint_key(endpoint)
        self._current_checkpoint_key = checkpoint_key

        checkpoint_value = self.checkpoint_manager.get_checkpoint(checkpoint_key)
        if checkpoint_value:
            logger.info(f"Loaded checkpoint for {endpoint}: {checkpoint_value}")
            self._last_checkpoint_value = checkpoint_value
        else:
            logger.info(f"No checkpoint found for {endpoint}")
            self._last_checkpoint_value = None

        return checkpoint_value

    def _save_checkpoint(self, endpoint: str, checkpoint_value: str) -> None:
        """
        Save a checkpoint value for the given endpoint.

        Parameters
        ----------
        endpoint : str
            API endpoint
        checkpoint_value : str
            Checkpoint value to save
        """
        if not self.checkpoint_manager:
            return

        checkpoint_key = self._generate_checkpoint_key(endpoint)
        self.checkpoint_manager.set_checkpoint(checkpoint_key, checkpoint_value)
        self._latest_checkpoint_value = checkpoint_value

        logger.info(f"Saved checkpoint for {endpoint}: {checkpoint_value}")

    def _extract_checkpoint_value(self, records: List[Dict[str, Any]]) -> Optional[str]:
        """
        Extract checkpoint value from records based on the configured checkpoint field.

        Parameters
        ----------
        records : List[Dict[str, Any]]
            Records to extract checkpoint value from

        Returns
        -------
        Optional[str]
            Extracted checkpoint value or None
        """
        if not self.checkpoint_field or not records:
            return None

        # Find the latest/highest checkpoint value from all records
        checkpoint_values = []
        for record in records:
            if self.checkpoint_field in record:
                value = record[self.checkpoint_field]
                if value is not None:
                    checkpoint_values.append(str(value))

        if not checkpoint_values:
            return None

        # For timestamp fields, get the latest one
        # For ID fields, get the highest one
        # We'll try to be smart about this by checking if it looks like a timestamp
        try:
            # Try to parse as timestamp first
            import dateutil.parser

            parsed_values = []
            for val in checkpoint_values:
                try:
                    parsed_values.append((dateutil.parser.parse(val), val))
                except Exception:
                    # If parsing fails, fall back to string comparison
                    return max(checkpoint_values)

            # Return the latest timestamp
            if parsed_values:
                return max(parsed_values, key=lambda x: x[0])[1]
        except ImportError:
            # If dateutil is not available, fall back to string/numeric comparison
            try:
                # Try numeric comparison
                return str(max(float(val) for val in checkpoint_values))
            except (ValueError, TypeError):
                # Fall back to string comparison
                return max(checkpoint_values)

        return None

    def _extract_data(self) -> List[Dict[str, Any]]:
        """Extract all data using pagination."""
        all_records = []

        for batch in self._stream_extract_internal():
            all_records.extend(batch)

        return all_records

    def extract_endpoint(
        self,
        endpoint: str,
        incremental_field: Optional[str] = None,
        last_checkpoint: Optional[str] = None,
        auto_checkpoint: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract data from a specific endpoint with checkpoint support.

        Parameters
        ----------
        endpoint : str
            API endpoint to extract from
        incremental_field : str, optional
            Field for incremental extraction (overrides instance checkpoint_field)
        last_checkpoint : str, optional
            Last checkpoint value for incremental extraction (overrides stored checkpoint)
        auto_checkpoint : bool, default True
            Whether to automatically save checkpoints during extraction

        Returns
        -------
        Dict[str, Any]
            Extracted data
        """
        # Store current endpoint for extraction
        self._current_endpoint = endpoint
        self._incremental_field = incremental_field or self.checkpoint_field
        self._auto_checkpoint = auto_checkpoint

        # Load checkpoint if not explicitly provided and auto_checkpoint is enabled
        if last_checkpoint is None and auto_checkpoint:
            last_checkpoint = self._load_checkpoint(endpoint)

        self._last_checkpoint = last_checkpoint

        # Extract data
        result = self.extract()

        # Save final checkpoint if auto_checkpoint is enabled and we have a latest value
        if auto_checkpoint and self._latest_checkpoint_value:
            self._save_checkpoint(endpoint, self._latest_checkpoint_value)

        return result

    def stream_extract(
        self,
        endpoint: str,
        batch_size: int = 500,
        incremental_field: Optional[str] = None,
        last_checkpoint: Optional[str] = None,
        auto_checkpoint: bool = True,
        checkpoint_frequency: int = 1,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream data from API in Dictionary batches with checkpoint support.

        Parameters
        ----------
        endpoint : str
            API endpoint to extract from
        batch_size : int, default 500
            Records per batch
        incremental_field : str, optional
            Field for incremental extraction (overrides instance checkpoint_field)
        last_checkpoint : str, optional
            Last checkpoint value (overrides stored checkpoint)
        auto_checkpoint : bool, default True
            Whether to automatically save checkpoints during extraction
        checkpoint_frequency : int, default 1
            Save checkpoint every N batches (1 = every batch)

        Yields
        ------
        Dict[str, Any]
            Batches of data as Dictionaries
        """
        self._current_endpoint = endpoint
        self._incremental_field = incremental_field or self.checkpoint_field
        self._auto_checkpoint = auto_checkpoint
        self._checkpoint_frequency = checkpoint_frequency

        # Load checkpoint if not explicitly provided and auto_checkpoint is enabled
        if last_checkpoint is None and auto_checkpoint:
            last_checkpoint = self._load_checkpoint(endpoint)

        self._last_checkpoint = last_checkpoint

        batch_records = []
        batch_count = 0

        for records in self._stream_extract_internal():
            batch_records.extend(records)

            while len(batch_records) >= batch_size:
                # Extract batch and convert to Dict
                batch = batch_records[:batch_size]
                batch_records = batch_records[batch_size:]

                dict_batch = self._convert_to_dict(batch)
                batch_count += 1

                # Extract and save checkpoint for this batch if enabled
                if auto_checkpoint and self._incremental_field:
                    batch_checkpoint = self._extract_checkpoint_value(batch)
                    if batch_checkpoint:
                        self._latest_checkpoint_value = batch_checkpoint

                        # Save checkpoint based on frequency
                        if batch_count % checkpoint_frequency == 0:
                            self._save_checkpoint(endpoint, batch_checkpoint)

                logger.debug(
                    f"Yielding batch {batch_count} with {dict_batch.get('count', 0)} records"
                )
                yield dict_batch

        # Yield remaining records
        if batch_records:
            batch = batch_records
            dict_batch = self._convert_to_dict(batch)
            batch_count += 1

            # Extract and save checkpoint for final batch if enabled
            if auto_checkpoint and self._incremental_field:
                batch_checkpoint = self._extract_checkpoint_value(batch)
                if batch_checkpoint:
                    self._latest_checkpoint_value = batch_checkpoint
                    self._save_checkpoint(endpoint, batch_checkpoint)

            logger.debug(
                f"Yielding final batch {batch_count} with {dict_batch.get('count', 0)} records"
            )
            yield dict_batch

        # Save final checkpoint if we have one and auto_checkpoint is enabled
        if auto_checkpoint and self._latest_checkpoint_value:
            self._save_checkpoint(endpoint, self._latest_checkpoint_value)

    def _stream_extract_internal(self) -> Generator[List[Dict[str, Any]], None, None]:
        """Internal generator for streaming raw records with checkpoint tracking."""
        endpoint = getattr(self, "_current_endpoint", "/")
        incremental_field = getattr(self, "_incremental_field", None)
        last_checkpoint = getattr(self, "_last_checkpoint", None)
        auto_checkpoint = getattr(self, "_auto_checkpoint", False)

        page_count = 0

        # Reset pagination state
        self._current_page = 1
        self._next_token = None
        self._current_offset = 0

        while True:
            # Check page limit
            if self.max_pages and page_count >= self.max_pages:
                logger.info(f"Reached maximum pages limit: {self.max_pages}")
                break

            # Build request parameters
            params = self._build_pagination_params()

            if incremental_field and last_checkpoint:
                params[incremental_field] = last_checkpoint

            # Make request
            try:
                response = self._make_request(endpoint, params=params)
                data = response.json()

            except Exception as e:
                logger.error(f"Failed to fetch page {page_count + 1}: {e}")
                break

            # Extract records from response
            records = self._extract_records_from_response(data)

            if not records:
                logger.info("No more records found, ending extraction")
                break

            # Extract checkpoint value from records if checkpoint field is configured
            if auto_checkpoint and incremental_field:
                page_checkpoint = self._extract_checkpoint_value(records)
                if page_checkpoint:
                    self._latest_checkpoint_value = page_checkpoint
                    logger.debug(
                        f"Updated checkpoint value from page {page_count + 1}: {page_checkpoint}"
                    )

            logger.info(f"Fetched {len(records)} records (page {page_count + 1})")
            yield records

            # Update pagination state
            if not self._update_pagination_state(data):
                logger.info("No more pages available")
                break

            page_count += 1

        logger.info(f"Completed extraction: {page_count} pages processed")

    def _build_pagination_params(self) -> Dict[str, Any]:
        """Build pagination parameters for the current page."""
        params = {}

        if self.pagination_type == "page":
            params[self.pagination_key] = self._current_page
            params["limit"] = self.page_size
        elif self.pagination_type == "cursor" and self._next_token:
            params[self.pagination_key] = self._next_token
            params["limit"] = self.page_size
        elif self.pagination_type == "offset":
            params["offset"] = self._current_offset
            params["limit"] = self.page_size
        elif self.pagination_type == "none":
            params["limit"] = self.page_size

        return params

    def _extract_records_from_response(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract records from API response, handling various formats."""
        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            # Try common data field names
            for key in ["data", "results", "items", "records", "entries"]:
                if key in data and isinstance(data[key], list):
                    return data[key]

            # If no list found, treat as single record
            return [data]

        logger.warning(f"Unexpected response format: {type(data)}")
        return []

    def _update_pagination_state(self, data: Dict[str, Any]) -> bool:
        """Update pagination state and return True if more pages available."""
        if self.pagination_type == "page":
            self._current_page += 1
            return True  # Will break on empty response

        elif self.pagination_type == "cursor":
            # Look for next token in various common fields
            next_token = None
            if isinstance(data, dict):
                for key in [
                    "next",
                    "next_page_token",
                    "next_cursor",
                    "continuation_token",
                ]:
                    if key in data:
                        next_token = data[key]
                        break

            if next_token:
                self._next_token = next_token
                return True
            return False

        elif self.pagination_type == "offset":
            self._current_offset += self.page_size
            return True  # Will break on empty response

        return False  # No pagination

    # ─────────────────────────────
    # Checkpoint Management Utilities
    # ─────────────────────────────

    def get_checkpoint_info(self, endpoint: str) -> Dict[str, Any]:
        """
        Get checkpoint information for a specific endpoint.

        Parameters
        ----------
        endpoint : str
            API endpoint

        Returns
        -------
        Dict[str, Any]
            Checkpoint information including key, value, and status
        """
        if not self.checkpoint_manager:
            return {
                "checkpoint_enabled": False,
                "message": "No checkpoint manager configured",
            }

        checkpoint_key = self._generate_checkpoint_key(endpoint)
        checkpoint_value = self.checkpoint_manager.get_checkpoint(checkpoint_key)

        return {
            "checkpoint_enabled": True,
            "endpoint": endpoint,
            "checkpoint_key": checkpoint_key,
            "checkpoint_value": checkpoint_value,
            "has_checkpoint": checkpoint_value is not None,
            "checkpoint_field": self.checkpoint_field,
            "latest_value": self._latest_checkpoint_value,
        }

    def delete_checkpoint(self, endpoint: str) -> bool:
        """
        Delete the checkpoint for a specific endpoint.

        Parameters
        ----------
        endpoint : str
            API endpoint

        Returns
        -------
        bool
            True if checkpoint was deleted, False if no checkpoint manager or no checkpoint found
        """
        if not self.checkpoint_manager:
            logger.warning("No checkpoint manager configured")
            return False

        checkpoint_key = self._generate_checkpoint_key(endpoint)

        try:
            self.checkpoint_manager.delete_checkpoint(checkpoint_key)
            logger.info(f"Deleted checkpoint for endpoint: {endpoint}")

            # Clear internal state if this was the current checkpoint
            if self._current_checkpoint_key == checkpoint_key:
                self._last_checkpoint_value = None
                self._latest_checkpoint_value = None
                self._current_checkpoint_key = None

            return True
        except Exception as e:
            logger.error(f"Failed to delete checkpoint for {endpoint}: {e}")
            return False

    def list_checkpoints(self) -> Dict[str, Any]:
        """
        List all checkpoints managed by this extractor.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing checkpoint information
        """
        if not self.checkpoint_manager:
            return {
                "checkpoint_enabled": False,
                "message": "No checkpoint manager configured",
                "checkpoints": {},
            }

        try:
            all_checkpoints = self.checkpoint_manager.list_checkpoints()

            # Filter checkpoints that belong to this extractor (match base_url pattern)
            clean_base_url = self.base_url.replace("://", "_").replace("/", "_")
            extractor_checkpoints = {}

            for key, value in all_checkpoints.items():
                if key.startswith(clean_base_url):
                    # Extract endpoint from key
                    try:
                        endpoint_part = key.split(":", 1)[1] if ":" in key else key
                        endpoint = endpoint_part.replace("_", "/")
                        extractor_checkpoints[endpoint] = {
                            "checkpoint_key": key,
                            "checkpoint_value": value,
                            "endpoint": endpoint,
                        }
                    except (IndexError, AttributeError):
                        # If key format doesn't match expected pattern, include anyway
                        extractor_checkpoints[key] = {
                            "checkpoint_key": key,
                            "checkpoint_value": value,
                            "endpoint": "unknown",
                        }

            return {
                "checkpoint_enabled": True,
                "base_url": self.base_url,
                "checkpoint_count": len(extractor_checkpoints),
                "checkpoints": extractor_checkpoints,
            }

        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return {"checkpoint_enabled": True, "error": str(e), "checkpoints": {}}

    def reset_checkpoints(
        self, endpoints: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Reset (delete) checkpoints for specified endpoints or all endpoints.

        Parameters
        ----------
        endpoints : List[str], optional
            List of endpoints to reset. If None, resets all checkpoints for this extractor.

        Returns
        -------
        Dict[str, bool]
            Dictionary mapping endpoints to success status
        """
        if not self.checkpoint_manager:
            logger.warning("No checkpoint manager configured")
            return {}

        results = {}

        if endpoints is None:
            # Get all checkpoints for this extractor
            checkpoint_info = self.list_checkpoints()
            if checkpoint_info.get("checkpoint_enabled"):
                endpoints = list(checkpoint_info["checkpoints"].keys())
            else:
                return {}

        for endpoint in endpoints:
            success = self.delete_checkpoint(endpoint)
            results[endpoint] = success

        logger.info(f"Reset {sum(results.values())} out of {len(results)} checkpoints")
        return results

    def get_current_checkpoint_status(self) -> Dict[str, Any]:
        """
        Get the current checkpoint status of the extractor.

        Returns
        -------
        Dict[str, Any]
            Current checkpoint status and configuration
        """
        return {
            "checkpoint_manager_configured": self.checkpoint_manager is not None,
            "checkpoint_field": self.checkpoint_field,
            "checkpoint_key_template": self.checkpoint_key_template,
            "current_checkpoint_key": self._current_checkpoint_key,
            "last_checkpoint_value": self._last_checkpoint_value,
            "latest_checkpoint_value": self._latest_checkpoint_value,
            "auto_checkpoint_enabled": getattr(self, "_auto_checkpoint", False),
        }

    def set_checkpoint_manually(self, endpoint: str, checkpoint_value: str) -> bool:
        """
        Manually set a checkpoint value for an endpoint.

        Parameters
        ----------
        endpoint : str
            API endpoint
        checkpoint_value : str
            Checkpoint value to set

        Returns
        -------
        bool
            True if checkpoint was set successfully, False otherwise
        """
        if not self.checkpoint_manager:
            logger.warning("No checkpoint manager configured")
            return False

        try:
            self._save_checkpoint(endpoint, checkpoint_value)
            logger.info(f"Manually set checkpoint for {endpoint}: {checkpoint_value}")
            return True
        except Exception as e:
            logger.error(f"Failed to manually set checkpoint for {endpoint}: {e}")
            return False

    @staticmethod
    def example_usage():
        """Demonstrate example usage of RESTAPIExtractor with checkpoint functionality."""

        example = dedent(
            """
            from ReusableComponents.Utilities.Services.ETLFramework.extractors.api_extractor import (
                RESTAPIExtractor, CheckpointManager, FileCheckpointBackend
            )
            
            # Basic usage without checkpoints
            with RESTAPIExtractor(
                    base_url="https://api.example.com",
                    pagination_type="page",
                    page_size=100,
                    max_pages=10
                ) as extractor:
                    result = extractor.extract_endpoint("/users")
                    print(f"Extracted {result.get('count', 0)} users")

            # Usage with checkpoint management
            backend = FileCheckpointBackend("./checkpoints")
            checkpoint_manager = CheckpointManager(backend)
            
            with RESTAPIExtractor(
                    base_url="https://api.example.com",
                    pagination_type="page",
                    page_size=100,
                    checkpoint_manager=checkpoint_manager,
                    checkpoint_field="updated_at"  # Field to use for incremental extraction
                ) as extractor:
                    # Extract with automatic checkpoint management
                    result = extractor.extract_endpoint("/users", auto_checkpoint=True)
                    
                    # Stream large datasets with checkpoints
                    for batch in extractor.stream_extract(
                        "/transactions", 
                        batch_size=500,
                        auto_checkpoint=True,
                        checkpoint_frequency=5  # Save checkpoint every 5 batches
                    ):
                        process_batch(batch)
                    
                    # Checkpoint management utilities
                    print("Checkpoint info:", extractor.get_checkpoint_info("/users"))
                    print("All checkpoints:", extractor.list_checkpoints())
                    
                    # Reset specific checkpoints if needed
                    extractor.reset_checkpoints(["/users"])
        """
        )

        print(example)
