"""

"""

# Import required libraries
import os
import time
import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Iterator, Optional, List, Union, Dict, Any

# Import ETL framework modules with proper error handling
try:
    from core.errors import (
        DataSourceConnectionError,
        DataSourceAuthenticationError,
        DataReadError,
        ExtractorError,
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
            DataReadError,
            ExtractorError,
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

class BaseQueueExtractor(BaseExtractor, ABC):
    """
    Base class for extracting data from queue systems.

    This class provides a foundation for queue-based extractors that need to
    extract data from message queues and return it as a dictionary.
    It delegates to _extract_data() for the actual extraction logic and handles
    data conversion to comply with the BaseExtractor interface.

    Attributes
    ----------
    config : dict
        Configuration dictionary for the queue connection
    checkpoint_manager : Optional[CheckpointManager]
        Manager for tracking extraction checkpoints
    """
    
    def __init__(self, config: dict):
        """
        Initialize the BaseQueueExtractor.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing queue connection details
        """
        super().__init__()
        self.config = config
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self._extraction_start_time: Optional[float] = None
        self._extraction_end_time: Optional[float] = None
        self._last_metadata: Optional[dict] = None
        
    @abstractmethod
    def _extract_data(self) -> Union[List[dict], dict, pd.DataFrame, str]:
        """
        Abstract method to extract raw data from the queue system.

        Subclasses must implement this method to define their specific
        extraction logic. The data will be converted to Dict by extract().

        Returns
        -------
        Union[List[dict], dict, pd.DataFrame, str]
            Raw extracted data from the queue

        Raises
        ------
        DataSourceConnectionError
            If connection to the queue fails
        DataSourceAuthenticationError
            If authentication fails
        DataReadError
            If reading from the queue fails
        """
        pass
    
    # ─────────────────────────────
    # Data Conversion and Metadata
    # ─────────────────────────────

    def _convert_to_dict(
        self, data: Union[List[Dict], Dict, pd.DataFrame, str]
    ) -> Dict[str, Any]:
        """
        Convert various data formats to Dictionary.

        Parameters
        ----------
        data : Union[List[Dict], Dict, pd.DataFrame, str]
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
            elif isinstance(data, str):
                return {"data": data, "type": "string"}
            else:
                raise DataReadError(f"Cannot convert {type(data)} to Dictionary")
        except Exception as e:
            logger.error(f"Dictionary conversion failed: {e}")
            raise DataReadError(f"Error converting data to Dictionary: {e}") from e

    def _create_metadata(self, result_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create metadata for the extracted data.

        Parameters
        ----------
        result_dict : Dict[str, Any]
            The extracted data as a dictionary

        Returns
        -------
        Dict[str, Any]
            Metadata including extraction time, record count, etc.
        """
        try:
            # Calculate record count
            record_count = 0
            if "count" in result_dict:
                record_count = result_dict["count"]
            elif "data" in result_dict:
                if isinstance(result_dict["data"], list):
                    record_count = len(result_dict["data"])
                elif isinstance(result_dict["data"], str):
                    record_count = 1
            elif isinstance(result_dict, dict):
                record_count = len(result_dict)
            
            # Calculate extraction duration
            extraction_duration = 0
            if self._extraction_start_time and self._extraction_end_time:
                extraction_duration = round(
                    self._extraction_end_time - self._extraction_start_time, 3
                )
            
            metadata = {
                "extraction_time": datetime.now().isoformat(),
                "source_type": "queue",
                "extraction_start_time": self._extraction_start_time,
                "extraction_end_time": self._extraction_end_time,
                "extraction_duration_sec": extraction_duration,
                "data_info": {
                    "record_count": record_count,
                    "top_level_keys": list(result_dict.keys()) if isinstance(result_dict, dict) else [],
                },
            }
            
            logger.debug(f"Created metadata: {metadata['data_info']}")
            return metadata
            
        except Exception as e:
            logger.warning(f"Metadata creation failed: {e}")
            return {
                "extraction_time": datetime.now().isoformat(),
                "source_type": "queue",
                "error": f"Metadata creation failed: {e}",
            }

    def _validate_config(self):
        """
        Validate the configuration dictionary.

        This method should be overridden by subclasses to implement
        specific validation logic for their queue type.

        Raises
        ------
        ValueError
            If any required configuration parameter is missing or invalid
        """
        if not self.config:
            raise ValueError("Configuration dictionary cannot be empty")
        
    def extract(self) -> Dict[str, Any]:
        """
        Extract data from the queue system and return as Dictionary.

        This method ensures interface compliance with BaseExtractor by always
        returning a dictionary. It delegates to _extract_data() for the
        actual extraction logic and handles data conversion.

        Returns
        -------
        Dict[str, Any]
            Extracted data as a dictionary containing:
            - data: The actual extracted data (can be list, dict, DataFrame records, or string)
            - count: Number of records (if applicable)
            - Additional fields based on data type

        Raises
        ------
        ExtractorError
            If extraction fails
        DataSourceConnectionError
            If connection to the queue fails
        DataSourceAuthenticationError
            If authentication fails
        DataReadError
            If data cannot be read or converted to Dictionary
        """
        try:
            # Validate configuration
            self._validate_config()
            
            # Track extraction timing
            self._extraction_start_time = time.time()

            # Extract raw data
            data = self._extract_data()
            
            # Convert to Dictionary
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
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata from the last extraction.

        Returns
        -------
        Dict[str, Any]
            Metadata dictionary containing extraction details, or a message
            if no extraction has been performed yet
        """
        if self._last_metadata is None:
            return {"message": "No extraction performed yet"}
        return self._last_metadata.copy()


class RabbitMQExtractor(BaseQueueExtractor):
    """
    Extractor for RabbitMQ queues.

    This class implements the extraction logic specific to RabbitMQ.
    """

    def _validate_config(self):
        """
        Validate RabbitMQ-specific configuration parameters.

        Raises
        ------
        ValueError
            If any required configuration parameter is missing or invalid
        """
        required_keys = ["host", "port", "queue_name", "username", "password"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config parameter: {key}")
            
    def _connect(config: dict):
        """
        Establish connection to RabbitMQ.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing connection details

        Returns
        -------
        pika.BlockingConnection
            Established RabbitMQ connection

        Raises
        ------
        DataSourceConnectionError
            If connection to RabbitMQ fails
        DataSourceAuthenticationError
            If authentication fails
        """
        pass

    def _extract_data(self) -> List[dict]:
        """
        Extract data from the RabbitMQ queue.

        Returns
        -------
        List[dict]
            List of messages extracted from the queue

        Raises
        ------
        DataSourceConnectionError
            If connection to RabbitMQ fails
        DataSourceAuthenticationError
            If authentication fails
        DataReadError
            If reading from the queue fails
        """
        pass