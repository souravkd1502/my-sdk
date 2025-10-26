""" """

# Import required libraries
import os
import logging
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Iterator, Optional, List, Union

# Cloud Providers Specific imports
# AWS S3
# ----------------------------------------------
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

# Google Cloud Storage
# ----------------------------------------------


# Azure Blob Storage
# ----------------------------------------------
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
from azure.core.exceptions import AzureError, ResourceNotFoundError

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


class CloudProvider(Enum):
    """Supported cloud storage providers."""

    S3 = "s3"
    AZURE = "azure"
    GCS = "gcs"


@dataclass
class FileMetadata:
    """Metadata for a cloud storage file."""

    name: str
    size: int
    last_modified: datetime
    content_type: str
    etag: Optional[str] = None
    path: Optional[str] = None


@dataclass
class ExtractionResult:
    """Result of file extraction operation."""

    file_metadata: FileMetadata
    content: Union[bytes, Iterator[bytes]]
    success: bool
    error_message: Optional[str] = None


class CloudExtractor(BaseExtractor, ABC):
    """
    Abstract base class for cloud data extractors.

    This class provides a template for implementing extractors that interact with cloud storage services.
    It includes methods for connecting to the cloud service, listing available data sources, and extracting data.

    Subclasses must implement the abstract methods defined in this class.

    Attributes
    ----------
    config : dict
        Configuration parameters for the extractor, such as authentication credentials and connection settings.

    Methods
    -------
    connect()
        Establishes a connection to the cloud service.
    list_data_sources()
        Lists available data sources in the cloud storage.
    extract(source, **kwargs)
        Extracts data from the specified source.

    Raises
    ------
    CloudExtractorError
        Base exception for all cloud extractor related errors.
    DataSourceConnectionError
        Raised when connection to the cloud service fails.
    DataSourceAuthenticationError
        Raised when authentication with the cloud service fails.
    DataReadError
        Raised when extraction fails due to issues with reading data from the cloud source.

    Examples
    --------
    >>> class MyCloudExtractor(CloudExtractor):
    ...     def connect(self):
    ...         # Implement connection logic here
    ...         pass
    ...
    ...     def list_data_sources(self):
    ...         # Implement logic to list data sources here
    ...         return ["source1", "source2"]
    ...
    ...     def extract(self, source, **kwargs):
    ...         # Implement extraction logic here
    ...         return f"Data from {source}"
    ...
    >>> extractor = MyCloudExtractor(config={"api_key": "my_api_key"})
    >>> extractor.connect()
    >>> sources = extractor.list_data_sources()
    >>> data = extractor.extract(sources[0])

    See Also
    --------
    - `BaseExtractor`: The base class for all extractors in the ETL framework.
    - `CloudExtractorError`: The base exception class for all cloud extractor related errors.
    - `DataSourceConnectionError`: Raised when connection to the cloud service fails.
    - `DataSourceAuthenticationError`: Raised when authentication with the cloud service fails.
    - `DataReadError`: Raised when extraction fails due to issues with reading data from the cloud source.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config: dict = config
        self.checkpoint_manager: Optional[CheckpointManager] = None

    @abstractmethod
    def list_files(
        self,
        source_name: str,
        prefix: Optional[str] = None,
        last_modified_after: Optional[datetime] = None,
    ) -> List[FileMetadata]:
        """
        Lists files in the specified data source with optional filtering by prefix
        and last modified timestamp. Supports checkpointing to skip already processed files.

        Parameters
        ----------
        source_name : str
            The identifier of the data source (e.g., bucket or container).
        prefix : Optional[str], default None
            Filter files by prefix or virtual directory.
        last_modified_after : Optional[datetime], default None
            Filter files modified after this timestamp.

        Returns
        -------
        List[FileMetadata]
            A list of file metadata objects.

        Raises
        ------
        DataSourceConnectionError
            If unable to connect to the cloud service.
        """
        pass

    @abstractmethod
    def get_metadata(self, source_name: str, file_path: str) -> FileMetadata:
        """
        Retrieves metadata for a specific file in the data source.

        Parameters
        ----------
        source_name : str
            The identifier of the data source.
        file_path : str
            The path or key of the file.

        Returns
        -------
        FileMetadata
            Metadata of the file including size, last modified date, content type, etc.

        Raises
        ------
        DataSourceConnectionError
            If unable to connect to the cloud service to retrieve metadata.
        """
        pass

    @abstractmethod
    def extract(
        self, source_name: str, file_metadata: FileMetadata, stream: bool = True
    ) -> ExtractionResult:
        """
        Extracts a single file from the cloud storage.

        Supports checkpointing and streaming of large files in chunks.

        Parameters
        ----------
        source_name : str
            The identifier of the data source.
        file_metadata : FileMetadata
            Metadata of the file to extract.
        stream : bool, default True
            If True, content is streamed in chunks. If False, content is read fully.

        Returns
        -------
        ExtractionResult
            The result containing metadata, content, and success status.

        Raises
        ------
        DataReadError
            If extraction fails due to reading issues.
        """
        pass

    @abstractmethod
    def extract_prefix(
        self,
        source_name: str,
        prefix: str,
        stream: bool = True,
        last_modified_after: Optional[datetime] = None,
    ) -> Iterator[ExtractionResult]:
        """
        Extracts all files under a given prefix, with optional streaming and checkpointing.

        Parameters
        ----------
        source_name : str
            The identifier of the data source.
        prefix : str
            Prefix or virtual directory path to extract files from.
        stream : bool, default True
            If True, content is streamed in chunks. If False, content is read fully.
        last_modified_after : Optional[datetime], default None
            Filter files modified after this timestamp.

        Yields
        ------
        Iterator[ExtractionResult]
            Extraction results for each file.

        Raises
        ------
        DataReadError
            If extraction fails due to reading issues.
        """
        pass


class S3Extractor(CloudExtractor):
    """
    Concrete implementation of CloudExtractor for AWS S3.

    This class provides methods to connect to AWS S3, list available buckets and objects, and extract data from S3 objects.

    Raises
    ------
    CloudExtractorError
        Base exception for all cloud extractor related errors.
    DataSourceConnectionError
        Raised when connection to AWS S3 fails.
    DataSourceAuthenticationError
        Raised when authentication with AWS S3 fails.
    DataReadError
        Raised when extraction fails due to issues with reading data from the S3 object.

    Examples
    --------
    >>> s3_extractor = S3Extractor(config={"aws_access_key_id": "my_key", "aws_secret_access_key": "my_secret"})
    >>> s3_extractor.connect()
    >>> buckets = s3_extractor.list_data_sources()
    >>> data = s3_extractor.extract("my-bucket/my-object")

    See Also
    --------
    - `CloudExtractor`: The abstract base class for cloud data extractors.
    - `CloudExtractorError`: The base exception class for all cloud extractor related errors.
    - `DataSourceConnectionError`: Raised when connection to AWS S3 fails.
    - `DataSourceAuthenticationError`: Raised when authentication with AWS S3 fails.
    - `DataReadError`: Raised when extraction fails due to issues with reading data from the S3 object.
    """

    def __init__(
        self,
        config,
        chunk_size=8192,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        """
        Initializes the S3Extractor with the given configuration and chunk size.
        Establishes a connection to AWS S3.

        This method uses the provided configuration to authenticate and connect to AWS S3.

        Args
        ------
        config : dict
            Configuration parameters for AWS S3 connection, including access keys and region.
        chunk_size : int, optional
            The size of data chunks to read from S3 objects (default is 8192 bytes).
        checkpoint_manager : Optional[CheckpointManager], optional
            An optional CheckpointManager instance for managing extraction checkpoints.

        Raises
        ------
        DataSourceConnectionError
            If the connection to AWS S3 fails.
        DataSourceAuthenticationError
            If authentication with AWS S3 fails.

        Notes
        -----
        Ensure that the AWS credentials provided in the config are valid and have the necessary permissions.
        Format for config specific to AWS S3:
        {
            "aws_access_key_id": "your_access_key",
            "aws_secret_access_key": "your_secret_key",
            "region_name": "your_region"  # Optional, defaults to 'us-east-1'
        }
        """
        super().__init__(config)
        self.s3_client = None
        self.chunk_size = chunk_size
        self.checkpoint_manager = checkpoint_manager
        try:
            if not boto3:
                raise CloudExtractorError(
                    "boto3 not installed. Install with: pip install boto3"
                )

            # Validate config
            self._validate_s3_config()

            # Establish connection to S3
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=self.config.get("aws_access_key_id")
                or os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=self.config.get("aws_secret_access_key")
                or os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=self.config.get("region_name")
                or os.getenv("AWS_REGION_NAME", "us-east-1"),
            )
            logger.info("Successfully connected to AWS S3.")
        except (NoCredentialsError, PartialCredentialsError) as e:
            logger.error("Authentication with AWS S3 failed.")
            raise DataSourceAuthenticationError("Invalid AWS credentials.") from e
        except ClientError as e:
            logger.error("Connection to AWS S3 failed.")
            raise DataSourceConnectionError("Could not connect to AWS S3.") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise CloudExtractorError(
                f"An unexpected error occurred while connecting to AWS S3: {e}"
            ) from e

    def _validate_s3_config(self):
        """
        Validates the AWS S3 configuration parameters.
        """
        required_keys = ["aws_access_key_id", "aws_secret_access_key"]
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise CloudExtractorError(
                f"Missing required AWS S3 config keys: {', '.join(missing_keys)}"
            )

    def list_files(
        self,
        bucket_name: str,
        prefix: Optional[str] = None,
        last_modified_after: Optional[datetime] = None,
    ) -> List[FileMetadata]:
        """
        List all unprocessed files in the specified S3 bucket and prefix.

        Parameters
        ----------
        bucket_name : str
            The name of the S3 bucket.
        prefix : Optional[str], default None
            The prefix (directory path) to list files from.
        last_modified_after : Optional[datetime], default None
            Filter files by last modified timestamp.

        Returns
        -------
        List[FileMetadata]
            A list of metadata for the files found.
        """
        try:
            kwargs = {"Bucket": bucket_name}
            if prefix:
                kwargs["Prefix"] = prefix

            files: List[FileMetadata] = []
            paginator = self.s3_client.get_paginator("list_objects_v2")

            for page in paginator.paginate(**kwargs):
                for obj in page.get("Contents", []):
                    last_modified = obj["LastModified"].replace(tzinfo=None)

                    if last_modified_after and last_modified <= last_modified_after:
                        continue

                    # Skip if checkpoint already exists
                    if (
                        self.checkpoint_manager
                        and self.checkpoint_manager.get_checkpoint(obj["Key"])
                    ):
                        continue

                    files.append(
                        FileMetadata(
                            name=obj["Key"].split("/")[-1],
                            size=obj["Size"],
                            last_modified=last_modified,
                            content_type=obj.get(
                                "ContentType", "application/octet-stream"
                            ),
                            etag=obj.get("ETag", "").strip('"'),
                            path=obj["Key"],
                        )
                    )

            return files
        except (ClientError, NoCredentialsError) as e:
            raise CloudExtractorError(f"S3 error: {str(e)}")

    def get_metadata(self, bucket_name: str, file_path: str) -> FileMetadata:
        """
        Retrieve metadata for the specified S3 object.

        Parameters
        ----------
        bucket_name : str
            The name of the S3 bucket.
        file_path : str
            The S3 object path in the format 'object_key'.

        Returns
        -------
        FileMetadata
            Metadata of the specified S3 object.

        Raises
        ------
        DataSourceConnectionError
            If unable to connect to AWS S3 to retrieve file metadata.
        """
        try:
            response = self.s3_client.head_object(Bucket=bucket_name, Key=file_path)

            return FileMetadata(
                name=file_path.split("/")[-1],
                size=response["ContentLength"],
                last_modified=response["LastModified"].replace(tzinfo=None),
                content_type=response.get("ContentType", "application/octet-stream"),
                etag=response.get("ETag", "").strip('"'),
                path=file_path,
            )
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Failed to retrieve metadata for {file_path}: {e}")
            raise DataSourceConnectionError(
                f"Could not retrieve metadata for {file_path}."
            ) from e

    def extract(
        self, bucket_name: str, file_metadata: FileMetadata, stream: bool = True
    ) -> ExtractionResult:
        """
        Extract a single file from S3 with checkpointing support.

        Parameters
        ----------
        bucket_name : str
            The name of the S3 bucket.
        file_metadata : FileMetadata
            Metadata of the file to extract.
        stream : bool, default True
            If True, file content is streamed in chunks. If False, content is read fully.

        Returns
        -------
        ExtractionResult
            The result containing metadata, content, and success status.
        """
        try:
            response = self.s3_client.get_object(
                Bucket=bucket_name, Key=file_metadata.path
            )
            content = (
                self._stream_content(response["Body"])
                if stream
                else response["Body"].read()
            )

            result = ExtractionResult(
                file_metadata=file_metadata,
                content=content,
                success=True,
            )

            # Save checkpoint only after successful extraction
            if self.checkpoint_manager:
                self.checkpoint_manager.set_checkpoint(
                    file_metadata.path, str(file_metadata.last_modified)
                )

            return result
        except (ClientError, NoCredentialsError) as e:
            return ExtractionResult(
                file_metadata=file_metadata,
                content=iter([]) if stream else b"",
                success=False,
                error_message=f"S3 extraction error: {str(e)}",
            )

    def extract_prefix(
        self,
        bucket_name: str,
        prefix: str,
        stream: bool = True,
        last_modified_after: Optional[datetime] = None,
    ) -> Iterator[ExtractionResult]:
        """
        Extract all files under a given prefix with checkpointing support.

        Parameters
        ----------
        bucket_name : str
            The name of the S3 bucket.
        prefix : str
            The prefix (directory path) to extract files from.
        stream : bool, default True
            If True, file content is streamed in chunks. If False, content is read fully.
        last_modified_after : Optional[datetime], default None
            Filter files by last modified timestamp.

        Yields
        ------
        Iterator[ExtractionResult]
            Extraction results for each file.
        """
        for file_meta in self.list_files(bucket_name, prefix, last_modified_after):
            yield self.extract_file(bucket_name, file_meta, stream=stream)

    def _stream_content(self, body) -> Iterator[bytes]:
        """
        Stream file content in chunks.

        Parameters
        ----------
        body : botocore.response.StreamingBody
            The streaming body of the S3 object.

        Yields
        ------
        Iterator[bytes]
            File content in chunks.
        """
        try:
            while True:
                chunk = body.read(self.chunk_size)
                if not chunk:
                    break
                yield chunk
        finally:
            body.close()

    @staticmethod
    def example() -> str:
        """
        Provides an example configuration for the S3Extractor.

        Returns
        -------
        str
            Example configuration in JSON format.
        """

        return """
    
        # Example usage
        s3_config = {
            "aws_access_key_id": "your_access_key",
            "aws_secret_access_key": "your_secret_key",
            "region_name": "us-east-1",
        }

        from ..core.checkpoint import FileCheckpointBackend

        checkpoint = FileCheckpointBackend(file_path="s3_checkpoints.json")
        extractor = S3Extractor(config=s3_config, checkpoint_manager=checkpoint)

        data = extractor.extract_prefix(
            bucket_name="your-bucket-name",
            prefix="data/",
            stream=False,
            last_modified_after=datetime(2023, 1, 1),
        )
        
        for i in data:
            print(i)
    """


class AzureBlobExtractor(CloudExtractor):
    """
    Concrete implementation of CloudExtractor for Azure Blob Storage.

    This class provides methods to connect to Azure Blob Storage, list containers and blobs,
    and extract data with checkpointing and streaming support.

    Raises
    ------
    CloudExtractorError
        Base exception for all cloud extractor related errors.
    DataSourceConnectionError
        Raised when connection to Azure Blob Storage fails.
    DataSourceAuthenticationError
        Raised when authentication with Azure Blob Storage fails.
    DataReadError
        Raised when extraction fails due to issues with reading data from blobs.

    Examples
    --------
    >>> azure_extractor = AzureBlobExtractor(config={"connection_string": "my_connection_string"})
    >>> azure_extractor.connect()
    >>> files = azure_extractor.list_files("my-container", prefix="data/")
    >>> result = azure_extractor.extract_file("my-container", files[0])

    See Also
    --------
    - `CloudExtractor`: The abstract base class for cloud data extractors.
    - `CloudExtractorError`: The base exception class for all cloud extractor related errors.
    - `DataSourceConnectionError`: Raised when connection to Azure Blob Storage fails.
    - `DataSourceAuthenticationError`: Raised when authentication with Azure Blob Storage fails.
    - `DataReadError`: Raised when extraction fails due to issues with reading data from Azure Blob Storage.
    """

    def __init__(
        self,
        config,
        chunk_size: int = 8192,
        checkpoint_manager: Optional["CheckpointManager"] = None,
    ):
        """
        Initializes the AzureBlobExtractor with the given configuration and chunk size.
        Establishes a connection to Azure Blob Storage.

        Args
        ----
        config : dict
            Configuration parameters for Azure Blob Storage connection.
            Must include `connection_string`.
        chunk_size : int, optional
            The size of data chunks to read from blobs (default is 8192 bytes).
        checkpoint_manager : Optional[CheckpointManager], optional
            An optional CheckpointManager instance for managing extraction checkpoints.

        Raises
        ------
        DataSourceConnectionError
            If the connection to Azure Blob Storage fails.
        DataSourceAuthenticationError
            If authentication with Azure Blob Storage fails.

        Notes
        -----
        Ensure that the connection string provided is valid and has the necessary permissions.
        Example config:
        {
            "connection_string": "your_connection_string"
        }
        """
        super().__init__(config)
        self.blob_service_client: Optional[BlobServiceClient] = None
        self.chunk_size = chunk_size
        self.checkpoint_manager = checkpoint_manager
        try:
            connection_string = self.config.get("connection_string")
            if not connection_string:
                raise CloudExtractorError(
                    "Missing required config key: connection_string"
                )

            self.blob_service_client = BlobServiceClient.from_connection_string(
                connection_string
            )
            logger.info("Successfully connected to Azure Blob Storage.")
        except AzureError as e:
            logger.error("Connection to Azure Blob Storage failed.")
            raise DataSourceConnectionError(
                f"Could not connect to Azure Blob Storage: {e}"
            ) from e
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise CloudExtractorError(
                f"Unexpected error while connecting to Azure Blob Storage: {e}"
            ) from e

    def list_files(
        self,
        container_name: str,
        prefix: Optional[str] = None,
        last_modified_after: Optional[datetime] = None,
    ) -> List["FileMetadata"]:
        """
        List all unprocessed files in the specified Azure Blob container and prefix.

        Parameters
        ----------
        container_name : str
            The name of the Azure Blob container.
        prefix : Optional[str], default None
            The prefix (virtual directory path) to list files from.
        last_modified_after : Optional[datetime], default None
            Filter files by last modified timestamp.

        Returns
        -------
        List[FileMetadata]
            A list of metadata for the files found.
        """
        try:
            container_client: ContainerClient = (
                self.blob_service_client.get_container_client(container_name)
            )
            files: List[FileMetadata] = []

            blobs = container_client.list_blobs(name_starts_with=prefix)

            for blob in blobs:
                last_modified = blob.last_modified.replace(tzinfo=None)

                if last_modified_after and last_modified <= last_modified_after:
                    continue

                # Skip if checkpoint already exists
                if self.checkpoint_manager and self.checkpoint_manager.get_checkpoint(
                    blob.name
                ):
                    continue

                files.append(
                    FileMetadata(
                        name=blob.name.split("/")[-1],
                        size=blob.size,
                        last_modified=last_modified,
                        content_type=(
                            blob.content_settings.content_type
                            if blob.content_settings
                            else "application/octet-stream"
                        ),
                        etag=blob.etag.strip('"') if blob.etag else "",
                        path=blob.name,
                    )
                )

            return files
        except AzureError as e:
            raise CloudExtractorError(f"Azure Blob error: {str(e)}")

    def get_metadata(self, container_name: str, blob_path: str) -> "FileMetadata":
        """
        Retrieve metadata for the specified Azure Blob.

        Parameters
        ----------
        container_name : str
            The name of the Azure Blob container.
        blob_path : str
            The blob path (object key) in the container.

        Returns
        -------
        FileMetadata
            Metadata of the specified Azure Blob.

        Raises
        ------
        DataSourceConnectionError
            If unable to connect to Azure Blob Storage to retrieve file metadata.
        """
        try:
            blob_client: BlobClient = self.blob_service_client.get_blob_client(
                container=container_name, blob=blob_path
            )
            props = blob_client.get_blob_properties()

            return FileMetadata(
                name=blob_path.split("/")[-1],
                size=props.size,
                last_modified=props.last_modified.replace(tzinfo=None),
                content_type=(
                    props.content_settings.content_type
                    if props.content_settings
                    else "application/octet-stream"
                ),
                etag=props.etag.strip('"') if props.etag else "",
                path=blob_path,
            )
        except (AzureError, ResourceNotFoundError) as e:
            logger.error(f"Failed to retrieve metadata for {blob_path}: {e}")
            raise DataSourceConnectionError(
                f"Could not retrieve metadata for {blob_path}."
            ) from e

    def extract(
        self, container_name: str, file_metadata: "FileMetadata", stream: bool = True
    ) -> "ExtractionResult":
        """
        Extract a single file from Azure Blob Storage with checkpointing support.

        Parameters
        ----------
        container_name : str
            The name of the Azure Blob container.
        file_metadata : FileMetadata
            Metadata of the file to extract.
        stream : bool, default True
            If True, file content is streamed in chunks. If False, content is read fully.

        Returns
        -------
        ExtractionResult
            The result containing metadata, content, and success status.
        """
        try:
            blob_client: BlobClient = self.blob_service_client.get_blob_client(
                container=container_name, blob=file_metadata.path
            )

            logger.info(
                f"Extracting blob: {file_metadata.path} from container: {container_name}"
            )
            logger.info(f"Blob Client: {blob_client.exists()}")

            downloader = blob_client.download_blob()
            content = (
                self._stream_content(downloader) if stream else downloader.readall()
            )

            result = ExtractionResult(
                file_metadata=file_metadata,
                content=content,
                success=True,
            )

            # Save checkpoint only after successful extraction
            if self.checkpoint_manager:
                self.checkpoint_manager.set_checkpoint(
                    file_metadata.path, str(file_metadata.last_modified)
                )

            logger.info(f"Successfully extracted blob: {file_metadata.path}")
            return result
        except AzureError as e:
            logger.error(f"Failed to extract blob: {file_metadata.path}: {e}")
            return ExtractionResult(
                file_metadata=file_metadata,
                content=iter([]) if stream else b"",
                success=False,
                error_message=f"Azure Blob extraction error: {str(e)}",
            )

    def extract_prefix(
        self,
        container_name: str,
        prefix: str = None,
        stream: bool = True,
        last_modified_after: Optional[datetime] = None,
    ) -> Iterator["ExtractionResult"]:
        """
        Extract all files under a given prefix with checkpointing support.

        Parameters
        ----------
        container_name : str
            The name of the Azure Blob container.
        prefix : str, default None
            The prefix (virtual directory path) to extract files from.
        stream : bool, default True
            If True, file content is streamed in chunks. If False, content is read fully.
        last_modified_after : Optional[datetime], default None
            Filter files by last modified timestamp.

        Yields
        ------
        Iterator[ExtractionResult]
            Extraction results for each file.
        """
        logger.info(
            f"Extracting files from container: {container_name} with prefix: {prefix}"
        )
        for file_meta in self.list_files(container_name, prefix, last_modified_after):
            logger.info(f"Processing file: {file_meta.path}")
            yield self.extract(container_name, file_meta, stream=stream)

    def _stream_content(self, downloader) -> Iterator[bytes]:
        """
        Stream file content in chunks.

        Parameters
        ----------
        downloader : StorageStreamDownloader
            The downloader for the Azure Blob.

        Yields
        ------
        Iterator[bytes]
            File content in chunks.
        """
        try:
            stream = downloader.chunks()
            for chunk in stream:
                yield chunk
        finally:
            downloader.close()

    @staticmethod
    def example() -> str:
        """
        Provides an example configuration for the AzureBlobExtractor.

        Returns
        -------
        str
            Example configuration in JSON format.
        """
        return """
        # Example usage
        azure_config = {
            "connection_string": "your_connection_string",
            # OR use account_name and account_key if preferred:
            # "account_name": "your_account_name",
            # "account_key": "your_account_key",
        }

        from core.checkpoint import FileCheckpointBackend

        # Initialize checkpoint manager
        file_backend = FileCheckpointBackend(directory="azure_checkpoints/")
        print("Using checkpoint file at:", file_backend._get_directory())
        
        checkpoint = CheckpointManager(backend=file_backend)

        # Create Azure Blob extractor
        extractor = AzureBlobExtractor(config=azure_config, checkpoint_manager=checkpoint)
            
        data = extractor.extract_prefix(
            container_name="your-container-name",
        )
        
        for i in data:
            print(i)
        """
