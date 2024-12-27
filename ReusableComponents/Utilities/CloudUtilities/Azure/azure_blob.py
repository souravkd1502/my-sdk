"""
azure_blob_utils.py
--------------------
This module provides utility functions for interacting with Azure Blob Storage.

Functions:
----------
    - get_blob_service_client: Initializes and returns an Azure BlobServiceClient based on the provided connection method.
    - upload_files_to_blob: Uploads a file to a specified container in Azure Blob Storage.
    - download_files_from_blob: Downloads a file from a specified container in Azure Blob Storage.
    - list_blobs_in_container: Lists all blobs in a specified Azure Blob Storage container.
    - delete_blob_from_container: Deletes a specified blob from an Azure Blob Storage container.
    - download_blob_file_to_local: Downloads a blob file from Azure Blob Storage to a local file path.
    - download_file_to_tmp_dir: Downloads a file from an Azure Blob Storage container to a temporary file.
    
Usage:
------
    - Before using any methods, ensure that the Azure Blob Storage Python SDK is installed.
    - Initialize the class and use the
        provided methods with the appropriate parameters to perform file operations on Azure Blob Storage.
        

Features:
---------
    - Supports uploading, downloading, listing, and deleting blob files in Azure Blob Storage.
    - Provides methods for handling Azure Blob Storage operations using connection strings or Azure AD credentials.
    - Supports logging to provide informative messages for successful operations and error handling.


Requirements:
-------------
- azure-core==1.18.0
- azure-identity==1.6.0
- azure-storage-blob==12.8.1

TODO:
-----
- Add async support for Azure Blob Storage operations.
- Add support for other Azure Storage services (e.g., Azure File Storage, Azure Table Storage).
- Add support for more advanced operations (e.g., copying blobs, setting metadata).
- Add support for batch operations (e.g., uploading multiple files at once).

FIXME:
------

Author:
-------------
Sourav Das

Date:
-------------
2024-12-24
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Import dependencies
import logging
import tempfile
from azure.core.exceptions import AzureError
from azure.storage.blob import BlobServiceClient
from azure.identity import ClientSecretCredential

from typing import List

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class AzureBlobStorage:
    """
    A utility class for managing file operations in Azure Blob Storage.

    This class provides several methods for interacting with Azure Blob Storage,
    allowing for the uploading, downloading, listing, and deleting of blob files
    within a specified storage container. The operations are performed using the
    Azure Blob Storage Python SDK and require a valid Azure Blob Storage connection string.

    Methods:
        - upload_files_to_blob: Uploads a file to a specified container in Azure Blob Storage.
        - download_files_from_blob: Downloads a file from a specified container in Azure Blob Storage.
        - list_blobs_in_container: Lists all blobs in a specified Azure Blob Storage container.
        - delete_blob_from_container: Deletes a specified blob from an Azure Blob Storage container.
        - download_blob_file_to_local: Downloads a blob file from Azure Blob Storage to a local file path.

    Usage:
        - Before using any methods, ensure that the Azure Blob Storage Python SDK is installed.
        - Initialize the class and use the provided methods with the appropriate parameters
            to perform file operations on Azure Blob Storage.

    Logging:
        - Logging is set up to provide informative messages for successful operations
            and to log any errors encountered during the processes.

    Note:
        - Ensure all required parameters are provided and valid to avoid ValueErrors.
        - Handle raised exceptions where appropriate to ensure graceful error handling in the application.
    """

    @staticmethod
    def get_blob_service_client(
        connection_str: str = None,
        tenant_id: str = None,
        client_id: str = None,
        client_secret: str = None,
        blob_service_endpoint: str = None,
    ) -> BlobServiceClient:
        """
        Initializes and returns an Azure BlobServiceClient based on the provided connection method.

        This method supports two ways of initializing the `BlobServiceClient`:
        1. Using a connection string.
        2. Using Azure Active Directory (AAD) credentials.

        Args:
            connection_str (str): The connection string for the Azure Blob storage account.
                                    Must not be provided with tenant_id, client_id, and client_secret.
            tenant_id (str): Azure Active Directory tenant ID.
                                Required for AAD authentication.
            client_id (str): Azure Active Directory client ID.
                                Required for AAD authentication.
            client_secret (str): Azure Active Directory client secret.
                                    Required for AAD authentication.
            blob_service_endpoint (str): The Blob service endpoint URL. Required for AAD authentication.

        Returns:
            BlobServiceClient: An instance of `BlobServiceClient` for interacting with Azure Blob Storage.

        Raises:
            ValueError: If both connection string and Azure credentials are provided or if neither is provided.
            Exception: If initialization fails due to invalid credentials or configuration issues.

        Example:
            ```python
            # Using connection string
            blob_client = AzureBlobClient.get_blob_service_client(
                connection_str="your-connection-string",
                tenant_id="",
                client_id="",
                client_secret="",
                blob_service_endpoint=""
            )

            # Using AAD credentials
            blob_client = AzureBlobClient.get_blob_service_client(
                connection_str="",
                tenant_id="your-tenant-id",
                client_id="your-client-id",
                client_secret="your-client-secret",
                blob_service_endpoint="https://<account-name>.blob.core.windows.net/"
            )
            ```
        """
        try:
            # Validate input
            if connection_str and any([tenant_id, client_id, client_secret]):
                _logger.error(
                    "Both connection string and Azure credentials were provided. Please provide only one."
                )
                raise ValueError(
                    "Both connection string and Azure credentials were provided. Please provide one."
                )

            # Authenticate using Azure AD credentials
            if tenant_id and client_id and client_secret:
                _logger.info(
                    "Azure credentials provided. Initializing ClientSecretCredential."
                )
                credential = ClientSecretCredential(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    client_secret=client_secret,
                )

                blob_service_client = BlobServiceClient(
                    account_url=blob_service_endpoint, credential=credential
                )
                _logger.info(
                    "BlobServiceClient initialized with Azure credentials successfully."
                )

            # Authenticate using connection string
            elif connection_str:
                _logger.info(
                    "Connection string provided. Initializing BlobServiceClient from connection string."
                )
                blob_service_client = BlobServiceClient.from_connection_string(
                    connection_str
                )
                _logger.info(
                    "BlobServiceClient initialized with connection string successfully."
                )

            # Handle missing credentials
            else:
                _logger.error(
                    "Neither connection string nor Azure credentials were provided."
                )
                raise ValueError(
                    "Neither connection string nor Azure credentials were provided. Please provide one."
                )

            return blob_service_client

        except ValueError as ve:
            _logger.exception(f"ValueError encountered: {ve}")
            raise
        except Exception as ex:
            _logger.exception(f"Failed to initialize BlobServiceClient: {ex}")
            raise AzureError(
                f"Failed to initialize BlobServiceClient due to error: {ex}"
            ) from ex

    @classmethod
    def upload_files_to_blob(
        cls,
        container_name: str,
        file_name: str,
        data: bytes,
        overwrite: bool,
        blob_service_client: BlobServiceClient,
    ) -> None:
        """
        Uploads a file to Azure Blob Storage.

        This method uploads a file to a specified container in Azure Blob Storage using
        the provided connection string, container name, file name, and data. It allows
        overwriting existing files based on the 'overwrite' parameter.

        Args:
            container_name (str): The name of the container to upload the file to.
            file_name (str): The name of the file to be created or overwritten in the blob.
            data (bytes): The data/content of the file to be uploaded.
            overwrite (bool): A flag indicating whether to overwrite an existing file with the same name.
            blob_service_client (BlobServiceClient): An instance of `BlobServiceClient` for interacting with Azure Blob Storage.


        Raises:
            ValueError: If any of the required parameters are missing or empty.
            Exception: If an error occurs during the upload process.

        Example Usage:

        ```Python
        from azure.storage.blob import BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string("your-connection-string")
        AzureBlobStorage.upload_files_to_blob(
            container_name=<container_name>,
            blob_name=key,
            data=byte_data,
            overwrite=True,
            blob_service_client=blob_service_client
        )
        ```

        """
        # Check if all required values are provided
        if not all([container_name, file_name, data, blob_service_client]):
            raise ValueError(
                "All parameters (container_name, file_name, data, blob_service_client) must be provided and non-empty."
            )

        try:

            # Get a BlobClient for the specified container and file name
            blob_client = blob_service_client.get_blob_client(
                container=container_name, blob=file_name
            )

            # Upload the file data to the blob; overwrite if specified
            blob_client.upload_blob(data, overwrite=overwrite)

            # Log successful upload
            _logger.info(
                f"File '{file_name}' uploaded to Azure Blob Storage in container '{container_name}'."
            )

        except Exception as e:
            # Log any exceptions that occur
            _logger.error(
                f"An error occurred while uploading '{file_name}' to container '{container_name}': {e}"
            )
            raise e  # Rethrow the exception for the caller to handle

    @classmethod
    def download_files_from_blob(
        cls, container_name: str, file_name: str, blob_service_client: BlobServiceClient
    ) -> bytes:
        """
        Downloads a file from Azure Blob Storage.

        This method downloads a file from a specified container in Azure Blob Storage using
        the provided connection string, container name, and file name.

        Args:
            container_name (str): The name of the container to download the file from.
            file_name (str): The name of the file to be downloaded from the blob.
            blob_service_client (BlobServiceClient): An instance of `BlobServiceClient` for interacting with Azure Blob Storage.


        Returns:
            bytes: The data/content of the downloaded file.

        Raises:
            ValueError: If any of the required parameters are missing or empty.
            Exception: If an error occurs during the download process.

        Example Usage:

        ```Python
        from azure.storage.blob import BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string("your-connection-string")
        data = AzureBlobStorage.download_files_from_blob(
            container_name=<container_name>,
            blob_name=key,
            blob_service_client=blob_service_client
        )
        ```

        """
        # Check if all required values are provided
        if not all([container_name, file_name, blob_service_client]):
            raise ValueError(
                "All parameters (container_name, file_name, blob_service_client) must be provided and non-empty."
            )

        try:

            # Get a BlobClient for the specified container and file name
            blob_client = blob_service_client.get_blob_client(
                container=container_name, blob=file_name
            )

            # Download the file data from the blob
            data = blob_client.download_blob().readall()

            # Log successful download
            _logger.info(
                f"File '{file_name}' downloaded from Azure Blob Storage in container '{container_name}'."
            )

            return data

        except Exception as e:
            # Log any exceptions that occur
            _logger.error(
                f"An error occurred while downloading '{file_name}' from container '{container_name}': {e}"
            )
            raise e

    @classmethod
    def list_blobs_in_container(
        cls, container_name: str, blob_service_client: BlobServiceClient
    ) -> List[str]:
        """
        Lists all blobs in a specified Azure Blob Storage container.

        This method retrieves a list of all blobs in a specified container in Azure Blob Storage using
        the provided connection string and container name.

        Args:
            container_name (str): The name of the container to list the blobs from.
            BlobServiceClient (BlobServiceClient): An instance of `BlobServiceClient` for interacting with Azure Blob Storage.
            blob_service_client (BlobServiceClient): An instance of `BlobServiceClient` for interacting with Azure Blob Storage.

        Returns:
            List[str]: A list of blob names in the specified container.

        Raises:
            ValueError: If any of the required parameters are missing or empty.
            Exception: If an error occurs during the listing process.

        Example Usage:

        ```Python
        from azure.storage.blob import BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string("your-connection-string")
        blobs = AzureBlobStorage.list_blobs_in_container(
            container_name=<container_name>,
            blob_service_client=blob_service_client
        )
        ```

        """
        # Check if all required values are provided
        if not all([container_name, blob_service_client]):
            raise ValueError(
                "All parameters (container_name) must be provided and non-empty."
            )

        try:

            # Get a list of all blobs in the specified container
            blob_list = []
            for blob in blob_service_client.get_container_client(
                container_name
            ).list_blobs():
                blob_list.append(blob.name)

            # Log successful listing
            _logger.info(f"List of blobs in container '{container_name}': {blob_list}")

            return blob_list

        except Exception as e:
            # Log any exceptions that occur
            _logger.error(
                f"An error occurred while listing blobs in container '{container_name}': {e}"
            )
            raise e

    @classmethod
    def delete_blob_from_container(
        cls, container_name: str, blob_name: str, blob_service_client: BlobServiceClient
    ) -> None:
        """
        Deletes a specified blob from an Azure Blob Storage container.

        This method deletes a specified blob from a specified container in Azure Blob Storage using
        the provided connection string, container name, and blob name.

        Args:
            container_name (str): The name of the container to delete the blob from.
            blob_name (str): The name of the blob to be deleted.
            blob_service_client (BlobServiceClient): An instance of `BlobServiceClient` for interacting with Azure Blob Storage.

        Raises:
            ValueError: If any of the required parameters are missing or empty.
            Exception: If an error occurs during the deletion process.

        Example Usage:

        ```Python
        from azure.storage.blob import BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string("your-connection-string")
        AzureBlobStorage.delete_blob_from_container(
            container_name=<container_name>,
            blob_name=key,
            blob_service_client=blob_service_client
        )
        ```

        """

        # Check if all required values are provided
        if not all([container_name, blob_name, blob_service_client]):
            raise ValueError(
                "All parameters (container_name, blob_name, blob_service_client) must be provided and non-empty."
            )

        try:

            # Get a BlobClient for the specified container and blob name
            blob_client = blob_service_client.get_blob_client(
                container=container_name, blob=blob_name
            )

            # Delete the blob from the container
            blob_client.delete_blob()

            # Log successful deletion
            _logger.info(
                f"Blob '{blob_name}' deleted from container '{container_name}'."
            )

        except Exception as e:
            # Log any exceptions that occur
            _logger.error(
                f"An error occurred while deleting blob '{blob_name}' from container '{container_name}': {e}"
            )
            raise e

    @classmethod
    def download_blob_file_to_local(
        cls, container_name: str, file_name: str, file_path: str, blob_service_client: BlobServiceClient
    ) -> None:
        """
        Downloads a blob file from Azure Blob Storage to a local file path.

        This method downloads a specified blob file from a container in Azure Blob Storage
        and saves it to the provided local file path.

        Args:
            container_name (str): The name of the container from which the blob file will be downloaded.
            file_name (str): The name of the blob file to download.
            file_path (str): The local file path where the blob file will be saved.
            blob_service_client (BlobServiceClient): An instance of `BlobServiceClient` for interacting with Azure Blob Storage.

        Raises:
            ValueError: If any of the required parameters are missing or empty.
            Exception: If an error occurs during the download process.

        Example Usage:

        ```Python
        from azure.storage.blob import BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string("your-connection-string")
        AzureBlobStorage.download_blob_file_to_local(
            container_name=<container_name>,
            file_name=key,
            file_path=<local_file_path>,
            blob_service_client=blob_service_client
        )
        ```

        """
        # Check if all required values are provided
        if not all([container_name, file_name, file_path, blob_service_client]):
            raise ValueError(
                "All parameters (container_name, file_name, file_path) must be provided and non-empty."
            )

        try:

            blob_client = blob_service_client.get_blob_client(
                container=container_name, blob=file_name
            )

            # Write the downloaded data to the local file path
            _logger.info(f"Attempting to write blob into the file: {file_path}")
            with open(file_path, "wb") as file:
                download_stream = blob_client.download_blob()
                file.write(download_stream.readall())

            # Log successful download
            _logger.info(
                f"Blob '{file_name}' downloaded from container '{container_name}' to local path '{file_path}'."
            )

        except Exception as e:
            # Log any exceptions that occur
            _logger.error(
                f"An error occurred while downloading '{file_name}' from container '{container_name}': {e}"
            )
            raise e

    @classmethod
    def download_file_to_tmp_dir(
        cls,
        container_name: str,
        file_name: str,
        blob_service_client: BlobServiceClient,
    ) -> str:
        """
        Downloads a file from an Azure Blob Storage container to a temporary file.
        The temporary file will be deleted automatically when closed.

        Args:
            container_name (str): Name of the Azure Blob Storage container.
            file_name (str): Name of the file (blob) to be downloaded.
            blob_service_client (BlobServiceClient): An instance of `BlobServiceClient` for interacting with Azure Blob Storage.

        Raises:
            ValueError: If any required parameter is missing or invalid.
            Exception: If an error occurs during the download process.

        Returns:
            str: The file path of the temporary file.

        Example:
            ```python
            from azure.storage.blob import BlobServiceClient

            blob_service_client = BlobServiceClient.from_connection_string("your-connection-string")
            temp_file_path = AzureBlobUtility.download_file_to_tmp_dir(
                container_name="my-container",
                file_name="example.txt",
                blob_service_client=blob_service_client
            )
            print(f"File downloaded to: {temp_file_path}")
            ```
        """
        # Validate required parameters
        if not all([container_name, file_name, blob_service_client]):
            _logger.error(
                "Missing required parameters: container_name, file_name, or blob_service_client."
            )
            raise ValueError(
                "All parameters (container_name, file_name, blob_service_client) must be provided and non-empty."
            )

        # Get file extension to add to temp file
        extension = file_name.split(".")[-1]

        try:
            # Get a BlobClient for the specified container and file
            _logger.info(
                f"Fetching BlobClient for container '{container_name}' and file '{file_name}'."
            )
            blob_client = blob_service_client.get_blob_client(
                container=container_name, blob=file_name
            )

            # Create a temporary file path
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file_path = temp_file.name
                temp_file_path = temp_file_path + "." + extension
                _logger.info(f"Created temporary file at '{temp_file_path}'.")

            # Download the file data and write to the temp file
            _logger.info(
                f"Downloading file '{file_name}' from container '{container_name}'."
            )
            download_stream = blob_client.download_blob()

            with open(temp_file_path, "wb") as file:
                file.write(download_stream.readall())
            _logger.info(
                f"File '{file_name}' successfully downloaded to temporary file '{temp_file_path}'."
            )

            return temp_file_path

        except Exception as e:
            # Log and re-raise the exception for external handling
            _logger.error(
                f"An error occurred while downloading '{file_name}' from container '{container_name}': {e}"
            )
            raise AzureError(
                f"Failed to download '{file_name}' to a temporary file."
            ) from e
