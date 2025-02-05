"""
azure_key_vaults.py: 
---------------------
A utility class for interacting with Azure Key Vault. This class provides methods to securely manage secrets within an Azure Key Vault.
It supports retrieving and setting secrets using the Azure Key Vault SDK and requires proper Azure credentials (client ID, client secret, 
tenant ID) and a Key Vault URL.

This class is intended to be used as a reusable component for securely managing secrets in Azure Key Vault.

Functions:
-----------
- create_key: Creates a key based on user input.
- get_secret: gets a secret in the Azure Key Vault.
- set_secret: sets a secret in the Azure Key Vault.
- update_secret_properties: Updates the properties of a secret in the Azure Key Vault.
- delete_secret: Deletes a secret from the Azure Key Vault.
- list_secrets: Lists all secrets in the Azure Key Vault.


Usage:
------
- Ensure that the Azure Identity and Azure Key Vault SDKs are installed before using this class:
    `pip install azure-identity azure-keyvault-keys`.
- Initialize the class with the required Azure credentials and vault URL, or set them using environment variables.
- Use `get_secret` and `set_secret` methods to interact with secrets stored in the Azure Key Vault.

Features:
---------
- Create a new key in the Azure Key Vault.
- Retrieve a secret from the Azure Key Vault.
- Set a secret in the Azure Key Vault.
- Update the properties of a secret in the Azure Key Vault.
- Delete a secret from the Azure Key Vault.
- List all secrets in the Azure Key Vault.

References:
----------- 
- https://docs.microsoft.com/en-us/azure/key-vault/keys/quick-create-python


Requirements:
-------------
- azure-core==1.32.0
- azure-identity==1.6.0
- azure-keyvault-secrets==4.9.0
- azure-keyvault-keys==4.10.0

TODO:
-----
- Add support to get and set keys in the Azure Key Vault.
- Add support to list and delete keys in the Azure Key Vault.
- Create a separate class for managing keys in the Azure Key Vault.

Author:
-------
Sourav Das

Date:
-----
2024-12-25
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Import dependencies
import os
import logging

from azure.keyvault.keys import KeyClient
from azure.core.exceptions import AzureError
from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretProperties, SecretClient

from typing import List

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class AzureKeyVault:
    """
    A utility class for interacting with Azure Key Vault.

    This class provides methods to securely manage secrets within an Azure Key Vault.
    It supports retrieving and setting secrets using the Azure Key Vault SDK and
    requires proper Azure credentials (client ID, client secret, tenant ID) and a Key Vault URL.

    Attributes:
        client_id (str): The client ID for Azure Active Directory authentication.
        client_secret (str): The client secret for Azure Active Directory authentication.
        tenant_id (str): The tenant ID for Azure Active Directory authentication.
        vault_url (str): The URL of the Azure Key Vault.

    Methods:
        - create_key: Creates a key based on user input.
        - get_secret: gets a secret in the Azure Key Vault.

    Usage:
        - Ensure that the Azure Identity and Azure Key Vault SDKs are installed before using this class:
            `pip install azure-identity azure-keyvault-keys`.
        - Initialize the class with the required Azure credentials and vault URL, or set them using environment variables.
        - Use `get_secret` and `set_secret` methods to interact with secrets stored in the Azure Key Vault.

    """

    def __init__(
        self,
        client_id: str = os.getenv("CLIENT_ID"),
        client_secret: str = os.getenv("CLIENT_SECRET"),
        tenant_id: str = os.getenv("TENANT_ID"),
        vault_url: str = os.getenv("VAULT_URL"),
    ) -> None:
        """
        Initializes the AzureKeyVault instance with Azure credentials and Key Vault URL.

        Args:
            client_id (str, optional): Azure client ID. Defaults to os.getenv("CLIENT_ID").
            client_secret (str, optional): Azure client secret. Defaults to os.getenv("CLIENT_SECRET").
            tenant_id (str, optional): Azure tenant ID. Defaults to os.getenv("TENANT_ID").
            vault_url (str, optional): Azure Key Vault URL. Defaults to os.getenv("VAULT_URL").

        Raises:
            ValueError: If any required credentials (client_id, client_secret, tenant_id, vault_url) are missing.
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.vault_url = vault_url

        # Check if all required values are provided
        if not all(
            [self.client_id, self.client_secret, self.tenant_id, self.vault_url]
        ):
            _logger.error(
                "All parameters (client_id, client_secret, tenant_id, vault_url) must be provided and non-empty."
            )
            raise ValueError(
                "All parameters (client_id, client_secret, tenant_id, vault_url) must be provided and non-empty."
            )

        # Initialize the Azure credentials and Key Vault client
        try:
            self.credentials = ClientSecretCredential(
                client_id=self.client_id,
                client_secret=self.client_secret,
                tenant_id=self.tenant_id,
            )
            self.key_client = KeyClient(
                vault_url=self.vault_url, credential=self.credentials
            )
            self.secret_client = SecretClient(
                vault_url=self.vault_url, credential=self.credentials
            )
            _logger.info("Successfully authenticated with Azure Key Vault.")

        except AzureError as e:
            _logger.error(f"Failed to authenticate with Azure Key Vault: {e}")
            raise e
        except Exception as e:
            _logger.error(f"An unexpected error occurred during initialization: {e}")
            raise e

    def create_key(self, key_name: str, key_type: str) -> None:
        """
        Sets or updates a secret in the Azure Key Vault.

        Args:
            key_name (str): The name of the secret to set or update.
            key_type (str): The type of the secret to set. Possible values: 'RSA Key', 'EC key'.

        Raises:
            AzureError: If there is an issue accessing the Azure Key Vault.
        """
        if key_type.lower() not in ["rsa key", "ec key"]:
            _logger.error(
                "Invalid secret type. Please provide a valid secret type: 'RSA Key', 'EC key'."
            )
            raise ValueError(
                "Invalid secret type. Please provide a valid secret type: 'RSA Key', 'EC key'."
            )

        try:
            if key_type.lower() == "rsa key":
                key_size = 2048
                key_ops = [
                    "encrypt",
                    "decrypt",
                    "sign",
                    "verify",
                    "wrapKey",
                    "unwrapKey",
                ]
                rsa_key = self.client.create_rsa_key(
                    key_name, size=key_size, key_operations=key_ops
                )
                _logger.info(
                    f"RSA Key with name '{rsa_key.name}' created of type '{rsa_key.key_type}'."
                )

            elif key_type.lower() == "ec key":
                key_curve = "P-256"
                ec_key = self.client.create_ec_key(key_name, curve=key_curve)
                _logger.info(
                    f"EC Key with name '{ec_key.name}' created of type '{ec_key.key_type}'."
                )

        except AzureError as e:
            _logger.error(f"Failed to create key '{key_name}' in Azure Key Vault: {e}")
            raise e
        except Exception as e:
            _logger.error(
                f"An unexpected error occurred while creating key '{key_name}': {e}"
            )
            raise e

    def get_secret(self, secret_name: str) -> str:
        """
        Retrieves a secret value from the Azure Key Vault.

        Args:
            secret_name (str): The name of the secret to retrieve from the Azure Key Vault.

        Returns:
            str: The value of the secret.

        Raises:
            AzureError: If there is an issue accessing the Azure Key Vault or retrieving the secret.
            Exception: For any other unforeseen errors that occur.
        """
        try:
            # Attempt to retrieve the secret from Azure Key Vault
            secret = self.secret_client.get_secret(secret_name)
            _logger.info(
                f"Successfully retrieved secret '{secret_name}' from Azure Key Vault."
            )
            return secret.value

        except AzureError as e:
            # Log and raise Azure-specific errors
            _logger.error(
                f"Failed to retrieve secret '{secret_name}' from Azure Key Vault: {e}"
            )
            raise e
        except Exception as e:
            # Log and raise any other exceptions
            _logger.error(
                f"An unexpected error occurred while retrieving secret '{secret_name}': {e}"
            )
            raise e

    def set_secret(self, secret_name: str, secret_value: str) -> None:
        """
        Sets a secret value and secret name in the Azure Key Vault.

        Args:
            secret_name (str): The name of the secret to set in the Azure Key Vault.
            secret_value (str): The secret value to set in the Azure Key Vault.

        Returns:
            None

        Raises:
            AzureError: If there is an issue accessing the Azure Key Vault or retrieving the secret.
            Exception: For any other unforeseen errors that occur.
        """
        try:
            # Attempt to retrieve the secret from Azure Key Vault
            secret = self.secret_client.set_secret(name=secret_name, value=secret_value)
            _logger.info(
                f"Successfully retrieved secret '{secret_name}' from Azure Key Vault."
            )
            return secret.value

        except AzureError as e:
            # Log and raise Azure-specific errors
            _logger.error(
                f"Failed to retrieve secret '{secret_name}' from Azure Key Vault: {e}"
            )
            raise e
        except Exception as e:
            # Log and raise any other exceptions
            _logger.error(
                f"An unexpected error occurred while retrieving secret '{secret_name}': {e}"
            )
            raise e

    def update_secret_properties(
        self, secret_name: str, content_type: str, enabled: bool = False
    ) -> SecretProperties:
        """
        Updates a secret's metadata in Azure Key Vault.

        This method updates the metadata properties of a secret in the Azure Key Vault. It cannot change the secret's value;
        to change the value, use the `set_secret` method.

        Args:
            secret_name (str): The name of the secret whose properties are to be updated.
            content_type (str): The content type of the secret.
            enabled (bool, optional): A flag indicating whether the secret is enabled or disabled. Defaults to False.

        Returns:
            SecretProperties: The updated properties of the secret.

        Raises:
            AzureError: If there is an issue accessing the Azure Key Vault or updating the secret properties.
            Exception: For any other unforeseen errors that occur.
        """
        try:
            # Update the secret's properties
            updated_secret_properties = self.client.update_secret_properties(
                secret_name, content_type=content_type, enabled=enabled
            )
            _logger.info(f"Successfully updated properties for secret '{secret_name}'.")

            # Return updated secret properties
            return updated_secret_properties

        except AzureError as e:
            _logger.error(
                f"Failed to update properties for secret '{secret_name}': {e}"
            )
            raise e
        except Exception as e:
            _logger.error(
                f"An unexpected error occurred while updating properties for secret '{secret_name}': {e}"
            )
            raise e

    def delete_secret(self, secret_name: str) -> None:
        """
        Deletes a secret from Azure Key Vault.

        This method requests Key Vault to delete a secret. If soft-delete is enabled, this method allows for
        immediate deletion by waiting for the process to complete.

        Args:
            secret_name (str): The name of the secret to delete.

        Raises:
            AzureError: If there is an issue accessing the Azure Key Vault or deleting the secret.
            Exception: For any other unforeseen errors that occur.
        """
        try:
            # Begin the deletion process for the secret
            deleted_secret = self.client.begin_delete_secret(secret_name).result()
            _logger.info(
                f"Successfully deleted secret '{deleted_secret.name}' from Azure Key Vault."
            )

        except AzureError as e:
            _logger.error(f"Failed to delete secret '{secret_name}': {e}")
            raise e
        except Exception as e:
            _logger.error(
                f"An unexpected error occurred while deleting secret '{secret_name}': {e}"
            )
            raise e

    def list_secrets(self) -> List[str]:
        """
        Lists the names of all secrets in Azure Key Vault.

        This method retrieves a list of the names of all secrets in the client's vault. It does not include
        the values or versions of the secrets.

        Returns:
            List[str]: A list of secret names in the Azure Key Vault.

        Raises:
            AzureError: If there is an issue accessing the Azure Key Vault or listing the secrets.
            Exception: For any other unforeseen errors that occur.
        """
        try:
            # List properties of all secrets in the vault
            secret_properties = self.secret_client.list_properties_of_secrets()
            secret_names = [
                secret_property.name for secret_property in secret_properties
            ]

            _logger.info("Successfully listed all secrets in Azure Key Vault.")
            return secret_names

        except AzureError as e:
            _logger.error(f"Failed to list secrets in Azure Key Vault: {e}")
            raise e
        except Exception as e:
            _logger.error(
                f"An unexpected error occurred while listing secrets in Azure Key Vault: {e}"
            )
            raise e
