"""
azure_cosmos.py
----------------

This module contains the utility functions for Azure Cosmos DB.

Prerequisites
-------------
1. An Azure Cosmos account -
    https://docs.microsoft.com/azure/cosmos-db/create-sql-api-python#create-a-database-account
2. Microsoft Azure Cosmos PyPi package -
    https://pypi.python.org/pypi/azure-cosmos/

Sample - demonstrates the basic CRUD operations on a Database resource for Azure Cosmos
----------------------------------------------------------------------------------------------------------

    1. Query for Database (QueryDatabases)
    2. Create Database (CreateDatabase)
    3. Get a Database by its Id property (ReadDatabase)
    4. List all Database resources on an account (ReadDatabases)
    5. Delete a Database given its Id property (DeleteDatabase)
    
Sample - demonstrates the basic CRUD operations on a Container resource for Azure Cosmos
----------------------------------------------------------------------------------------------------------

    1. Query for Container
    2. Create Container
        2.1 - Basic Create
        2.2 - Create container with custom IndexPolicy
        2.3 - Create container with provisioned throughput set
        2.4 - Create container with unique key
        2.5 - Create Container with partition key V2
        2.6 - Create Container with partition key V1
        2.7 - Create Container with analytical store enabled
    3. Manage Container Provisioned Throughput
        3.1 - Get Container provisioned throughput (RU/s)
        3.2 - Change provisioned throughput (RU/s)
    4. Get a Container by its Id property
    5. List all Container resources in a Database
    6. Delete Container

Note
----
Running this sample will create (and delete) multiple Containers on your account.
Each time a Container is created the account will be billed for 1 hour of usage based on
the provisioned throughput (RU/s) of that account.

Requirements
------------
- azure-cosmos==4.9.0
- azure-core==1.32.0
- azure-identity==1.6.0

TODO:
-----

FIXME:
------

Author
------
Sourav Das

Date
------
2024-26-12
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Import dependencies
import logging
import azure.cosmos.exceptions as exceptions
from azure.cosmos import ThroughputProperties
from azure.identity import ClientSecretCredential
from azure.cosmos.partition_key import PartitionKey
from azure.cosmos import (
    CosmosClient as AzureCosmosClient,
    DatabaseProxy,
    ContainerProxy,
)

from typing import Optional, Tuple, List, Dict, Any

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# CONSTANT
CALL_CONNECT_MESSAGE = "Client is not connected. Call 'connect()' first."


class CosmosClient:
    """
    A class to manage the Azure Cosmos DB Client with support for multiple authentication methods:
        1. Azure Active Directory (AAD) authentication.
        2. Host and Master Key authentication.

    This class provides methods to connect to the Cosmos DB account, initialize database and container proxies,
    and retrieve the CosmosClient object.

    Attributes:
    -----------
        host (str): The Cosmos DB account endpoint.
        master_key (str): The Cosmos DB account master key.
        tenant_id (str): Azure Active Directory tenant ID.
        client_id (str): Azure Active Directory client ID.
        client_secret (str): Azure Active Directory client secret.
        database_id (str): Cosmos DB database ID.
        container_id (str): Cosmos DB container ID.
        partition_key (str): Cosmos DB partition key.
        client (AzureCosmosClient): The Cosmos DB client.
        database (DatabaseProxy): The database proxy.
        container (ContainerProxy): The container proxy.

    Methods:
    --------
        connect(auth_method: str): Establish a connection to the Cosmos DB account.
        initialize_database_and_container(): Initialize the database and container proxies.
        initialize_database(): Initialize the database proxy.
        initialize_container(): Initialize the container proxy.
        get_client(): Get the CosmosClient object.

    Notes:
    ------
    Learn more about operations for these authorization methods:
    https://aka.ms/cosmos-native-rbac
    """

    # Supported authentication methods
    AUTH_METHOD_AAD = "aad"
    AUTH_METHOD_KEY = "key"

    def __init__(
        self,
        host: Optional[str] = None,
        master_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        database_id: Optional[str] = None,
        container_id: Optional[str] = None,
        partition_key: Optional[str] = None,
    ):
        """
        Initialize the CosmosClient with Azure Cosmos DB account details.

        Args:
            host (str, optional): The Cosmos DB account endpoint.
            master_key (str, optional): The Cosmos DB account master key.
            tenant_id (str, optional): Azure Active Directory tenant ID.
            client_id (str, optional): Azure Active Directory client ID.
            client_secret (str, optional): Azure Active Directory client secret.
            database_id (str, optional): Cosmos DB database ID.
            container_id (str, optional): Cosmos DB container ID.
            partition_key (str, optional): Cosmos DB partition key.
        """
        self.host = host
        self.master_key = master_key
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.database_id = database_id
        self.container_id = container_id
        self.partition_key = partition_key
        self.client: Optional[AzureCosmosClient] = None
        self.database: Optional[DatabaseProxy] = None
        self.container: Optional[ContainerProxy] = None

    def connect(self, auth_method: str) -> None:
        """
        Establish a connection to the Cosmos DB account and set the `self.client` attribute.

        Args:
            auth_method (str): Authentication method, either 'aad' or 'key'.

        Raises:
            ValueError: If required credentials for the chosen authentication method are missing.
        """
        if self.client:
            _logger.info("Client is already connected.")
            return

        _logger.info(f"Connecting with auth method: {auth_method}")
        if auth_method == self.AUTH_METHOD_KEY:
            if not self.host or not self.master_key:
                raise ValueError(
                    "Host and Master Key are required for key-based authentication."
                )
            _logger.info(f"Using host: {self.host}, master_key: {self.master_key}")
            self.client = AzureCosmosClient(url=self.host, credential=self.master_key)

        elif auth_method == self.AUTH_METHOD_AAD:
            if not (
                self.host and self.tenant_id and self.client_id and self.client_secret
            ):
                raise ValueError(
                    "Host, Tenant ID, Client ID, and Client Secret are required for AAD authentication."
                )

            credential = ClientSecretCredential(
                tenant_id=self.tenant_id,
                client_id=self.client_id,
                client_secret=self.client_secret,
            )
            self.client = AzureCosmosClient(url=self.host, credential=credential)
        else:
            raise ValueError(
                f"Unsupported authentication method: {auth_method}. Only '{self.AUTH_METHOD_AAD}' and '{self.AUTH_METHOD_KEY}' are supported."
            )
        _logger.info("Connected to Cosmos DB.")

    def initialize_database_and_container(self) -> Tuple[DatabaseProxy, ContainerProxy]:
        """
        Initialize the database and container proxies after connection.

        Raises:
            ValueError: If database_id or container_id is not provided.
            RuntimeError: If the client is not connected.
        """
        if self.database_id and self.container_id:
            return self.database, self.container

        if not self.database_id or not self.container_id:
            raise ValueError("Both database_id and container_id are required.")
        if not self.client:
            raise RuntimeError(CALL_CONNECT_MESSAGE)

        _logger.info(f"Initializing database: {self.database_id}")
        self.database = self.client.get_database_client(self.database_id)
        _logger.info(f"Initializing container: {self.container_id}")
        self.container = self.database.get_container_client(self.container_id)

        return self.database, self.container

    def initialize_database(self) -> DatabaseProxy:
        """
        Initialize the database proxy after connection.

        Raises:
            ValueError: If database_id is not provided.
            RuntimeError: If the client is not connected.

        Returns:
            DatabaseProxy: The database proxy object.
        """
        if self.database:
            return self.database

        if not self.database_id:
            raise ValueError("database_id is required.")
        if not self.client:
            raise RuntimeError(CALL_CONNECT_MESSAGE)
        _logger.info(f"Initializing database: {self.database_id}")
        self.database = self.client.get_database_client(self.database_id)
        return self.database

    def initialize_container(self) -> ContainerProxy:
        """
        Initialize the container proxy after connection.

        Raises:
            ValueError: If container_id is not provided.
            RuntimeError: If the client is not connected.

        Returns:
            ContainerProxy: The container proxy object.
        """
        if self.container:
            return self.container

        if not self.container_id:
            raise ValueError("container_id is required.")
        if not self.client:
            raise RuntimeError(CALL_CONNECT_MESSAGE)
        _logger.info(f"Initializing container: {self.container_id}")
        self.container = self.database.get_container_client(self.container_id)
        return self.container

    def get_client(self):
        """
        Get the CosmosClient object.

        Returns
        -------
        cosmos_client.CosmosClient
            The CosmosClient object.
        """
        return self.client


class CosmosContainerManager:
    """
    A class to manage the Azure Cosmos DB Container. This class provides the following functionalities:
        1. Query for Container (QueryContainers)
        2. Create Container (CreateContainer)
        3. Get a Container by its Id property (ReadContainer)
        4. List all Container resources in a Database (ReadContainers)
        5. Delete a Container given its Id property (DeleteContainer)
        6. Manage Container Provisioned Throughput (GetContainerProvisionedThroughput, ChangeContainerProvisionedThroughput)
    """

    def __init__(
        self,
        host: Optional[str] = None,
        master_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        container_id: Optional[str] = None,
        database_id: Optional[str] = None,
        auth_method: Optional[str] = None,
    ):
        """
        Initialize the CosmosContainerManager with a CosmosClient object.

        Args:
            host (str, optional): The Cosmos DB account endpoint.
            master_key (str, optional): The Cosmos DB account master key.
            tenant_id (str, optional): Azure Active Directory tenant ID.
            client_id (str, optional): Azure Active Directory client ID.
            client_secret (str, optional): Azure Active Directory client secret.
            container_id (str, optional): Cosmos DB container ID.
            database_id (str, optional): Cosmos DB database ID.
            auth_method (str, optional): Authentication method, either 'aad' or 'key'.
        """
        self.client = CosmosClient(
            host=host,
            master_key=master_key,
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
            container_id=container_id,
            database_id=database_id,
        )

        self.client.connect(auth_method)
        self.db, self.container = self.client.initialize_database_and_container()

    def find_container(self, container_id: str) -> List:
        """
        Find a container by its ID.

        Args:
            container_id (str): The ID of the container to find.

        Returns:
            list: A list of containers matching the given ID.
        """
        # Query the database for containers with the specified ID
        if not self.db:
            raise RuntimeError(CALL_CONNECT_MESSAGE)
        containers = list(
            self.db.query_containers(
                query="SELECT * FROM c WHERE c.id=@id",
                parameters=[dict(name="@id", value=container_id)],
            )
        )

        # Check if any containers were found
        if len(containers) > 0:
            _logger.info("Container with id '{0}' was found".format(container_id))
        else:
            _logger.info("No container with id '{0}' was found".format(container_id))

        return containers

    def create_container(
        self, id: str, return_properties: bool = False
    ) -> Optional[Dict]:
        """
        Create containers with different configurations such as custom indexing, throughput, partitioning, and unique keys.

        Args:
            id (str): Base Container ID. Additional containers will be created with specific configurations using this ID as a prefix.
            return_properties (bool, optional): Whether to return the properties of the last created container (analytical store). Defaults to False.

        Returns:
            Optional[Dict]: If return_properties is True, returns the properties of the last created container; otherwise, returns None.
        Raises:
            RuntimeError: If the database connection is not initialized.
        """
        if not self.db:
            raise RuntimeError(CALL_CONNECT_MESSAGE)

        partition_key = PartitionKey(path="/id", kind="Hash")

        def create_container_with_config(
            container_id: str, **kwargs: Any
        ) -> Optional[Dict]:
            try:
                container = self.db.create_container(
                    id=container_id, partition_key=partition_key, **kwargs
                )
                _logger.info(f"Container '{container_id}' created successfully.")
                return container.read() if return_properties else None
            except exceptions.CosmosResourceExistsError:
                _logger.warning(f"Container '{container_id}' already exists.")
                return None
            except Exception as e:
                _logger.error(f"Failed to create container '{container_id}': {e}")
                raise

        configurations = [
            {"suffix": "", "kwargs": {}},
            {
                "suffix": "_custom_index_policy",
                "kwargs": {"indexing_policy": {"automatic": False}},
            },
            {"suffix": "_custom_throughput", "kwargs": {"offer_throughput": 400}},
            {
                "suffix": "_unique_keys",
                "kwargs": {
                    "unique_key_policy": {
                        "uniqueKeys": [{"paths": ["/field1/field2", "/field3"]}]
                    }
                },
            },
            {"suffix": "_analytical_store", "kwargs": {"analytical_storage_ttl": -1}},
        ]

        properties = None
        for config in configurations:
            container_id = id + config["suffix"]
            _logger.info(
                f"Creating container: {container_id} with config: {config['kwargs']}"
            )
            properties = create_container_with_config(container_id, **config["kwargs"])

        return properties if return_properties else None

    def manage_provisioned_throughput(self, id: str):
        """
        Manage provisioned throughput of a container.

        Retrieves the current provisioned throughput of the container and updates it by adding 100 RUs.

        :param id: Container ID.
        """
        if not self.db:
            raise RuntimeError(CALL_CONNECT_MESSAGE)
        try:
            # Log the initiation of retrieving the container's throughput
            _logger.info(f"Retrieving container '{id}' throughput...")

            # Retrieve the container client and get its current throughput
            container = self.db.get_container_client(container=id)
            offer = container.get_throughput()

            # Log the current throughput
            _logger.info(
                f"Container '{id}' throughput: {offer['offerThroughput']} RUs."
            )

            # Log the initiation of updating the container's throughput
            _logger.info("Updating container throughput...")

            # Update the container's throughput by adding 100 RUs
            offer = container.replace_throughput(offer["offerThroughput"] + 100)

            # Log the updated throughput
            _logger.info(f"Updated throughput: {offer['offerThroughput']} RUs.")
        except exceptions.CosmosResourceNotFoundError:
            # Log an error if the container does not exist
            _logger.error(f"Container with id '{id}' does not exist.")

    def read_container(self, id: str):
        """
        Read a container by its ID.

        This method attempts to read a container from the database using its ID.
        It logs the process of reading the container and handles cases where the
        container does not exist.

        :param id: str - The ID of the container to read.
        """
        if not self.db:
            raise RuntimeError(CALL_CONNECT_MESSAGE)
        try:
            # Log the initiation of the container reading process
            _logger.info(f"Reading container '{id}'...")

            # Retrieve the container client
            container = self.db.get_container_client(id)

            # Attempt to read the container's properties
            container.read()

            # Log the success of finding the container
            _logger.info(f"Container '{id}' found.")
        except exceptions.CosmosResourceNotFoundError:
            # Log an error if the container does not exist
            _logger.error(f"Container with id '{id}' does not exist.")

    def delete_container(self, id: str) -> None:
        """
        Delete a container by its ID.

        This method attempts to delete a container from the database using its ID.
        It logs the process of deleting the container and handles cases where the
        container does not exist.

        Args:
            id (str): The ID of the container to delete.
        Raises:
            RuntimeError: If the database connection is not initialized.
        """
        if not self.db:
            raise RuntimeError(CALL_CONNECT_MESSAGE)
        try:
            _logger.info(f"Deleting container '{id}'...")
            self.db.delete_container(id)
            _logger.info(f"Container '{id}' deleted successfully.")
        except exceptions.CosmosResourceNotFoundError as e:
            _logger.error(f"Container with id '{id}' does not exist. Error: {e}")


class CosmosDatabaseManager:
    """
    A class to manage the Azure Cosmos DB Database. This class provides the following functionalities:
        1. Query for Database (QueryDatabases)
        2. Create Database (CreateDatabase)
        3. Get a Database by its Id property (ReadDatabase)
        4. List all Database resources on an account (ReadDatabases)
        5. Delete a Database given its Id property (DeleteDatabase)
    """

    def __init__(
        self,
        host: Optional[str] = None,
        master_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        container_id: Optional[str] = None,
        database_id: Optional[str] = None,
        auth_method: Optional[str] = None,
    ):
        """
        Initialize the CosmosContainerManager with a CosmosClient object.

        Args:
            host (str, optional): The Cosmos DB account endpoint.
            master_key (str, optional): The Cosmos DB account master key.
            tenant_id (str, optional): Azure Active Directory tenant ID.
            client_id (str, optional): Azure Active Directory client ID.
            client_secret (str, optional): Azure Active Directory client secret.
            container_id (str, optional): Cosmos DB container ID.
            database_id (str, optional): Cosmos DB database ID.
            auth_method (str, optional): Authentication method, either 'aad' or 'key'.
        """
        self.client = CosmosClient(
            host=host,
            master_key=master_key,
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret,
            container_id=container_id,
            database_id=database_id,
        )

        self.client.connect(auth_method)
        self.cosmos_client = self.client.get_client()

    def find_database(self, id: str) -> List:
        """
        Find a database by its ID.

        Args:
            id (str): The ID of the database to find.

        Returns:
            list: A list of databases matching the given ID.
        """
        # Log the initiation of the database query process
        _logger.info("Query for Database")

        if not self.cosmos_client:
            raise RuntimeError(CALL_CONNECT_MESSAGE)

        # Query the Cosmos DB for databases with the specified ID
        databases = list(
            self.cosmos_client.query_databases(
                {
                    "query": "SELECT * FROM r WHERE r.id=@id",
                    "parameters": [{"name": "@id", "value": id}],
                }
            )
        )

        # Check if any databases were found
        if len(databases) > 0:
            _logger.info("Database with id '{0}' was found".format(id))
        else:
            _logger.info("No database with id '{0}' was found".format(id))

        # Return the list of found databases
        return databases

    def create_database(self, id) -> None:
        """
        Create a database with the specified ID.

        This method attempts to create a new database in the Azure Cosmos DB.
        If a database with the specified ID already exists, it logs a warning.
        The method also attempts to create the database with auto scale settings.

        Args:
            id (str): The ID of the database to create.
        Returns:
            None
        Raises:
            RuntimeError: If the client is not connected.
        """
        if not self.cosmos_client:
            raise RuntimeError(CALL_CONNECT_MESSAGE)
        # Attempt to create the database
        _logger.info(f"Attempting to create database '{id}'...")
        try:
            self.cosmos_client.create_database(id=id)
            _logger.info(f"Database with id '{id}' created.")
        except exceptions.CosmosResourceExistsError:
            _logger.warning(f"A database with id '{id}' already exists.")

        # Attempt to create the database with auto scale settings
        _logger.info(
            f"Attempting to create database '{id}' with auto scale settings..."
        )
        try:
            self.cosmos_client.create_database(
                id=id,
                offer_throughput=ThroughputProperties(
                    auto_scale_max_throughput=5000, auto_scale_increment_percent=0
                ),
            )
            _logger.info(f"Database with id '{id}' created with auto scale settings.")
        except exceptions.CosmosResourceExistsError:
            _logger.warning(
                f"A database with id '{id}' already exists with auto scale settings."
            )

    def read_database(self, id):
        """
        Read a database by its ID.

        This method attempts to read a database in the Azure Cosmos DB using its ID.
        If the database does not exist, it logs an error.

        Args:
            id (str): The ID of the database to read.

        Returns:
            dict: The database data if found, otherwise None.
        """
        if not self.cosmos_client:
            raise RuntimeError(CALL_CONNECT_MESSAGE)

        _logger.info("Get a Database by id")

        try:
            # Get the database client using its ID
            database = self.cosmos_client.get_database_client(id)

            # Read the database properties
            data = database.read()

            # Log success if the database is found
            _logger.info(
                "Database with id '{0}' was found, it's link is {1}".format(
                    id, database.database_link
                )
            )

        except exceptions.CosmosResourceNotFoundError:
            # Log an error if the database does not exist
            _logger.info("A database with id '{0}' does not exist".format(id))

        # Return the database data if found, otherwise None
        return data if data else None

    def list_databases(self):
        """
        List all databases on an account

        This method lists all databases on the Azure Cosmos account using the provided
        connection string. It logs the database IDs if any databases are found.

        Returns:
            List[dict]: List of database data if found, otherwise an empty list.
        """
        if not self.cosmos_client:
            raise RuntimeError(CALL_CONNECT_MESSAGE)

        _logger.info("List all Databases on an account")
        _logger.info("Databases:")
        databases = list(self.cosmos_client.list_databases())

        if not databases:
            _logger.info("No databases found.")
            return []

        for database in databases:
            _logger.info(database["id"])

        return databases

    def delete_database(self, id):
        """
        Deletes a database from the Azure Cosmos account.

        This method deletes a database with the specified ID from the Azure Cosmos
        account using the provided connection string. It logs the deletion of the
        database if it exists.

        Args:
            id (str): The ID of the database to delete.

        Raises:
            CosmosResourceNotFoundError: If the database does not exist.
        """
        if not self.cosmos_client:
            raise RuntimeError(CALL_CONNECT_MESSAGE)
        _logger.info("Delete a Database by id")

        try:
            self.cosmos_client.delete_database(id)

            _logger.info("Database with id '{0}' was deleted".format(id))

        except exceptions.CosmosResourceNotFoundError:
            _logger.info("A database with id '{0}' does not exist".format(id))


class CosmosItemManager:
    """Not implemented yet."""

    pass


class CosmosIndexManager:
    """Not implemented yet."""

    pass
