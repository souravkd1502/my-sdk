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

classes
-------
- CosmosClient: A class to manage the Azure Cosmos DB Client with support for multiple authentication methods.
- CosmosContainerManager: A class to manage the Azure Cosmos DB Container.
- CosmosDatabaseManager: A class to manage the Azure Cosmos DB Database.
- CosmosItemManager: A class to manage the Azure Cosmos DB Item.
- CosmosIndexManager: A class to manage the Azure Cosmos DB Index.
- CosmosQuerySupport: A class to manage the Azure Cosmos DB Query.
- CosmosStoredProcedureManager: A class to manage the Azure Cosmos DB Stored Procedure.
- CosmosTriggerManager: A class to manage the Azure Cosmos DB Trigger.
- CosmosBulkExecutor: A class to manage the Azure Cosmos DB Bulk Executor.
- CosmosUtilManager: A class to manage the Azure Cosmos DB Utility functions.
- CosmosWrapper: A class to manage the Azure Cosmos DB Wrapper functions. (High-level interface that combines the other classes.)

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
    CosmosClient,
    DatabaseProxy,
    ContainerProxy,
    exceptions,
    PartitionKey,
)

from typing import Optional, Tuple, List, Dict, Any, Union

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# CONSTANT
CALL_CONNECT_MESSAGE = "Client is not connected. Call 'connect()' first."


class CosmosClientManager:
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
        partition_key: Optional[str] = None,
        auth_method: str = None,
    ):
        """
        Initialize the CosmosClient with Azure Cosmos DB account details.

        Args:
            host (str, optional): The Cosmos DB account endpoint.
            master_key (str, optional): The Cosmos DB account master key.
            tenant_id (str, optional): Azure Active Directory tenant ID.
            client_id (str, optional): Azure Active Directory client ID.
            client_secret (str, optional): Azure Active Directory client secret.
            partition_key (str, optional): Cosmos DB partition key.
            auth_method (str, optional): The authentication method to use. Defaults to None.
        """
        self.host = host
        self.master_key = master_key
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.partition_key = partition_key
        self.client: Optional[CosmosClient] = None
        self.database: Optional[DatabaseProxy] = None
        self.container: Optional[ContainerProxy] = None
        self.auth_method = auth_method

    def _connect(self) -> None:
        """
        Establish a connection to the Cosmos DB account and set the `self.client` attribute.
        
        Raises:
            ValueError: If required credentials for the chosen authentication method are missing.
        """
        if self.client:
            _logger.info("Client is already connected.")
            return

        _logger.info(f"Connecting with auth method: {self.auth_method}")
        if self.auth_method == self.AUTH_METHOD_KEY:
            if not self.host or not self.master_key:
                raise ValueError(
                    "Host and Master Key are required for key-based authentication."
                )
            _logger.info(f"Using host: {self.host}, master_key: {self.master_key}")
            self.client = CosmosClient(url=self.host, credential=self.master_key)

        elif self.auth_method == self.AUTH_METHOD_AAD:
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
            self.client = CosmosClient(url=self.host, credential=credential)
        else:
            raise ValueError(
                f"Unsupported authentication method: {self.auth_method}. Only '{self.AUTH_METHOD_AAD}' and '{self.AUTH_METHOD_KEY}' are supported."
            )
        _logger.info("Connected to Cosmos DB.")

    def _initialize_database_and_container(
        self,
    ) -> Tuple[DatabaseProxy, ContainerProxy]:
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

    def _initialize_database(self, database_id: str) -> DatabaseProxy:
        """
        Initialize the database proxy after connection.

        Args:
            database_id (str): The ID of the database to initialize.

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
        _logger.info(f"Initializing database: {database_id}")
        self.database = self.client.get_database_client(self.database_id)
        return self.database

    def _initialize_container(self, container_id: str, database_id: str) -> ContainerProxy:
        """
        Initialize the container proxy after connection.
        
        Args:
            container_id (str): The ID of the container to initialize.
            database_id (str): The ID of the database containing the container.

        Raises:
            ValueError: If container_id is not provided.
            RuntimeError: If the client is not connected.

        Returns:
            ContainerProxy: The container proxy object.
        """
        if self.container:
            return self.container

        if not container_id:
            raise ValueError("container_id is required.")
        if not self.client:
            raise RuntimeError(CALL_CONNECT_MESSAGE)
        _logger.info(f"Initializing container: {container_id}")
        
        if not self.database:
            self.database = self.client.get_database_client(database_id)
        self.container = self.database.get_container_client(container_id)
        return self.container

    def _get_client(self) -> CosmosClient:
        """
        Get the CosmosClient object.

        Returns
        -------
        cosmos_client.CosmosClient
            The CosmosClient object.
        """
        self._connect()
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
        db_client: DatabaseProxy,
    ):
        """
        Initialize the CosmosContainerManager with a CosmosClient object.

        Args:
            db_client (DatabaseProxy): The DatabaseProxy object.
        """
        self.db = db_client

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

    def create_new_container(
        self,
        container_id: str,
    ) -> Optional[Dict]:
        """
        Create containers with different configurations such as custom indexing, throughput, partitioning, and unique keys.

        Args:
            container_id (str): Base Container ID. Additional containers will be created with specific configurations using this ID as a prefix.

        Raises:
            RuntimeError: If the database connection is not initialized.
        """
        if not self.db:
            raise RuntimeError(CALL_CONNECT_MESSAGE)

        partition_key = PartitionKey(path="/id", kind="Hash")

        try:
            container = self.db.create_container(
                id=container_id,
                partition_key=partition_key,
            )
            _logger.info(f"Container '{container_id}' created successfully.")
            return container.read()
        except exceptions.CosmosResourceExistsError:
            _logger.warning(f"Container '{container_id}' already exists.")
            return None
        except Exception as e:
            _logger.error(f"Failed to create container '{container_id}': {e}")
            raise

    def manage_provisioned_throughput(self, container_id: str):
        """
        Manage provisioned throughput of a container.

        Retrieves the current provisioned throughput of the container and updates it by adding 100 RUs.

        :param container_id: Container ID.
        """
        if not self.db:
            raise RuntimeError(CALL_CONNECT_MESSAGE)
        try:
            # Log the initiation of retrieving the container's throughput
            _logger.info(f"Retrieving container '{container_id}' throughput...")

            # Retrieve the container client and get its current throughput
            container = self.db.get_container_client(container=container_id)
            offer = container.get_throughput()

            # Log the current throughput
            _logger.info(
                f"Container '{container_id}' throughput: {offer['offerThroughput']} RUs."
            )

            # Log the initiation of updating the container's throughput
            _logger.info("Updating container throughput...")

            # Update the container's throughput by adding 100 RUs
            offer = container.replace_throughput(offer["offerThroughput"] + 100)

            # Log the updated throughput
            _logger.info(f"Updated throughput: {offer['offerThroughput']} RUs.")
        except exceptions.CosmosResourceNotFoundError:
            # Log an error if the container does not exist
            _logger.error(f"Container with id '{container_id}' does not exist.")

    def read_container(self, container_id: str) -> Dict[str, Any]:
        """
        Read a container by its ID.

        This method attempts to read a container from the database using its ID.
        It logs the process of reading the container and handles cases where the
        container does not exist.

        :param container_id: str - The ID of the container to read.
        """
        if not self.db:
            raise RuntimeError(CALL_CONNECT_MESSAGE)
        try:
            # Log the initiation of the container reading process
            _logger.info(f"Reading container '{container_id}'...")

            # Retrieve the container client
            container = self.db.get_container_client(container_id)

            # Log the success of finding the container
            _logger.info(f"Container '{container_id}' found.")
            # Attempt to read the container's properties
            return container.read()
        except exceptions.CosmosResourceNotFoundError:
            # Log an error if the container does not exist
            _logger.error(f"Container with id '{container_id}' does not exist.")

    def delete_container(self, container_id: str) -> None:
        """
        Delete a container by its ID.

        This method attempts to delete a container from the database using its ID.
        It logs the process of deleting the container and handles cases where the
        container does not exist.

        Args:
            container_id (str): The ID of the container to delete.
        Raises:
            RuntimeError: If the database connection is not initialized.
        """
        if not self.db:
            raise RuntimeError(CALL_CONNECT_MESSAGE)
        try:
            _logger.info(f"Deleting container '{container_id}'...")
            self.db.delete_container(container_id)
            _logger.info(f"Container '{container_id}' deleted successfully.")
        except exceptions.CosmosResourceNotFoundError as e:
            _logger.error(f"Container with id '{container_id}' does not exist. Error: {e}")


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
        client: CosmosClient,
    ):
        """
        Initialize the CosmosContainerManager with a CosmosClient object.

        Args:
            client (CosmosClient): The CosmosClient object.
        """
        self.client = client

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

        if not self.client:
            raise RuntimeError(CALL_CONNECT_MESSAGE)

        # Query the Cosmos DB for databases with the specified ID
        databases = list(
            self.client.query_databases(
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

    def create_database(self, id, properties: Dict[str, Any] = None) -> None:
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
        if not self.client:
            raise RuntimeError(CALL_CONNECT_MESSAGE)
        # Attempt to create the database
        _logger.info(f"Attempting to create database '{id}'...")
        if not properties:
            try:
                self.client.create_database(id=id)
                _logger.info(f"Database with id '{id}' created.")
            except exceptions.CosmosResourceExistsError:
                _logger.warning(f"A database with id '{id}' already exists.")
        else:
            # Attempt to create the database with defined properties
            _logger.info(
                f"Attempting to create database '{id}' with properties {properties}..."
            )
            try:
                self.client.create_database(
                    id=id,
                    offer_throughput=ThroughputProperties(
                        auto_scale_max_throughput=properties[
                            "auto_scale_max_throughput"
                        ],
                        auto_scale_increment_percent=properties[
                            "auto_scale_increment_percent"
                        ],
                    ),
                )
                _logger.info(
                    f"Database with id '{id}' created with auto scale settings."
                )
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
        if not self.client:
            raise RuntimeError(CALL_CONNECT_MESSAGE)

        _logger.info("Get a Database by id")

        try:
            # Get the database client using its ID
            database = self.client.get_database_client(id)

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
        if not self.client:
            raise RuntimeError(CALL_CONNECT_MESSAGE)

        _logger.info("List all Databases on an account")
        _logger.info("Databases:")
        databases = list(self.client.list_databases())

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
        if not self.client:
            raise RuntimeError(CALL_CONNECT_MESSAGE)
        _logger.info("Delete a Database by id")

        try:
            self.client.delete_database(id)

            _logger.info("Database with id '{0}' was deleted".format(id))

        except exceptions.CosmosResourceNotFoundError:
            _logger.info("A database with id '{0}' does not exist".format(id))


class CosmosItemManager:
    """
    A class for managing CRUD operations on items in an Azure Cosmos DB container,
    with additional support for counting items and retrieving container metrics.

    Args:
        cosmos_client (CosmosClient): An instance of the CosmosWrapper or CosmosClient.
        database_id (str): The ID of the database to use.
        container_id (str): The ID of the container to use.
    """

    def __init__(self, container_client: ContainerProxy, ):
        """
        Initialize the CosmosItemManager with a CosmosClient, database ID, and container ID.

        Args:
            container_client (ContainerProxy): The ContainerProxy object.

        """
        
        # Initialize container client using the provided container ID
        self.container_client = container_client

    def create_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates a new item in the container.

        Args:
            item (Dict[str, Any]): The item to create.
            partition_key (Optional[str]): The partition key for the item. If not specified, the partition key
                from the item will be used.

        Returns:
            Dict[str, Any]: The created item.

        Raises:
            RuntimeError: If an error occurs during the create operation.
        """
        try:
            # Create the item in the container
            created_item = self.container_client.create_item(body=item)
            print(f"Item created successfully: {created_item}")
            return created_item
        except exceptions.CosmosHttpResponseError as e:
            # Raise an error if an exception occurs during the create operation
            print(f"Failed to create item: {e.message}")
            raise RuntimeError(f"Failed to create item: {e.message}")

    def read_item(self, item_id: str) -> Dict[str, Any]:
        """
        Reads an item from the container.

        Args:
            item_id (str): The ID of the item to read.

        Returns:
            Dict[str, Any]: The read item.

        Raises:
            RuntimeError: If an error occurs during the read operation.
        """
        try:
            print(f"Attempting to read item with ID '{item_id}'.")
            # Read the item from the container using the item ID and partition key
            item = self.container_client.read_item(item=item_id, partition_key=item_id)
            print(f"Successfully read item: {item}")
            return item
        except exceptions.CosmosHttpResponseError as e:
            # Raise an error if an exception occurs during the read operation
            print(f"Failed to read item with ID '{item_id}': {e.message}")
            raise RuntimeError(f"Failed to read item with ID '{item_id}': {e.message}")

    def query_items(self, query: str, parameters: Optional[Dict[str, Union[str, Any]]] = None) -> list:
        """
        Queries items from the container based on a SQL-like query and optional parameters.

        Args:
            query (str): The SQL-like query string used to fetch items from the container.
            parameters (Optional[Dict[str, Union[str, Any]]]): Optional dictionary of parameters to include in the query.

        Returns:
            list: A list of items matching the query criteria.

        Raises:
            RuntimeError: If an error occurs during the query operation.
        """
        print(f"Querying items using query '{query}' with parameters: {parameters}")
        try:
            # Execute the query with optional parameters, enabling cross-partition querying
            results = self.container_client.query_items(
                query=query,
                parameters=parameters or [],
                enable_cross_partition_query=True
            )
            # Convert the results to a list and return
            print(f"Successfully queried {len(list(results))} items.")
            return list(results)
        except exceptions.CosmosHttpResponseError as e:
            # Raise a RuntimeError if an exception occurs during the query
            print(f"Failed to query items: {e.message}")
            raise RuntimeError(f"Failed to query items: {e.message}")

    def update_item(self, item_id: str, partition_key: str, updated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates an existing item in the container.

        Args:
            item_id (str): The ID of the item to be updated.
            partition_key (str): The partition key of the item to be updated.
            updated_data (Dict[str, Any]): The dictionary of updated key-value pairs to be applied to the item.

        Returns:
            Dict[str, Any]: The updated item.

        Raises:
            RuntimeError: If an error occurs during the update operation.
        """
        try:
            print(f"Attempting to update item with ID '{item_id}' and partition key '{partition_key}'.")
            # Read the existing item to be updated
            existing_item = self.read_item(item_id)
            print(f"Read existing item: {existing_item}")

            # Update the existing item with the provided updated key-value pairs
            existing_item.update(updated_data)
            print(f"Updated item: {existing_item}")

            # Replace the existing item with the updated data
            return self.container_client.replace_item(item=existing_item, body=existing_item)
        except exceptions.CosmosHttpResponseError as e:
            # Raise a RuntimeError if an exception occurs during the update
            print(f"Failed to update item with ID '{item_id}': {e.message}")
            raise RuntimeError(f"Failed to update item with ID '{item_id}': {e.message}")

    def delete_item(self, item_id: str, partition_key: str) -> None:
        """
        Deletes a specified item from the container.

        Args:
            item_id (str): The ID of the item to be deleted.
            partition_key (str): The partition key of the item to be deleted.

        Raises:
            RuntimeError: If an error occurs during the delete operation.
        """
        print(f"Attempting to delete item with ID '{item_id}' and partition key '{partition_key}'.")
        try:
            # Attempt to delete the item using its ID and partition key
            self.container_client.delete_item(item=item_id, partition_key=partition_key)
            print(f"Item with ID '{item_id}' successfully deleted.")
        except exceptions.CosmosHttpResponseError as e:
            # Raise a RuntimeError if deletion fails
            print(f"Failed to delete item with ID '{item_id}': {e.message}")
            raise RuntimeError(f"Failed to delete item with ID '{item_id}': {e.message}")


    def count_items(self, query: Optional[str] = None) -> int:
        """
        Counts the total number of items in the container or matching a query.

        Args:
            query (str, optional): A SQL query to filter the items to count.

        Returns:
            int: The total count of items.

        Raises:
            RuntimeError: If the query fails to execute or retrieve the results.
        """
        try:
            # If no query is provided, use a query to count the total number of items
            if query is None:
                query = "SELECT VALUE COUNT(1) FROM c"
            print(f"Executing count query: {query}")

            # Execute the query and retrieve the results
            result = self.container_client.query_items(query=query, enable_cross_partition_query=True)
            print("Query executed successfully. Retrieving count from results.")

            # Return the count, which is the first element of the result
            count = list(result)[0]
            print(f"Total items counted: {count}")
            return count
        except exceptions.CosmosHttpResponseError as e:
            # Raise a RuntimeError if the query fails or retrieval fails
            print(f"Failed to count items: {e.message}")
            raise RuntimeError(f"Failed to count items: {e.message}")


    def get_container_metrics(self) -> Dict[str, Any]:
        """
        Retrieves metrics for the container, such as storage usage and request units.

        Returns:
            dict: A dictionary containing the container metrics.
        
        Note:
        The returned dictionary contains the following metrics:

        - id (str): The ID of the container.
        - partition_key (str): The partition key for the container.
        - document_count (int): The number of documents in the container.
        - usage_size_in_mb (int): The total storage size of the container in megabytes.
        - resource_units_per_second (int): The total request units per second allocated to the container.

        Raises:
            RuntimeError: If the request fails to retrieve the container metrics.
        """
        try:
            print("Attempting to retrieve container metrics.")
            properties = self.container_client.read()
            print(f"Container properties: {properties}")
            metrics = {
                "id": properties.get("id"),
                "partition_key": properties.get("partitionKey"),
                "document_count": properties.get("documentCount"),
                "usage_size_in_mb": properties.get("usageSizeInMB"),
                "resource_units_per_second": properties.get("resourceUnitsPerSecond"),
            }
            print(f"Container metrics: {metrics}")
            return metrics
        except exceptions.CosmosHttpResponseError as e:
            print(f"Failed to retrieve container metrics: {e.message}")
            raise RuntimeError(f"Failed to retrieve container metrics: {e.message}")


class CosmosIndexManager:
    """Not implemented yet."""

    pass


class CosmosTriggerManager:
    """Not implemented yet."""

    pass


class CosmosBulkExecutor:
    """Not implemented yet."""

    pass


class CosmosUtilManager:
    """Not implemented yet."""

    pass


cosmos_obj = CosmosClientManager(
    host="https://localhost:8081",
    master_key="C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw==",
    auth_method="key",
)

cosmos_obj._connect()
client = cosmos_obj._initialize_container(container_id="test_container", database_id="test_db")


# Initialize CosmosItemManager
item_manager = CosmosItemManager(
    container_client=client,
)

# # Create an item
# item = {"id": "3", "name": "Test Item3", "partitionKey": "testPartition"}
# created_item = item_manager.create_item(item)

# Read the item
retrieved_item = item_manager.read_item(item_id="2")
print("-------------------------------------------------------------------------------")
print(f"Retrieved Item: {retrieved_item}")
print("-------------------------------------------------------------------------------")

# # Query items
# query = "SELECT * FROM c WHERE c.partitionKey = @partitionKey"
# parameters = [{"name": "@partitionKey", "value": "testPartition"}]
# queried_items = item_manager.query_items(query, parameters)
# retrieved_item = item_manager.read_item(item_id="1", partition_key="testPartition")
# print("-------------------------------------------------------------------------------")
# print(f"Query Item: {retrieved_item}")
# print("-------------------------------------------------------------------------------")

# # Update the item
# updated_item = item_manager.update_item(item_id="1", partition_key="testPartition", updated_data={"name": "Updated Item"})
# print("-------------------------------------------------------------------------------")
# print(f"updated Item: {updated_item}")
# print("-------------------------------------------------------------------------------")

# # Delete the item
# item_manager.delete_item(item_id="1", partition_key="testPartition")

# # Count all items in the container
# total_items = item_manager.count_items()
# print("-------------------------------------------------------------------------------")
# print(f"Total items: {total_items}")
# print("-------------------------------------------------------------------------------")

# # Count items matching a specific condition
# query = "SELECT * FROM c WHERE c.partitionKey = @partitionKey"
# parameters = [{"name": "@partitionKey", "value": "testPartition"}]
# filtered_count = item_manager.count_items(query)
# print("-------------------------------------------------------------------------------")
# print(f"Filtered item count: {filtered_count}")
# print("-------------------------------------------------------------------------------")

# # Get container metrics
# metrics = item_manager.get_container_metrics()
# print("-------------------------------------------------------------------------------")
# print(f"Container Metrics: {metrics}")
# print("-------------------------------------------------------------------------------")