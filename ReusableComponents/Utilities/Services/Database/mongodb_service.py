"""
mongodb_service.py
-------------------

Description
------------
This module implements the MongoDB service class that provides methods for interacting with a MongoDB database.

"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Import dependencies
import os
import json
import logging
from bson import ObjectId
from dotenv import load_dotenv

from pymongo.errors import PyMongoError
from pymongo.server_api import ServerApi
from pymongo.mongo_client import MongoClient
from jsonschema import validate, ValidationError

from typing import Any, Optional, Dict, List, Union

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Load Environment variables
load_dotenv(override=True)


class MongoDBService:
    """
    MongoDBService class provides methods for interacting with a MongoDB database.
    """

    def __init__(
        self,
        cluster_url: str = None,
        user: str = None,
        password: str = None,
        cluster_name: str = None,
        db_name: str = None,
        collection_name: str = None,
        uri: str = None,
        max_pool_size: int = 5,
        wait_queue_timeout_ms: int = 100,
    ) -> None:
        """
        Initialize MongoDBService class with the required parameters.

        Args:
        -----
            - cluster_url (str[Optional]): The MongoDB cluster URL.
            - user (str[Optional]): The MongoDB user.
            - password (str[Optional]): The MongoDB password.
            - cluster_name (str[Optional]): The MongoDB cluster name.
            - db_name (str[Optional]): The MongoDB database name.
            - collection_name (str[Optional]): The MongoDB collection name.
            - uri (str[Optional]): The MongoDB URI.
            - maxPoolSize (int[Optional]): The maximum number of connections in the connection pool. Defaults to 5.
            - waitQueueTimeoutMS (int[Optional]): The maximum time in milliseconds that a thread waits for a connection to become available. Defaults to 100.

        Returns:
        --------
            - None

        Raises:
        -------
            - ValueError: If no parameters are provided.
            - ValueError: If both URI and cluster URL, user, cluster_name and password are provided.
            - ValueError: If database or collection names are not provided.

        Notes:
        ------
        - If uri is provided, it will be used to connect to the MongoDB cluster.
        - If cluster_url, user, cluster_name and password are provided, they will be used to connect to the MongoDB cluster.
        - If no parameters are provided, the connection will be established using the environment variables.

        Example:
        --------
        >>> mongo_db_service = MongoDBService(uri="mongodb://localhost:27017", collection_name="my_collection", db_name="my_database")
                                                # Connect to MongoDB cluster using URI
        >>> mongo_db_service = MongoDBService(
            cluster_url="localhost:27017",
            user="admin",
            password="admin",
            cluster_name="my_cluster",
            collection_name="my_collection",
            db_name="my_database"
        )
                                                # Connect to MongoDB cluster using cluster URL, user, cluster_name and password
        >>> mongo_db_service = MongoDBService() # Connect to MongoDB cluster using environment variables
        """

        # Assign values from function arguments or environment variables
        self.cluster_url = cluster_url or os.getenv("MONGODB_CLUSTER_URL")
        self.user = user or os.getenv("MONGODB_USER")
        self.password = password or os.getenv("MONGODB_PASSWORD")
        self.cluster_name = cluster_name or os.getenv("MONGODB_CLUSTER_NAME")
        self.uri = uri or os.getenv("MONGODB_URI")

        self.db_name = db_name or os.getenv("MONGODB_DATABASE")
        self.collection_name = collection_name or os.getenv("MONGODB_COLLECTION")

        self.max_pool_size = max_pool_size
        self.wait_queue_timeout_ms = wait_queue_timeout_ms

        # Validate the connection parameters
        self._validate_connection_parameters()

        # Connect to MongoDB cluster
        self.connection = self._connect_to_mongodb()

        # Get the database and collection
        self.db = self.connection.get_database(self.db_name)
        if self.db is None:
            _logger.error(f"Database {self.db_name} not found.")
            raise ValueError(f"Database {self.db_name} not found.")
        self.collection = self.db.get_collection(self.collection_name)
        if self.collection is None:
            _logger.info(f"Collection {self.collection_name} not found.")
            raise ValueError(f"Collection {self.collection_name} not found.")

    # Connection Management

    def _validate_connection_parameters(self) -> None:
        """
        Validates the connection parameters.

        Args:
        -----
            - None

        Returns:
        --------
            - None

        Raises:
        -------
            - ValueError: If no parameters are provided.
            - ValueError: If both URI and cluster URL, user, cluster_name and password are provided.
            - ValueError: If database or collection names are not provided.

        Notes:
        ------
        - If uri is provided, it will be used to connect to the MongoDB cluster.
        - If cluster_url, user, cluster_name and password are provided, they will be used to connect to the MongoDB cluster.
        - If no parameters are provided, the connection will be established using the environment variables.

        Example:
        --------
        >>> mongo_db_service = MongoDBService(uri="mongodb://localhost:27017", collection_name="my_collection", db_name="my_database")
                                                # Connect to MongoDB cluster using URI
        >>> mongo_db_service = MongoDBService(
            cluster_url="localhost:27017",
            user="XXXXX",
            password="XXXXX",
            cluster_name="my_cluster",
            collection_name="my_collection",
            db_name="my_database"
        )
                                                # Connect to MongoDB cluster using cluster URL, user, cluster_name and password
        >>> mongo_db_service = MongoDBService() # Connect to MongoDB cluster using environment variables
        """
        if (
            not self.uri
            and not self.cluster_url
            and not self.user
            and not self.password
            and not self.cluster_name
        ):
            raise ValueError("No connection parameters provided.")
        elif (
            self.uri
            and self.cluster_url
            and self.user
            and self.password
            and self.cluster_name
        ):
            raise ValueError(
                "Both URI and cluster URL, user, cluster_name and password are provided. Please provide only one of them."
            )
        elif not self.db_name or not self.collection_name:
            raise ValueError("Database or collection names are not provided.")

    def _connect_to_mongodb(self) -> MongoClient:
        """
        Connect to the MongoDB cluster.

        Args:
        -----
            - None

        Returns:
        --------
            - MongoClient: The MongoDB client.
        """
        if self.uri:
            return MongoClient(
                self.uri,
                server_api=ServerApi("1"),
                maxPoolSize=self.max_pool_size,
                waitQueueTimeoutMS=self.wait_queue_timeout_ms,
            )
        else:
            connection_url = f"mongodb+srv://{self.user}:{self.password}@{self.cluster_url}/?retryWrites=true&w=majority&appName={self.cluster_name}"
            return MongoClient(
                connection_url,
                server_api=ServerApi("1"),
                maxPoolSize=self.max_pool_size,
                waitQueueTimeoutMS=self.wait_queue_timeout_ms,
            )

    def _close_connection(self) -> None:
        """
        Close the connection to the MongoDB cluster.

        Args:
        -----
            - None

        Returns:
        --------
            - None
        """
        try:
            self.connection.close()
            print("Connection to MongoDB cluster closed.")
        except PyMongoError as e:
            print(f"Error closing connection: {e}")

    # Utility Methods
    def _validate_document(
        self, schema: Dict[str, Any], document: Dict[str, Any]
    ) -> bool:
        """
        Validates a document against a given schema.

        Args:
        -----
            schema (Dict[str, Any]): The schema to validate against.
            document (Dict[str, Any]): The document to validate.

        Returns:
        --------
            bool: True if the document is valid, False otherwise.
        """
        try:
            validate(document, schema)
            return True
        except ValidationError:
            return False

    # Database Operations
    def _list_databases(self) -> List[str]:
        """
        Lists all databases in the MongoDB cluster.

        Args:
        -----
            - None

        Returns:
        --------
            - List[str]: A list of database names.

        Raises:
        -------
            - PyMongoError: If there is an error while listing the databases.

        Example:
        --------
        ```python
        # List all databases
        databases = mongo_db_service.list_databases()
        print("Databases:", databases)
        ```
        """
        try:
            return self.connection.list_database_names()
        except PyMongoError as e:
            print(f"Error listing databases: {e}")
            return []

    def _list_collections(self) -> List[str]:
        """
        Lists all collections in the database.

        Args:
        -----
            - None

        Returns:
        --------
            - List[str]: A list of collection names.

        Raises:
        -------
            - PyMongoError: If there is an error while listing the collections.

        Example:
        --------
        ```python
        # List all collections
        collections = mongo_db_service.list_collections()
        print("Collections:", collections)
        ```
        """
        try:
            return self.db.list_collection_names()
        except PyMongoError as e:
            print(f"Error listing collections: {e}")
            return []

    def _create_collections(self, collection_name: str | List[str]) -> None:
        """
        Creates a collection in the database if it doesn't exist.

        Args:
        -----
            - collection_name (str | List[str]): The name of the collection to create.

        Returns:
        --------
            - None

        Raises:
        -------
            - PyMongoError: If there is an error while creating the collection.

        Example:
        --------
        ```python
        # Create a collection
        mongo_db_service._create_collections("new_collection")
        ```
        """
        try:
            if isinstance(collection_name, list):
                for name in collection_name:
                    self._create_collections(name)
                    print(f"Collection {name} created.")
            self.db.create_collection(collection_name)
            print(f"Collection {collection_name} created.")
        except PyMongoError as e:
            print(f"Error creating collection: {e}")

    def _drop_collections(self, collection_name: str | List[str]) -> None:
        """
        Drops a collection from the database.

        Args:
        -----
            - collection_name (str | List[str]): The name of the collection to drop.

        Returns:
        --------
            - None

        Raises:
        -------
            - PyMongoError: If there is an error while dropping the collection.

        Example:
        --------
        ```python
        # Drop a collection
        mongo_db_service._drop_collections("old_collection")
        ```
        """
        try:
            if isinstance(collection_name, list):
                for name in collection_name:
                    self._drop_collections(name)
                    print(f"Collection {name} dropped.")
            self.db.drop_collection(collection_name)
            print(f"Collection {collection_name} dropped.")
        except PyMongoError as e:
            print(f"Error dropping collection: {e}")

    # Basic CRUD operations
    def insert_one(
        self,
        document: Dict[str, Any],
        validate_document: bool = False,
        schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Insert a single document into the specified collection.

        Args:
        -----
            - document (Dict[str, Any]): The document to be inserted.
            - validate_document (bool): If True, validates the document against the schema. Defaults to False.
            - schema (Optional[Dict[str, Any]]): The schema to validate the document against. Defaults to None.

        Returns:
        --------
            - str: The ID of the inserted document.

        Raises:
        -------
            - ValidationError: If the document is not valid against the schema.
            - PyMongoError: If there is an error while inserting the document.

        Example:
        --------
        ```python
        # Insert a single document
        doc_id = mongo_db_service.insert_one(
            {"name": "John Doe", "email": "W0HdD@example.com"}
        )
        print("Inserted Document ID:", doc_id)
        ```
        """
        print(f"Inserting document {document}")
        if validate_document:
            # Validate the document against the schema
            if not self._validate_document(schema, document):
                raise ValidationError("Document validation failed.")
        try:
            # Insert the document into the collection
            doc = self.collection.insert_one(document)
            # Return the ID of the inserted document
            print(f"Inserted document with id: {doc.inserted_id}")
            return str(doc.inserted_id)
        except PyMongoError as e:
            # Raise a PyMongoError if there is an error while inserting the document
            print(f"Error inserting document: {e}")
            raise PyMongoError(f"Error inserting document: {e}") from e

    def insert_many(
        self,
        documents: List[Dict[str, Any]],
        validate_document: bool = False,
        schema: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Insert multiple documents into the specified collection.

        Args:
        -----
            - documents (List[Dict[str, Any]]): The documents to be inserted.
            - validate_document (bool): If True, validates the documents against the schema. Defaults to False.
            - schema (Optional[Dict[str, Any]]): The schema to validate the documents against. Defaults to None.

        Returns:
        --------
            - List[str]: The IDs of the inserted documents.

        Raises:
        -------
            - ValidationError: If any document is not valid against the schema.
            - PyMongoError: If there is an error while inserting the documents.

        Example:
        --------
        ```python
        # Insert Documents
        docs = [
            {
                "item": "journal",
                "qty": 25,
                "tags": ["blank", "blue"],
                "size": {"h": 14, "w": 21, "uom": "cm"},
            },
            {
                "item": "mat",
                "qty": 85,
                "tags": ["gray"],
                "size": {"h": 27.9, "w": 35.5, "uom": "cm"},
            },
            {
                "item": "mousepad",
                "qty": 25,
                "tags": ["gel", "blue"],
                "size": {"h": 19, "w": 22.85, "uom": "cm"},
            },
        ]

        docs = mongo_db_service.insert_many(documents=docs)
        print(f"Inserted documents: {docs}")
        ```
        """
        print(f"Inserting documents {documents}")
        if validate_document:
            # Validate the documents against the schema
            for doc in documents:
                if not self._validate_document(schema, doc):
                    print(
                        f"Document {doc} is not valid against the schema. Document validation failed for document {doc}"
                    )
                    raise ValidationError("Document validation failed.")
        try:
            # Insert the documents into the collection
            docs = self.collection.insert_many(documents)
            # Return the IDs of the inserted documents
            print(f"Inserted documents with ids: {docs.inserted_ids}")
            return [str(doc) for doc in docs.inserted_ids]
        except PyMongoError as e:
            # Raise a PyMongoError if there is an error while inserting the documents
            print(f"Error inserting documents: {e}")
            raise PyMongoError(f"Error inserting documents: {e}") from e

    def find_documents(
        self,
        query: Optional[Dict[str, Any]] = None,
        single: bool = False,
        sort: Optional[List[tuple]] = None,
        page: int = 1,
        limit: int = 10,
        include_soft_deleted: bool = False,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
        """
        Finds documents in the collection based on the query.

        Args:
        -----
            query (Optional[Dict[str, Any]]): Query filter for the search. Defaults to None.
            single (bool): If True, returns a single document. Defaults to False.
            sort (Optional[List[tuple]]): List of tuples for sorting, e.g., [("field", 1)]. Defaults to None.
            page (int): Page number for pagination (1-based index). Defaults to 1.
            limit (int): Number of documents to return per page. Defaults to 10.
            include_soft_deleted (bool): If True, includes soft-deleted documents. Defaults to False.

        Returns:
        --------
            Union[Dict[str, Any], List[Dict[str, Any]], None]: A single document, a list of documents, or None if no results.

        Raises:
        -------
            PyMongoError: If there is an error while finding the documents.

        Example:
        --------
        ```python
        # Find a single document by ID
        single_doc = mongo_db_service.find_documents(
            query={"_id": "678dff9d2599d30e184794bd"}, single=True
        )
        print("Single Document:", single_doc)

        # Find multiple documents with pagination and sorting
        multiple_docs = mongo_db_service.find_documents(query={}, page=1, limit=2)
        print("Multiple Documents:", multiple_docs)
        ```
        """
        try:
            # Initialize the query if None
            if query is None:
                query = {}

            # Exclude soft-deleted documents if include_soft_deleted is True
            if include_soft_deleted:
                query["deleted"] = {"$eq": True}

            # Return a single document if 'single' is True
            if single:
                return self.collection.find_one(query)

            # Query the collection with pagination and sorting
            cursor = self.collection.find(query)

            # Apply sorting if specified
            if sort:
                cursor = cursor.sort(sort)

            # Apply pagination by skipping documents and limiting the number of results
            cursor = cursor.skip((page - 1) * limit).limit(limit)

            # Convert cursor to a list of documents
            return list(cursor)

        except PyMongoError as e:
            # Log and return None if an error occurs
            print(f"Error finding documents: {e}")
            return None

    def count_documents(self, query: Optional[Dict[str, Any]] = None) -> int:
        """
        Count the number of documents in the collection that match the given query filter.

        Args:
            query (Optional[Dict[str, Any]]): A dictionary representing the query filter.
                                                If None, counts all documents in the collection.

        Returns:
            int: The total count of documents that match the query filter. Returns 0 if an error occurs.

        Raises:
            PyMongoError: Catches exceptions raised by PyMongo operations.

        Example:
        --------
        ```python
        # Count documents
        doc_count = mongo_db_service.count_documents(query={})
        print("Document Count:", doc_count)
        ```
        """
        try:
            # Use an empty dictionary as the default to count all documents
            print(f"Counting documents with query: {query}")
            count = self.collection.count_documents(query or {})
            print(f"Counted {count} documents.")
            return count
        except PyMongoError as e:
            # Log the error and return 0 if an exception occurs
            print(f"Error counting documents: {e}")
            return 0

    def update_documents(
        self,
        query: Dict[str, Any],
        update: Dict[str, Any],
        single: bool = False,
    ) -> Union[int, None]:
        """
        Updates documents in the collection based on the query.

        Args:
            query (Dict[str, Any]): The query filter for selecting documents.
            update (Dict[str, Any]): The update operation to apply.
            single (bool): If True, updates only a single document. Defaults to False.

        Returns:
            Union[int, None]: The count of documents modified or None if an error occurred.

        Raises:
            PyMongoError: Catches exceptions raised by PyMongo operations.

        Example:
        --------
        ```python
        # Update single document
        single_update = mongo_db_service.update_documents(
            query={"item": "journal"},
            update={"$set": {"qty": 50}},
            single=True,
        )
        print("Single Document Updated Count:", single_update)

        # Update multiple documents
        multiple_update = mongo_db_service.update_documents(
            query={"tags": {"$in": ["blue"]}},
            update={"$set": {"qty": 100}},
        )
        print("Multiple Documents Updated Count:", multiple_update)
        ```
        """
        try:
            print(f"Updating documents with query: {query}")
            print(f"Update operation: {update}")
            # Update one document if 'single' is True
            if single:
                result = self.collection.update_one(query, update)
                print(f"Updated {result.modified_count} documents.")
                return result.modified_count

            # Update multiple documents if 'single' is False
            else:
                result = self.collection.update_many(query, update)
                print(f"Updated {result.modified_count} documents.")
                return result.modified_count

        except PyMongoError as e:
            # Log the error and return None if an exception occurs
            print(f"Error updating documents: {e}")
            return None

    def delete_documents(
        self,
        query: Dict[str, Any],
        single: bool = False,
    ) -> Union[int, None]:
        """
        Deletes documents in the collection based on the query.

        Args:
            query (Dict[str, Any]): The query filter for selecting documents.
            single (bool): If True, deletes only a single document. Defaults to False.

        Returns:
            Union[int, None]: The count of documents deleted or None if an error occurred.

        Raises:
            PyMongoError: Catches exceptions raised by PyMongo operations.

        Example:
        --------
        ```python
        # Delete a single document by query
        single_delete = mongo_db_service.delete_documents(
            query={"item": "mousepad"},
            single=True,
        )
        print("Single Document Deleted Count:", single_delete)

        # Delete multiple documents
        multiple_delete = mongo_db_service.delete_documents(
            query={"qty": {"$gte": 85}},
        )
        print("Multiple Documents Deleted Count:", multiple_delete)
        ```
        """
        try:
            print(f"Deleting documents with query: {query}")
            print(f"Single: {single}")

            # Delete one document if 'single' is True
            if single:
                result = self.collection.delete_one(query)
                print(f"Deleted {result.deleted_count} documents.")
                return result.deleted_count

            # Delete multiple documents if 'single' is False
            else:
                result = self.collection.delete_many(query)
                print(f"Deleted {result.deleted_count} documents.")
                return result.deleted_count

        except PyMongoError as e:
            # Log the error and return None if an exception occurs
            print(f"Error deleting documents: {e}")
            return None

    def soft_delete_documents(self, query: Dict[str, Any], single: bool = False) -> int:
        """
        Perform a soft delete by setting a `deleted` flag in the matching documents.

        Args:
        -----
            query (Dict[str, Any]): The query to match documents.
            single (bool): If True, update only the first matching document. Defaults to False.

        Returns:
        --------
            int: The count of documents updated.

        Raises:
        -------
            - PyMongoError: If there is an error during the soft delete operation.

        Example:
        --------
        ```python
        # Soft delete a single document
        single_delete = mongo.soft_delete_documents(
            query={"item": "journal"},
            single=True,
        )
        print("Single Document Soft Deleted Count:", single_delete)

        # Soft delete multiple documents
        multiple_delete = mongo.soft_delete_documents(
            query={"qty": {"$lt": 30}},
        )
        print("Multiple Documents Soft Deleted Count:", multiple_delete)
        ```
        """
        update = {"$set": {"deleted": True}}
        try:
            if single:
                result = self.collection.update_one(query, update)
                return result.modified_count
            else:
                result = self.collection.update_many(query, update)
                return result.modified_count
        except PyMongoError as e:
            print(f"Error during soft delete: {e}")
            return 0

    def aggregate_documents(
        self,
        pipeline: List[Dict[str, Any]],
        allow_disk_use: bool = True,
        explain: bool = False,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any], None]:
        """
        Perform an aggregation pipeline query.

        Args:
        -----
            pipeline (List[Dict[str, Any]]): The aggregation pipeline as a list of stages.
            allow_disk_use (bool): Allows disk usage for aggregation operations. Defaults to True.
            explain (bool): If True, returns the aggregation execution plan. Defaults to False.

        Returns:
        --------
            Union[List[Dict[str, Any]], Dict[str, Any], None]: Aggregated results or execution plan.

        Raises:
        -------
            PyMongoError: If there is an error during the aggregation.

        Example:
        --------
        ```python
        # Custom aggregation pipeline
        pipeline = [
            {"$match": {"status": "active"}},
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]
        results = mongo_db.aggregate_documents(pipeline)
        print("Aggregation Results:", results)
        ```
        """
        try:
            if explain:
                return self.collection.aggregate(
                    pipeline, allowDiskUse=allow_disk_use, explain=True
                )

            results = self.collection.aggregate(pipeline, allowDiskUse=allow_disk_use)
            return list(results)

        except PyMongoError as e:
            print(f"Error during aggregation: {e}")
            return None

    def common_aggregations(
        self, operation: str, field: str, query: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Provide helper methods for common aggregation operations.

        Args:
        -----
            operation (str): The aggregation operation (e.g., "count", "sum", "average").
            field (str): The field to perform the aggregation on.
            query (Optional[Dict[str, Any]]): An optional filter query. Defaults to None.

        Returns:
        --------
            Optional[List[Dict[str, Any]]]: The results of the aggregation or None if an error occurs.

        Raises:
        -------
            ValueError: If the operation is not supported.

        Example:
        --------
        ```python
        # Count aggregation
        count_results = mongo_db.common_aggregations("count", "status", {"status": "active"})
        print("Count Results:", count_results)

        # Sum aggregation
        sum_results = mongo_db.common_aggregations("sum", "qty")
        print("Sum of Quantities:", sum_results)
        ```
        """
        if query is None:
            query = {}

        MATCH_STRING = "$match"
        GROUP_STRING = "$group"

        try:
            if operation == "count":
                pipeline = [
                    {MATCH_STRING: query},
                    {GROUP_STRING: {"_id": None, "count": {"$sum": 1}}},
                ]
            elif operation == "sum":
                pipeline = [
                    {MATCH_STRING: query},
                    {GROUP_STRING: {"_id": None, "total": {"$sum": f"${field}"}}},
                ]
            elif operation == "average":
                pipeline = [
                    {MATCH_STRING: query},
                    {GROUP_STRING: {"_id": None, "average": {"$avg": f"${field}"}}},
                ]
            elif operation == "max":
                pipeline = [
                    {MATCH_STRING: query},
                    {GROUP_STRING: {"_id": None, "max": {"$max": f"${field}"}}},
                ]
            elif operation == "min":
                pipeline = [
                    {MATCH_STRING: query},
                    {GROUP_STRING: {"_id": None, "min": {"$min": f"${field}"}}},
                ]
            else:
                raise ValueError(f"Unsupported aggregation operation: {operation}")

            return self.aggregate_documents(pipeline)

        except ValueError as ve:
            print(f"Validation Error: {ve}")
            return None
        except PyMongoError as e:
            print(f"Error during common aggregation: {e}")
            return None

    # Bulk Operations

    # Export-Import Operations
    def export_to_json(
        self, file_path: str, query: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Export data from the collection to a JSON file.

        Args:
        -----
            file_path (str): Path to save the JSON file.
            query (Optional[Dict[str, Any]]): Query filter to select documents. Defaults to None.

        Returns:
        --------
            bool: True if the export was successful, False otherwise.
        """
        try:
            # If no query is provided, export all documents
            query = query or {}
            documents = list(self.collection.find(query))

            # Convert ObjectId to string for JSON serialization
            for doc in documents:
                doc["_id"] = str(doc["_id"])

            # Write documents to a JSON file
            with open(file_path, "w") as json_file:
                json.dump(documents, json_file, indent=4)
            print(f"Data exported successfully to {file_path}")
            return True
        except (PyMongoError, IOError) as e:
            print(f"Error exporting data: {e}")
            return False

    def import_from_json(self, file_path: str, replace_existing: bool = False) -> bool:
        """
        Import data from a JSON file into the collection.

        Args:
        -----
            file_path (str): Path to the JSON file to import.
            replace_existing (bool): If True, replace existing documents with the same `_id`. Defaults to False.

        Returns:
        --------
            bool: True if the import was successful, False otherwise.
        """
        try:
            # Read the JSON file
            with open(file_path, "r") as json_file:
                documents = json.load(json_file)

            # If replacing existing documents, use `replace_one`
            if replace_existing:
                for doc in documents:
                    doc["_id"] = ObjectId(doc["_id"])
                    self.collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)
            else:
                # Otherwise, insert all documents, ignoring duplicate `_id` errors
                for doc in documents:
                    doc["_id"] = ObjectId(doc["_id"])
                self.collection.insert_many(documents, ordered=False)

            print(f"Data imported successfully from {file_path}")
            return True
        except (PyMongoError, IOError) as e:
            print(f"Error importing data: {e}")
            return False
