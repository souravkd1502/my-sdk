"""
postgressql_service.py
----------------------
This module contains the implementation of the PostgresSQLService class.

Description:
------------

Usage:
------

Requirements:
-------------

Environment Variables:
-----------------------

TODO:
-----

FIXME:
------

Author:
-------
Sourav Das

Version:
--------
1.0

Date:
------
27.01.2025
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Import dependencies
import os
import json
import yaml
import logging
import psycopg2
from uuid import UUID
from psycopg2 import pool
from decimal import Decimal
from dotenv import load_dotenv
from datetime import datetime, date
from psycopg2.extensions import cursor

from typing import Any, Optional, Dict, List, Union, Tuple, Type

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Load Environment variables
load_dotenv(override=True)

# Constants
JSON =".json"

class PostgresSQLService:
    """
    This class provides the implementation of the PostgresSQLService class.
    """

    _pool = None  # Connection pool for reusing connections

    # Initialization and Configuration
    def __init__(self, config: Dict[str, Union[str, int]]) -> None:
        """
        Initialize the database wrapper with configuration parameters.

        Args:
            config (Dict[str, Union[str, int]]): Configuration dictionary containing database connection details.
                Required keys:
                    host (str): Hostname or IP address of the PostgreSQL server
                    port (int): Port number of the PostgreSQL server
                    user (str): Username for database authentication
                    password (str): Password for database authentication
                    database (str): Name of the database to connect to

        Raises:
            ValueError: If required configuration parameters are missing
        """
        required_keys = ["host", "port", "user", "password", "database"]
        missing_keys = [key for key in required_keys if key not in config]

        if missing_keys:
            raise ValueError(
                f"Missing required configuration keys: {', '.join(missing_keys)}"
            )

        self.config = config
        self.connection = None
        _logger.info("Postgres SQL Service initialized")

    def __del__(self):
        """
        Destructor to ensure proper cleanup of database connections.

        This method is automatically called when the instance is garbage collected.
        It attempts to close the database connection if it exists.
        """
        try:
            # Check if the connection exists
            if self.connection is not None:
                # Close the database connection
                self.connection.close()
                _logger.info("Database connection closed during cleanup")
        except Exception as e:
            # Log any exception that occurs during cleanup
            _logger.error(f"Error during cleanup: {str(e)}")

    @classmethod
    def from_env(cls):
        """
        Create an instance using environment variables for configuration.

        Environment variables used:
            - DB_HOST
            - DB_PORT
            - DB_USER
            - DB_PASSWORD
            - DB_NAME

        Returns:
            DatabaseWrapper: An instance of the wrapper configured with environment variables

        Raises:
            ValueError: If required environment variables are missing
        """
        config = {
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "database": os.getenv("DB_NAME"),
        }

        # Remove None values
        config = {k: v for k, v in config.items() if v is not None}

        if len(config) < 5:
            missing_vars = [
                k
                for k, v in {
                    "DB_HOST": os.getenv("DB_HOST"),
                    "DB_PORT": os.getenv("DB_PORT"),
                    "DB_USER": os.getenv("DB_USER"),
                    "DB_PASSWORD": os.getenv("DB_PASSWORD"),
                    "DB_NAME": os.getenv("DB_NAME"),
                }.items()
                if v is None
            ]
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

        return cls(config)

    @classmethod
    def load_config(cls, config_file: str):
        """
        Load configuration from a file and create an instance.

        Args:
            config_file (str): Path to the configuration file (JSON or YAML)

        Returns:
            DatabaseWrapper: An instance of the wrapper configured with the file contents

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If the file format is not supported or file is invalid
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        file_ext = os.path.splitext(config_file)[1].lower()

        try:
            with open(config_file, "r") as f:
                if file_ext == JSON:
                    config = json.load(f)
                elif file_ext in (".yml", ".yaml"):
                    config = yaml.safe_load(f)
                else:
                    raise ValueError(
                        f"Unsupported configuration file format: {file_ext}"
                        " (supported formats: .json, .yml, .yaml)"
                    )

            return cls(config)

        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Invalid configuration file format: {str(e)}")

    @classmethod
    def initialize_pool(cls, min_conn: int, max_conn: int, config: dict):
        """
        Initialize the connection pool.

        Args:
            min_conn (int): Minimum number of connections in the pool.
            max_conn (int): Maximum number of connections in the pool.
            config (dict): Configuration for connecting to the database.

        Raises:
            psycopg2.Error: If the connection pool cannot be created.
        """
        try:
            cls._pool = pool.SimpleConnectionPool(min_conn, max_conn, **config)
            _logger.info("Connection pool initialized successfully.")
        except psycopg2.Error as e:
            _logger.error(f"Failed to initialize connection pool: {e}")
            raise

    def _connect(self) -> bool:
        """
        Establish a direct database connection if not already connected.

        Returns:
            bool: True if connection successful, False otherwise.
        """
        try:
            if not self.connection or self.connection.closed:
                self.connection = psycopg2.connect(**self.config)
                _logger.info("Database connection established.")
                return True
        except psycopg2.Error as e:
            _logger.error(f"Connection error: {e}")
            return False
        return True

    def _disconnect(self) -> bool:
        """
        Close the active database connection if it exists.

        Returns:
            bool: True if disconnection successful, False otherwise.
        """
        try:
            if self.connection and not self.connection.closed:
                self.connection.close()
                self.connection = None
                _logger.info("Database connection closed.")
            return True
        except psycopg2.Error as e:
            _logger.error(f"Disconnect error: {e}")
            return False

    @classmethod
    def get_connection(cls):
        """
        Get a connection from the connection pool.

        Returns:
            psycopg2.extensions.connection: A database connection object.

        Raises:
            RuntimeError: If the connection pool is not initialized.
            psycopg2.Error: If a connection cannot be retrieved.
        """
        if cls._pool is None:
            _logger.warning("Connection pool is not initialized.")
            raise RuntimeError("Connection pool not initialized")

        try:
            conn = cls._pool.getconn()
            _logger.info("Connection retrieved from pool.")
            return conn
        except psycopg2.Error as e:
            _logger.error(f"Failed to get connection from pool: {e}")
            raise

    @classmethod
    def release_connection(cls, conn):
        """
        Release a connection back to the pool.

        Args:
            conn (psycopg2.extensions.connection): The connection to release.
        """
        if cls._pool and conn:
            cls._pool.putconn(conn)
            _logger.info("Connection released back to pool.")

    def is_connected(self) -> bool:
        """
        Check if the database connection is active.

        Returns:
            bool: True if connected, False otherwise.
        """
        if not self.connection:
            return False

        try:
            # Test connection with a simple query
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            return True
        except psycopg2.Error as e:
            _logger.warning(f"Connection test failed: {e}")
            return False

    @classmethod
    def close_pool(cls):
        """
        Close all connections in the connection pool.
        """
        if cls._pool:
            cls._pool.closeall()
            cls._pool = None
            _logger.info("Connection pool closed.")

    # CRUD Operations
    def create(self, table: str, data: dict) -> int:
        """
        Inserts a new record into the specified table and returns the ID of the inserted row.

        Args:
            table (str): The table to insert data into.
            data (dict): A dictionary of column names and values to insert.

        Returns:
            int: The ID of the inserted row.
        """
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["%s"] * len(data))
        values = list(data.values())

        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) RETURNING id"
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, values)
                self.connection.commit()
                return cursor.fetchone()[0]
        except psycopg2.Error as e:
            _logger.error(f"Error inserting into {table}: {e}")
            self.connection.rollback()
            raise

    def read(self, table: str, filters: dict = None) -> list:
        """
        Fetches rows from the specified table based on optional filter criteria.

        Args:
            table (str): The table to fetch data from.
            filters (dict): Optional dictionary of column names and their filter values.

        Returns:
            list: A list of dictionaries representing the fetched rows.
        """
        where_clause, values = self._build_where_clause(filters)
        query = f"SELECT * FROM {table} {where_clause}"

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, values)
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except psycopg2.Error as e:
            _logger.error(f"Error reading from {table}: {e}")
            raise

    def update(self, table: str, data: dict, filters: dict) -> int:
        """
        Updates records in the specified table based on filter conditions.

        Args:
            table (str): The table to update data in.
            data (dict): A dictionary of column names and their new values.
            filters (dict): A dictionary of column names and their filter values.

        Returns:
            int: The number of rows affected.
        """
        set_clause = ", ".join([f"{col} = %s" for col in data.keys()])
        where_clause, where_values = self._build_where_clause(filters)
        values = list(data.values()) + where_values

        query = f"UPDATE {table} SET {set_clause} {where_clause}"
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, values)
                self.connection.commit()
                return cursor.rowcount
        except psycopg2.Error as e:
            _logger.error(f"Error updating {table}: {e}")
            self.connection.rollback()
            raise

    def delete(self, table: str, filters: dict) -> int:
        """
        Deletes records from the specified table based on filter conditions.

        Args:
            table (str): The table to delete data from.
            filters (dict): A dictionary of column names and their filter values.

        Returns:
            int: The number of rows deleted.
        """
        where_clause, values = self._build_where_clause(filters)
        query = f"DELETE FROM {table} {where_clause}"

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, values)
                self.connection.commit()
                return cursor.rowcount
        except psycopg2.Error as e:
            _logger.error(f"Error deleting from {table}: {e}")
            self.connection.rollback()
            raise

    # Query Execution
    def execute_query(self, query: str, params: tuple = None) -> any:
        """
        Executes a single SQL query with optional parameters and returns the result.

        Args:
            query (str): The SQL query to execute.
            params (tuple): A tuple of parameters to pass into the query.

        Returns:
            any: The result of the query execution (e.g., fetched rows, affected rows count).
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                if query.strip().lower().startswith("select"):
                    return cursor.fetchall()  # Fetch results for SELECT queries
                self.connection.commit()  # Commit for INSERT, UPDATE, DELETE
                return (
                    cursor.rowcount
                )  # Return number of affected rows for non-SELECT queries
        except psycopg2.Error as e:
            _logger.error(
                f"Error executing query: {query} | Params: {params} | Error: {e}"
            )
            self.connection.rollback()
            raise

    def execute_many(self, query: str, param_list: list) -> int:
        """
        Executes a query multiple times with a list of parameter sets for batch operations.

        Args:
            query (str): The SQL query to execute.
            param_list (list): A list of tuples, where each tuple is a parameter set for one execution.

        Returns:
            int: The total number of rows affected by all executions.
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.executemany(query, param_list)
                self.connection.commit()
                return cursor.rowcount  # Total number of rows affected
        except psycopg2.Error as e:
            _logger.error(
                f"Error executing batch query: {query} | Params: {param_list} | Error: {e}"
            )
            self.connection.rollback()
            raise

    def execute_raw(self, query: str) -> List[Dict[str, Any]]:
        """
        Executes a raw SQL query without parameterization (use with caution).

        Args:
            query (str): The raw SQL query to execute.

        Returns:
            any: The result of the query execution.
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                if query.strip().lower().startswith("select"):
                    return self._format_query_results(cursor.fetchall(), cursor)
                self.connection.commit()  # Commit for non-SELECT queries
                return cursor.rowcount  # Return number of affected rows
        except psycopg2.Error as e:
            _logger.error(f"Error executing raw query: {query} | Error: {e}")
            self.connection.rollback()
            raise

    # Query Builder
    def _build_read_query(
        self,
        table: str,
        columns: List[str],
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        group_by: Optional[List[str]] = None,
        aggregates: Optional[Dict[str, str]] = None,
    ) -> Union[str, List[Any]]:
        """
        Helper function to build a SQL query for reading data from a table.

        Args:
            table (str): The name of the table to read data from.
            columns (List[str]): A list of column names to select.
            filters (Optional[Dict[str, Any]]): A dictionary of column names and their filter values.
            order_by (Optional[str]): The column to order the results by.
            limit (Optional[int]): The maximum number of rows to return.
            group_by (Optional[List[str]]): A list of column names to group the results by.
            aggregates (Optional[Dict[str, str]]): A dictionary of column names and their aggregation functions
                (e.g., {"total": "SUM", "price": "AVG"}).

        Returns:
            Union[str, List[Any]]: A tuple containing the SQL query string and a list of parameterized values.
        """
        # Base SELECT clause
        select_clause = ", ".join(columns)

        # Add aggregate functions to SELECT clause, if provided
        if aggregates:
            aggregate_clause = ", ".join(
                [
                    f"{func}({col}) AS {col}_{func.lower()}"
                    for col, func in aggregates.items()
                ]
            )
            select_clause = (
                f"{select_clause}, {aggregate_clause}" if columns else aggregate_clause
            )

        # WHERE Clause
        where_clause, where_values = self._build_where_clause(filters)

        # GROUP BY Clause
        group_by_clause = f"GROUP BY {', '.join(group_by)}" if group_by else ""

        # ORDER BY Clause
        order_by_clause = f"ORDER BY {order_by}" if order_by else ""

        # LIMIT Clause
        limit_clause = f"LIMIT {limit}" if limit else ""

        # Final Query Construction
        query = f"SELECT {select_clause} FROM {table} {where_clause} {group_by_clause} {order_by_clause} {limit_clause}"
        return self._format_query(query.strip(), where_values)

    # Helper Functions
    def _format_query(self, base_query: str, params: list) -> str:
        """
        Dynamically inserts parameterized values into a query for display or debugging purposes.
        This is NOT for actual query execution; use parameterized queries for safety.

        Args:
            base_query (str): The SQL query with placeholders (e.g., `%s`).
            params (list): The list of parameterized values to insert into the query.

        Returns:
            str: The fully formatted query as a string.
        """
        # Replace `%s` placeholders with safely escaped values from `params`
        formatted_query = base_query
        for value in params:
            if isinstance(value, str):
                # Escape single quotes manually, then wrap the string in quotes
                value = "'" + value.replace("'", "''") + "'"
            formatted_query = formatted_query.replace("%s", str(value), 1)
        return formatted_query

    def _build_where_clause(
        self, filters: Optional[Dict[str, Any]]
    ) -> Union[str, List[Any]]:
        """
        Helper function to build a WHERE clause for SQL queries.

        Args:
            filters (Optional[Dict[str, Any]]): A dictionary of column names and their filter values.

        Returns:
            Union[str, List[Any]]: A string representing the WHERE clause and a list of parameterized values.
        """
        if not filters:
            return "", []

        clauses = []
        values = []

        for col, value in filters.items():
            if (
                isinstance(value, tuple)
                and len(value) == 2
                and value[0].lower() in ["in", "not in"]
            ):
                # Handle IN and NOT IN conditions
                placeholders = ", ".join(["%s"] * len(value[1]))
                clauses.append(f"{col} {value[0].upper()} ({placeholders})")
                values.extend(value[1])
            elif isinstance(value, tuple) and len(value) == 2:
                # Handle operators like >, <, >=, <=
                clauses.append(f"{col} {value[0]} %s")
                values.append(value[1])
            else:
                # Default equality condition
                clauses.append(f"{col} = %s")
                values.append(value)

        where_clause = "WHERE " + " AND ".join(clauses)
        return where_clause, values

    # Export and Import
    def bulk_insert(self, table: str, data: List[Dict[str, Any]]) -> List[int]:
        """
        Bulk insert a list of dictionaries into the specified table.

        Args:
            table (str): The table to insert data into.
            data (List[Dict[str, Any]]): A list of dictionaries representing the data to be inserted.

        Returns:
            List[int]: A list of IDs of the inserted rows.
        """
        if not data:
            raise ValueError("Data list is empty. Nothing to insert.")

        # Dynamically construct the columns and placeholders
        columns = ", ".join(data[0].keys())
        placeholders = ", ".join(["%s"] * len(data[0]))
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) RETURNING id"

        # Prepare the data as a list of tuples
        values = [tuple(row.values()) for row in data]

        try:
            with self.connection.cursor() as cursor:
                # Execute the batch insert and fetch IDs of inserted rows
                cursor.executemany(query, values)
                self.connection.commit()
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self.connection.rollback()
            raise RuntimeError(f"Failed to perform bulk insert: {e}")

    def export_query(
        self,
        query: str,
        file_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Export data from a table based on specified filters.

        Args:
            query (str): The SQL query to execute.
            file_path (Optional[str]): The file path to export data to. If None, no file will be created.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the exported data.
        """
        # Step 1: Fetch data using the `read` method
        data = self.execute_raw(query)

        # Step 2: Handle file export if `file_path` is provided
        if file_path:
            # Validate the file format
            if not any(
                file_path.endswith(ext) for ext in [".csv", ".json", ".xlsx", ".xml"]
            ):
                raise ValueError(
                    "Invalid file format. Supported formats are .csv, .json, .xlsx, and .xml"
                )

            try:
                # Export to JSON
                if file_path.endswith(JSON):

                    def custom_serializer(obj):
                        # Handle date and datetime
                        if isinstance(obj, (date, datetime)):
                            return obj.isoformat()  # Convert to ISO 8601 string
                        # Handle Decimal
                        if isinstance(obj, Decimal):
                            return float(obj)  # Convert Decimal to float
                        # Handle UUID
                        if isinstance(obj, UUID):
                            return str(obj)  # Convert UUID to string
                        # Handle bytes
                        if isinstance(obj, bytes):
                            return obj.decode("utf-8")  # Convert bytes to string
                        # For unsupported types, raise a TypeError
                        raise TypeError(f"Type {type(obj)} is not JSON serializable")

                    with open(file_path, mode="w", encoding="utf-8") as json_file:
                        json.dump(data, json_file, indent=4, default=custom_serializer)

                # Handle other formats (CSV, XML, XLSX) as before
                elif file_path.endswith(".csv"):
                    # CSV logic here (see previous implementation)
                    pass
                elif file_path.endswith(".xlsx"):
                    # XLSX logic here (see previous implementation)
                    pass
                elif file_path.endswith(".xml"):
                    # XML logic here (see previous implementation)
                    pass

            except Exception as e:
                raise RuntimeError(f"Failed to export data to {file_path}: {e}")

        # Step 3: Return the data as a Python list
        return data

    # Utility Functions
    def _format_query_results(
        self, records: List[Tuple], cursor: cursor
    ) -> List[Dict[str, Any]]:
        """
        Converts a query result into a list of dictionaries.

        Args:
            records (List[Tuple]): The query result to be converted.
            cursor (Cursor): The cursor to Pyodbc connection instance, used to query DB.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the query result.
        """
        # Extract column names from the cursor description
        column_names = [desc[0] for desc in cursor.description]

        # Initialize an empty list to store dictionaries
        records_as_dicts = []

        # Iterate over each record in the records
        for record in records:
            # Create a dictionary for the current record
            record_dict = {}
            for i, column_name in enumerate(column_names):
                # Add key-value pairs to the dictionary
                record_dict[column_name] = record[i]

            # Append the dictionary to the list
            records_as_dicts.append(record_dict)

        return records_as_dicts

    def get_schema(self, table: str) -> List[Dict[str, Any]]:
        """
        Retrieves the detailed schema of the specified table.

        Args:
            table (str): The table name.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents a column
            with its details such as name, data type, nullable, default value, and constraints.
        """
        query = """
        SELECT 
            col.column_name AS name,
            col.data_type AS data_type,
            col.is_nullable AS nullable,
            col.column_default AS default_value,
            tc.constraint_type AS constraint_type,
            kcu.column_name AS constraint_column
        FROM information_schema.columns col
        LEFT JOIN information_schema.key_column_usage kcu 
            ON col.table_name = kcu.table_name AND col.column_name = kcu.column_name
        LEFT JOIN information_schema.table_constraints tc 
            ON kcu.table_name = tc.table_name AND kcu.constraint_name = tc.constraint_name
        WHERE col.table_name = %s
        ORDER BY col.ordinal_position
        """
        
        with self.connection.cursor() as cursor:
            cursor.execute(query, (table,))
            result = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        
        # Format the results as a list of dictionaries
        detailed_schema = [dict(zip(columns, row)) for row in result]
        return detailed_schema

    @staticmethod
    def validate_data(data: dict, schema: dict):
        """
        Validates input data against a specified schema.

        Args:
            data (dict): The input data to validate.
            schema (dict): The schema with column names and data types.

        Raises:
            ValueError: If a field is missing or its type is incorrect.
        """
        for field, expected_type in schema.items():
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
            if not PostgresSQLService._is_type_valid(data[field], expected_type):
                raise ValueError(
                    f"Invalid type for field '{field}'. Expected {expected_type}, got {type(data[field]).__name__}."
                )

    @staticmethod
    def map_row_to_object(row: dict, obj_type: Type) -> Any:
        """
        Converts a database row (dictionary) into a Python object of the specified type.

        Args:
            row (dict): A dictionary representing a database row.
            obj_type (type): The type of object to map the row to.

        Returns:
            Any: An instance of the specified type with attributes set based on the row.
        """
        obj = obj_type()
        for key, value in row.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
        return obj

    @staticmethod
    def _is_type_valid(value: Any, expected_type: str) -> bool:
        """
        Checks if a value matches the expected data type from the schema.

        Args:
            value (Any): The value to check.
            expected_type (str): The expected type as a string (from the database schema).

        Returns:
            bool: True if the value matches the type, False otherwise.
        """
        type_mapping = {
            "integer": int,
            "bigint": int,
            "smallint": int,
            "character varying": str,
            "text": str,
            "boolean": bool,
            "numeric": float,
            "real": float,
            "double precision": float,
            "timestamp without time zone": str,  # Can validate further if needed
            "timestamp with time zone": str,  # Can validate further if needed
            "date": str,  # Use ISO format validation if required
        }
        python_type = type_mapping.get(expected_type.lower())
        if not python_type:
            return False  # Unknown type in schema
        return isinstance(value, python_type)
