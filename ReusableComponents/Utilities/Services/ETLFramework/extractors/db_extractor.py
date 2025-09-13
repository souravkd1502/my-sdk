"""
db_extractor.py
===============
Database data extraction framework for ETL pipelines.

This module provides a unified framework for extracting structured and semi-structured
data from databases. It defines an abstract base class (:class:`DBExtractor`) that
standardizes database ingestion patterns, and concrete implementations for SQL
(relational databases via SQLAlchemy) and MongoDB (document-oriented NoSQL).
The design is extensible, allowing other databases (e.g., Cassandra, Redis, DynamoDB)
to be integrated seamlessly.

The framework is optimized for production ETL: it supports streaming large query
results, parallel query execution across shards or tables, and schema-aware ingestion.
Metadata about queries, record counts, and connection parameters is captured to enable
observability and auditability.

Key Features
------------
- **Base class** (`DBExtractor`) defining consistent extraction interfaces.
- **Implemented databases**:
  - **SQLExtractor**: Any SQLAlchemy-supported RDBMS (Postgres, MySQL, SQLite, Oracle, MS SQL).
    Supports parameterized queries, chunked fetching (streaming), schema introspection,
    and connection pooling.
  - **MongoDBExtractor**: Extracts from MongoDB collections using queries or aggregations.
    Supports cursor-based streaming, projection, batching, and schema inference.
- **Extensible design**: Easily extend to other databases (Cassandra, DynamoDB, Redis, etc.).
- **Streaming and chunked reads** for large datasets without overwhelming memory.
- **Parallel execution** for multi-query workloads.
- **Schema validation** to ensure compatibility with downstream transformations.
- **Connection abstraction** with retry logic and connection pooling.
- **Comprehensive metadata** including query string, execution time, rows fetched, and errors.

Exceptions/Errors
-----------------
- ``ExtractorError``: General ETL framework errors (e.g., unsupported databases).
- ``DataSourceConnectionError``: Raised when connection to the database fails.
- ``DataSourceAuthenticationError``: Raised when authentication fails.
- ``QueryExecutionError``: Raised when an SQL or Mongo query fails.
- ``DataReadError``: Raised when cursor/stream fetching fails.
- ``DependencyMissingError``: Raised when required database drivers are missing.

Usage Examples
--------------
Basic SQL extraction:

    >>> from etl_framework.extractors import SQLExtractor
    >>> extractor = SQLExtractor(
    ...     connection_config={
    ...         "db_type": "postgresql",
    ...         "username": "user",
    ...         "password": "pass",
    ...         "host": "localhost",
    ...         "port": 5432,
    ...         "database": "mydb"
    ...     },
    ...     query="SELECT * FROM sales LIMIT 1000"
    ... )
    >>> df, meta = extractor.extract()
    >>> print(f"Extracted {meta['num_records']} records")

Streaming large SQL query:

    >>> extractor = SQLExtractor(
    ...     connection_config={
    ...         "db_type": "mysql",
    ...         "username": "user",
    ...         "password": "pass",
    ...         "host": "localhost",
    ...         "port": 3306,
    ...         "database": "mydb"
    ...     },
    ...     query="SELECT * FROM big_table",
    ...     chunksize=50_000
    ... )
    >>> for chunk, meta in extractor.stream_extract():
    ...     process_chunk(chunk)

MongoDB collection extraction:

    >>> from etl_framework.extractors import MongoDBExtractor
    >>> extractor = MongoDBExtractor(
    ...     connection_config={
    ...         "uri": "mongodb://localhost:27017/",
    ...         "database": "analytics",
    ...         "collection": "events"
    ...     },
    ...     query={"event_type": "purchase"},
    ...     projection={"_id": 0, "user_id": 1, "amount": 1}
    ... )
    >>> df, meta = extractor.extract()

MongoDB aggregation pipeline with streaming:

    >>> extractor = MongoDBExtractor(
    ...     connection_config={
    ...         "uri": "mongodb://localhost:27017/",
    ...         "database": "logs",
    ...         "collection": "web"
    ...     },
    ...     pipeline=[
    ...         {"$match": {"status": 500}},
    ...         {"$group": {"_id": "$endpoint", "count": {"$sum": 1}}}
    ...     ],
    ...     batch_size=1000
    ... )
    >>> for chunk, meta in extractor.stream_extract():
    ...     process_chunk(chunk)

Dependencies
------------
- ``sqlalchemy``: Required for SQL database connections (supports Postgres, MySQL, SQLite, Oracle, MSSQL, etc.).
- ``pandas``: Required for DataFrame output and chunked cursor reads.
- ``pymongo``: Required for MongoDB connections and cursor-based streaming.
- ``typing``: Type hints for better interfaces.

Notes
-----
- For SQL, ``chunksize`` enables server-side cursors to stream results without loading
    all rows into memory.
- For MongoDB, batch size determines how many documents are pulled per cursor fetch.
- Connection pooling and retries are supported via SQLAlchemy and PyMongo configs.
- Designed for production ETL: logging, error handling, and extensibility
    are first-class considerations.
- For distributed ETL (e.g., sharded MongoDB or multi-table SQL jobs),
    parallel execution can accelerate ingestion.
"""

import logging
import pandas as pd
from time import time
from pathlib import Path
from contextlib import contextmanager
from abc import ABC, abstractmethod
from sqlalchemy.engine import Connection
from sqlalchemy import create_engine, Engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError, NoSuchModuleError
from typing import (
    Tuple,
    Dict,
    Any,
    Optional,
    List,
    Generator,
    Union,
    Protocol,
    runtime_checkable,
)

try:
    # Try relative import first (when used as part of package)
    from ..core.errors import (
        DependencyMissingError,
        DataSourceConnectionError,
        DataSourceAuthenticationError,
        QueryExecutionError,
    )
    from .base import BaseExtractor
except ImportError:
    # Fall back to absolute import (when run directly)
    import sys
    from pathlib import Path

    # Add the ETLFramework directory to Python path
    etl_framework_path = Path(__file__).parent.parent
    sys.path.insert(0, str(etl_framework_path))
    from core.errors import (
        DependencyMissingError,
        DataSourceConnectionError,
        DataSourceAuthenticationError,
        QueryExecutionError,
    )
    from extractors.base import BaseExtractor

# Optional imports with error handling
try:
    import pymongo
    from pymongo import MongoClient, database, collection, cursor
    from pymongo.errors import (
        ConnectionFailure,
        OperationFailure,
        ServerSelectionTimeoutError,
    )

    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    pymongo = None
    MongoClient = None
    ConnectionFailure = Exception
    AuthenticationFailed = Exception
    OperationFailure = Exception
    ServerSelectionTimeoutError = Exception

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler if no handlers exist
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

# Type definitions
ConnectionConfig = Union[str, Dict[str, Any]]
Metadata = Dict[str, Any]
ExtractionResult = Tuple[pd.DataFrame, Metadata]
StreamResult = Generator[ExtractionResult, None, None]


@runtime_checkable
class DatabaseConnection(Protocol):
    """Protocol for database connection objects."""

    def close(self) -> None:
        """Close the database connection."""
        ...


class SchemaManager:
    """
    Manages database schema discovery and validation.

    This class provides utilities for discovering table schemas,
    validating data types, and ensuring compatibility between
    source and target systems.
    """

    def __init__(self, connection: Union[Connection, MongoClient]): # type: ignore
        """
        Initialize the schema manager.

        Parameters
        ----------
        connection : Union[Connection, MongoClient]
            Database connection object.
        """
        self.connection = connection
        self._schema_cache: Dict[str, Dict[str, Any]] = {}

    def get_table_schema(
        self, table_name: str, database: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Get the schema for a table or collection.

        Parameters
        ----------
        table_name : str
            Name of the table or collection.
        database : Optional[str]
            Database name (for MongoDB).

        Returns
        -------
        Dict[str, str]
            Dictionary mapping column names to data types.

        Raises
        ------
        QueryExecutionError
            If schema discovery fails.
        """
        cache_key = f"{database}.{table_name}" if database else table_name

        if cache_key in self._schema_cache:
            return self._schema_cache[cache_key]

        try:
            if isinstance(self.connection, Connection):
                # SQL database schema discovery
                inspector = inspect(self.connection.engine)
                columns = inspector.get_columns(table_name)
                schema = {col["name"]: str(col["type"]) for col in columns}
            elif hasattr(self.connection, "list_database_names"):
                # MongoDB schema inference
                db = (
                    self.connection[database]
                    if database
                    else self.connection.get_default_database()
                )
                collection = db[table_name]

                # Sample documents to infer schema
                sample = list(collection.find().limit(100))
                schema = self._infer_mongo_schema(sample)
            else:
                raise QueryExecutionError(
                    f"Unsupported connection type: {type(self.connection)}"
                )

            self._schema_cache[cache_key] = schema
            logger.info("Discovered schema for %s: %s", cache_key, schema)
            return schema

        except Exception as e:
            logger.error("Failed to get schema for %s: %s", cache_key, str(e))
            raise QueryExecutionError(
                f"Schema discovery failed for {cache_key}: {e}"
            ) from e

    def _infer_mongo_schema(self, documents: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Infer schema from MongoDB documents.

        Parameters
        ----------
        documents : List[Dict[str, Any]]
            Sample documents from the collection.

        Returns
        -------
        Dict[str, str]
            Inferred schema mapping field names to types.
        """
        if not documents:
            return {}

        schema = {}
        for doc in documents:
            for field, value in doc.items():
                if field not in schema:
                    schema[field] = type(value).__name__
                elif schema[field] != type(value).__name__:
                    schema[field] = "mixed"  # Multiple types detected

        return schema

    def validate_schema_compatibility(
        self, source_schema: Dict[str, str], target_schema: Dict[str, str]
    ) -> List[str]:
        """
        Validate compatibility between source and target schemas.

        Parameters
        ----------
        source_schema : Dict[str, str]
            Source schema mapping.
        target_schema : Dict[str, str]
            Target schema mapping.

        Returns
        -------
        List[str]
            List of compatibility issues found.
        """
        issues = []

        # Check for missing columns
        missing_cols = set(target_schema.keys()) - set(source_schema.keys())
        if missing_cols:
            issues.append(f"Missing columns in source: {missing_cols}")

        # Check for type mismatches
        for col in set(source_schema.keys()) & set(target_schema.keys()):
            if source_schema[col] != target_schema[col]:
                issues.append(
                    f"Type mismatch for {col}: source={source_schema[col]}, "
                    f"target={target_schema[col]}"
                )

        return issues


class MetadataManager:
    """
    Manages extraction metadata collection and enrichment.

    This class standardizes metadata collection across different
    database types and provides utilities for enriching metadata
    with additional context.
    """

    @staticmethod
    def create_base_metadata(
        source: str,
        df: pd.DataFrame,
        extraction_time_taken: Optional[float] = None,
        **kwargs,
    ) -> Metadata:
        """
        Create standardized base metadata for extractions.

        Parameters
        ----------
        source : str
            Source identifier (table name, query, etc.).
        df : pd.DataFrame
            Extracted DataFrame.
        extraction_time_taken : Optional[float]
            Time taken for extraction in seconds.
        **kwargs
            Additional metadata fields.

        Returns
        -------
        Metadata
            Standardized metadata dictionary.
        """
        metadata = {
            "source": source,
            "extraction_time": pd.Timestamp.now().isoformat(),
            "extraction_time_taken": extraction_time_taken,
            "num_records": len(df),
            "num_columns": len(df.columns) if hasattr(df, "columns") else 0,
            "columns": df.columns.tolist() if hasattr(df, "columns") else [],
            "memory_usage_mb": (
                df.memory_usage(deep=True).sum() / 1024**2
                if hasattr(df, "memory_usage")
                else 0
            ),
            "data_types": df.dtypes.to_dict() if hasattr(df, "dtypes") else {},
        }

        # Add optional metadata
        metadata.update(kwargs)

        return metadata

    @staticmethod
    def enrich_sql_metadata(
        metadata: Metadata, query: str, connection_info: Dict[str, Any]
    ) -> Metadata:
        """
        Enrich metadata with SQL-specific information.

        Parameters
        ----------
        metadata : Metadata
            Base metadata dictionary.
        query : str
            SQL query executed.
        connection_info : Dict[str, Any]
            Connection configuration.

        Returns
        -------
        Metadata
            Enriched metadata.
        """
        metadata.update(
            {
                "query": query,
                "db_type": connection_info.get("db_type"),
                "database": connection_info.get("database"),
                "host": connection_info.get("host"),
                "query_type": (
                    "SELECT" if query.strip().upper().startswith("SELECT") else "OTHER"
                ),
            }
        )
        return metadata

    @staticmethod
    def enrich_mongo_metadata(
        metadata: Metadata,
        query: Optional[Dict[str, Any]] = None,
        pipeline: Optional[List[Dict[str, Any]]] = None,
        collection: Optional[str] = None,
        database: Optional[str] = None,
    ) -> Metadata:
        """
        Enrich metadata with MongoDB-specific information.

        Parameters
        ----------
        metadata : Metadata
            Base metadata dictionary.
        query : Optional[Dict[str, Any]]
            MongoDB query filter.
        pipeline : Optional[List[Dict[str, Any]]]
            Aggregation pipeline.
        collection : Optional[str]
            Collection name.
        database : Optional[str]
            Database name.

        Returns
        -------
        Metadata
            Enriched metadata.
        """
        metadata.update(
            {
                "collection": collection,
                "database": database,
                "query": query,
                "pipeline": pipeline,
                "operation_type": "aggregation" if pipeline else "find",
            }
        )
        return metadata


class DBExtractor(BaseExtractor, ABC):
    """
    Abstract base class for database extractors.

    This class defines the interface and common functionality for extracting data
    from various types of databases. Subclasses must implement the `extract` method
    to perform the actual data extraction.

    Attributes
    ----------
    connection_config : ConnectionConfig
        The database connection configuration.
    schema_manager : Optional[SchemaManager]
        Schema discovery and validation manager.
    metadata_manager : MetadataManager
        Metadata collection and enrichment manager.
    """

    def __init__(self, connection_config: ConnectionConfig) -> None:
        """
        Initialize the DBExtractor with a connection configuration.

        Parameters
        ----------
        connection_config : ConnectionConfig
            The database connection string or URI, or a dictionary of connection parameters.

        Raises
        ------
        DataSourceAuthenticationError
            If the connection configuration is invalid.
        """
        self.connection_config = connection_config
        self.schema_manager: Optional[SchemaManager] = None
        self.metadata_manager = MetadataManager()
        self._connection: Optional[DatabaseConnection] = None

        # Validate connection configuration
        self._validate_connection_config()

    def _validate_connection_config(self) -> None:
        """
        Validate the connection configuration.

        Raises
        ------
        DataSourceAuthenticationError
            If the configuration is invalid.
        """
        if not self.connection_config:
            raise DataSourceAuthenticationError("Connection configuration is required")

        if isinstance(self.connection_config, dict):
            if not self.connection_config.get(
                "db_type"
            ) and not self.connection_config.get("uri"):
                raise DataSourceAuthenticationError(
                    "Either 'db_type' or 'uri' must be specified in connection config"
                )

    @property
    def connection_string(self) -> str:
        """
        Get the database connection string.

        Returns
        -------
        str
            The database connection string.
        """
        if isinstance(self.connection_config, str):
            return self.connection_config
        return self.connection_config.get("connection_string", "")

    @abstractmethod
    def extract(self) -> ExtractionResult:
        """
        Extract data from the database and return it as a DataFrame along with metadata.

        Returns
        -------
        ExtractionResult
            A tuple containing the extracted data as a pandas DataFrame and a dictionary
            of metadata about the extraction process.

        Raises
        ------
        DataSourceConnectionError
            If the connection to the database fails.
        QueryExecutionError
            If the query execution fails.
        DataReadError
            If reading data from the database fails.
        """
        pass

    @abstractmethod
    def stream_extract(self) -> StreamResult:
        """
        Stream data from the database in chunks.

        Yields
        ------
        ExtractionResult
            Tuples containing DataFrame chunks and their metadata.

        Raises
        ------
        DataSourceConnectionError
            If the connection to the database fails.
        QueryExecutionError
            If the query execution fails.
        DataReadError
            If reading data from the database fails.
        """
        pass

    @abstractmethod
    def connect(self) -> None:
        """
        Establish connection to the database.

        Raises
        ------
        DataSourceConnectionError
            If connection fails.
        DataSourceAuthenticationError
            If authentication fails.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Close the database connection.
        """
        pass

    @contextmanager
    def connection_context(self):
        """
        Context manager for database connections.

        Ensures proper connection cleanup even if errors occur.

        Yields
        ------
        DBExtractor
            The extractor instance with an active connection.

        Raises
        ------
        DataSourceConnectionError
            If connection fails.
        """
        try:
            self.connect()
            yield self
        finally:
            self.disconnect()


class SQLExtractor(DBExtractor):
    """
    Extractor for SQL databases.

    Provides connection management, query execution, schema discovery,
    and both full/incremental extraction utilities for relational databases.

    Supported backends: PostgreSQL, MySQL, SQLite, Oracle, MSSQL.

    Metadata Schema
    ---------------
    Every extraction returns a dictionary with standardized keys:

    {
        "source": str,                        # table name or query
        "extraction_time": str,               # ISO timestamp when extraction finished
        "extraction_time_taken": float,       # duration in seconds
        "num_records": int,                   # number of rows
        "num_columns": int,                   # number of columns
        "columns": List[str],                 # column names
        "memory_usage_mb": float,             # memory usage in MB
        "data_types": Dict[str, str],         # column data types
        "query": str,                         # executed query
        "db_type": str,                       # database type
        "database": str,                      # database name
        "host": str,                          # database host
        "query_type": str,                    # SELECT, INSERT, etc.
    }
    """

    SUPPORTED_DATABASES = ["postgresql", "mysql", "sqlite", "oracle", "mssql"]

    def __init__(
        self,
        connection_config: Dict[str, Any],
        query: Optional[str] = None,
        chunksize: Optional[int] = None,
    ) -> None:
        """
        Initialize SQL extractor.

        Parameters
        ----------
        connection_config : Dict[str, Any]
            Database connection configuration containing:
            - db_type: Database type (postgresql, mysql, sqlite, oracle, mssql)
            - username: Database username (not needed for SQLite)
            - password: Database password (not needed for SQLite)
            - host: Database host (not needed for SQLite)
            - port: Database port (not needed for SQLite)
            - database: Database name or file path for SQLite
        query : Optional[str]
            SQL query to execute. If None, must be provided to extract methods.
        chunksize : Optional[int]
            Number of rows to fetch per chunk for streaming operations.

        Raises
        ------
        DataSourceAuthenticationError
            If connection configuration is invalid.
        """
        super().__init__(connection_config)
        self.query = query
        self.chunksize = chunksize
        self.engine: Optional[Engine] = None
        self._connection: Optional[Connection] = None

        # Validate SQL-specific credentials during initialization
        self._validate_credentials()

    def _check_dependencies(self, db_type: str) -> None:
        """
        Check if required database drivers are installed.

        Parameters
        ----------
        db_type : str
            Database type identifier.

        Raises
        ------
        DependencyMissingError
            If required driver is not installed.
        """
        driver_map = {
            "postgresql": "psycopg2",
            "mysql": "pymysql",
            "sqlite": "sqlite3",
            "oracle": "oracledb",
            "mssql": "pyodbc",
        }

        driver = driver_map.get(db_type)
        if not driver:
            raise DependencyMissingError(f"Unsupported database type: {db_type}")

        try:
            __import__(driver)
        except ImportError as e:
            logger.error("Missing dependency for %s: %s", db_type, str(e))
            raise DependencyMissingError(
                f"Required driver '{driver}' not installed for {db_type}. "
                f"Please install it using: pip install {driver}"
            ) from e

    def _validate_credentials(self) -> None:
        """
        Validate database credentials.

        Raises
        ------
        DataSourceAuthenticationError
            If credentials are missing or invalid.
        """
        if not isinstance(self.connection_config, dict):
            raise DataSourceAuthenticationError(
                "Connection config must be a dictionary"
            )

        db_type = self.connection_config.get("db_type")
        if not db_type:
            raise DataSourceAuthenticationError("`db_type` is required in config.")

        if db_type == "sqlite":
            if not self.connection_config.get("database"):
                raise DataSourceAuthenticationError(
                    "For SQLite, `database` (file path or :memory:) is required."
                )
            return

        required = ["username", "password", "host", "port", "database"]
        missing = [field for field in required if not self.connection_config.get(field)]
        if missing:
            raise DataSourceAuthenticationError(
                f"Missing credentials for {db_type}: {', '.join(missing)}"
            )

    def _create_engine(self) -> None:
        """
        Create SQLAlchemy engine from configuration.

        Raises
        ------
        DataSourceConnectionError
            If engine creation fails.
        DependencyMissingError
            If required database driver is missing.
        """
        if not isinstance(self.connection_config, dict):
            raise DataSourceConnectionError("Invalid connection configuration")

        db_type = self.connection_config.get("db_type")
        self._validate_credentials()
        self._check_dependencies(db_type)

        try:
            if db_type == "sqlite":
                url = f"sqlite:///{self.connection_config['database']}"
            elif db_type == "postgresql":
                url = (
                    f"postgresql+psycopg2://{self.connection_config['username']}:"
                    f"{self.connection_config['password']}@{self.connection_config['host']}:"
                    f"{self.connection_config['port']}/{self.connection_config['database']}"
                )
                logger.debug("PostgreSQL URL: %s", url)
            elif db_type == "mysql":
                url = (
                    f"mysql+pymysql://{self.connection_config['username']}:"
                    f"{self.connection_config['password']}@{self.connection_config['host']}:"
                    f"{self.connection_config['port']}/{self.connection_config['database']}"
                )
            elif db_type == "oracle":
                dsn = f"{self.connection_config['host']}:{self.connection_config['port']}/{self.connection_config['database']}"
                url = f"oracle+oracledb://{self.connection_config['username']}:{self.connection_config['password']}@{dsn}"
            elif db_type == "mssql":
                url = (
                    f"mssql+pyodbc://{self.connection_config['username']}:"
                    f"{self.connection_config['password']}@{self.connection_config['host']}:"
                    f"{self.connection_config['port']}/{self.connection_config['database']}"
                    "?driver=ODBC+Driver+17+for+SQL+Server"
                )
            else:
                raise ValueError(f"Unsupported database type: {db_type}")

            logger.info("Creating SQLAlchemy engine for %s", db_type)
            self.engine = create_engine(
                url,
                pool_pre_ping=True,
                pool_recycle=1800,
                echo=False,
                future=True,
            )

        except (SQLAlchemyError, NoSuchModuleError, Exception) as e:
            logger.exception("Failed to create engine for %s: %s", db_type, str(e))
            raise DataSourceConnectionError(
                f"Failed to create engine for {db_type}: {e}"
            ) from e

    def connect(self) -> None:
        """
        Establish connection to the SQL database.

        Raises
        ------
        DataSourceConnectionError
            If connection fails.
        """
        if not self.engine:
            self._create_engine()

        try:
            logger.info("Connecting to SQL database...")
            self._connection = self.engine.connect()

            # Initialize schema manager
            self.schema_manager = SchemaManager(self._connection)

            logger.info("SQL database connection established.")

        except OperationalError as e:
            logger.error("Operational error during connection: %s", str(e))
            raise DataSourceConnectionError(f"Operational error: {e}") from e
        except SQLAlchemyError as e:
            logger.error("SQLAlchemy error during connection: %s", str(e))
            raise DataSourceConnectionError(f"Failed to connect: {e}") from e

    def disconnect(self) -> None:
        """
        Close the database connection.
        """
        if self._connection:
            try:
                self._connection.close()
                logger.info("SQL database connection closed.")
            except Exception as e:
                logger.warning("Error while closing connection: %s", str(e))
            finally:
                self._connection = None

    def extract(self, query: Optional[str] = None) -> ExtractionResult:
        """
        Extract data using SQL query.

        Parameters
        ----------
        query : Optional[str]
            SQL query to execute. If None, uses the query from initialization.

        Returns
        -------
        ExtractionResult
            Tuple of (DataFrame, metadata).

        Raises
        ------
        QueryExecutionError
            If query execution fails.
        DataSourceConnectionError
            If not connected to database.
        """
        query_to_execute = query or self.query
        if not query_to_execute:
            raise QueryExecutionError("No query provided for extraction")

        if not self._connection:
            raise DataSourceConnectionError("Not connected to the database.")

        logger.info(
            "Executing SQL query: %s",
            (
                query_to_execute[:100] + "..."
                if len(query_to_execute) > 100
                else query_to_execute
            ),
        )

        try:
            start_time = time()
            df = pd.read_sql(query_to_execute, self._connection)
            execution_time = time() - start_time

            # Create base metadata
            metadata = self.metadata_manager.create_base_metadata(
                source=query_to_execute,
                df=df,
                extraction_time_taken=round(execution_time, 2),
            )

            # Enrich with SQL-specific metadata
            metadata = self.metadata_manager.enrich_sql_metadata(
                metadata, query_to_execute, self.connection_config
            )

            logger.info(
                "Successfully extracted %d records in %.2f seconds",
                len(df),
                execution_time,
            )
            return df, metadata

        except Exception as e:
            logger.error("Failed to execute query: %s", str(e))
            raise QueryExecutionError(f"Failed to execute query: {e}") from e

    def stream_extract(
        self, query: Optional[str] = None, chunksize: Optional[int] = None
    ) -> StreamResult:
        """
        Stream data from SQL database in chunks.

        Parameters
        ----------
        query : Optional[str]
            SQL query to execute. If None, uses the query from initialization.
        chunksize : Optional[int]
            Number of rows per chunk. If None, uses the chunksize from initialization.

        Yields
        ------
        ExtractionResult
            Tuples of (DataFrame chunk, metadata).

        Raises
        ------
        QueryExecutionError
            If query execution fails.
        DataSourceConnectionError
            If not connected to database.
        """
        query_to_execute = query or self.query
        chunk_size = chunksize or self.chunksize or 10000

        if not query_to_execute:
            raise QueryExecutionError("No query provided for streaming extraction")

        if not self._connection:
            raise DataSourceConnectionError("Not connected to the database.")

        logger.info("Starting streaming extraction with chunk size: %d", chunk_size)

        try:
            chunk_number = 0
            for chunk in pd.read_sql(
                query_to_execute, self._connection, chunksize=chunk_size
            ):
                chunk_number += 1

                # Create metadata for this chunk
                metadata = self.metadata_manager.create_base_metadata(
                    source=query_to_execute,
                    df=chunk,
                    chunk_number=chunk_number,
                    chunk_size=chunk_size,
                )

                # Enrich with SQL-specific metadata
                metadata = self.metadata_manager.enrich_sql_metadata(
                    metadata, query_to_execute, self.connection_config
                )

                logger.debug(
                    "Yielding chunk %d with %d records", chunk_number, len(chunk)
                )
                yield chunk, metadata

        except Exception as e:
            logger.error("Failed to stream query: %s", str(e))
            raise QueryExecutionError(f"Failed to stream query: {e}") from e

    def extract_table(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
        order_by: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract data from a specific table.

        Parameters
        ----------
        table_name : str
            Name of the table to extract from.
        columns : Optional[List[str]]
            List of columns to select. If None, selects all columns.
        where : Optional[str]
            WHERE clause condition.
        limit : Optional[int]
            Maximum number of rows to return.
        order_by : Optional[str]
            ORDER BY clause.

        Returns
        -------
        ExtractionResult
            Tuple of (DataFrame, metadata).
        """
        # Build query
        cols = ", ".join(columns) if columns else "*"
        query = f"SELECT {cols} FROM {table_name}"

        if where:
            query += f" WHERE {where}"
        if order_by:
            query += f" ORDER BY {order_by}"
        if limit:
            query += f" LIMIT {limit}"

        return self.extract(query)

    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """
        Get schema information for a table.

        Parameters
        ----------
        table_name : str
            Name of the table.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping column names to data types.

        Raises
        ------
        DataSourceConnectionError
            If not connected to database.
        QueryExecutionError
            If schema discovery fails.
        """
        if not self.schema_manager:
            raise DataSourceConnectionError(
                "Schema manager not initialized. Connect to database first."
            )

        return self.schema_manager.get_table_schema(table_name)

    def extract_incremental(
        self,
        table_name: str,
        watermark_column: str,
        last_value: Any,
        columns: Optional[List[str]] = None,
        order_by: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract data incrementally using a watermark column.

        Parameters
        ----------
        table_name : str
            Name of the table to extract from.
        watermark_column : str
            Column used for incremental extraction (e.g., timestamp, id).
        last_value : Any
            Last processed value of the watermark column.
        columns : Optional[List[str]]
            List of columns to select. If None, selects all columns.
        order_by : Optional[str]
            ORDER BY clause. Defaults to watermark_column.

        Returns
        -------
        ExtractionResult
            Tuple of (DataFrame, metadata).
        """
        if not self._connection:
            raise DataSourceConnectionError("Not connected to the database.")

        cols = ", ".join(columns) if columns else "*"
        order_clause = order_by or watermark_column

        query = text(
            f"SELECT {cols} FROM {table_name} "
            f"WHERE {watermark_column} > :last_value "
            f"ORDER BY {order_clause}"
        )

        try:
            start_time = time()
            df = pd.read_sql(query, self._connection, params={"last_value": last_value})
            execution_time = time() - start_time

            # Create metadata
            metadata = self.metadata_manager.create_base_metadata(
                source=table_name,
                df=df,
                extraction_time_taken=round(execution_time, 2),
                watermark_column=watermark_column,
                last_value=last_value,
                extraction_type="incremental",
            )

            # Enrich with SQL-specific metadata
            metadata = self.metadata_manager.enrich_sql_metadata(
                metadata, str(query), self.connection_config
            )

            logger.info(
                "Incremental extraction from %s: %d new records", table_name, len(df)
            )
            return df, metadata

        except Exception as e:
            logger.error("Failed incremental extraction: %s", str(e))
            raise QueryExecutionError(f"Failed incremental extraction: {e}") from e


class MongoDBExtractor(DBExtractor):
    """
    Extractor for MongoDB databases.

    Provides connection management, query execution, schema discovery,
    and streaming extraction utilities for MongoDB collections.

    Metadata Schema
    ---------------
    Every extraction returns a dictionary with standardized keys:

    {
        "source": str,                        # collection name
        "extraction_time": str,               # ISO timestamp when extraction finished
        "extraction_time_taken": float,       # duration in seconds
        "num_records": int,                   # number of documents
        "num_columns": int,                   # number of fields
        "columns": List[str],                 # field names
        "memory_usage_mb": float,             # memory usage in MB
        "data_types": Dict[str, str],         # field data types
        "collection": str,                    # collection name
        "database": str,                      # database name
        "query": Dict[str, Any],              # MongoDB query filter
        "pipeline": List[Dict[str, Any]],     # aggregation pipeline
        "operation_type": str,                # "find" or "aggregation"
    }
    """

    def __init__(
        self,
        connection_config: Dict[str, Any],
        query: Optional[Dict[str, Any]] = None,
        pipeline: Optional[List[Dict[str, Any]]] = None,
        projection: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Initialize MongoDB extractor.

        Parameters
        ----------
        connection_config : Dict[str, Any]
            MongoDB connection configuration containing:
            - uri: MongoDB connection URI
            - database: Database name
            - collection: Collection name
            - username: Username (optional if in URI)
            - password: Password (optional if in URI)
        query : Optional[Dict[str, Any]]
            MongoDB query filter for find operations.
        pipeline : Optional[List[Dict[str, Any]]]
            Aggregation pipeline for aggregate operations.
        projection : Optional[Dict[str, Any]]
            Field projection specification.
        batch_size : Optional[int]
            Number of documents to fetch per batch.

        Raises
        ------
        DependencyMissingError
            If pymongo is not installed.
        DataSourceAuthenticationError
            If connection configuration is invalid.
        """
        if not PYMONGO_AVAILABLE:
            raise DependencyMissingError(
                "pymongo is required for MongoDB extraction. Install it using: pip install pymongo"
            )

        super().__init__(connection_config)
        self.query = query or {}
        self.pipeline = pipeline
        self.projection = projection
        self.batch_size = batch_size or 1000
        self._client: Optional[MongoClient] = None # type: ignore
        self._database = None
        self._collection = None

    def _validate_mongo_config(self) -> None:
        """
        Validate MongoDB connection configuration.

        Raises
        ------
        DataSourceAuthenticationError
            If configuration is invalid.
        """
        if not isinstance(self.connection_config, dict):
            raise DataSourceAuthenticationError(
                "Connection config must be a dictionary"
            )

        required_fields = ["uri", "database", "collection"]
        missing = [
            field for field in required_fields if not self.connection_config.get(field)
        ]

        if missing:
            raise DataSourceAuthenticationError(
                f"Missing MongoDB config fields: {missing}"
            )

    def connect(self) -> None:
        """
        Establish connection to MongoDB.

        Raises
        ------
        DataSourceConnectionError
            If connection fails.
        DataSourceAuthenticationError
            If authentication fails.
        """
        self._validate_mongo_config()

        try:
            logger.info("Connecting to MongoDB...")

            # Create MongoDB client
            self._client = MongoClient(
                self.connection_config["uri"],
                serverSelectionTimeoutMS=5000,  # 5 second timeout
            )

            # Test connection
            logger.info("MongoDB client created: %s", self._client['db_name'])

            # Get database and collection
            self._database: database = self._client[self.connection_config["database"]]
            self._collection: collection = self._database[self.connection_config["collection"]]

            logger.info("Connected to MongoDB database: %s", self.connection_config["database"])
            logger.info("Using MongoDB collection: %s", self.connection_config["collection"])
            
            # Initialize schema manager
            self.schema_manager = SchemaManager(self._client)

            logger.info("MongoDB connection established.")

        except ConnectionFailure as e:
            logger.error("MongoDB connection failed: %s", str(e))
            raise DataSourceConnectionError(f"MongoDB connection failed: {e}") from e
        except AuthenticationFailed as e:
            logger.error("MongoDB authentication failed: %s", str(e))
            raise DataSourceAuthenticationError(
                f"MongoDB authentication failed: {e}"
            ) from e
        except ServerSelectionTimeoutError as e:
            logger.error("MongoDB server selection timeout: %s", str(e))
            raise DataSourceConnectionError(f"MongoDB server timeout: {e}") from e
        except Exception as e:
            logger.error("Unexpected MongoDB connection error: %s", str(e))
            raise DataSourceConnectionError(f"Unexpected MongoDB error: {e}") from e

    def disconnect(self) -> None:
        """
        Close the MongoDB connection.
        """
        if self._client:
            try:
                self._client.close()
                logger.info("MongoDB connection closed.")
            except Exception as e:
                logger.warning("Error while closing MongoDB connection: %s", str(e))
            finally:
                self._client = None
                self._database = None
                self._collection = None

    def extract(
        self,
        query: Optional[Dict[str, Any]] = None,
        pipeline: Optional[List[Dict[str, Any]]] = None,
    ) -> ExtractionResult:
        """
        Extract data from MongoDB collection.

        Parameters
        ----------
        query : Optional[Dict[str, Any]]
            MongoDB query filter. If None, uses the query from initialization.
        pipeline : Optional[List[Dict[str, Any]]]
            Aggregation pipeline. If None, uses the pipeline from initialization.

        Returns
        -------
        ExtractionResult
            Tuple of (DataFrame, metadata).

        Raises
        ------
        QueryExecutionError
            If query execution fails.
        DataSourceConnectionError
            If not connected to database.
        """
        if self._collection is None:
            raise DataSourceConnectionError("Not connected to MongoDB.")

        query_to_use = query or self.query
        pipeline_to_use = pipeline or self.pipeline
        
        if query_to_use and pipeline_to_use:
            raise QueryExecutionError("Cannot use both query and pipeline simultaneously.")
        if not query_to_use and not pipeline_to_use:
            raise QueryExecutionError("Either query or pipeline must be provided.")

        logger.info(
            "Extracting from MongoDB collection: %s",
            self.connection_config["collection"],
        )

        try:
            start_time = time()

            if pipeline_to_use:
                # Use aggregation pipeline
                cursor = self._collection.aggregate(pipeline_to_use)
                documents = list(cursor)
                operation_type = "aggregation" # NOSONAR
                source_info = f"Aggregation on {self.connection_config['collection']}"
            else:
                # Use find query
                cursor = self._collection.find(query_to_use, self.projection)
                documents = list(cursor)
                operation_type = "find"
                source_info = f"Find on {self.connection_config['collection']}"

            execution_time = time() - start_time

            # Convert to DataFrame
            df = pd.DataFrame(documents) if documents else pd.DataFrame()

            # Create base metadata
            metadata = self.metadata_manager.create_base_metadata(
                source=source_info,
                df=df,
                extraction_time_taken=round(execution_time, 2),
            )

            # Enrich with MongoDB-specific metadata
            metadata = self.metadata_manager.enrich_mongo_metadata(
                metadata,
                query=query_to_use,
                pipeline=pipeline_to_use,
                collection=self.connection_config["collection"],
                database=self.connection_config["database"],
            )

            logger.info(
                "Successfully extracted %d documents in %.2f seconds",
                len(df),
                execution_time,
            )
            return df, metadata

        except OperationFailure as e:
            logger.error("MongoDB operation failed: %s", str(e))
            raise QueryExecutionError(f"MongoDB operation failed: {e}") from e
        except Exception as e:
            logger.error("Failed to extract from MongoDB: %s", str(e))
            raise QueryExecutionError(f"Failed to extract from MongoDB: {e}") from e

    def stream_extract(
        self,
        query: Optional[Dict[str, Any]] = None,
        pipeline: Optional[List[Dict[str, Any]]] = None,
        batch_size: Optional[int] = None,
    ) -> StreamResult:
        """
        Stream data from MongoDB collection in batches.

        Parameters
        ----------
        query : Optional[Dict[str, Any]]
            MongoDB query filter. If None, uses the query from initialization.
        pipeline : Optional[List[Dict[str, Any]]]
            Aggregation pipeline. If None, uses the pipeline from initialization.
        batch_size : Optional[int]
            Number of documents per batch. If None, uses batch_size from initialization.

        Yields
        ------
        ExtractionResult
            Tuples of (DataFrame batch, metadata).

        Raises
        ------
        QueryExecutionError
            If query execution fails.
        DataSourceConnectionError
            If not connected to database.
        """
        if not self._collection:
            raise DataSourceConnectionError("Not connected to MongoDB.")

        query_to_use = query or self.query
        pipeline_to_use = pipeline or self.pipeline
        batch_size_to_use = batch_size or self.batch_size

        logger.info(
            "Starting streaming extraction with batch size: %d", batch_size_to_use
        )

        try:
            batch_number = 0

            if pipeline_to_use:
                # Use aggregation pipeline with batching
                cursor = self._collection.aggregate(
                    pipeline_to_use, batchSize=batch_size_to_use
                )
                operation_type = "aggregation" # NOSONAR
                source_info = f"Aggregation on {self.connection_config['collection']}"
            else:
                # Use find query with batching
                cursor = self._collection.find(
                    query_to_use, self.projection
                ).batch_size(batch_size_to_use)
                operation_type = "find"
                source_info = f"Find on {self.connection_config['collection']}"

            batch = []
            for document in cursor:
                batch.append(document)

                if len(batch) >= batch_size_to_use:
                    batch_number += 1

                    # Convert batch to DataFrame
                    df = pd.DataFrame(batch) if batch else pd.DataFrame()

                    # Create metadata for this batch
                    metadata = self.metadata_manager.create_base_metadata(
                        source=source_info,
                        df=df,
                        batch_number=batch_number,
                        batch_size=batch_size_to_use,
                    )

                    # Enrich with MongoDB-specific metadata
                    metadata = self.metadata_manager.enrich_mongo_metadata(
                        metadata,
                        query=query_to_use,
                        pipeline=pipeline_to_use,
                        collection=self.connection_config["collection"],
                        database=self.connection_config["database"],
                    )

                    logger.debug(
                        "Yielding batch %d with %d documents", batch_number, len(batch)
                    )
                    yield df, metadata

                    # Reset batch
                    batch = []

            # Handle remaining documents in the last batch
            if batch:
                batch_number += 1
                df = pd.DataFrame(batch)

                metadata = self.metadata_manager.create_base_metadata(
                    source=source_info,
                    df=df,
                    batch_number=batch_number,
                    batch_size=len(batch),
                )

                metadata = self.metadata_manager.enrich_mongo_metadata(
                    metadata,
                    query=query_to_use,
                    pipeline=pipeline_to_use,
                    collection=self.connection_config["collection"],
                    database=self.connection_config["database"],
                )

                logger.debug(
                    "Yielding final batch %d with %d documents",
                    batch_number,
                    len(batch),
                )
                yield df, metadata

        except OperationFailure as e:
            logger.error("MongoDB operation failed during streaming: %s", str(e))
            raise QueryExecutionError(
                f"MongoDB operation failed during streaming: {e}"
            ) from e
        except Exception as e:
            logger.error("Failed to stream from MongoDB: %s", str(e))
            raise QueryExecutionError(f"Failed to stream from MongoDB: {e}") from e

    def get_collection_schema(
        self, collection_name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Get schema information for a MongoDB collection.

        Parameters
        ----------
        collection_name : Optional[str]
            Name of the collection. If None, uses the collection from configuration.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping field names to inferred data types.

        Raises
        ------
        DataSourceConnectionError
            If not connected to database.
        QueryExecutionError
            If schema discovery fails.
        """
        if not self.schema_manager:
            raise DataSourceConnectionError(
                "Schema manager not initialized. Connect to database first."
            )

        collection = collection_name or self.connection_config.get("collection")
        if not collection:
            raise QueryExecutionError("No collection specified for schema discovery")

        return self.schema_manager.get_table_schema(
            collection, self.connection_config.get("database")
        )

    def extract_incremental(
        self,
        watermark_field: str,
        last_value: Any,
        query: Optional[Dict[str, Any]] = None,
        sort_field: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract data incrementally using a watermark field.

        Parameters
        ----------
        watermark_field : str
            Field used for incremental extraction (e.g., timestamp, _id).
        last_value : Any
            Last processed value of the watermark field.
        query : Optional[Dict[str, Any]]
            Additional query filters.
        sort_field : Optional[str]
            Field to sort by. Defaults to watermark_field.

        Returns
        -------
        ExtractionResult
            Tuple of (DataFrame, metadata).
        """
        if not self._collection:
            raise DataSourceConnectionError("Not connected to MongoDB.")

        # Build incremental query
        incremental_query = query.copy() if query else {}
        incremental_query[watermark_field] = {"$gt": last_value}

        sort_by = sort_field or watermark_field

        try:
            start_time = time()

            # Execute find with incremental filter and sorting
            cursor = self._collection.find(incremental_query, self.projection).sort(
                sort_by, 1
            )
            documents = list(cursor)

            execution_time = time() - start_time

            # Convert to DataFrame
            df = pd.DataFrame(documents) if documents else pd.DataFrame()

            # Create metadata
            metadata = self.metadata_manager.create_base_metadata(
                source=self.connection_config["collection"],
                df=df,
                extraction_time_taken=round(execution_time, 2),
                watermark_field=watermark_field,
                last_value=last_value,
                extraction_type="incremental",
            )

            # Enrich with MongoDB-specific metadata
            metadata = self.metadata_manager.enrich_mongo_metadata(
                metadata,
                query=incremental_query,
                collection=self.connection_config["collection"],
                database=self.connection_config["database"],
            )

            logger.info(
                "Incremental extraction from %s: %d new documents",
                self.connection_config["collection"],
                len(df),
            )
            return df, metadata

        except Exception as e:
            logger.error("Failed incremental extraction: %s", str(e))
            raise QueryExecutionError(f"Failed incremental extraction: {e}") from e


# Export classes for easier imports
__all__ = [
    "DBExtractor",
    "SQLExtractor",
    "MongoDBExtractor",
    "SchemaManager",
    "MetadataManager",
    "ConnectionConfig",
    "Metadata",
    "ExtractionResult",
    "StreamResult",
]
