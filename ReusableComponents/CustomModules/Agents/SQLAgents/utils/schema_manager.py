"""
schema_manager.py
-----------------
This module contains the SchemaManager class which is used to manage the schema of a database.

Description:
------------
The SchemaManager class is responsible for defining the schema of a database and creating a table if it does not exist.

Requirements:
-------------

Author:
-------
Sourav Das

Version:
-------
1.0

Date:
-----
07.02.2025
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Importing necessary libraries and modules
import logging
from dotenv import load_dotenv

from typing import Dict

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Load environment variables
load_dotenv(override=True)

# Constant
INVALID_DIALECT_ERROR = "Invalid dialect provided. Please provide a valid dialect."

class SchemaManager:
    """
    SchemaManager class is responsible for defining the schema of a database and creating a table if it does not exist.
    """
    
    def __init__(self, config: Dict) -> None:
        """
        Constructor method to initialize the SchemaManager class.
        
        Parameters:
        -----------
        config: Dict
            A dictionary containing the configuration details of the database.
        """
        self.config = config
        
    def _validate_schema(self) -> None:
        """
        Validate the schema of the database.
        """
        dialect = self.config.get("dialect")
        if dialect == "mssql":
            self._validate_mssql_schema()
        elif dialect == "postgresql":
            self._validate_postgresql_schema()
        else:
            _logger.error(INVALID_DIALECT_ERROR)
            raise ValueError(INVALID_DIALECT_ERROR)
        
    def _validate_mssql_schema(self) -> None:
        """
        Validate the schema of the database for MSSQL.
        """
        for key in self.config.keys():
            if key not in ["server", "database", "user", "password"]:
                _logger.error(f"Invalid key: {key}")
                raise ValueError(f"Invalid key: {key}")
    
    def _validate_postgresql_schema(self) -> None:
        """
        Validate the schema of the database for PostgreSQL.
        """
        for key in self.config.keys():
            if key not in ["host", "port", "user", "password", "database"]:
                _logger.error(f"Invalid key: {key}")
                raise ValueError(f"Invalid key: {key}")
            
    def _get_db_client(self) -> None:
        """
        Get the database client based on the dialect.
        """
        dialect = self.config.get("dialect")
        if dialect == "mssql":
            return self._get_mssql_db_client()
        elif dialect == "postgresql":
            return self._get_postgresql_db_client()
        else:
            _logger.error(INVALID_DIALECT_ERROR)

    def _get_mssql_db_client(self) -> None:
        """
        Get the MSSQL database client.
        """
        
        import pyodbc
        
        local = self.config.get("local")
        if local:
            connection_string = (
                "DRIVER={ODBC Driver 17 for SQL Server};"
                f"SERVER={self.config.get('server')};"
                f"DATABASE={self.config.get('database')};"
                "trusted_connection=yes;"
            )
        else:
            connection_string = (
                "DRIVER={ODBC Driver 17 for SQL Server};"
                f"SERVER={self.config.get('server')};"
                f"DATABASE={self.config.get('database')};"
                f"UID={self.config.get('user')};"
                f"PWD={self.config.get('password')};"
                "TrustServerCertificate=no;"
                "Encrypt=yes;"
                "ConnectionTimeout=30;"
            )
            
        self.conn = pyodbc.connect(connection_string)
        
    def _get_postgresql_db_client(self) -> None:
        """
        Get the PostgreSQL database client.
        """
        
        import psycopg2
        
        connection_string = (
            f"host={self.config.get('host')} "
            f"port={self.config.get('port')} "
            f"user={self.config.get('user')} "
            f"password={self.config.get('password')} "
            f"dbname={self.config.get('database')}"
        )
        
        self.conn = psycopg2.connect(connection_string)
        
    def _get_schema(self) -> None:
        """
        Get the schema of the database.
        """
        