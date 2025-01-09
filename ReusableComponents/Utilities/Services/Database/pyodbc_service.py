"""
pyodbc_service.py
-----------------
This module contains the PyODBCService class, which is a service class that provides methods for interacting with a SQL Server database using the pyodbc library.

Classes:
--------
PyODBCService: A service class that provides methods for interacting with a SQL Server database using the pyodbc library.

Dependencies:
-------------

Requirements:
-------------
- python-dotenv==1.0.1
- pyodbc==5.1.0

Environment Variables:
----------------------

TODO:
-----

FIXME:
------

Author:
-------
Sourav Das

Date:
-----
09-01-2025
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Import dependencies
import os
import pyodbc
import logging
from pyodbc import Connection
from dotenv import load_dotenv

from typing import Any, Optional, Dict, List, Tuple

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Load Environment variables
load_dotenv(override=True)

class SQLDatabase:
    """
    A class to interact with an SQL database.

    Attributes:
        connection_string (str): The connection string for the database.
    """

    def __init__(
        self,
        server: str = None,
        database: str = None,
        user: str = None,
        password: str = None,
        port: int = None,
        local: bool = True,
    ) -> None:
        """
        Initializes the SQLDatabase object with the connection parameters retrieved from arguments or environment variables.

        Args:
            server (str, optional): The SQL server address. Defaults to the value of the 'AZURE_SQL_SERVER' environment variable if not provided.
            database (str, optional): The name of the database. Defaults to the value of the 'AZURE_SQL_DATABASE' environment variable if not provided.
            user (str, optional): The username for database authentication. Defaults to the value of the 'AZURE_SQL_USER' environment variable if not provided.
            password (str, optional): The password for database authentication. Defaults to the value of the 'AZURE_SQL_PASSWORD' environment variable if not provided.
            port (int, optional): The port for database authentication. Defaults to the value of the 'AZURE_SQL_PORT' environment variable if not provided.
            local (bool): Whether to use local database. Defaults to True (For now, subjected to change based on deployment plans).
        Raises:
            ValueError: If any of the required connection parameters (server, database, user, password) are not provided either through arguments or environment variables.
        """
        self.server = server or os.getenv("AZURE_SQL_SERVER")
        self.database = database or os.getenv("AZURE_SQL_DATABASE")
        self.user = user or os.getenv("AZURE_SQL_USER")
        self.password = password or os.getenv("AZURE_SQL_PASSWORD")
        self.port = port or os.getenv("AZURE_SQL_PORT")
        self.local = local

        if not self.local and not all(
            [self.server, self.database, self.user, self.password, self.port]
        ):
            raise ValueError(
                "All connection parameters must be provided, either through arguments or environment variables."
            )

        self.connection_string = self._construct_connection_string()
        _logger.info("Connecting to %s", self.database)

    def _construct_connection_string(self) -> str:
        """
        Constructs the connection string based on the provided connection parameters.

        Returns:
            str: The constructed connection string.
        """
        if self.local:
            connection_string = (
                "DRIVER={ODBC Driver 17 for SQL Server};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                "Trusted_Connection=yes;"
            )
        else:
            connection_string = (
                "DRIVER={ODBC Driver 17 for SQL Server};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.user};"
                f"PWD={self.password};"
                "TrustServerCertificate=no;"
                "Encrypt=yes;"
                "ConnectionTimeout=30;"
            )
        return connection_string

    def connect(self) -> Connection:
        """
        Connects to the database using the provided connection string.

        Returns:
            pyodbc.Connection or None: A connection object if successful, None otherwise.
        """
        try:
            conn = pyodbc.connect(self.connection_string)
            _logger.info("Connected to the database")
            return conn
        except Exception as e:
            _logger.error("Error connecting to the database: %s", e)
            return None

    def close(self, conn: Connection) -> None:
        """
        Closes the database connection.

        Args:
            conn (pyodbc.Connection): The connection object to be closed.
        """
        try:
            conn.close()
            _logger.info("Connection closed")
        except Exception as e:
            _logger.error("Error closing connection: %s", e)