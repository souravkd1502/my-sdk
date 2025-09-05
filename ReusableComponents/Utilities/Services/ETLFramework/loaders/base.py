"""
base.py
=================
Abstract base class for all Loaders in the ETL framework.

This module defines the `BaseLoader` abstract class, which enforces a contract
for implementing data loading logic. Loaders take a `pandas.DataFrame` as input
and persist it to a target destination (e.g., database, data warehouse, API, file).

Key Features
------------
- Provides a standard interface (`load`) for all loaders.
- Enforces consistent input type (`pandas.DataFrame`).
- Supports flexible destinations: relational DBs, cloud storage, APIs, files.

Usage Example
-------------
    >>> import pandas as pd
    >>> from etl_framework.loaders.base import BaseLoader
    >>>
    >>> class CSVSaver(BaseLoader):
    ...     def __init__(self, filepath: str):
    ...         self.filepath = filepath
    ...
    ...     def load(self, df: pd.DataFrame) -> None:
    ...         df.to_csv(self.filepath, index=False)
    >>>
    >>> loader = CSVSaver("output.csv")
    >>> df = pd.DataFrame({"a": [1, 2, 3]})
    >>> loader.load(df)

Dependencies
------------
- ``pandas``: Required for consuming tabular data.
- Destination-specific libraries may also be required
    (e.g., ``sqlalchemy`` for databases, ``boto3`` for AWS S3).

Notes
-----
- All custom loaders **must** inherit from `BaseLoader` and implement `load`.
- Loaders should handle destination-specific connection and write errors gracefully.
"""

import pandas as pd
from abc import ABC, abstractmethod


class BaseLoader(ABC):
    """
    Abstract base class for all loaders in the ETL framework.

    Subclasses must implement the `load` method to persist
    data into a target system.
    """

    @abstractmethod
    def load(self, df: pd.DataFrame) -> None:
        """
        Load data into a target destination.

        Args:
            df (pd.DataFrame): The DataFrame containing data to load.

        Raises:
            Exception: Subclasses should raise appropriate ETL exceptions
                        (e.g., DataWriteError, DestinationConnectionError).
        """
        pass
