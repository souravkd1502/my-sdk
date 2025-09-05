"""
base.py
=================
Abstract base class for all Extractors in the ETL framework.

This module defines the `BaseExtractor` abstract class, which enforces a
standardized interface for implementing data extraction logic. Extractors are
responsible for retrieving data from various sources (e.g., files, databases,
APIs) and returning it as a `pandas.DataFrame`.

Key Features
------------
- Provides a contract (`extract`) that all extractors must implement.
- Ensures extracted data is returned in a standardized format (`pd.DataFrame`).
- Promotes consistency and interoperability within ETL pipelines.

Usage Example
-------------
    >>> import pandas as pd
    >>> from etl_framework.extractors.base import BaseExtractor
    >>> from etl_framework.extractors.exceptions import DataReadError
    >>>
    >>> class CSVExtractor(BaseExtractor):
    ...     def __init__(self, filepath: str):
    ...         self.filepath = filepath
    ...
    ...     def extract(self) -> pd.DataFrame:
    ...         try:
    ...             return pd.read_csv(self.filepath)
    ...         except Exception as e:
    ...             raise DataReadError(f"Failed to read CSV: {e}")
    >>>
    >>> extractor = CSVExtractor("data.csv")
    >>> df = extractor.extract()
    >>> print(df.head())

Dependencies
------------
- ``pandas``: Required for representing tabular data.
- ``abc``: Used for defining abstract base classes.

Notes
-----
- All custom extractors **must** inherit from `BaseExtractor` and implement `extract`.
- Extractors should raise specific exceptions (`DataSourceConnectionError`, `DataReadError`) 
    to allow pipelines to differentiate between failure types.
"""

import pandas as pd
from abc import ABC, abstractmethod

class BaseExtractor(ABC):
    """
    Abstract base class for all extractors in the ETL framework.

    This class enforces the implementation of the `extract` method in all
    subclasses, ensuring that extracted data is returned as a `pandas.DataFrame`.
    """

    @abstractmethod
    def extract(self) -> pd.DataFrame:
        """
        Extract data from the source system.

        This method must be implemented by subclasses to retrieve data
        from their respective sources (files, databases, APIs, etc.).

        Returns:
            pd.DataFrame: A DataFrame containing the extracted data.

        Raises:
            ExtractorError: For general extraction failures.
            DataSourceConnectionError: If connection to the source fails.
            DataReadError: If reading data from the source fails.
        """
        pass
