"""
base.py
=================
Abstract base class for all Transformers in the ETL framework.

This module defines the `BaseTransformer` abstract class, which enforces a
contract for implementing transformation logic in ETL pipelines. Transformers
take a `pandas.DataFrame` as input, apply data cleaning, enrichment, or feature
engineering, and return the transformed DataFrame.

Key Features
------------
- Provides a standard interface (`transform`) for all transformers.
- Enforces consistent input/output type (`pandas.DataFrame`).
- Allows chaining multiple transformers in sequence within a pipeline.

Usage Example
-------------
    >>> import pandas as pd
    >>> from etl_framework.transformers.base import BaseTransformer
    >>>
    >>> class DropNullsTransformer(BaseTransformer):
    ...     def transform(self, df: pd.DataFrame) -> pd.DataFrame:
    ...         return df.dropna()
    >>>
    >>> transformer = DropNullsTransformer()
    >>> df = pd.DataFrame({"a": [1, None, 3]})
    >>> print(transformer.transform(df))

Dependencies
------------
- ``pandas``: Required for handling tabular data transformations.

Notes
-----
- All custom transformers **must** inherit from `BaseTransformer` and implement `transform`.
- Transformers should be stateless when possible for easier testing and reusability.
"""

import pandas as pd
from abc import ABC, abstractmethod


class BaseTransformer(ABC):
    """
    Abstract base class for all transformers in the ETL framework.

    Subclasses must implement the `transform` method to modify or enrich
    input data and return a new DataFrame.
    """

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformation logic to a DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.

        Raises:
            Exception: Subclasses should raise appropriate ETL exceptions
                        (e.g., DataValidationError, TransformationError).
        """
        pass
