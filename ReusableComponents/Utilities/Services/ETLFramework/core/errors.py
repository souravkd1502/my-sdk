"""
errors.py
=========
Custom exception classes for an ETL (Extract, Transform, Load) framework.
Provides specific, hierarchical exceptions for error handling in ETL pipelines.

This module defines exceptions for all layers of the ETL process:
- Extractor: Errors related to data extraction (e.g., missing sources, connection failures).
- Transformer: Errors during data transformation (e.g., schema mismatches, validation failures).
- Loader: Errors during data loading (e.g., destination connection issues, duplicates).
- Pipeline: Errors during pipeline orchestration (e.g., timeouts, config issues).
- System: Environment-level errors (e.g., missing dependencies, disk/memory issues).

Each exception class is designed to be caught at the appropriate layer,
enabling granular error handling and debugging.

Exceptions/Errors
-----------------
Extractor Layer
---------------------
- ``DataSourceNotFoundError``: Raised when a file, table, or API endpoint does not exist.
- ``DataSourceConnectionError``: Raised when connection to the source fails.
- ``DataSourceAuthenticationError``: Raised when credentials for the data source are invalid.
- ``DataReadError``: Raised when extraction fails due to bad format or corrupted data.

Transformer Layer
---------------------
- ``TransformationLogicError``: Raised when custom transformation logic fails.
- ``SchemaMismatchError``: Raised when the DataFrame schema does not match expectations.
- ``DataValidationError``: Raised when data fails validation rules.

Loader Layer
-----------------
- ``DataLoadError``: Raised when data cannot be loaded into the target.
- ``DestinationConnectionError``: Raised when connection to the destination fails.
- ``DuplicateDataError``: Raised when duplicate or conflicting data is detected.

Pipeline Layer
---------------
- ``InvalidPipelineConfigError``: Raised when the pipeline configuration is invalid.
- ``StepTimeoutError``: Raised when a step exceeds its execution time limit.

System Layer
-------------
- ``DependencyMissingError``: Raised when a required library is missing.
- ``DiskSpaceError``: Raised when there is insufficient disk space.
- ``NetworkError``: Raised when network issues occur during extract/load.

Usage Example
-------------
    >>> from errors import DataSourceNotFoundError
    >>>
    >>> try:
    ...     extractor.extract("missing_file.csv")
    ... except DataSourceNotFoundError as e:
    ...     print(f"Error: {e}")
    ...     # Handle missing source

Dependencies
------------
- None (built-in ``Exception`` class is sufficient).
"""


class ETLException(Exception):
    """Base class for all ETL-related exceptions."""

    pass


# ─────────────────────────────
# Extractor Exceptions
# ─────────────────────────────
class ExtractorError(ETLException):
    """Base class for extractor-related errors."""

    pass


class DataSourceNotFoundError(ExtractorError):
    """Raised when a file, table, or API endpoint does not exist."""

    pass


class DataSourceConnectionError(ExtractorError):
    """Raised when connection to source (DB, API, etc.) fails."""

    pass


class DataSourceAuthenticationError(ExtractorError):
    """Raised when credentials for data source are invalid."""

    pass


class DataReadError(ExtractorError):
    """Raised when extraction fails due to bad file format, schema mismatch, or corrupted data."""

    pass

class QueryExecutionError(ExtractorError):
    """Raised when query execution fails to execute."""

    pass

# ─────────────────────────────
# Transformer Exceptions
# ─────────────────────────────
class TransformerError(ETLException):
    """Base class for transformer-related errors."""

    pass


class TransformationLogicError(TransformerError):
    """Raised when custom transformation logic fails (bad function, division by zero, etc.)."""

    pass


class SchemaMismatchError(TransformerError):
    """Raised when DataFrame schema is not as expected."""

    pass


class DataValidationError(TransformerError):
    """Raised when data fails validation rules (e.g., missing required fields, wrong types)."""

    pass

class CloudExtractorError(Exception):
    """Base exception for cloud extractor errors."""
    pass

# ─────────────────────────────
# Loader Exceptions
# ─────────────────────────────
class LoaderError(ETLException):
    """Base class for loader-related errors."""

    pass


class DataLoadError(LoaderError):
    """Raised when data cannot be loaded into the target (insert/update failure)."""

    pass


class DestinationConnectionError(LoaderError):
    """Raised when connection to destination fails (DB, S3, etc.)."""

    pass


class DestinationAuthenticationError(LoaderError):
    """Raised when credentials for destination are invalid."""

    pass


class DuplicateDataError(LoaderError):
    """Raised when duplicate or conflicting data prevents loading."""

    pass


# ─────────────────────────────
# Pipeline Exceptions
# ─────────────────────────────
class PipelineError(ETLException):
    """Base class for pipeline orchestration errors."""

    pass


class InvalidPipelineConfigError(PipelineError):
    """Raised when pipeline configuration file is invalid or missing fields."""

    pass


class PipelineExecutionError(PipelineError):
    """Raised when an unexpected error occurs during pipeline execution."""

    pass


class StepTimeoutError(PipelineError):
    """Raised when a step (extract/transform/load) exceeds execution time limit."""

    pass


class RetryLimitExceededError(PipelineError):
    """Raised when retries for a failing step are exhausted."""

    pass


# ─────────────────────────────
# System-Level Exceptions
# ─────────────────────────────
class ETLSystemError(ETLException):
    """Base class for system/environment-level ETL issues."""

    pass


class DependencyMissingError(ETLSystemError):
    """Raised when a required library or dependency is missing."""

    pass


class DiskSpaceError(ETLSystemError):
    """Raised when there is insufficient disk space during extract/load."""

    pass


class MemoryOverflowError(ETLSystemError):
    """Raised when processing exceeds memory limits (e.g., huge DataFrame)."""

    pass


class NetworkError(ETLSystemError):
    """Raised when network-related issues occur during extract/load."""

    pass
