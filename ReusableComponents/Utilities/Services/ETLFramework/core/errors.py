"""
errors.py
----------

Custom exception classes for ETL framework components: Extractor, Transformer, Loader, and Pipeline.
Each exception class provides clear, specific error handling for various failure scenarios.

Exception Coverage
-------------------

- Extractor layer
----------------------------
    -- Source missing (file/table/API not found)
    -- Connection/authentication failure
    -- Bad/corrupt data format
    -- Transformer layer
    -- Logic bugs in transformations
    -- Schema mismatch (e.g., expecting a date column but missing)
    -- Data validation failures (wrong types, business rule violations)

- Loader layer
----------------------------
    -- Destination connection/auth failure
    -- Duplicate/conflicting records
    -- Storage errors (disk full, DB constraint violation)

- Pipeline orchestration
----------------------------
    -- Bad config
    -- Step timeout
    -- Retry exhaustion
    -- Unexpected exceptions bubbled up

- System-level issues
----------------------------
    -- Missing dependency (psycopg2, pandas)
    -- Disk/memory exhaustion
    -- Network outage
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
