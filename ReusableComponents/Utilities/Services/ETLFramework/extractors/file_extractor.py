"""
file_extractor.py
=================
File-based data extraction framework for ETL pipelines.

This module provides a unified framework for extracting structured and semi-structured
data from files. It defines an abstract base class (:class:`FileExtractor`) that
standardizes file ingestion patterns, and concrete implementations for CSV, JSON
(including JSON Lines/NDJSON), Excel, and Parquet. The design is extensible, allowing
additional formats such as XML, DOCX, or Avro to be integrated seamlessly.

The framework is optimized for scalability and production use: it supports
single files, lists of files, or entire directory trees (with recursive discovery),
and offers both in-memory and streaming modes for efficient handling of datasets.
Optional parallelism accelerates ingestion when working with many small/medium files.
Extractors also support schema validation, column projection, and partition enrichment
from folder naming schemes (e.g., ``year=2025/month=09``).

Key Features
------------
- **Base class** (`FileExtractor`) defining consistent extraction interfaces.
- **Implemented formats**:
  - **CSVExtractor**: flexible delimiter handling, chunked reading, schema validation.
  - **JSONExtractor**: standard JSON, JSON Lines (NDJSON), streaming with `ijson`,
    nested structure flattening, partition enrichment.
  - **ExcelExtractor**: multi-sheet ingestion, selective sheet extraction, chunking.
  - **ParquetExtractor**: schema validation across files, row-group streaming,
    partition-aware ingestion, and parallel reads.
- **Extensible design**: easily extend to XML, DOCX, Avro, or custom formats.
- **Input flexibility**: single file, multiple files, or recursive folder discovery.
- **Streaming and chunked reads** for memory-efficient processing of large files.
- **Parallel processing** for multi-file ingestion.
- **Schema validation** across files with optional strict typing.
- **Partition key enrichment** from directory naming conventions (``key=value``).
- **Comprehensive metadata** for tracking extracted datasets.

Exceptions/Errors
-----------------
- ``ETLException``: General ETL framework errors (e.g., unsupported file types).
- ``DataSourceNotFoundError``: Raised when no valid input files are found.
- ``DataReadError``: Raised when a file cannot be read or parsed.
- ``DependencyMissingError``: Raised if an optional dependency (e.g., ijson, openpyxl) is missing.
- ``ExtractorError``: Generic catch-all for extraction failures.

Usage Examples
--------------
Basic CSV extraction:

    >>> from etl_framework.extractors import CSVExtractor
    >>> extractor = CSVExtractor("data.csv")
    >>> df, meta = extractor.extract()
    >>> print(f"Extracted {meta['num_records']} records")

Streaming large JSONL file:

    >>> from etl_framework.extractors import JSONExtractor
    >>> extractor = JSONExtractor("data.jsonl", json_lines=True, chunksize=50_000)
    >>> for chunk, meta in extractor.stream_extract():
    ...     process_chunk(chunk)

Excel multi-sheet ingestion:

    >>> from etl_framework.extractors import ExcelExtractor
    >>> extractor = ExcelExtractor("workbook.xlsx", sheet_name=None)
    >>> df, meta = extractor.extract()

Partitioned Parquet ingestion:

    >>> from etl_framework.extractors import ParquetExtractor
    >>> extractor = ParquetExtractor("dataset/", recursive=True, parallel=True, validate_schema=True)
    >>> df, meta = extractor.extract()

Row-group streaming from Parquet:

    >>> extractor = ParquetExtractor("large.parquet", chunksize=1)
    >>> for chunk, meta in extractor.extract():
    ...     process_chunk(chunk)

Dependencies
------------
- ``pandas``: Required for DataFrame operations and chunked file reading.
- ``pathlib``: Cross-platform file path operations.
- ``concurrent.futures``: Parallel file processing.
- ``ijson`` (optional): Streaming JSON parser for large JSON arrays.
- ``openpyxl`` or ``xlrd``: Required for Excel ingestion.
- ``pyarrow`` or ``fastparquet``: Required for Parquet ingestion.
- ``typing``: Type hints for better interfaces.

Notes
-----
- Use ``stream_extract()`` for very large files to avoid memory pressure.
- When ``chunksize`` is specified, extractors yield dataframes in chunks
    instead of loading everything into memory at once.
- For Parquet, row-group streaming requires ``pyarrow``.
- For Excel, chunking requires conversion into CSV-like streams.
- Designed for production ETL: logging, error handling, and extensibility
    are first-class considerations.
- For massive or distributed workloads, delegate orchestration to Dask, Polars, or Spark.

TODO:
-------
1. Properly test Recursive functionality in all file based extractor classes.
2. Add filtering functionality to all file based extractor classes.
"""

import logging
import pandas as pd
from pathlib import Path
from textwrap import dedent
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Tuple, Dict, Any, Optional, Union, List, Generator, Iterable


try:
    # Try relative import first (when used as part of package)
    from ..core.errors import (
        ETLException,
        DataReadError,
        DataSourceNotFoundError,
        DependencyMissingError,
        ExtractorError,
    )
except ImportError:
    # Fall back to absolute import (when run directly)
    import sys
    from pathlib import Path

    # Add the ETLFramework directory to Python path
    etl_framework_path = Path(__file__).parent.parent
    sys.path.insert(0, str(etl_framework_path))
    from core.errors import (
        ETLException,
        DataReadError,
        DataSourceNotFoundError,
        DependencyMissingError,
        ExtractorError,
    )


# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FileExtractor(ABC):
    """
    Abstract base class for all file-based extractors.

    This class provides common validation and metadata creation logic
    for extractors that read data from file systems (e.g., CSV, Excel, JSON).

    Subclasses must implement the `extract` method to handle
    format-specific file reading.
    """

    SUPPORTED_FILE_TYPES = ["csv", "xlsx", "json", "parquet"]

    def __init__(self, filepath: str) -> None:
        """
        Initialize the FileExtractor with the given file path.

        Args
        ----
        filepath : str
            The path to the file to be extracted.

        Raises
        ------
        ValueError
            If the file path is invalid or empty.
        ETLException
            If the file type is unsupported.
        """
        if not isinstance(filepath, str) or not filepath:
            logger.error("Invalid file path provided: %s", filepath)
            raise ValueError("Filepath must be a non-empty string.")

        if not filepath.endswith(tuple(self.SUPPORTED_FILE_TYPES)):
            logger.error("Unsupported file type: %s", filepath)
            raise ETLException(
                f"Unsupported file type. Only {self.SUPPORTED_FILE_TYPES} are supported."
            )

        self.filepath = filepath

    @property
    def _get_filepath(self) -> str:
        """
        Get the file path associated with this extractor.

        Returns
        -------
        str
            The file path.
        """
        return self.filepath

    @abstractmethod
    def extract(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extract data from the file and return a DataFrame with metadata.

        Subclasses must override this method with format-specific logic.

        Returns
        -------
        Tuple[pandas.DataFrame, Dict[str, Any]]
            - DataFrame containing extracted data.
            - Metadata dictionary with details of the extraction.

        Raises
        ------
        DataReadError
            If the file cannot be read.
        """
        pass

    def _create_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create metadata for the extracted DataFrame.

        This method can be overridden by subclasses to provide
        additional metadata specific to the file type.

        Args
        ----
        df : pandas.DataFrame
            Extracted data.

        Returns
        -------
        Dict[str, Any]
            Metadata dictionary containing:
            - source: file path
            - extraction_time: timestamp
            - num_records: number of rows
            - num_columns: number of columns
            - columns: list of column names
            - summary: statistical summary
        """
        try:
            metadata: Dict[str, Any] = {
                "source": self.filepath,
                "extraction_time": pd.Timestamp.now().isoformat(),
                "num_records": len(df),
                "num_columns": len(df.columns),
                "columns": df.columns.tolist(),
                "summary": df.describe(include="all").to_dict() if not df.empty else {},
            }
            return metadata
        except Exception as e:
            logger.warning("Metadata creation failed: %s", str(e))
            return {
                "source": self.filepath,
                "extraction_time": pd.Timestamp.now().isoformat(),
                "num_records": len(df),
                "num_columns": len(df.columns) if hasattr(df, "columns") else 0,
                "columns": df.columns.tolist() if hasattr(df, "columns") else [],
                "summary": {},
            }


class CSVExtractor(FileExtractor):
    """
    Extractor for CSV files.

    Inherits from FileExtractor and implements the extract method to read
    one or more CSV files into a pandas.DataFrame.

    Provides functionality to load data from single or multiple CSV files,
    supporting:
    - Single file or multiple files
    - Folder-based extraction (with optional recursion)
    - Chunked streaming for very large files
    - Parallel reading of multiple CSVs
    - Schema validation for multiple files

    This class is designed for flexible ETL pipelines, offering both batch
    (regular) and streaming modes with optional schema validation.
    """

    def __init__(
        self,
        input_source: Union[str, List[str]],
        delimiter: str = ",",
        encoding: Optional[str] = "utf-8",
        na_values: Optional[List[str]] = None,
        recursive: bool = False,
        parallel: bool = False,
        chunksize: Optional[int] = None,
        validate_schema: bool = True,
        strict_schema: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the CSVExtractor with input source and parsing options.

        Args
        ----
        input_source : Union[str, List[str]]
            File path, folder path, or list of file/folder paths.
        delimiter : str, default=','
            Delimiter used in the CSV files.
        encoding : Optional[str], default='utf-8'
            File encoding.
        na_values : Optional[List[str]], default=None
            Additional strings to recognize as NA/NaN.
        recursive : bool, default=False
            Whether to search for CSV files recursively in folders.
        parallel : bool, default=False
            Whether to read multiple files in parallel.
        chunksize : Optional[int], default=None
            If set, read large CSVs in chunks of given row size.
        validate_schema : bool, default=True
            Whether to validate schema compatibility across multiple files.
        strict_schema : bool, default=False
            If True, enforces strict data type matching. If False, allows
            compatible data types (e.g., int64 and float64).
        kwargs : dict
            Additional keyword arguments passed to `pandas.read_csv`.

        Raises
        ------
        ValueError
            If no valid CSV files are found in `input_source`.
        """
        # Discover CSV files from given input first
        self.filepaths: List[str] = self._discover_files(input_source)

        # For single file, call parent init. For multiple files, set filepath to first file
        if len(self.filepaths) == 1:
            super().__init__(self.filepaths[0])
        else:
            # For multiple files, we bypass parent validation and set filepath manually
            self.filepath = self.filepaths[0]  # Use first file as representative

        self.input_source = input_source
        self.delimiter = delimiter
        self.encoding = encoding
        self.na_values = na_values
        self.recursive = recursive
        self.parallel = parallel
        self.chunksize = chunksize
        self.validate_schema = validate_schema
        self.strict_schema = strict_schema
        self.read_csv_kwargs = kwargs

        # Schema validation attributes
        self.master_schema: Optional[Dict[str, Any]] = None
        self.schema_validated: bool = False

    def _discover_files(self, source: Union[str, List[str]]) -> List[str]:
        """
        Discover CSV files from the given input source.

        Args
        ----
        source : Union[str, List[str]]
            File path, folder path, or list of them.

        Returns
        -------
        List[str]
            List of absolute file paths to discovered CSV files.

        Raises
        ------
        ValueError
            If no CSV files are found in the given source.
        """
        filepaths: List[str] = []

        # Normalize to list for uniform iteration
        if isinstance(source, str):
            source = [source]

        for path in source:
            p = Path(path)

            if p.is_file() and p.suffix.lower() == ".csv":
                filepaths.append(str(p.resolve()))

            elif p.is_dir():
                # Use recursive or non-recursive globbing
                pattern = "**/*.csv" if self.recursive else "*.csv"
                for file in p.glob(pattern):
                    filepaths.append(str(file.resolve()))

            else:
                logger.warning("Invalid source path skipped: %s", path)

        if not filepaths:
            raise ValueError(f"No CSV files found in input source: {source}")

        logger.info("Discovered %d CSV files.", len(filepaths))
        return filepaths

    def _extract_schema(self, filepath: str) -> Dict[str, Any]:
        """
        Extract schema information from a CSV file.

        Args
        ----
        filepath : str
            Path to the CSV file.

        Returns
        -------
        Dict[str, Any]
            Schema dictionary containing:
            - columns: list of column names
            - dtypes: dictionary of column data types
            - num_columns: number of columns
            - filepath: source file path

        Raises
        ------
        DataReadError
            If the file cannot be read for schema extraction.
        """
        try:
            # Read just the first few rows to extract schema
            read_params: Dict[str, Any] = {
                "delimiter": self.delimiter,
                "encoding": self.encoding,
                "na_values": self.na_values,
                "nrows": 100,  # Read only first 100 rows for schema
                **self.read_csv_kwargs,
            }

            logger.debug("Extracting schema from CSV file: %s", filepath)
            sample_df = pd.read_csv(filepath, **read_params)

            schema = {
                "columns": sample_df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in sample_df.dtypes.items()},
                "num_columns": len(sample_df.columns),
                "filepath": filepath,
            }

            logger.debug(
                "Schema extracted from %s: %d columns", filepath, schema["num_columns"]
            )
            return schema

        except Exception as e:
            logger.error("Failed to extract schema from %s: %s", filepath, e)
            raise DataReadError(f"Schema extraction failed for {filepath}: {e}") from e

    def _are_dtypes_compatible(self, dtype1: str, dtype2: str) -> bool:
        """
        Check if two pandas data types are compatible for concatenation.

        Args
        ----
        dtype1 : str
            First data type as string.
        dtype2 : str
            Second data type as string.

        Returns
        -------
        bool
            True if data types are compatible, False otherwise.
        """
        if self.strict_schema:
            return dtype1 == dtype2

        # Define compatible data type groups
        numeric_types = {
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float16",
            "float32",
            "float64",
            "Float32",
            "Float64",
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "UInt8",
            "UInt16",
            "UInt32",
            "UInt64",
        }

        string_types = {"object", "string", "category"}

        datetime_types = {
            "datetime64[ns]",
            "datetime64[ns, UTC]",
            "period[D]",
            "timedelta64[ns]",
        }

        boolean_types = {"bool", "boolean"}

        # Check if both types belong to the same compatible group
        if dtype1 in numeric_types and dtype2 in numeric_types:
            return True
        elif dtype1 in string_types and dtype2 in string_types:
            return True
        elif dtype1 in datetime_types and dtype2 in datetime_types:
            return True
        elif dtype1 in boolean_types and dtype2 in boolean_types:
            return True
        else:
            return dtype1 == dtype2

    def _validate_schema_compatibility(
        self, schemas: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate schema compatibility across multiple CSV files.

        Args
        ----
        schemas : List[Dict[str, Any]]
            List of schema dictionaries from multiple files.

        Returns
        -------
        Dict[str, Any]
            Master schema dictionary with unified column information.

        Raises
        ------
        DataReadError
            If schemas are incompatible across files.
        """
        if not schemas:
            raise DataReadError("No schemas provided for validation.")

        if len(schemas) == 1:
            logger.info("Single file detected, schema validation skipped.")
            return schemas[0]

        logger.info("Validating schema compatibility across %d files.", len(schemas))

        # Use first file as reference schema
        master_schema = schemas[0].copy()
        master_columns = set(master_schema["columns"])
        incompatible_files = []

        for i, schema in enumerate(schemas[1:], 1):
            current_columns = set(schema["columns"])
            filepath = schema["filepath"]

            # Check column names
            if master_columns != current_columns:
                missing_in_current = master_columns - current_columns
                extra_in_current = current_columns - master_columns

                error_msg = f"Schema mismatch in {filepath}:"
                if missing_in_current:
                    error_msg += f" Missing columns: {list(missing_in_current)}."
                if extra_in_current:
                    error_msg += f" Extra columns: {list(extra_in_current)}."

                incompatible_files.append(error_msg)
                continue

            # Check data types for each column
            for col in master_schema["columns"]:
                master_dtype = master_schema["dtypes"][col]
                current_dtype = schema["dtypes"][col]

                if not self._are_dtypes_compatible(master_dtype, current_dtype):
                    error_msg = (
                        f"Data type mismatch in {filepath} for column '{col}': "
                        f"expected {master_dtype}, got {current_dtype}."
                    )
                    incompatible_files.append(error_msg)

        if incompatible_files:
            error_summary = "Schema validation failed:\n" + "\n".join(
                incompatible_files
            )
            logger.error("Schema validation failed: %s", error_summary)
            raise DataReadError(error_summary)

        logger.info("Schema validation successful: All files have compatible schemas.")
        return master_schema

    def _read_file(self, filepath: str) -> pd.DataFrame:
        """
        Read a single CSV file into a DataFrame.

        Args
        ----
        filepath : str
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the file's contents.

        Raises
        ------
        DataReadError
            If the file cannot be read due to missing file, parse error,
            or unexpected issues.
        """
        try:
            # Prepare read_csv parameters
            read_params: Dict[str, Any] = {
                "delimiter": self.delimiter,
                "encoding": self.encoding,
                "na_values": self.na_values,
                **self.read_csv_kwargs,
            }
            if self.chunksize is not None:
                read_params["chunksize"] = self.chunksize

            logger.debug("Reading CSV file: %s with params %s", filepath, read_params)
            df = pd.read_csv(filepath, **read_params)
            return df

        except FileNotFoundError as fnf_error:
            logger.error("CSV file not found: %s", filepath)
            raise DataReadError(f"File not found: {filepath}") from fnf_error

        except pd.errors.ParserError as parse_error:
            logger.error("CSV parsing failed for %s: %s", filepath, parse_error)
            raise DataReadError(
                f"Parsing error while reading {filepath}: {parse_error}"
            ) from parse_error

        except Exception as e:
            logger.exception("Unexpected error during CSV extraction from %s", filepath)
            raise DataReadError(f"Failed to read {filepath}: {e}") from e

    def extract(
        self,
    ) -> Union[
        Tuple[pd.DataFrame, Dict[str, Any]],
        Generator[Tuple[pd.DataFrame, Dict[str, Any]], None, None],
    ]:
        """
        Extract data from one or more CSV files.

        Returns
        -------
        Union[Tuple[pandas.DataFrame, Dict[str, Any]],
                Generator[Tuple[pandas.DataFrame, Dict[str, Any]], None, None]]
            - If `chunksize` is None:
                Combined DataFrame containing extracted CSV data and metadata.
            - If `chunksize` is set:
                Generator yielding (DataFrame chunk, metadata) tuples.

        Raises
        ------
        DataReadError
            If any file cannot be read successfully.
        """
        if self.chunksize:
            logger.info(
                "Extracting CSVs in streaming mode with chunksize=%s", self.chunksize
            )
            return self._extract_streaming()
        else:
            logger.info(
                "Extracting CSVs in regular mode from %d files", len(self.filepaths)
            )
            return self._extract_regular()

    def _extract_regular(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extract data in regular (non-streaming) mode.

        Reads all discovered CSV files into memory and concatenates them
        into a single DataFrame. Performs schema validation if enabled
        and multiple files are being processed.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any]]
            Combined DataFrame and metadata.
        """
        # Perform schema validation if enabled and multiple files exist
        if (
            self.validate_schema
            and len(self.filepaths) > 1
            and not self.schema_validated
        ):
            logger.info(
                "Performing schema validation for %d files.", len(self.filepaths)
            )
            schemas = []
            for fp in self.filepaths:
                schema = self._extract_schema(fp)
                schemas.append(schema)

            # Validate and store master schema
            self.master_schema = self._validate_schema_compatibility(schemas)
            self.schema_validated = True
            logger.info("Schema validation completed successfully.")

        dfs: List[pd.DataFrame] = []

        if self.parallel:
            logger.info("Reading files in parallel using ThreadPoolExecutor.")
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(self._read_file, fp): fp for fp in self.filepaths
                }
                for future in as_completed(futures):
                    fp = futures[future]
                    try:
                        df = future.result()
                        dfs.append(df)
                        logger.info("Successfully read file: %s", fp)
                    except Exception as e:
                        logger.error("Failed to read file %s: %s", fp, e)
                        raise
        else:
            for fp in self.filepaths:
                df = self._read_file(fp)
                dfs.append(df)
                logger.info("Successfully read file: %s", fp)

        if not dfs:
            raise DataReadError("No data extracted from the provided CSV files.")

        combined_df = pd.concat(dfs, ignore_index=True)
        metadata = self._create_metadata(combined_df)

        # Add schema information to metadata if validation was performed
        if self.master_schema:
            metadata["schema_validation"] = {
                "validated": True,
                "master_schema": self.master_schema,
                "strict_mode": self.strict_schema,
                "num_files_validated": len(self.filepaths),
            }
        else:
            metadata["schema_validation"] = {
                "validated": False,
                "reason": "Single file or validation disabled",
            }

        logger.info(
            "CSV extraction completed: %d records combined from %d files.",
            metadata["num_records"],
            len(self.filepaths),
        )
        return combined_df, metadata

    def _extract_streaming(
        self,
    ) -> Generator[Tuple[pd.DataFrame, Dict[str, Any]], None, None]:
        """
        Extract data in streaming mode using chunks.

        Iterates through each file and yields chunks of data with metadata.
        Performs schema validation before streaming if enabled.

        Yields
        ------
        Generator[Tuple[pd.DataFrame, Dict[str, Any]], None, None]
            Each item is a tuple of (chunk DataFrame, metadata dict).
        """
        # Perform schema validation if enabled and multiple files exist
        if (
            self.validate_schema
            and len(self.filepaths) > 1
            and not self.schema_validated
        ):
            logger.info(
                "Performing schema validation for %d files before streaming.",
                len(self.filepaths),
            )
            schemas = []
            for fp in self.filepaths:
                schema = self._extract_schema(fp)
                schemas.append(schema)

            # Validate and store master schema
            self.master_schema = self._validate_schema_compatibility(schemas)
            self.schema_validated = True
            logger.info("Schema validation completed successfully.")

        for fp in self.filepaths:
            logger.info("Streaming file: %s", fp)
            result = self._read_file(fp)

            # When chunksize is set, result is a pandas TextFileReader
            for chunk in result:  # type: ignore
                logger.debug("Yielding chunk of size %d from %s", len(chunk), fp)
                chunk_metadata = self._create_metadata(chunk)

                # Add schema information to metadata if validation was performed
                if self.master_schema:
                    chunk_metadata["schema_validation"] = {
                        "validated": True,
                        "master_schema": self.master_schema,
                        "strict_mode": self.strict_schema,
                        "num_files_validated": len(self.filepaths),
                    }
                else:
                    chunk_metadata["schema_validation"] = {
                        "validated": False,
                        "reason": "Single file or validation disabled",
                    }

                yield chunk, chunk_metadata

    @staticmethod
    def example_usage() -> None:
        """
        Print detailed usage examples for CSVExtractor.

        This demonstrates different ways to use the class,
        covering single-file, multi-file, folder-based,
        recursive, parallel, streaming, and schema validation modes.
        """

        examples = dedent(
            """
        ============================================================
        CSVExtractor USAGE EXAMPLES
        ============================================================

        1. Extract data from a single CSV file
        ------------------------------------------------------------
        extractor = CSVExtractor("data.csv", delimiter=",", encoding="utf-8")
        df, metadata = extractor.extract()
        print(metadata)

        2. Extract from multiple CSV files (batch mode)
        ------------------------------------------------------------
        extractor = CSVExtractor(["file1.csv", "file2.csv"])
        df, metadata = extractor.extract()
        print(f"Combined {metadata['num_records']} rows")

        3. Extract all CSV files from a folder
        ------------------------------------------------------------
        extractor = CSVExtractor("data_folder/")
        df, metadata = extractor.extract()
        print(metadata)

        4. Extract CSV files from a folder (recursive search)
        ------------------------------------------------------------
        extractor = CSVExtractor("data_folder/", recursive=True)
        df, metadata = extractor.extract()
        print(f"Found {len(df)} rows from recursive search")

        5. Extract multiple CSVs in parallel
        ------------------------------------------------------------
        extractor = CSVExtractor("data_folder/", parallel=True)
        df, metadata = extractor.extract()
        print(f"Parallel extraction complete: {metadata}")

        6. Streaming large CSV with chunks (memory-efficient)
        ------------------------------------------------------------
        extractor = CSVExtractor("large_file.csv", chunksize=100000)
        for chunk, chunk_metadata in extractor.extract():
            print(f"Processed chunk with {chunk_metadata['num_records']} rows")

        7. Schema validation across multiple files
        ------------------------------------------------------------
        # a) Normal schema validation (columns + types compatible)
        extractor = CSVExtractor(["file1.csv", "file2.csv"], validate_schema=True)
        df, metadata = extractor.extract()
        print(metadata["schema_validation"])

        # b) Strict schema validation (columns + exact dtypes match)
        extractor = CSVExtractor(["file1.csv", "file2.csv"], validate_schema=True, strict_schema=True)
        df, metadata = extractor.extract()

        # c) Disable schema validation (default)
        extractor = CSVExtractor(["file1.csv", "file2.csv"], validate_schema=False)
        df, metadata = extractor.extract()

        ============================================================
        NOTES
        ============================================================
        - Metadata includes number of records, number of columns, and column names.
        - Streaming mode is best for very large files.
        - Parallel mode speeds up reading when many small/medium files exist.
        - Schema validation ensures consistency across multiple input files.
        ============================================================
        """
        )
        print(examples)


class JSONExtractor(FileExtractor):
    """
    Production-grade extractor for JSON and JSON Lines (NDJSON).

    This class discovers one or more JSON/JSONL files, reads them in batch or
    streaming mode, optionally flattens nested structures, and returns a
    combined :class:`pandas.DataFrame` with extraction metadata. For out-of-core
    parsing of huge JSON arrays, it can use ``ijson`` if available.

    Parameters
    ----------
    input_source : Union[str, List[str]]
        File path, folder path, or list of paths. Folders will be scanned
        for JSON/JSONL files; set ``recursive=True`` to traverse subfolders.
    json_lines : Optional[bool], default=None
        If ``True``, treat all files as JSONL/NDJSON. If ``False``, treat all
        as standard JSON. If ``None``, attempt to auto-detect per file
        (based on extension and file sniffing).
    recursive : bool, default=False
        Recursively discover files under directories.
    parallel : bool, default=False
        Parallelize reading for many files (not used in streaming mode).
    max_workers : Optional[int], default=None
        Max worker threads for parallel reads. Defaults to ``ThreadPoolExecutor`` heuristics.
    chunksize : Optional[int], default=None
        Chunk size for streaming. For JSONL, this controls the number of
        lines per chunk. For JSON array streaming with ``ijson``, a batching
        size of ``chunksize`` is used.
    flatten : bool, default=True
        Flatten nested JSON objects using ``pandas.json_normalize``.
    select_columns : Optional[List[str]], default=None
        If provided, select only these columns from the final DataFrame/chunk.
    engine : Literal["pandas", "ijson"], default="pandas"
        Engine for standard JSON (non-JSONL). ``"ijson"`` enables streaming
        over large JSON arrays; requires the ``ijson`` package.
    file_extensions : Optional[List[str]], default=None
        File extensions to consider. Defaults to [".json", ".jsonl", ".ndjson"].

    Raises
    ------
    DataSourceNotFoundError
        If no matching files are found.
    DependencyMissingError
        If ``engine="ijson"`` is selected but ``ijson`` is not installed.
    """

    def __init__(
        self,
        input_source: Union[str, List[str]],
        *,
        json_lines: Optional[bool] = None,
        recursive: bool = False,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        chunksize: Optional[int] = None,
        flatten: bool = True,
        select_columns: Optional[List[str]] = None,
        engine: Literal["pandas", "ijson"] = "pandas",
        file_extensions: Optional[List[str]] = None,
    ) -> None:
        # NOTE: FileExtractor's constructor validates single file paths.
        # For multi-input/folder workflows we perform our own discovery/validation here.
        # We still call super().__init__ with a representative string for consistency.
        super().__init__(
            str(input_source) if isinstance(input_source, str) else "multiple-inputs"
        )

        self.json_lines: Optional[bool] = json_lines
        self.recursive: bool = recursive
        self.parallel: bool = parallel
        self.max_workers: Optional[int] = max_workers
        self.chunksize: Optional[int] = chunksize
        self.flatten: bool = flatten
        self.select_columns: Optional[List[str]] = select_columns
        self.engine: Literal["pandas", "ijson"] = engine
        self.file_extensions: List[str] = file_extensions or [
            ".json",
            ".jsonl",
            ".ndjson",
        ]

        # Discover candidate files
        self.filepaths: List[Path] = self._discover_files(input_source)

        if not self.filepaths:
            logger.error(
                "No JSON/JSONL files discovered in input_source=%s", input_source
            )
            raise DataSourceNotFoundError(
                f"No JSON/JSONL files found in: {input_source}"
            )

        # Validate engine dependencies
        if self.engine == "ijson":
            try:
                import ijson  # noqa: F401
            except Exception as exc:
                logger.error("ijson not available but engine='ijson' was requested.")
                raise DependencyMissingError(
                    "The 'ijson' package is required for engine='ijson' streaming."
                ) from exc

        logger.info(
            "JSONExtractor initialized with %d file(s), recursive=%s, parallel=%s, engine=%s, chunksize=%s",
            len(self.filepaths),
            self.recursive,
            self.parallel,
            self.engine,
            str(self.chunksize),
        )

    # ---------------------------------------------------------------------
    # Discovery & detection
    # ---------------------------------------------------------------------
    def _discover_files(self, source: Union[str, List[str]]) -> List[Path]:
        """
        Discover JSON/JSONL files from input source(s).

        Returns
        -------
        List[pathlib.Path]
            Absolute paths of discovered files filtered by configured extensions.
        """
        paths: List[Path] = []
        inputs: List[str] = [source] if isinstance(source, str) else list(source)

        for raw in inputs:
            p = Path(raw)
            if p.is_file() and self._is_supported_file(p):
                paths.append(p.resolve())
            elif p.is_dir():
                pattern = "**/*" if self.recursive else "*"
                for fp in p.glob(pattern):
                    if fp.is_file() and self._is_supported_file(fp):
                        paths.append(fp.resolve())
            else:
                logger.warning(
                    "Input path skipped (not a file/dir or missing): %s", raw
                )

        # De-duplicate while preserving order
        seen: set = set()
        unique = []
        for fp in paths:
            if fp not in seen:
                unique.append(fp)
                seen.add(fp)
        return unique

    def _is_supported_file(self, path: Path) -> bool:
        """
        Check if the file has a supported extension.
        Args
        -----
        path : pathlib.Path
            File path to check.
        Returns
        -------
        bool
            True if the file has a supported extension, False otherwise.
        """
        return path.suffix.lower() in set(self.file_extensions)

    def _detect_is_jsonl(self, path: Path) -> bool:
        """
        Heuristic detection of JSON Lines vs standard JSON.

        - If extension is ``.jsonl`` or ``.ndjson`` -> treat as JSONL.
        - If user forced ``json_lines`` -> honor it.
        - Otherwise, peek the first few non-empty bytes/lines:
            if multiple JSON objects separated by newlines -> JSONL.
        """
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            return True
        if self.json_lines is not None:
            return self.json_lines

        try:
            with path.open("rb") as f:
                # Peek a small buffer to guess (avoid loading whole file)
                head = f.read(4096)
            # If we see multiple newlines with JSON object starts -> likely JSONL
            sample = head.decode(errors="ignore")
            lines = [ln.strip() for ln in sample.splitlines() if ln.strip()]
            if (
                len(lines) >= 2
                and lines[0].startswith("{")
                and lines[1].startswith("{")
            ):
                return True
            # Fallback: assume standard JSON
            return False
        except Exception as exc:
            logger.warning(
                "Failed to sniff file kind for %s; assuming standard JSON. Reason: %s",
                path,
                exc,
            )
            return False

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def extract(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load all data into memory and return a combined DataFrame with metadata.

        Prefer :meth:`stream_extract` for very large data to avoid memory pressure.

        Returns
        -------
        Tuple[pandas.DataFrame, Dict[str, Any]]

        Raises
        ------
        DataReadError
            If any file cannot be read (the first failure is raised).
        """
        logger.info(
            "Starting JSON extraction (batch mode) from %d file(s).",
            len(self.filepaths),
        )

        # Streaming mode is not compatible with a single combined DataFrame in memory
        if self.engine == "ijson" and self.chunksize:
            msg = "Use stream_extract() for engine='ijson' with chunksize to avoid loading all data into memory."
            logger.error(msg)
            raise ExtractorError(msg)

        if self.json_lines and self.chunksize:
            msg = "Use stream_extract() for JSON Lines with chunksize to avoid loading all data into memory."
            logger.error(msg)
            raise ExtractorError(msg)

        dfs: List[pd.DataFrame] = []

        if self.parallel and len(self.filepaths) > 1:
            logger.info(
                "Reading files in parallel (max_workers=%s).", str(self.max_workers)
            )
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                fut_to_fp = {
                    ex.submit(self._read_file_batch, fp): fp for fp in self.filepaths
                }
                for fut in as_completed(fut_to_fp):
                    fp = fut_to_fp[fut]
                    try:
                        df = fut.result()
                        if not df.empty:
                            df = self._maybe_select_columns(df)
                            dfs.append(df)
                        logger.info("Read OK: %s (rows=%d)", fp, len(df))
                    except Exception as exc:
                        logger.error("Failed to read %s: %s", fp, exc)
                        raise
        else:
            for fp in self.filepaths:
                df = self._read_file_batch(fp)  # raises DataReadError on failure
                if not df.empty:
                    df = self._maybe_select_columns(df)
                    dfs.append(df)
                logger.info("Read OK: %s (rows=%d)", fp, len(df))

        if not dfs:
            logger.warning("No data extracted from discovered files.")
            return pd.DataFrame(), self._create_metadata(pd.DataFrame())

        combined = pd.concat(dfs, ignore_index=True)
        metadata = self._create_metadata(combined)
        metadata.update(
            {
                "files_read": len(self.filepaths),
                "engine": self.engine,
                "json_lines": bool(self.json_lines),
                "parallel": self.parallel,
                "recursive": self.recursive,
            }
        )
        logger.info(
            "JSON extraction completed. Total rows=%d, columns=%d.",
            len(combined),
            len(combined.columns),
        )
        return combined, metadata

    def stream_extract(
        self,
    ) -> Generator[Tuple[pd.DataFrame, Dict[str, Any]], None, None]:
        """
        Stream records as DataFrame chunks (generator), yielding (chunk, metadata).

        Use this for very large inputs that cannot fit in memory. Each yielded
        chunk includes its own metadata (row count, columns, etc.).

        Yields
        ------
        Generator[Tuple[pandas.DataFrame, Dict[str, Any]], None, None]

        Raises
        ------
        DataReadError
            If any file cannot be read.
        """
        logger.info(
            "Starting JSON streaming extraction over %d file(s). engine=%s, chunksize=%s",
            len(self.filepaths),
            self.engine,
            str(self.chunksize),
        )

        if not self.chunksize:
            # Provide a sensible default batch size for streaming if user omitted it
            self.chunksize = 50_000
            logger.info(
                "No chunksize provided; defaulting to chunksize=%d", self.chunksize
            )

        for fp in self.filepaths:
            is_jsonl = self._detect_is_jsonl(fp)
            try:
                if is_jsonl:
                    for chunk in self._stream_jsonl(fp, self.chunksize):
                        yield self._maybe_select_columns(chunk), self._create_metadata(
                            chunk
                        )
                else:
                    for chunk in self._stream_json_array(fp, self.chunksize):
                        yield self._maybe_select_columns(chunk), self._create_metadata(
                            chunk
                        )
            except Exception as exc:
                logger.error("Streaming failed for %s: %s", fp, exc)
                raise

    # ---------------------------------------------------------------------
    # Internal: reading strategies
    # ---------------------------------------------------------------------
    def _read_file_batch(self, fp: Path) -> pd.DataFrame:
        """
        Read a single file fully into memory (batch mode).

        This method auto-detects JSON vs JSONL if not explicitly configured.
        """
        is_jsonl = self._detect_is_jsonl(fp)

        if is_jsonl:
            # Batch mode: read all lines at once (no chunksize)
            try:
                df = pd.read_json(fp, lines=True, orient="records")
            except ValueError as ve:
                logger.error("JSONL parsing error in %s: %s", fp, ve)
                raise DataReadError(f"JSONL parsing error in {fp}: {ve}") from ve
            except FileNotFoundError as fnf:
                logger.error("File not found: %s", fp)
                raise DataReadError(f"File not found: {fp}") from fnf
            except Exception as exc:
                logger.exception("Unexpected error reading JSONL %s: %s", fp, exc)
                raise DataReadError(f"Failed to read JSONL {fp}: {exc}") from exc
        else:
            # Standard JSON: engine determines approach
            if self.engine == "pandas":
                try:
                    # pandas handles objects/arrays; returns DataFrame or Series
                    df = pd.read_json(fp)
                    if isinstance(df, pd.Series) or (not isinstance(df, pd.DataFrame)):
                        # Normalize into tabular structure if pandas returned a Series or irregular result
                        df = pd.json_normalize(df.to_dict())  # type: ignore
                except ValueError as ve:
                    logger.error("JSON parsing error in %s: %s", fp, ve)
                    raise DataReadError(f"JSON parsing error in {fp}: {ve}") from ve
                except FileNotFoundError as fnf:
                    logger.error("File not found: %s", fp)
                    raise DataReadError(f"File not found: {fp}") from fnf
                except Exception as exc:
                    logger.exception("Unexpected error reading JSON %s: %s", fp, exc)
                    raise DataReadError(f"Failed to read JSON {fp}: {exc}") from exc
            elif self.engine == "ijson":
                # If user chose ijson engine but didn't use streaming API, we still aggregate into memory.
                # We'll stream and accumulate batches, then concat.
                try:
                    batches = list(
                        self._stream_json_array(
                            fp, batch_size=self.chunksize or 100_000
                        )
                    )
                    if not batches:
                        return pd.DataFrame()
                    df = pd.concat(batches, ignore_index=True)
                except DataReadError:
                    raise
                except Exception as exc:
                    logger.exception("ijson aggregation failed for %s: %s", fp, exc)
                    raise DataReadError(
                        f"Failed to stream-aggregate JSON {fp}: {exc}"
                    ) from exc
            else:
                raise ExtractorError(f"Unsupported engine: {self.engine}")

        # Optional flatten & partition enrichment
        df = self._maybe_flatten(df)
        df = self._maybe_enrich_partitions(df, fp)

        return df

    def _stream_jsonl(self, fp: Path, batch_size: int) -> Iterable[pd.DataFrame]:
        """
        Stream a JSON Lines file in chunks of ``batch_size`` lines.
        """
        try:
            # pandas will return an iterator of DataFrames if chunksize is set
            for chunk in pd.read_json(
                fp, lines=True, chunksize=batch_size, orient="records"
            ):
                if not isinstance(chunk, pd.DataFrame):
                    continue
                yield self._maybe_flatten(chunk)
        except FileNotFoundError as fnf:
            logger.error("File not found: %s", fp)
            raise DataReadError(f"File not found: {fp}") from fnf
        except ValueError as ve:
            logger.error("JSONL parsing error in %s: %s", fp, ve)
            raise DataReadError(f"JSONL parsing error in {fp}: {ve}") from ve
        except Exception as exc:
            logger.exception("Unexpected error streaming JSONL %s: %s", fp, exc)
            raise DataReadError(f"Failed to stream JSONL {fp}: {exc}") from exc

    def _stream_json_array(self, fp: Path, batch_size: int) -> Iterable[pd.DataFrame]:
        """
        Stream a large JSON array of objects using ``ijson`` in batches.

        Each yielded batch is a DataFrame constructed from a list of objects.
        """
        if self.engine != "ijson":
            # If not using ijson engine, fall back to pandas full-load (not a stream)
            logger.debug("Engine is not 'ijson'; falling back to batch read for %s", fp)
            yield self._read_file_batch(fp)
            return

        try:
            import ijson  # type: ignore
        except Exception as exc:
            logger.error("ijson is required to stream JSON arrays.")
            raise DependencyMissingError(
                "The 'ijson' package is required for streaming JSON arrays."
            ) from exc

        try:
            buffer: List[Dict[str, Any]] = []
            with fp.open("rb") as f:
                # 'item' iterates over array elements when root is a JSON array
                for obj in ijson.items(f, "item"):
                    buffer.append(obj)
                    if len(buffer) >= batch_size:
                        chunk = (
                            pd.json_normalize(buffer)
                            if self.flatten
                            else pd.DataFrame(buffer)
                        )
                        chunk = self._maybe_enrich_partitions(chunk, fp)
                        yield chunk
                        buffer.clear()
                # Flush remaining
                if buffer:
                    chunk = (
                        pd.json_normalize(buffer)
                        if self.flatten
                        else pd.DataFrame(buffer)
                    )
                    chunk = self._maybe_enrich_partitions(chunk, fp)
                    yield chunk
        except FileNotFoundError as fnf:
            logger.error("File not found: %s", fp)
            raise DataReadError(f"File not found: {fp}") from fnf
        except ijson.JSONError as je:  # type: ignore[attr-defined]
            logger.error("ijson parsing error in %s: %s", fp, je)
            raise DataReadError(f"ijson parsing error in {fp}: {je}") from je
        except Exception as exc:
            logger.exception("Unexpected error streaming JSON array %s: %s", fp, exc)
            raise DataReadError(f"Failed to stream JSON array {fp}: {exc}") from exc

    # ---------------------------------------------------------------------
    # Helpers: flattening, selection, partitions
    # ---------------------------------------------------------------------
    def _maybe_flatten(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten nested columns if ``flatten`` is enabled.
        """
        if not self.flatten or df.empty:
            return df

        # Detect nested dict/list columns to decide if normalization is needed
        def _has_nested(row_sample: pd.Series) -> bool:
            return any(isinstance(v, (dict, list)) for v in row_sample)

        try:
            if len(df) > 0 and _has_nested(df.iloc[0]):
                # Use json_normalize for nested fields
                normalized = pd.json_normalize(df.to_dict(orient="records"))
                return normalized
        except Exception as exc:
            logger.warning(
                "Flattening failed; continuing with original DataFrame. Reason: %s", exc
            )
        return df

    def _maybe_select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select only requested columns if ``select_columns`` is set.
        """
        if self.select_columns and not df.empty:
            missing = [c for c in self.select_columns if c not in df.columns]
            if missing:
                logger.warning(
                    "Some requested columns are missing and will be ignored: %s",
                    missing,
                )
            present = [c for c in self.select_columns if c in df.columns]
            if present:
                return df.loc[:, present]
        return df

    def _maybe_enrich_partitions(self, df: pd.DataFrame, fp: Path) -> pd.DataFrame:
        """
        Add partition columns derived from directory names like 'key=value'.
        """
        if df.empty:
            return df
        parts = self._extract_partitions(fp)
        if not parts:
            return df
        try:
            for k, v in parts.items():
                if k not in df.columns:
                    df[k] = v
        except Exception as exc:
            logger.warning("Partition enrichment failed for %s: %s", fp, exc)
        return df

    def _extract_partitions(self, fp: Path) -> Dict[str, str]:
        """
        Extract partition key-value pairs from ancestors of ``fp``.
        Example path: /data/year=2025/month=09/part-0001.json -> {'year': '2025', 'month': '09'}
        """
        parts: Dict[str, str] = {}
        try:
            for parent in fp.parents:
                name = parent.name
                if "=" in name:
                    key, _, val = name.partition("=")
                    if key and val:
                        parts[key] = val
        except Exception as exc:
            logger.debug("Partition parsing skipped for %s (%s)", fp, exc)
        return parts

    @staticmethod
    def example_usage() -> None:
        """
        Print detailed usage examples for JSONExtractor.

        Demonstrates different ways to use the class, covering:
        - Single and multi-file inputs
        - JSON vs JSON Lines (NDJSON)
        - Recursive discovery
        - Parallel extraction
        - Streaming with chunks
        - Out-of-core parsing with ijson
        - Flattening nested objects
        - Partition enrichment
        - Column selection
        """

        examples = dedent(
            """
        ============================================================
        JSONExtractor USAGE EXAMPLES
        ============================================================

        1. Extract data from a single JSON file
        ------------------------------------------------------------
        extractor = JSONExtractor("data.json")
        df, metadata = extractor.extract()
        print(metadata)

        2. Extract from a JSON Lines (NDJSON) file
        ------------------------------------------------------------
        extractor = JSONExtractor("data.jsonl", json_lines=True)
        df, metadata = extractor.extract()
        print(f"Extracted {metadata['num_records']} rows")

        3. Extract from multiple files (mixed JSON + JSONL)
        ------------------------------------------------------------
        extractor = JSONExtractor(["file1.json", "file2.jsonl"])
        df, metadata = extractor.extract()
        print(metadata)

        4. Extract all JSON/JSONL files from a folder
        ------------------------------------------------------------
        extractor = JSONExtractor("data_folder/")
        df, metadata = extractor.extract()
        print(f"Discovered {metadata['files_read']} files")

        5. Recursive discovery with partition enrichment
        ------------------------------------------------------------
        extractor = JSONExtractor(
            "data_partitioned/",
            recursive=True,
        )
        df, metadata = extractor.extract()
        print("Partition columns:", [c for c in df.columns if "=" not in c])

        6. Parallel extraction of multiple JSONL files
        ------------------------------------------------------------
        extractor = JSONExtractor("data_folder/", json_lines=True, parallel=True)
        df, metadata = extractor.extract()
        print(f"Parallel extraction complete: {metadata}")

        7. Streaming large JSONL with chunks (memory-efficient)
        ------------------------------------------------------------
        extractor = JSONExtractor("large.jsonl", json_lines=True, chunksize=100000)
        for chunk, chunk_metadata in extractor.stream_extract():
            print(f"Processed chunk with {chunk_metadata['num_records']} rows")

        8. Streaming a large JSON array with ijson (out-of-core)
        ------------------------------------------------------------
        extractor = JSONExtractor("large_array.json", engine="ijson", chunksize=50000)
        for chunk, chunk_metadata in extractor.stream_extract():
            print(f"Processed {chunk_metadata['num_records']} rows")

        9. Flattening nested JSON structures
        ------------------------------------------------------------
        extractor = JSONExtractor("nested.json", flatten=True)
        df, metadata = extractor.extract()
        print("Flattened columns:", df.columns.tolist())

        10. Selecting only specific columns
        ------------------------------------------------------------
        extractor = JSONExtractor("data.jsonl", json_lines=True, select_columns=["id", "name"])
        df, metadata = extractor.extract()
        print(df.head())

        ============================================================
        NOTES
        ============================================================
        - JSON vs JSONL is auto-detected unless forced with ``json_lines``.
        - ``engine="ijson"`` is recommended for very large JSON arrays.
        - Streaming mode yields (chunk, metadata) instead of loading all data.
        - Flattening expands nested JSON into normalized tabular form.
        - Partition enrichment extracts keys (e.g., year=2025) from folder paths.
        - Column selection lets you restrict to only required fields.
        - Parallel mode accelerates batch reading of many small/medium files.
        ============================================================
        """
        )
        print(examples)


class ExcelExtractor(FileExtractor):
    """
    Production-grade extractor for Excel files (.xlsx, .xls).

    This class provides comprehensive functionality for extracting data from one or more
    Excel files into pandas DataFrames. It supports single files, multiple files, and
    entire directory structures with recursive discovery. The extractor offers both
    batch (in-memory) and streaming modes for memory-efficient processing of large datasets.

    Key Features
    ------------
    - **Multi-file support**: Single file, multiple files, or recursive folder discovery
    - **Sheet selection**: Extract from specific sheets by name or index
    - **Streaming mode**: Memory-efficient chunked reading for large Excel files
    - **Parallel processing**: Accelerated multi-file ingestion with threading
    - **Schema validation**: Ensures consistency across multiple Excel files
    - **Flexible engines**: Support for openpyxl, xlrd, and other pandas Excel engines
    - **Rich metadata**: Comprehensive extraction statistics and schema information
    - **Error handling**: Robust error management with detailed logging

    Excel-Specific Capabilities
    ---------------------------
    - Multiple sheet extraction (specify sheet by name or index)
    - Date parsing and automatic type inference
    - Custom converters for complex data transformations
    - Support for merged cells and Excel-specific formatting
    - Engine selection based on file format (.xlsx vs .xls)

    Parameters
    ----------
    input_source : Union[str, List[str]]
        File path, folder path, or list of file/folder paths. Folders will be
        scanned for Excel files; set ``recursive=True`` to traverse subfolders.
    sheet_name : Union[str, int, List[Union[str, int]]], default=0
        Sheet(s) to extract. Can be sheet name (str), sheet index (int), or list
        of names/indices for multiple sheets. Default extracts first sheet.
    header : Union[int, List[int], None], default=0
        Row number(s) to use as column headers. Use None for no header.
    usecols : Optional[Union[str, List[str], List[int]]], default=None
        Column names, indices, or Excel column letters to extract (e.g., "A:E").
    skiprows : Optional[Union[List[int], int]], default=None
        Row numbers to skip (0-indexed) or number of rows to skip from top.
    dtype : Optional[Dict[str, Any]], default=None
        Data type for data or columns. E.g., {'a': np.float64, 'b': np.int32}.
    converters : Optional[Dict[str, callable]], default=None
        Dict of functions for converting values in certain columns.
    parse_dates : Optional[Union[bool, List[str]]], default=None
        If True, parse index. If list, parse specified columns as dates.
    date_format : Optional[callable], default=None
        Function to parse string dates. Deprecated, use pd.to_datetime instead.
    na_values : Optional[List[str]], default=None
        Additional strings to recognize as NA/NaN values.
    engine : str, default="openpyxl"
        Excel parsing engine. Options: 'openpyxl', 'xlrd', 'odf', 'pyxlsb'.
        'openpyxl' for .xlsx, 'xlrd' for .xls files.
    recursive : bool, default=False
        Whether to search for Excel files recursively in directories.
    parallel : bool, default=False
        Whether to read multiple files in parallel using ThreadPoolExecutor.
    max_workers : Optional[int], default=None
        Maximum number of worker threads for parallel processing.
    chunksize : Optional[int], default=None
        If set, simulate chunked reading by processing files in row chunks.
        Note: Excel doesn't support native chunking like CSV.
    validate_schema : bool, default=True
        Whether to validate schema compatibility across multiple files.
    strict_schema : bool, default=False
        If True, enforce exact data type matching. If False, allow compatible
        types (e.g., int64 and float64) across files.
    **kwargs : Any
        Additional keyword arguments passed to ``pandas.read_excel``.

    Raises
    ------
    DataSourceNotFoundError
        If no Excel files are found in the specified input source.
    DataReadError
        If files cannot be read due to corruption, access issues, or parsing errors.
    ExtractorError
        For general extraction failures or configuration errors.

    Notes
    -----
    - Excel files don't support native chunked reading like CSV files. The chunking
        is simulated by reading data in row-based chunks using skiprows and nrows.
    - For very large Excel files, consider converting to CSV or Parquet first.
    - The 'openpyxl' engine is recommended for .xlsx files, 'xlrd' for .xls files.
    - Schema validation compares column names and data types across multiple files.
    - Use streaming mode for memory-constrained environments with large datasets.

    Examples
    --------
    Basic single file extraction:

        >>> extractor = ExcelExtractor("sales_data.xlsx", sheet_name="Q1_Sales")
        >>> df, metadata = extractor.extract()
        >>> print(f"Extracted {metadata['num_records']} records")

    Multi-file extraction with schema validation:

        >>> extractor = ExcelExtractor(
        ...     ["file1.xlsx", "file2.xlsx"],
        ...     validate_schema=True,
        ...     parallel=True
        ... )
        >>> df, metadata = extractor.extract()

    Streaming large Excel file:

        >>> extractor = ExcelExtractor("large_file.xlsx", chunksize=10000)
        >>> for chunk, chunk_meta in extractor.extract():
        ...     process_chunk(chunk)

    Directory-based extraction:

        >>> extractor = ExcelExtractor(
        ...     "reports/",
        ...     recursive=True,
        ...     sheet_name="Summary",
        ...     usecols="A:F"
        ... )
        >>> df, metadata = extractor.extract()
    """

    def __init__(
        self,
        input_source: Union[str, List[str]],
        sheet_name: Union[str, int, List[Union[str, int]]] = 0,
        header: Union[int, List[int], None] = 0,
        usecols: Optional[Union[str, List[str], List[int]]] = None,
        skiprows: Optional[Union[List[int], int]] = None,
        dtype: Optional[Dict[str, Any]] = None,
        converters: Optional[Dict[str, callable]] = None,
        parse_dates: Optional[Union[bool, List[str]]] = None,
        date_format: Optional[callable] = None,
        na_values: Optional[List[str]] = None,
        engine: str = "openpyxl",
        recursive: bool = False,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        chunksize: Optional[int] = None,
        validate_schema: bool = True,
        strict_schema: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ExcelExtractor with input source and parsing options.

        Args
        ----
        input_source : Union[str, List[str]]
            File path, folder path, or list of file/folder paths to process.
        sheet_name : Union[str, int, List[Union[str, int]]], default=0
            Sheet(s) to extract. Can be sheet name, index, or list for multiple sheets.
        header : Union[int, List[int], None], default=0
            Row number(s) to use as column headers. None for no header.
        usecols : Optional[Union[str, List[str], List[int]]], default=None
            Columns to extract (names, indices, or Excel range like "A:E").
        skiprows : Optional[Union[List[int], int]], default=None
            Rows to skip from the beginning of the file.
        dtype : Optional[Dict[str, Any]], default=None
            Data type specifications for columns.
        converters : Optional[Dict[str, callable]], default=None
            Column-specific converter functions.
        parse_dates : Optional[Union[bool, List[str]]], default=None
            Date parsing configuration.
        date_format : Optional[callable], default=None
            Custom date parsing function.
        na_values : Optional[List[str]], default=None
            Additional NA value representations.
        engine : str, default="openpyxl"
            Excel parsing engine to use.
        recursive : bool, default=False
            Whether to search directories recursively.
        parallel : bool, default=False
            Whether to process multiple files in parallel.
        max_workers : Optional[int], default=None
            Maximum threads for parallel processing.
        chunksize : Optional[int], default=None
            Chunk size for simulated streaming (Excel limitation).
        validate_schema : bool, default=True
            Whether to validate schema compatibility across files.
        strict_schema : bool, default=False
            Whether to enforce strict data type matching.
        **kwargs : Any
            Additional arguments for pandas.read_excel.

        Raises
        ------
        DataSourceNotFoundError
            If no valid Excel files are found in input_source.
        ValueError
            If configuration parameters are invalid.
        """
        # Discover Excel files from given input first
        self.filepaths: List[str] = self._discover_files(input_source)

        # For single file, call parent init. For multiple files, set filepath to first file
        if len(self.filepaths) == 1:
            super().__init__(self.filepaths[0])
        else:
            # For multiple files, we bypass parent validation and set filepath manually
            self.filepath = self.filepaths[0]  # Use first file as representative

        self.input_source = input_source
        self.sheet_name = sheet_name
        self.header = header
        self.usecols = usecols
        self.skiprows = skiprows
        self.dtype = dtype
        self.converters = converters
        self.parse_dates = parse_dates
        self.date_format = date_format
        self.na_values = na_values
        self.engine = engine
        self.recursive = recursive
        self.parallel = parallel
        self.max_workers = max_workers
        self.chunksize = chunksize
        self.validate_schema = validate_schema
        self.strict_schema = strict_schema
        self.read_excel_kwargs = kwargs

        # Schema validation attributes
        self.master_schema: Optional[Dict[str, Any]] = None
        self.schema_validated: bool = False

        logger.info(
            "ExcelExtractor initialized with %d file(s), engine=%s, parallel=%s, chunksize=%s",
            len(self.filepaths),
            self.engine,
            self.parallel,
            str(self.chunksize),
        )

    def _discover_files(self, source: Union[str, List[str]]) -> List[str]:
        """
        Discover Excel files from the given input source.

        Args
        ----
        source : Union[str, List[str]]
            File path, folder path, or list of paths to search.

        Returns
        -------
        List[str]
            List of absolute file paths to discovered Excel files.

        Raises
        ------
        DataSourceNotFoundError
            If no Excel files are found in the given source.
        """
        filepaths: List[str] = []

        # Normalize to list for uniform iteration
        if isinstance(source, str):
            source = [source]

        for path in source:
            p = Path(path)

            if p.is_file() and p.suffix.lower() in [".xls", ".xlsx", ".xlsm"]:
                filepaths.append(str(p.resolve()))

            elif p.is_dir():
                # Use recursive or non-recursive globbing
                pattern = "**/*.xl*" if self.recursive else "*.xl*"
                for file in p.glob(pattern):
                    if file.suffix.lower() in [".xls", ".xlsx", ".xlsm"]:
                        filepaths.append(str(file.resolve()))

            else:
                logger.warning("Invalid source path skipped: %s", path)

        if not filepaths:
            raise DataSourceNotFoundError(
                f"No Excel files found in input source: {source}"
            )

        logger.info("Discovered %d Excel files.", len(filepaths))
        return filepaths

    def _extract_schema(self, filepath: str) -> Dict[str, Any]:
        """
        Extract schema information from an Excel file.

        Args
        ----
        filepath : str
            Path to the Excel file.

        Returns
        -------
        Dict[str, Any]
            Schema dictionary containing:
            - columns: list of column names
            - dtypes: dictionary of column data types
            - num_columns: number of columns
            - filepath: source file path
            - sheet_name: name of the extracted sheet

        Raises
        ------
        DataReadError
            If the file cannot be read for schema extraction.
        """
        try:
            # Read just the first few rows to extract schema
            read_params: Dict[str, Any] = {
                "sheet_name": self.sheet_name,
                "header": self.header,
                "usecols": self.usecols,
                "skiprows": self.skiprows,
                "dtype": self.dtype,
                "converters": self.converters,
                "parse_dates": self.parse_dates,
                "date_format": self.date_format,
                "na_values": self.na_values,
                "engine": self.engine,
                "nrows": 100,  # Read only first 100 rows for schema
                **self.read_excel_kwargs,
            }

            logger.debug("Extracting schema from Excel file: %s", filepath)
            sample_df = pd.read_excel(filepath, **read_params)

            schema = {
                "columns": sample_df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in sample_df.dtypes.items()},
                "num_columns": len(sample_df.columns),
                "filepath": filepath,
                "sheet_name": self.sheet_name,
            }

            logger.debug(
                "Schema extracted from %s: %d columns", filepath, schema["num_columns"]
            )
            return schema

        except Exception as e:
            logger.error("Failed to extract schema from %s: %s", filepath, e)
            raise DataReadError(f"Schema extraction failed for {filepath}: {e}") from e

    def _are_dtypes_compatible(self, dtype1: str, dtype2: str) -> bool:
        """
        Check if two pandas data types are compatible for concatenation.

        This method uses the same compatibility logic as CSVExtractor but is
        adapted for Excel-specific data types.

        Args
        ----
        dtype1 : str
            First data type as string.
        dtype2 : str
            Second data type as string.

        Returns
        -------
        bool
            True if data types are compatible, False otherwise.
        """
        if self.strict_schema:
            return dtype1 == dtype2

        # Define compatible data type groups
        numeric_types = {
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float16",
            "float32",
            "float64",
            "Float32",
            "Float64",
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "UInt8",
            "UInt16",
            "UInt32",
            "UInt64",
        }

        string_types = {"object", "string", "category"}

        datetime_types = {
            "datetime64[ns]",
            "datetime64[ns, UTC]",
            "period[D]",
            "timedelta64[ns]",
        }

        boolean_types = {"bool", "boolean"}

        # Check if both types belong to the same compatible group
        if dtype1 in numeric_types and dtype2 in numeric_types:
            return True
        elif dtype1 in string_types and dtype2 in string_types:
            return True
        elif dtype1 in datetime_types and dtype2 in datetime_types:
            return True
        elif dtype1 in boolean_types and dtype2 in boolean_types:
            return True
        else:
            return dtype1 == dtype2

    def _validate_schema_compatibility(
        self, schemas: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate schema compatibility across multiple Excel files.

        Args
        ----
        schemas : List[Dict[str, Any]]
            List of schema dictionaries from multiple files.

        Returns
        -------
        Dict[str, Any]
            Master schema dictionary with unified column information.

        Raises
        ------
        DataReadError
            If schemas are incompatible across files.
        """
        if not schemas:
            raise DataReadError("No schemas provided for validation.")

        if len(schemas) == 1:
            logger.info("Single file detected, schema validation skipped.")
            return schemas[0]

        logger.info("Validating schema compatibility across %d files.", len(schemas))

        # Use first file as reference schema
        master_schema = schemas[0].copy()
        master_columns = set(master_schema["columns"])
        incompatible_files = []

        for i, schema in enumerate(schemas[1:], 1):
            current_columns = set(schema["columns"])
            filepath = schema["filepath"]

            # Check column names
            if master_columns != current_columns:
                missing_in_current = master_columns - current_columns
                extra_in_current = current_columns - master_columns

                error_msg = f"Schema mismatch in {filepath}:"
                if missing_in_current:
                    error_msg += f" Missing columns: {list(missing_in_current)}."
                if extra_in_current:
                    error_msg += f" Extra columns: {list(extra_in_current)}."

                incompatible_files.append(error_msg)
                continue

            # Check data types for each column
            for col in master_schema["columns"]:
                master_dtype = master_schema["dtypes"][col]
                current_dtype = schema["dtypes"][col]

                if not self._are_dtypes_compatible(master_dtype, current_dtype):
                    error_msg = (
                        f"Data type mismatch in {filepath} for column '{col}': "
                        f"expected {master_dtype}, got {current_dtype}."
                    )
                    incompatible_files.append(error_msg)

        if incompatible_files:
            error_summary = "Schema validation failed:\n" + "\n".join(
                incompatible_files
            )
            logger.error("Schema validation failed: %s", error_summary)
            raise DataReadError(error_summary)

        logger.info("Schema validation successful: All files have compatible schemas.")
        return master_schema

    def _read_file(self, filepath: str) -> pd.DataFrame:
        """
        Read a single Excel file into a DataFrame.

        Args
        ----
        filepath : str
            Path to the Excel file.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the file's contents.

        Raises
        ------
        DataReadError
            If the file cannot be read due to missing file, parse error,
            or unexpected issues.
        """
        try:
            # Prepare read_excel parameters
            read_params: Dict[str, Any] = {
                "sheet_name": self.sheet_name,
                "header": self.header,
                "usecols": self.usecols,
                "skiprows": self.skiprows,
                "dtype": self.dtype,
                "converters": self.converters,
                "parse_dates": self.parse_dates,
                "date_format": self.date_format,
                "na_values": self.na_values,
                "engine": self.engine,
                **self.read_excel_kwargs,
            }

            logger.debug("Reading Excel file: %s with params %s", filepath, read_params)
            df = pd.read_excel(filepath, **read_params)
            return df

        except FileNotFoundError as fnf_error:
            logger.error("Excel file not found: %s", filepath)
            raise DataReadError(f"File not found: {filepath}") from fnf_error

        except PermissionError as perm_error:
            logger.error("Permission denied accessing Excel file: %s", filepath)
            raise DataReadError(f"Permission denied: {filepath}") from perm_error

        except Exception as e:
            logger.exception(
                "Unexpected error during Excel extraction from %s", filepath
            )
            raise DataReadError(f"Failed to read {filepath}: {e}") from e

    def extract(
        self,
    ) -> Union[
        Tuple[pd.DataFrame, Dict[str, Any]],
        Generator[Tuple[pd.DataFrame, Dict[str, Any]], None, None],
    ]:
        """
        Extract data from one or more Excel files.

        Returns
        -------
        Union[Tuple[pandas.DataFrame, Dict[str, Any]],
                Generator[Tuple[pandas.DataFrame, Dict[str, Any]], None, None]]
            - If `chunksize` is None:
                Combined DataFrame containing extracted Excel data and metadata.
            - If `chunksize` is set:
                Generator yielding (DataFrame chunk, metadata) tuples.

        Raises
        ------
        DataReadError
            If any file cannot be read successfully.
        """
        if self.chunksize:
            logger.info(
                "Extracting Excel files in streaming mode with chunksize=%s",
                self.chunksize,
            )
            return self._extract_streaming()
        else:
            logger.info(
                "Extracting Excel files in regular mode from %d files",
                len(self.filepaths),
            )
            return self._extract_regular()

    def _extract_regular(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extract data in regular (non-streaming) mode.

        Reads all discovered Excel files into memory and concatenates them
        into a single DataFrame. Performs schema validation if enabled
        and multiple files are being processed.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any]]
            Combined DataFrame and metadata.
        """
        # Perform schema validation if enabled and multiple files exist
        if (
            self.validate_schema
            and len(self.filepaths) > 1
            and not self.schema_validated
        ):
            logger.info(
                "Performing schema validation for %d files.", len(self.filepaths)
            )
            schemas = []
            for fp in self.filepaths:
                schema = self._extract_schema(fp)
                schemas.append(schema)

            # Validate and store master schema
            self.master_schema = self._validate_schema_compatibility(schemas)
            self.schema_validated = True
            logger.info("Schema validation completed successfully.")

        dfs: List[pd.DataFrame] = []

        if self.parallel:
            logger.info("Reading files in parallel using ThreadPoolExecutor.")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._read_file, fp): fp for fp in self.filepaths
                }
                for future in as_completed(futures):
                    fp = futures[future]
                    try:
                        df = future.result()
                        dfs.append(df)
                        logger.info("Successfully read file: %s", fp)
                    except Exception as e:
                        logger.error("Failed to read file %s: %s", fp, e)
                        raise
        else:
            for fp in self.filepaths:
                df = self._read_file(fp)
                dfs.append(df)
                logger.info("Successfully read file: %s", fp)

        if not dfs:
            raise DataReadError("No data extracted from the provided Excel files.")

        combined_df = pd.concat(dfs, ignore_index=True)
        metadata = self._create_metadata(combined_df)

        # Add schema information to metadata if validation was performed
        if self.master_schema:
            metadata["schema_validation"] = {
                "validated": True,
                "master_schema": self.master_schema,
                "strict_mode": self.strict_schema,
                "num_files_validated": len(self.filepaths),
            }
        else:
            metadata["schema_validation"] = {
                "validated": False,
                "reason": "Single file or validation disabled",
            }

        # Add Excel-specific metadata
        metadata.update(
            {
                "files_read": len(self.filepaths),
                "engine": self.engine,
                "sheet_name": self.sheet_name,
                "parallel": self.parallel,
                "recursive": self.recursive,
            }
        )

        logger.info(
            "Excel extraction completed: %d records combined from %d files.",
            metadata["num_records"],
            len(self.filepaths),
        )
        return combined_df, metadata

    def _extract_streaming(
        self,
    ) -> Generator[Tuple[pd.DataFrame, Dict[str, Any]], None, None]:
        """
        Extract data in streaming mode using simulated chunks.

        Iterates through each file and yields chunks of data with metadata.
        Performs schema validation before streaming if enabled.

        Note: Excel doesn't support native chunked reading like CSV. This method
        simulates chunking by reading data in row-based segments using skiprows
        and nrows parameters.

        Yields
        ------
        Generator[Tuple[pd.DataFrame, Dict[str, Any]], None, None]
            Each item is a tuple of (chunk DataFrame, metadata dict).
        """
        # Perform schema validation if enabled and multiple files exist
        if (
            self.validate_schema
            and len(self.filepaths) > 1
            and not self.schema_validated
        ):
            logger.info(
                "Performing schema validation for %d files before streaming.",
                len(self.filepaths),
            )
            schemas = []
            for fp in self.filepaths:
                schema = self._extract_schema(fp)
                schemas.append(schema)

            # Validate and store master schema
            self.master_schema = self._validate_schema_compatibility(schemas)
            self.schema_validated = True
            logger.info("Schema validation completed successfully.")

        for fp in self.filepaths:
            logger.info("Streaming file: %s", fp)

            # Simulate chunked reading for Excel files
            row_offset = 0
            chunk_count = 0

            while True:
                try:
                    read_params: Dict[str, Any] = {
                        "sheet_name": self.sheet_name,
                        "header": self.header if row_offset == 0 else None,
                        "usecols": self.usecols,
                        "skiprows": (self.skiprows or 0)
                        + row_offset
                        + (1 if row_offset > 0 and self.header is not None else 0),
                        "nrows": self.chunksize,
                        "dtype": self.dtype,
                        "converters": self.converters,
                        "parse_dates": self.parse_dates,
                        "date_format": self.date_format,
                        "na_values": self.na_values,
                        "engine": self.engine,
                        **self.read_excel_kwargs,
                    }

                    chunk = pd.read_excel(fp, **read_params)

                    # If chunk is empty, we've reached the end of the file
                    if chunk.empty:
                        break

                    # For subsequent chunks, set column names manually if needed
                    if (
                        row_offset > 0
                        and self.header is not None
                        and hasattr(self, "_chunk_columns")
                    ):
                        chunk.columns = self._chunk_columns
                    elif row_offset == 0 and self.header is not None:
                        # Store column names for subsequent chunks
                        self._chunk_columns = chunk.columns.tolist()

                    logger.debug(
                        "Yielding chunk %d of size %d from %s",
                        chunk_count,
                        len(chunk),
                        fp,
                    )
                    chunk_metadata = self._create_metadata(chunk)

                    # Add schema information to metadata if validation was performed
                    if self.master_schema:
                        chunk_metadata["schema_validation"] = {
                            "validated": True,
                            "master_schema": self.master_schema,
                            "strict_mode": self.strict_schema,
                            "num_files_validated": len(self.filepaths),
                        }
                    else:
                        chunk_metadata["schema_validation"] = {
                            "validated": False,
                            "reason": "Single file or validation disabled",
                        }

                    # Add chunk-specific metadata
                    chunk_metadata.update(
                        {
                            "chunk_number": chunk_count,
                            "row_offset": row_offset,
                            "engine": self.engine,
                            "sheet_name": self.sheet_name,
                        }
                    )

                    yield chunk, chunk_metadata

                    row_offset += len(chunk)
                    chunk_count += 1

                except Exception as e:
                    logger.error(
                        "Error reading chunk at offset %d from %s: %s",
                        row_offset,
                        fp,
                        e,
                    )
                    raise DataReadError(
                        f"Failed to read chunk from {fp} at offset {row_offset}: {e}"
                    ) from e

    @staticmethod
    def example_usage() -> None:
        """
        Print detailed usage examples for ExcelExtractor.

        This demonstrates different ways to use the class, covering:
        - Single and multi-file extraction
        - Sheet selection and column filtering
        - Recursive directory discovery
        - Parallel processing
        - Streaming mode for large files
        - Schema validation across multiple files
        - Excel-specific features (engines, date parsing, etc.)
        """

        examples = dedent(
            """
        ============================================================
        ExcelExtractor USAGE EXAMPLES
        ============================================================

        1. Extract data from a single Excel file
        ------------------------------------------------------------
        extractor = ExcelExtractor("data.xlsx", sheet_name="Sheet1")
        df, metadata = extractor.extract()
        print(f"Extracted {metadata['num_records']} records")

        2. Extract from multiple Excel files (batch mode)
        ------------------------------------------------------------
        extractor = ExcelExtractor(["file1.xlsx", "file2.xlsx"])
        df, metadata = extractor.extract()
        print(f"Combined {metadata['num_records']} rows from {metadata['files_read']} files")

        3. Extract all Excel files from a folder
        ------------------------------------------------------------
        extractor = ExcelExtractor("data_folder/")
        df, metadata = extractor.extract()
        print(f"Found {metadata['files_read']} Excel files")

        4. Extract Excel files from a folder (recursive search)
        ------------------------------------------------------------
        extractor = ExcelExtractor("data_folder/", recursive=True)
        df, metadata = extractor.extract()
        print(f"Recursive search found {len(df)} rows")

        5. Extract multiple Excel files in parallel
        ------------------------------------------------------------
        extractor = ExcelExtractor("data_folder/", parallel=True, max_workers=4)
        df, metadata = extractor.extract()
        print(f"Parallel extraction complete: {metadata}")

        6. Streaming large Excel file with chunks (memory-efficient)
        ------------------------------------------------------------
        extractor = ExcelExtractor("large_file.xlsx", chunksize=10000)
        for chunk, chunk_metadata in extractor.extract():
            print(f"Processed chunk {chunk_metadata['chunk_number']} with {chunk_metadata['num_records']} rows")

        7. Advanced Excel features
        ------------------------------------------------------------
        # a) Specific sheet and column selection
        extractor = ExcelExtractor(
            "report.xlsx",
            sheet_name="Summary",
            usecols="A:F",  # Extract only columns A through F
            skiprows=[0, 1],  # Skip first two rows
            header=2  # Use row 2 as header
        )
        df, metadata = extractor.extract()

        # b) Date parsing and custom converters
        extractor = ExcelExtractor(
            "data.xlsx",
            parse_dates=["date_column"],
            converters={"id": str, "amount": float},
            na_values=["N/A", "NULL", ""]
        )
        df, metadata = extractor.extract()

        # c) Engine selection for different file types
        extractor = ExcelExtractor("old_file.xls", engine="xlrd")  # For .xls files
        df, metadata = extractor.extract()

        8. Schema validation across multiple files
        ------------------------------------------------------------
        # a) Normal schema validation (compatible types allowed)
        extractor = ExcelExtractor(
            ["file1.xlsx", "file2.xlsx"],
            validate_schema=True,
            sheet_name="Data"
        )
        df, metadata = extractor.extract()
        print(metadata["schema_validation"])

        # b) Strict schema validation (exact type matching)
        extractor = ExcelExtractor(
            ["file1.xlsx", "file2.xlsx"],
            validate_schema=True,
            strict_schema=True
        )
        df, metadata = extractor.extract()

        # c) Disable schema validation
        extractor = ExcelExtractor(
            ["file1.xlsx", "file2.xlsx"],
            validate_schema=False
        )
        df, metadata = extractor.extract()

        9. Working with multiple sheets
        ------------------------------------------------------------
        # Note: For multiple sheets, you'll need to create separate extractors
        # or modify the implementation to handle multiple sheets per file
        sheets_to_extract = ["Summary", "Details", "Totals"]
        all_data = {}
        
        for sheet in sheets_to_extract:
            extractor = ExcelExtractor("workbook.xlsx", sheet_name=sheet)
            df, metadata = extractor.extract()
            all_data[sheet] = df

        ============================================================
        NOTES
        ============================================================
        - Excel chunking is simulated using skiprows + nrows since Excel doesn't
            support native chunked reading like CSV files.
        - For very large Excel files, consider converting to CSV or Parquet format.
        - Use 'openpyxl' engine for .xlsx files, 'xlrd' for .xls files.
        - Schema validation ensures column names and data types are consistent.
        - Parallel processing accelerates multi-file extraction.
        - Streaming mode helps with memory management for large datasets.
        - Date parsing and converters provide flexibility for data transformation.
        ============================================================
        """
        )
        print(examples)


class ParquetExtractor(FileExtractor):
    """
    Production-grade extractor for Parquet files (.parquet).

    This class provides comprehensive functionality for extracting data from one or more
    Parquet files into pandas DataFrames. It supports single files, multiple files, and
    entire directory structures with recursive discovery. The extractor offers both
    batch (in-memory) and streaming modes for memory-efficient processing of large datasets.

    Key Features
    ------------
    - **Multi-file support**: Single file, multiple files, or recursive folder discovery
    - **Partition discovery**: Read partitioned Parquet datasets (Hive-style or directory-based)
    - **Streaming mode**: Memory-efficient row group-based streaming for large Parquet files
    - **Parallel processing**: Accelerated multi-file ingestion with ThreadPoolExecutor
    - **Schema validation**: Ensures consistency across multiple Parquet files
    - **Flexible engines**: Support for pyarrow and fastparquet backends with auto-detection
    - **Rich metadata**: Comprehensive extraction statistics and schema information
    - **Error handling**: Robust error management with detailed logging and framework exceptions

    Parquet-Specific Capabilities
    -----------------------------
    - Native columnar reads (select subset of columns efficiently)
    - Predicate pushdown for filtered reads when supported by engine
    - Automatic compression and encoding detection (Snappy, Gzip, LZ4, etc.)
    - Partitioned dataset discovery and ingestion with partition key enrichment
    - Row group-based streaming for memory-efficient processing
    - Engine selection based on performance requirements or compatibility needs
    - Automatic schema inference and validation across multiple files

    Parameters
    ----------
    input_source : Union[str, List[str]]
        File path, folder path, or list of file/folder paths. Folders will be
        scanned for Parquet files; set ``recursive=True`` to traverse subfolders.
    columns : Optional[List[str]], default=None
        Subset of columns to read from the Parquet file(s). Enables efficient
        columnar reads by only loading required columns into memory.
    filters : Optional[List[Tuple]], default=None
        Row filter predicates for partitioned datasets. Format:
        [('column', 'operator', value)] where operator can be '=', '!=', '<',
        '<=', '>', '>=', 'in', 'not in'. E.g., [('year', '=', 2023), ('month', 'in', [1, 2, 3])].
    engine : str, default="pyarrow"
        Parquet parsing engine. Options:
        - 'pyarrow': Fast, feature-complete, recommended for most use cases
        - 'fastparquet': Alternative engine, may be faster for some operations
        - 'auto': Automatically select best available engine
    recursive : bool, default=False
        Whether to search for Parquet files recursively in directories.
        Useful for processing partitioned datasets with nested folder structures.
    parallel : bool, default=False
        Whether to read multiple files in parallel using ThreadPoolExecutor.
        Significantly speeds up processing when dealing with many small-medium files.
    max_workers : Optional[int], default=None
        Maximum number of worker threads for parallel processing. If None,
        defaults to ThreadPoolExecutor's default (min(32, os.cpu_count() + 4)).
    chunksize : Optional[int], default=None
        Row group-based streaming configuration. If set, enables streaming mode
        where data is yielded in chunks based on Parquet row groups.
    validate_schema : bool, default=True
        Whether to validate schema compatibility across multiple files.
        Ensures consistent column names and compatible data types.
    strict_schema : bool, default=False
        If True, enforce exact data type matching across files. If False, allow
        compatible types (e.g., int64 and float64, object and string).
    partition_columns : Optional[List[str]], default=None
        List of column names that are used for partitioning. These columns
        will be extracted from directory names in Hive-style partitioned datasets.
    use_nullable_dtypes : bool, default=False
        Whether to use pandas nullable data types (Int64, Float64, etc.) instead
        of traditional NumPy dtypes. Useful for better handling of missing values.
    **kwargs : Any
        Additional keyword arguments passed to ``pandas.read_parquet``.
        Common options include:
        - use_threads: bool, enable multi-threading within pyarrow
        - memory_map: bool, use memory mapping for better performance

    Raises
    ------
    DataSourceNotFoundError
        If no Parquet files are found in the specified input source.
    DataReadError
        If files cannot be read due to corruption, access issues, parsing errors,
        or schema incompatibilities.
    DependencyMissingError
        If the specified engine (pyarrow/fastparquet) is not installed.
    ExtractorError
        For general extraction failures or configuration errors.

    Notes
    -----
    - Parquet format supports efficient columnar reads and partition pruning.
    - pyarrow is generally faster and more feature-complete than fastparquet.
    - Schema validation compares column names and data types across multiple files.
    - Streaming mode leverages Parquet's row group structure for memory efficiency.
    - For very large datasets, consider using filters to reduce data transfer.
    - Partitioned datasets benefit from the recursive discovery and partition enrichment.

    Examples
    --------
    Basic single file extraction:

        >>> extractor = ParquetExtractor("sales_data.parquet")
        >>> df, metadata = extractor.extract()
        >>> print(f"Extracted {metadata['num_records']} records")

    Multi-file extraction with schema validation:

        >>> extractor = ParquetExtractor(
        ...     ["file1.parquet", "file2.parquet"],
        ...     validate_schema=True,
        ...     parallel=True
        ... )
        >>> df, metadata = extractor.extract()

    Streaming large Parquet file with row group chunks:

        >>> extractor = ParquetExtractor("large_file.parquet", chunksize=True)
        >>> for chunk, chunk_meta in extractor.extract():
        ...     process_chunk(chunk)

    Directory-based partitioned dataset extraction:

        >>> extractor = ParquetExtractor(
        ...     "partitioned_dataset/",
        ...     recursive=True,
        ...     filters=[("year", "=", 2023), ("month", "in", [1, 2])],
        ...     columns=["id", "value", "timestamp"]
        ... )
        >>> df, metadata = extractor.extract()

    Column subset with engine selection:

        >>> extractor = ParquetExtractor(
        ...     "data.parquet",
        ...     columns=["customer_id", "purchase_amount"],
        ...     engine="pyarrow",
        ...     use_nullable_dtypes=True
        ... )
        >>> df, metadata = extractor.extract()
    """

    def __init__(
        self,
        input_source: Union[str, List[str]],
        columns: Optional[List[str]] = None,
        filters: Optional[List[Tuple]] = None,
        engine: str = "pyarrow",
        recursive: bool = False,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        chunksize: Optional[bool] = None,
        validate_schema: bool = True,
        strict_schema: bool = False,
        partition_columns: Optional[List[str]] = None,
        use_nullable_dtypes: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ParquetExtractor with input source and parsing options.

        Args
        ----
        input_source : Union[str, List[str]]
            File path, folder path, or list of file/folder paths to process.
        columns : Optional[List[str]], default=None
            Subset of columns to read for efficient columnar access.
        filters : Optional[List[Tuple]], default=None
            Row filter predicates for partitioned datasets.
        engine : str, default="pyarrow"
            Parquet parsing engine ('pyarrow', 'fastparquet', 'auto').
        recursive : bool, default=False
            Whether to search directories recursively.
        parallel : bool, default=False
            Whether to process multiple files in parallel.
        max_workers : Optional[int], default=None
            Maximum threads for parallel processing.
        chunksize : Optional[bool], default=None
            Enable row group-based streaming if True.
        validate_schema : bool, default=True
            Whether to validate schema compatibility across files.
        strict_schema : bool, default=False
            Whether to enforce strict data type matching.
        partition_columns : Optional[List[str]], default=None
            Partition column names for Hive-style datasets.
        use_nullable_dtypes : bool, default=False
            Whether to use pandas nullable data types.
        **kwargs : Any
            Additional arguments for pandas.read_parquet.

        Raises
        ------
        DataSourceNotFoundError
            If no valid Parquet files are found in input_source.
        DependencyMissingError
            If the specified engine is not available.
        ValueError
            If configuration parameters are invalid.
        """
        # Discover Parquet files from given input first
        self.filepaths: List[str] = self._discover_files(input_source)

        # For single file, call parent init. For multiple files, set filepath to first file
        if len(self.filepaths) == 1:
            super().__init__(self.filepaths[0])
        else:
            # For multiple files, we bypass parent validation and set filepath manually
            self.filepath = self.filepaths[0]  # Use first file as representative

        self.input_source = input_source
        self.columns = columns
        self.filters = filters
        self.engine = engine
        self.recursive = recursive
        self.parallel = parallel
        self.max_workers = max_workers
        self.chunksize = chunksize
        self.validate_schema = validate_schema
        self.strict_schema = strict_schema
        self.partition_columns = partition_columns
        self.use_nullable_dtypes = use_nullable_dtypes
        self.read_parquet_kwargs = kwargs

        # Schema validation attributes
        self.master_schema: Optional[Dict[str, Any]] = None
        self.schema_validated: bool = False

        # Validate engine dependencies
        self._validate_engine_dependencies()

        logger.info(
            "ParquetExtractor initialized with %d file(s), engine=%s, parallel=%s, chunksize=%s",
            len(self.filepaths),
            self.engine,
            self.parallel,
            str(self.chunksize),
        )

    def _discover_files(self, source: Union[str, List[str]]) -> List[str]:
        """
        Discover Parquet files from the given input source.

        Args
        ----
        source : Union[str, List[str]]
            File path, folder path, or list of paths to search.

        Returns
        -------
        List[str]
            List of absolute file paths to discovered Parquet files.

        Raises
        ------
        DataSourceNotFoundError
            If no Parquet files are found in the given source.
        """
        filepaths: List[str] = []

        # Normalize to list for uniform iteration
        if isinstance(source, str):
            source = [source]

        for path in source:
            p = Path(path)

            if p.is_file() and p.suffix.lower() == ".parquet":
                filepaths.append(str(p.resolve()))

            elif p.is_dir():
                # Use recursive or non-recursive globbing
                pattern = "**/*.parquet" if self.recursive else "*.parquet"
                for file in p.glob(pattern):
                    filepaths.append(str(file.resolve()))

            else:
                logger.warning("Invalid source path skipped: %s", path)

        if not filepaths:
            raise DataSourceNotFoundError(
                f"No Parquet files found in input source: {source}"
            )

        logger.info("Discovered %d Parquet files.", len(filepaths))
        return filepaths

    def _validate_engine_dependencies(self) -> None:
        """
        Validate that the specified Parquet engine is available.

        Raises
        ------
        DependencyMissingError
            If the specified engine is not installed or available.
        """
        if self.engine == "auto":
            # Try pyarrow first, then fastparquet
            try:
                import pyarrow  # noqa: F401

                self.engine = "pyarrow"
                logger.debug("Auto-selected pyarrow engine")
                return
            except ImportError:
                pass

            try:
                import fastparquet  # noqa: F401

                self.engine = "fastparquet"
                logger.debug("Auto-selected fastparquet engine")
                return
            except ImportError:
                pass

            raise DependencyMissingError(
                "No Parquet engine available. Please install 'pyarrow' or 'fastparquet'."
            )

        elif self.engine == "pyarrow":
            try:
                import pyarrow  # noqa: F401

                logger.debug("Using pyarrow engine")
            except ImportError as exc:
                raise DependencyMissingError(
                    "pyarrow is required for engine='pyarrow'. Install with: pip install pyarrow"
                ) from exc

        elif self.engine == "fastparquet":
            try:
                import fastparquet  # noqa: F401

                logger.debug("Using fastparquet engine")
            except ImportError as exc:
                raise DependencyMissingError(
                    "fastparquet is required for engine='fastparquet'. Install with: pip install fastparquet"
                ) from exc

        else:
            raise ValueError(
                f"Unsupported engine '{self.engine}'. Choose from: 'pyarrow', 'fastparquet', 'auto'."
            )

    def _extract_schema(self, filepath: str) -> Dict[str, Any]:
        """
        Extract schema information from a Parquet file.

        Args
        ----
        filepath : str
            Path to the Parquet file.

        Returns
        -------
        Dict[str, Any]
            Schema dictionary containing:
            - columns: list of column names
            - dtypes: dictionary of column data types
            - num_columns: number of columns
            - filepath: source file path
            - parquet_metadata: additional Parquet-specific information

        Raises
        ------
        DataReadError
            If the file cannot be read for schema extraction.
        """
        try:
            # Read just a small sample to extract schema efficiently
            read_params: Dict[str, Any] = {
                "columns": self.columns,
                "engine": self.engine,
                "use_nullable_dtypes": self.use_nullable_dtypes,
                **self.read_parquet_kwargs,
            }

            logger.debug("Extracting schema from Parquet file: %s", filepath)

            # For schema extraction, we'll read just the first row group if using pyarrow
            if self.engine == "pyarrow":
                try:
                    import pyarrow.parquet as pq

                    parquet_file = pq.ParquetFile(filepath)
                    # Read first row group for schema
                    table = parquet_file.read_row_group(0, columns=self.columns)
                    sample_df = table.to_pandas()

                    # Extract additional Parquet metadata
                    try:
                        # Try to get compression info
                        compression = "unknown"
                        if parquet_file.schema_arrow.metadata:
                            metadata_dict = dict(parquet_file.schema_arrow.metadata)
                            compression = metadata_dict.get(
                                b"COMPRESSION", b"unknown"
                            ).decode("utf-8")
                    except Exception:
                        compression = "unknown"

                    parquet_metadata = {
                        "num_row_groups": parquet_file.num_row_groups,
                        "compression": compression,
                        "file_size_bytes": parquet_file.metadata.serialized_size,
                    }

                except Exception as pyarrow_exc:
                    logger.warning(
                        "Failed to read with pyarrow, falling back to pandas: %s",
                        pyarrow_exc,
                    )
                    # Fallback to regular pandas read
                    sample_df = pd.read_parquet(filepath, **read_params)
                    parquet_metadata = {"extraction_method": "pandas_fallback"}
            else:
                # Use pandas directly for fastparquet or other engines
                sample_df = pd.read_parquet(filepath, **read_params)
                parquet_metadata = {"extraction_method": f"pandas_{self.engine}"}

            schema = {
                "columns": sample_df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in sample_df.dtypes.items()},
                "num_columns": len(sample_df.columns),
                "filepath": filepath,
                "parquet_metadata": parquet_metadata,
            }

            logger.debug(
                "Schema extracted from %s: %d columns", filepath, schema["num_columns"]
            )
            return schema

        except Exception as e:
            logger.error("Failed to extract schema from %s: %s", filepath, e)
            raise DataReadError(f"Schema extraction failed for {filepath}: {e}") from e

    def _are_dtypes_compatible(self, dtype1: str, dtype2: str) -> bool:
        """
        Check if two pandas data types are compatible for concatenation.

        This method uses the same compatibility logic as CSVExtractor and ExcelExtractor
        but is adapted for Parquet-specific data types.

        Args
        ----
        dtype1 : str
            First data type as string.
        dtype2 : str
            Second data type as string.

        Returns
        -------
        bool
            True if data types are compatible, False otherwise.
        """
        if self.strict_schema:
            return dtype1 == dtype2

        # Define compatible data type groups (expanded for Parquet-specific types)
        numeric_types = {
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float16",
            "float32",
            "float64",
            "Float32",
            "Float64",
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "UInt8",
            "UInt16",
            "UInt32",
            "UInt64",
            # Parquet-specific decimal types
            "decimal128",
            "decimal256",
        }

        string_types = {"object", "string", "category", "large_string"}

        datetime_types = {
            "datetime64[ns]",
            "datetime64[ns, UTC]",
            "datetime64[us]",
            "datetime64[ms]",
            "period[D]",
            "timedelta64[ns]",
            "timestamp[ns]",
            "timestamp[us]",
            "timestamp[ms]",
        }

        boolean_types = {"bool", "boolean"}

        # Check if both types belong to the same compatible group
        if dtype1 in numeric_types and dtype2 in numeric_types:
            return True
        elif dtype1 in string_types and dtype2 in string_types:
            return True
        elif dtype1 in datetime_types and dtype2 in datetime_types:
            return True
        elif dtype1 in boolean_types and dtype2 in boolean_types:
            return True
        else:
            return dtype1 == dtype2

    def _validate_schema_compatibility(
        self, schemas: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate schema compatibility across multiple Parquet files.

        Args
        ----
        schemas : List[Dict[str, Any]]
            List of schema dictionaries from multiple files.

        Returns
        -------
        Dict[str, Any]
            Master schema dictionary with unified column information.

        Raises
        ------
        DataReadError
            If schemas are incompatible across files.
        """
        if not schemas:
            raise DataReadError("No schemas provided for validation.")

        if len(schemas) == 1:
            logger.info("Single file detected, schema validation skipped.")
            return schemas[0]

        logger.info("Validating schema compatibility across %d files.", len(schemas))

        # Use first file as reference schema
        master_schema = schemas[0].copy()
        master_columns = set(master_schema["columns"])
        incompatible_files = []

        for i, schema in enumerate(schemas[1:], 1):
            current_columns = set(schema["columns"])
            filepath = schema["filepath"]

            # Check column names
            if master_columns != current_columns:
                missing_in_current = master_columns - current_columns
                extra_in_current = current_columns - master_columns

                error_msg = f"Schema mismatch in {filepath}:"
                if missing_in_current:
                    error_msg += f" Missing columns: {list(missing_in_current)}."
                if extra_in_current:
                    error_msg += f" Extra columns: {list(extra_in_current)}."

                incompatible_files.append(error_msg)
                continue

            # Check data types for each column
            for col in master_schema["columns"]:
                master_dtype = master_schema["dtypes"][col]
                current_dtype = schema["dtypes"][col]

                if not self._are_dtypes_compatible(master_dtype, current_dtype):
                    error_msg = (
                        f"Data type mismatch in {filepath} for column '{col}': "
                        f"expected {master_dtype}, got {current_dtype}."
                    )
                    incompatible_files.append(error_msg)

        if incompatible_files:
            error_summary = "Schema validation failed:\n" + "\n".join(
                incompatible_files
            )
            logger.error("Schema validation failed: %s", error_summary)
            raise DataReadError(error_summary)

        logger.info("Schema validation successful: All files have compatible schemas.")
        return master_schema

    def _read_file(self, filepath: str) -> pd.DataFrame:
        """
        Read a single Parquet file into a DataFrame.

        Args
        ----
        filepath : str
            Path to the Parquet file.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the file's contents.

        Raises
        ------
        DataReadError
            If the file cannot be read due to missing file, parse error,
            or unexpected issues.
        """
        try:
            # Prepare read_parquet parameters
            read_params: Dict[str, Any] = {
                "columns": self.columns,
                "filters": self.filters,
                "engine": self.engine,
                "use_nullable_dtypes": self.use_nullable_dtypes,
                **self.read_parquet_kwargs,
            }

            logger.debug(
                "Reading Parquet file: %s with params %s", filepath, read_params
            )
            df = pd.read_parquet(filepath, **read_params)

            # Enrich with partition columns if configured
            df = self._maybe_enrich_partitions(df, Path(filepath))

            return df

        except FileNotFoundError as fnf_error:
            logger.error("Parquet file not found: %s", filepath)
            raise DataReadError(f"File not found: {filepath}") from fnf_error

        except PermissionError as perm_error:
            logger.error("Permission denied accessing Parquet file: %s", filepath)
            raise DataReadError(f"Permission denied: {filepath}") from perm_error

        except Exception as e:
            logger.exception(
                "Unexpected error during Parquet extraction from %s", filepath
            )
            raise DataReadError(f"Failed to read {filepath}: {e}") from e

    def extract(
        self,
    ) -> Union[
        Tuple[pd.DataFrame, Dict[str, Any]],
        Generator[Tuple[pd.DataFrame, Dict[str, Any]], None, None],
    ]:
        """
        Extract data from one or more Parquet files.

        Returns
        -------
        Union[Tuple[pandas.DataFrame, Dict[str, Any]],
                Generator[Tuple[pandas.DataFrame, Dict[str, Any]], None, None]]
            - If `chunksize` is None:
                Combined DataFrame containing extracted Parquet data and metadata.
            - If `chunksize` is True:
                Generator yielding (DataFrame chunk, metadata) tuples for row group streaming.

        Raises
        ------
        DataReadError
            If any file cannot be read successfully.
        """
        if self.chunksize:
            logger.info(
                "Extracting Parquet files in streaming mode with row group chunks"
            )
            return self._extract_streaming()
        else:
            logger.info(
                "Extracting Parquet files in regular mode from %d files",
                len(self.filepaths),
            )
            return self._extract_regular()

    def _extract_regular(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extract data in regular (non-streaming) mode.

        Reads all discovered Parquet files into memory and concatenates them
        into a single DataFrame. Performs schema validation if enabled
        and multiple files are being processed.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any]]
            Combined DataFrame and metadata.
        """
        # Perform schema validation if enabled and multiple files exist
        if (
            self.validate_schema
            and len(self.filepaths) > 1
            and not self.schema_validated
        ):
            logger.info(
                "Performing schema validation for %d files.", len(self.filepaths)
            )
            schemas = []
            for fp in self.filepaths:
                schema = self._extract_schema(fp)
                schemas.append(schema)

            # Validate and store master schema
            self.master_schema = self._validate_schema_compatibility(schemas)
            self.schema_validated = True
            logger.info("Schema validation completed successfully.")

        dfs: List[pd.DataFrame] = []

        if self.parallel:
            logger.info("Reading files in parallel using ThreadPoolExecutor.")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._read_file, fp): fp for fp in self.filepaths
                }
                for future in as_completed(futures):
                    fp = futures[future]
                    try:
                        df = future.result()
                        dfs.append(df)
                        logger.info("Successfully read file: %s", fp)
                    except Exception as e:
                        logger.error("Failed to read file %s: %s", fp, e)
                        raise
        else:
            for fp in self.filepaths:
                df = self._read_file(fp)
                dfs.append(df)
                logger.info("Successfully read file: %s", fp)

        if not dfs:
            raise DataReadError("No data extracted from the provided Parquet files.")

        combined_df = pd.concat(dfs, ignore_index=True)
        metadata = self._create_metadata(combined_df)

        # Add schema information to metadata if validation was performed
        if self.master_schema:
            metadata["schema_validation"] = {
                "validated": True,
                "master_schema": self.master_schema,
                "strict_mode": self.strict_schema,
                "num_files_validated": len(self.filepaths),
            }
        else:
            metadata["schema_validation"] = {
                "validated": False,
                "reason": "Single file or validation disabled",
            }

        # Add Parquet-specific metadata
        metadata.update(
            {
                "files_read": len(self.filepaths),
                "engine": self.engine,
                "parallel": self.parallel,
                "recursive": self.recursive,
                "columns_selected": self.columns,
                "filters_applied": self.filters is not None,
                "use_nullable_dtypes": self.use_nullable_dtypes,
            }
        )

        logger.info(
            "Parquet extraction completed: %d records combined from %d files.",
            metadata["num_records"],
            len(self.filepaths),
        )
        return combined_df, metadata

    def _extract_streaming(
        self,
    ) -> Generator[Tuple[pd.DataFrame, Dict[str, Any]], None, None]:
        """
        Extract data in streaming mode using row group chunks.

        Iterates through each file and yields chunks of data with metadata.
        Performs schema validation before streaming if enabled.

        This method leverages Parquet's row group structure for memory-efficient
        processing of large datasets.

        Yields
        ------
        Generator[Tuple[pd.DataFrame, Dict[str, Any]], None, None]
            Each item is a tuple of (chunk DataFrame, metadata dict).
        """
        # Perform schema validation if enabled and multiple files exist
        if (
            self.validate_schema
            and len(self.filepaths) > 1
            and not self.schema_validated
        ):
            logger.info(
                "Performing schema validation for %d files before streaming.",
                len(self.filepaths),
            )
            schemas = []
            for fp in self.filepaths:
                schema = self._extract_schema(fp)
                schemas.append(schema)

            # Validate and store master schema
            self.master_schema = self._validate_schema_compatibility(schemas)
            self.schema_validated = True
            logger.info("Schema validation completed successfully.")

        for fp in self.filepaths:
            logger.info("Streaming file: %s", fp)

            try:
                if self.engine == "pyarrow":
                    # Use pyarrow for row group-based streaming
                    import pyarrow.parquet as pq

                    parquet_file = pq.ParquetFile(fp)
                    total_row_groups = parquet_file.num_row_groups

                    for rg_idx in range(total_row_groups):
                        logger.debug(
                            "Reading row group %d/%d from %s",
                            rg_idx + 1,
                            total_row_groups,
                            fp,
                        )

                        # Read specific row group
                        table = parquet_file.read_row_group(
                            rg_idx,
                            columns=self.columns,
                            use_threads=True,
                        )
                        chunk = table.to_pandas()

                        # Apply filters if specified (post-read filtering)
                        if self.filters:
                            chunk = self._apply_filters(chunk, self.filters)

                        # Enrich with partition columns if configured
                        chunk = self._maybe_enrich_partitions(chunk, Path(fp))

                        chunk_metadata = self._create_metadata(chunk)

                        # Add schema information to metadata if validation was performed
                        if self.master_schema:
                            chunk_metadata["schema_validation"] = {
                                "validated": True,
                                "master_schema": self.master_schema,
                                "strict_mode": self.strict_schema,
                                "num_files_validated": len(self.filepaths),
                            }
                        else:
                            chunk_metadata["schema_validation"] = {
                                "validated": False,
                                "reason": "Single file or validation disabled",
                            }

                        # Add chunk-specific metadata
                        chunk_metadata.update(
                            {
                                "row_group": rg_idx,
                                "total_row_groups": total_row_groups,
                                "engine": self.engine,
                                "file_path": fp,
                            }
                        )

                        yield chunk, chunk_metadata

                else:
                    # For fastparquet or other engines, fall back to regular read
                    # (they don't support row group streaming as efficiently)
                    logger.warning(
                        "Row group streaming not supported with engine '%s'. "
                        "Reading entire file: %s",
                        self.engine,
                        fp,
                    )
                    df = self._read_file(fp)
                    chunk_metadata = self._create_metadata(df)
                    chunk_metadata.update(
                        {
                            "streaming_method": "full_file_fallback",
                            "engine": self.engine,
                            "file_path": fp,
                        }
                    )
                    yield df, chunk_metadata

            except Exception as e:
                logger.error("Error during streaming extraction from %s: %s", fp, e)
                raise DataReadError(f"Failed to stream data from {fp}: {e}") from e

    def _apply_filters(self, df: pd.DataFrame, filters: List[Tuple]) -> pd.DataFrame:
        """
        Apply row-level filters to a DataFrame.

        This is a post-read filtering mechanism for cases where the engine
        doesn't support predicate pushdown or when using row group streaming.

        Args
        ----
        df : pd.DataFrame
            DataFrame to filter.
        filters : List[Tuple]
            List of filter tuples in format (column, operator, value).

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame.
        """
        try:
            for column, operator, value in filters:
                if column not in df.columns:
                    logger.warning("Filter column '%s' not found in data", column)
                    continue

                if operator == "=":
                    df = df[df[column] == value]
                elif operator == "!=":
                    df = df[df[column] != value]
                elif operator == "<":
                    df = df[df[column] < value]
                elif operator == "<=":
                    df = df[df[column] <= value]
                elif operator == ">":
                    df = df[df[column] > value]
                elif operator == ">=":
                    df = df[df[column] >= value]
                elif operator == "in":
                    df = df[df[column].isin(value)]
                elif operator == "not in":
                    df = df[~df[column].isin(value)]
                else:
                    logger.warning("Unsupported filter operator: %s", operator)

            return df

        except Exception as e:
            logger.warning("Filter application failed: %s", e)
            return df

    def _maybe_enrich_partitions(self, df: pd.DataFrame, fp: Path) -> pd.DataFrame:
        """
        Add partition columns derived from directory names like 'key=value'.

        This method enriches the DataFrame with partition information extracted
        from Hive-style partitioned directory structures.

        Args
        ----
        df : pd.DataFrame
            DataFrame to enrich.
        fp : Path
            File path to extract partition information from.

        Returns
        -------
        pd.DataFrame
            DataFrame enriched with partition columns.
        """
        if df.empty:
            return df

        parts = self._extract_partitions(fp)
        if not parts:
            return df

        try:
            for k, v in parts.items():
                if k not in df.columns:
                    df[k] = v
                    logger.debug("Added partition column '%s' with value '%s'", k, v)
        except Exception as exc:
            logger.warning("Partition enrichment failed for %s: %s", fp, exc)

        return df

    def _extract_partitions(self, fp: Path) -> Dict[str, str]:
        """
        Extract partition key-value pairs from ancestors of ``fp``.

        This method parses Hive-style partition directories from the file path.

        Args
        ----
        fp : Path
            File path to extract partition information from.

        Returns
        -------
        Dict[str, str]
            Dictionary of partition key-value pairs.

        Examples
        --------
        Path: /data/year=2025/month=09/part-0001.parquet
        Returns: {'year': '2025', 'month': '09'}
        """
        parts: Dict[str, str] = {}
        try:
            for parent in fp.parents:
                name = parent.name
                if "=" in name:
                    key, _, val = name.partition("=")
                    if key and val:
                        parts[key] = val
        except Exception as exc:
            logger.debug("Partition parsing skipped for %s (%s)", fp, exc)

        return parts

    @staticmethod
    def example_usage() -> None:
        """
        Print detailed usage examples for ParquetExtractor.

        This demonstrates different ways to use the class, covering:
        - Single and multi-file extraction
        - Column subset and filtering
        - Recursive directory discovery
        - Parallel processing
        - Streaming mode for large files
        - Schema validation across multiple files
        - Parquet-specific features (engines, partitions, etc.)
        """

        examples = dedent(
            """
        ============================================================
        ParquetExtractor USAGE EXAMPLES
        ============================================================

        1. Extract data from a single Parquet file
        ------------------------------------------------------------
        extractor = ParquetExtractor("sales_data.parquet")
        df, metadata = extractor.extract()
        print(f"Extracted {metadata['num_records']} records")

        2. Extract from multiple Parquet files (batch mode)
        ------------------------------------------------------------
        extractor = ParquetExtractor(
            ["file1.parquet", "file2.parquet"],
            validate_schema=True,
            parallel=True
        )
        df, metadata = extractor.extract()
        print(f"Combined {metadata['num_records']} rows from {metadata['files_read']} files")

        3. Extract all Parquet files from a folder
        ------------------------------------------------------------
        extractor = ParquetExtractor("data_folder/")
        df, metadata = extractor.extract()
        print(f"Found {metadata['files_read']} Parquet files")

        4. Extract Parquet files from a folder (recursive search)
        ------------------------------------------------------------
        extractor = ParquetExtractor("data_folder/", recursive=True)
        df, metadata = extractor.extract()
        print(f"Recursive search found {len(df)} rows")

        5. Extract multiple Parquet files in parallel
        ------------------------------------------------------------
        extractor = ParquetExtractor("data_folder/", parallel=True, max_workers=4)
        df, metadata = extractor.extract()
        print(f"Parallel extraction complete: {metadata}")

        6. Streaming large Parquet file with row group chunks
        ------------------------------------------------------------
        extractor = ParquetExtractor("large_file.parquet", chunksize=True)
        for chunk, chunk_metadata in extractor.extract():
            print(f"Processed row group {chunk_metadata['row_group']} with {chunk_metadata['num_records']} rows")

        7. Advanced Parquet features
        ------------------------------------------------------------
        # a) Column subset selection for efficient reads
        extractor = ParquetExtractor(
            "data.parquet",
            columns=["customer_id", "purchase_amount", "timestamp"],
            engine="pyarrow"
        )
        df, metadata = extractor.extract()

        # b) Filtered reads with predicate pushdown
        extractor = ParquetExtractor(
            "partitioned_data/",
            filters=[("year", "=", 2023), ("month", "in", [1, 2, 3])],
            recursive=True
        )
        df, metadata = extractor.extract()

        # c) Engine selection and nullable dtypes
        extractor = ParquetExtractor(
            "data.parquet",
            engine="pyarrow",  # or "fastparquet", "auto"
            use_nullable_dtypes=True
        )
        df, metadata = extractor.extract()

        8. Schema validation across multiple files
        ------------------------------------------------------------
        # a) Normal schema validation (compatible types allowed)
        extractor = ParquetExtractor(
            ["file1.parquet", "file2.parquet"],
            validate_schema=True
        )
        df, metadata = extractor.extract()
        print(metadata["schema_validation"])

        # b) Strict schema validation (exact type matching)
        extractor = ParquetExtractor(
            ["file1.parquet", "file2.parquet"],
            validate_schema=True,
            strict_schema=True
        )
        df, metadata = extractor.extract()

        # c) Disable schema validation
        extractor = ParquetExtractor(
            ["file1.parquet", "file2.parquet"],
            validate_schema=False
        )
        df, metadata = extractor.extract()

        9. Partitioned dataset extraction
        ------------------------------------------------------------
        # Extract from Hive-style partitioned dataset
        extractor = ParquetExtractor(
            "warehouse/sales/",
            recursive=True,
            filters=[("year", ">=", 2022)],
            partition_columns=["year", "month", "day"]
        )
        df, metadata = extractor.extract()
        print("Partition columns automatically added:", [col for col in df.columns if col in ["year", "month", "day"]])

        10. Memory-efficient streaming for very large datasets
        ------------------------------------------------------------
        extractor = ParquetExtractor(
            "huge_dataset/",
            recursive=True,
            chunksize=True,
            columns=["id", "value"],  # Only load required columns
            filters=[("status", "=", "active")]
        )
        
        total_processed = 0
        for chunk, chunk_meta in extractor.extract():
            # Process each chunk separately
            result = process_chunk(chunk)
            total_processed += chunk_meta["num_records"]
            print(f"Processed {total_processed} records so far...")

        ============================================================
        NOTES
        ============================================================
        - Parquet supports efficient columnar reads - use 'columns' parameter for performance
        - pyarrow engine is generally faster and more feature-complete than fastparquet
        - Row group streaming leverages Parquet's internal structure for memory efficiency
        - Filters can be pushed down to the storage layer for better performance
        - Schema validation ensures consistency across multiple files
        - Parallel processing accelerates multi-file extraction
        - Partition enrichment automatically adds partition columns from directory structure
        - Use nullable dtypes for better handling of missing values
        ============================================================
        """
        )
        print(examples)
