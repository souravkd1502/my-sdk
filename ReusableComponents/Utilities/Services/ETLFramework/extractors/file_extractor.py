"""
file_extractor.py
=================
File-based data extraction framework for ETL pipelines.

This module provides a unified framework for extracting structured and semi-structured
data from files. It defines an abstract base class (:class:`FileExtractor`) that
standardizes file ingestion patterns, and concrete implementations for CSV and JSON
(including JSON Lines/NDJSON). The design is extensible, allowing additional formats
such as Excel, Parquet, XML, or DOCX to be integrated seamlessly.

The framework is optimized for scalability and production use: it supports
single files, lists of files, or entire directory trees (with recursive discovery),
and offers both in-memory and streaming modes for efficient handling of datasets.
Optional parallelism accelerates ingestion when working with many small/medium files.
JSON extractors also support nested structure flattening and partition enrichment
from folder naming schemes (e.g., ``year=2025/month=09``).

Key Features
------------
- **Base class** (`FileExtractor`) defining consistent extraction interfaces.
- **Implemented formats**: CSV, JSON, and JSON Lines (NDJSON).
- **Extensible design**: easily extend to Excel, Parquet, XML, DOCX, and more.
- **Input flexibility**: single file, multiple files, or recursive folder discovery.
- **Streaming and chunked reads** for memory-efficient processing of large files.
- **Parallel processing** for multi-file ingestion.
- **Flattening** of nested JSON structures with ``pandas.json_normalize``.
- **Selective column extraction** for efficiency.
- **Partition key enrichment** from directory naming conventions (``key=value``).
- **Comprehensive metadata** for tracking extracted datasets.

Exceptions/Errors
-----------------
- ``ETLException``: General ETL framework errors (e.g., unsupported file types).
- ``DataSourceNotFoundError``: Raised when no valid input files are found.
- ``DataReadError``: Raised when a file cannot be read or parsed.
- ``DependencyMissingError``: Raised if an optional dependency (e.g., ijson) is missing.
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

Parallel folder ingestion:

    >>> extractor = JSONExtractor(
    ...     input_source="data/logs/",
    ...     recursive=True,
    ...     json_lines=True,
    ...     parallel=True
    ... )
    >>> df, meta = extractor.extract()

Dependencies
------------
- ``pandas``: Required for DataFrame operations and chunked CSV/JSONL reading.
- ``pathlib``: Cross-platform file path operations.
- ``concurrent.futures``: Parallel file processing.
- ``ijson`` (optional): Streaming JSON parser for large JSON arrays.
- ``typing``: Type hints for better interfaces.

Notes
-----
- Use ``stream_extract()`` for very large files to avoid memory pressure.
- When ``chunksize`` is specified, extractors yield dataframes in chunks
    instead of loading everything into memory at once.
- Designed for production ETL: logging, error handling, and extensibility
    are first-class considerations.
- For massive or distributed workloads, delegate orchestration to Dask, Polars, or Spark.
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

    SUPPORTED_FILE_TYPES = ["csv", "xlsx", "json", "txt", "parquet", "xml", "docx"]

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
        raise NotImplementedError(
            "Subclasses of FileExtractor must implement the extract() method."
        )

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
            - columns: list of column names
            - summary: statistical summary
        """
        try:
            metadata: Dict[str, Any] = {
                "source": self.filepath,
                "extraction_time": pd.Timestamp.now().isoformat(),
                "num_records": len(df),
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
                "columns": df.columns.tolist(),
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
