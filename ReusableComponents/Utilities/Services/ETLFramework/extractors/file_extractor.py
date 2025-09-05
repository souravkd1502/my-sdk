""" """

import logging
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, Union, List, Generator, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed


try:
    # Try relative import first (when used as part of package)
    from ..core.errors import ETLException, DataReadError
except ImportError:
    # Fall back to absolute import (when run directly)
    import sys
    from pathlib import Path
    # Add the ETLFramework directory to Python path
    etl_framework_path = Path(__file__).parent.parent
    sys.path.insert(0, str(etl_framework_path))
    from core.errors import ETLException, DataReadError


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


import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# Custom exception classes (should exist in your framework)
class ETLException(Exception):
    """Base exception for ETL operations."""
    pass

class DataReadError(ETLException):
    """Raised when data extraction from a file fails."""
    pass

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    This class is designed for flexible ETL pipelines, offering both batch
    (regular) and streaming modes.
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
        kwargs : dict
            Additional keyword arguments passed to `pandas.read_csv`.

        Raises
        ------
        ValueError
            If no valid CSV files are found in `input_source`.
        """
        super().__init__(input_source)
        self.delimiter = delimiter
        self.encoding = encoding
        self.na_values = na_values
        self.recursive = recursive
        self.parallel = parallel
        self.chunksize = chunksize
        self.read_csv_kwargs = kwargs

        # Discover CSV files from given input
        self.filepaths: List[str] = self._discover_files(input_source)

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
            logger.info("Extracting CSVs in streaming mode with chunksize=%s", self.chunksize)
            return self._extract_streaming()
        else:
            logger.info("Extracting CSVs in regular mode from %d files", len(self.filepaths))
            return self._extract_regular()

    def _extract_regular(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extract data in regular (non-streaming) mode.

        Reads all discovered CSV files into memory and concatenates them
        into a single DataFrame.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any]]
            Combined DataFrame and metadata.
        """
        dfs: List[pd.DataFrame] = []

        if self.parallel:
            logger.info("Reading files in parallel using ThreadPoolExecutor.")
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(self._read_file, fp): fp for fp in self.filepaths}
                for future in as_completed(futures):
                    fp = futures[future]
                    try:
                        dfs.append(future.result())
                        logger.info("Successfully read file: %s", fp)
                    except Exception as e:
                        logger.error("Failed to read file %s: %s", fp, e)
                        raise
        else:
            for fp in self.filepaths:
                dfs.append(self._read_file(fp))
                logger.info("Successfully read file: %s", fp)

        if not dfs:
            raise DataReadError("No data extracted from the provided CSV files.")

        combined_df = pd.concat(dfs, ignore_index=True)
        metadata = self._create_metadata(combined_df)

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

        Yields
        ------
        Generator[Tuple[pd.DataFrame, Dict[str, Any]], None, None]
            Each item is a tuple of (chunk DataFrame, metadata dict).
        """
        for fp in self.filepaths:
            logger.info("Streaming file: %s", fp)
            result = self._read_file(fp)

            # When chunksize is set, result is a pandas TextFileReader
            for chunk in result:  # type: ignore
                logger.debug("Yielding chunk of size %d from %s", len(chunk), fp)
                yield chunk, self._create_metadata(chunk)




if __name__ == "__main__":
    file_path = "C:\\Users\\OMEN\\Downloads\\Age-sex-by-ethnic-group-grouped-total-responses-census-usually-resident-population-counts-2006-2013-2018-Censuses-RC-TA-SA2-DHB\\Data8277.csv"
    
    # Test without chunksize first to get a simple DataFrame and metadata
    print("Testing CSVExtractor without chunksize...")
    csv_extractor = CSVExtractor(file_path, parallel=False, low_memory=False)
    from time import time
    
    try:
        start = time()
        df, metadata = csv_extractor.extract()
        end = time()
        print(f"Extraction without chunking took {end - start:.2f} seconds")
        
        from pprint import pprint
        
        print("============================================================")
        print("Extraction successful! Metadata:\n")
        pprint(metadata)
        print("============================================================")
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        
    # Test with chunksize (streaming mode)
    print("\n\nTesting CSVExtractor with chunksize (streaming mode)...")
    start = time()
    csv_extractor_chunked = CSVExtractor(file_path, chunksize=1000000, parallel=False, low_memory=False)
    
    try:
        chunk_count = 0
        for chunk, chunk_metadata in csv_extractor_chunked.extract():
            chunk_count += 1
        print("============================================================")
        print(f"Successfully processed {chunk_count} chunks in streaming mode")
        end = time()
        print(f"Extraction with chunking took {end - start:.2f} seconds")
        print("============================================================")
        
    except Exception as e:
        print(f"Error during chunked extraction: {e}")
