"""
pipeline.py
=================
Core pipeline orchestration for ETL (Extract → Transform → Load) workflows.

This module defines the `Pipeline` class, which validates and coordinates extractors,
transformers, and loaders. It provides a structured, fault-tolerant execution flow for
ETL processes in the SDK, ensuring that data is reliably extracted, transformed, and
loaded into the desired destination.

Key Features
------------
- Validation of ETL components (extractor, transformers, loader).
- Logging and monitoring of each ETL step.
- Exception handling with custom ETL exceptions.
- Extensible design: plug in new extractors, transformers, or loaders.

Exceptions/Errors
-----------------
- ``PipelineExecutionError``: Raised when any pipeline step (extract, transform, load) fails.
- ``ExtractorError``: Raised during the extract step if the source fails.
- ``TransformerError``: Raised during transformations if logic or schema fails.
- ``LoaderError``: Raised during the load step if the destination fails.

Usage Example
-------------
    >>> from etl_framework.core.pipeline import Pipeline
    >>> from etl_framework.extractors.csv_extractor import CSVExtractor
    >>> from etl_framework.transformers.clean_nulls import CleanNullsTransformer
    >>> from etl_framework.loaders.csv_loader import CSVLoader
    >>>
    >>> pipeline = Pipeline(
    ...     extractor=CSVExtractor("input.csv"),
    ...     transformers=[CleanNullsTransformer()],
    ...     loader=CSVLoader("output.csv")
    ... )
    >>> pipeline.run()

Dependencies
------------
- ``pandas``: For handling tabular data in extract, transform, and load steps.
- ``logging``: For structured logging and monitoring of ETL steps.

Notes
-----
- This is the orchestration layer only. Extractors, transformers, and loaders
    must be implemented separately.
- All ETL components should inherit from their respective abstract base classes:
    `BaseExtractor`, `BaseTransformer`, and `BaseLoader`.
"""

from typing import List
import logging

from ..extractors.base import BaseExtractor
from ..transformers.base import BaseTransformer
from ..loaders.base import BaseLoader
from .errors import (
    ExtractorError,
    TransformerError,
    LoaderError,
    PipelineExecutionError,
)

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


"""
pipeline.py
=================
Pipeline orchestration for ETL (Extract → Transform → Load).

This module defines the :class:`Pipeline` class, which coordinates extractors,
transformers, and loaders to run end-to-end ETL jobs. It provides extensibility
via hooks, retry logic for fragile operations, structured logging, and metadata
tracking for observability.

Key Features
------------
- Orchestrates Extract → Transform → Load sequence.
- Pre/post hooks for custom actions.
- Retry logic for extractor/loader steps.
- Metadata tracking (record counts, timings, status).
- Granular exception handling with ETL-specific errors.

Exceptions/Errors
-----------------
- ``PipelineExecutionError``: Raised for any pipeline-level failure.
- ``ExtractorError``: Raised if extraction fails.
- ``TransformerError``: Raised if transformation fails.
- ``LoaderError``: Raised if loading fails.

Usage Example
-------------
    >>> from etl_framework.pipeline import Pipeline
    >>> pipeline = Pipeline(extractor, [transformer1, transformer2], loader)
    >>> pipeline.run()
    >>> print(pipeline.metadata)

"""

import logging
import time
from typing import List, Dict, Any, Callable
import pandas as pd

from ..extractors.base import BaseExtractor
from ..transformers.base import BaseTransformer
from ..loaders.base import BaseLoader
from .errors import (
    PipelineExecutionError,
    ExtractorError,
    TransformerError,
    LoaderError,
)


class Pipeline:
    """
    Coordinates the Extract → Transform → Load process.

    The pipeline validates components at initialization and executes them in
    sequence. Hooks are provided for customization, and metadata is tracked
    for monitoring.

    Attributes:
    -------------
        extractor (BaseExtractor): Extracts raw data from a source.
        transformers (List[BaseTransformer]): A sequence of transformations applied to the data.
        loader (BaseLoader): Loads the transformed data into a destination.
        metadata (Dict[str, Any]): Stores execution metadata for monitoring/auditing.
    """

    def __init__(
        self,
        extractor: BaseExtractor,
        transformers: List[BaseTransformer],
        loader: BaseLoader,
        retries: int = 3,
        retry_delay: int = 2,
    ) -> None:
        """
        Initialize a pipeline with extractor, transformers, and loader.

        Parameters
        ----------
        extractor : BaseExtractor
            An extractor instance implementing ``extract()``.
        transformers : List[BaseTransformer]
            A list of transformer instances implementing ``transform()``.
        loader : BaseLoader
            A loader instance implementing ``load()``.
        retries : int, optional
            Number of retries for extractor/loader steps (default is 3).
        retry_delay : int, optional
            Delay in seconds between retries (default is 2).

        Raises
        ------
        TypeError
            If extractor, transformers, or loader are invalid types.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.retries = retries
        self.retry_delay = retry_delay

        # Metadata for monitoring and auditing
        self.metadata: Dict[str, Any] = {
            "status": "initialized",
            "records_extracted": 0,
            "records_loaded": 0,
            "transformers_applied": [],
            "start_time": None,
            "end_time": None,
            "duration_sec": None,
        }

        # Validate extractor
        if not isinstance(extractor, BaseExtractor):
            raise TypeError(
                f"Extractor must be a BaseExtractor, got {type(extractor).__name__}"
            )
        self.extractor = extractor

        # Validate transformers
        if not isinstance(transformers, list):
            raise TypeError("Transformers must be a list")
        for t in transformers:
            if not isinstance(t, BaseTransformer):
                raise TypeError(
                    f"All transformers must be BaseTransformer, got {type(t).__name__}"
                )
        self.transformers = transformers

        # Validate loader
        if not isinstance(loader, BaseLoader):
            raise TypeError(f"Loader must be a BaseLoader, got {type(loader).__name__}")
        self.loader = loader

    # ------------------------------------------------------------------
    # Hook methods (designed to be overridden in subclasses if needed)
    # ------------------------------------------------------------------
    def before_extract(self) -> None: ...
    def after_extract(self, df: pd.DataFrame) -> None: ...
    def before_transform(self, df: pd.DataFrame) -> None: ...
    def after_transform(self, df: pd.DataFrame) -> None: ...
    def before_load(self, df: pd.DataFrame) -> None: ...
    def after_load(self, df: pd.DataFrame) -> None: ...

    # ------------------------------------------------------------------
    # Internal helper: retry wrapper
    # ------------------------------------------------------------------
    def _retry(self, func: Callable[[], Any], step_name: str) -> Any:
        """
        Retry wrapper for fragile steps like extract/load.

        Parameters
        ----------
        func : Callable
            The function to execute.
        step_name : str
            The pipeline step name (used in logs).

        Returns
        -------
        Any
            The function's return value.

        Raises
        ------
        Exception
            If all retries fail, the exception from the last attempt is raised.
        """
        for attempt in range(1, self.retries + 1):
            try:
                return func()
            except Exception as e:
                if attempt < self.retries:
                    self.logger.warning(
                        "[%s] attempt %d/%d failed: %s. Retrying in %ds...",
                        step_name,
                        attempt,
                        self.retries,
                        str(e),
                        self.retry_delay,
                    )
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error("[%s] failed after %d attempts.", step_name, attempt)
                    raise

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Execute the ETL pipeline: extract → transform → load.

        Raises
        ------
        PipelineExecutionError
            If any step (extract, transform, load) fails.
        """
        self.logger.info("Pipeline started.")
        self.metadata["status"] = "running"
        self.metadata["start_time"] = time.time()

        try:
            # Step 1: Extract
            self.before_extract()
            df: pd.DataFrame = self._retry(self.extractor.extract, "extract")
            self.metadata["records_extracted"] = len(df)
            self.after_extract(df)
            self.logger.info("Extraction successful: %d records retrieved.", len(df))

            if df.empty:
                self.logger.warning("Extractor returned empty dataset. Skipping transform/load.")
                self.metadata["status"] = "skipped_empty"
                return

            # Step 2: Transform
            for transformer in self.transformers:
                self.before_transform(df)
                df = transformer.transform(df)
                self.after_transform(df)
                self.metadata["transformers_applied"].append(transformer.__class__.__name__)
                self.logger.info("Applied transformer: %s", transformer.__class__.__name__)

            # Step 3: Load
            self.before_load(df)
            self._retry(lambda: self.loader.load(df), "load")
            self.after_load(df)
            self.metadata["records_loaded"] = len(df)
            self.logger.info("Loading completed successfully.")

            self.metadata["status"] = "success"

        except ExtractorError as e:
            self.logger.error("Extractor step failed: %s", str(e))
            self.metadata["status"] = "failed_extractor"
            raise PipelineExecutionError(f"Extractor step failed: {e}") from e
        except TransformerError as e:
            self.logger.error("Transformer step failed: %s", str(e))
            self.metadata["status"] = "failed_transformer"
            raise PipelineExecutionError(f"Transformer step failed: {e}") from e
        except LoaderError as e:
            self.logger.error("Loader step failed: %s", str(e))
            self.metadata["status"] = "failed_loader"
            raise PipelineExecutionError(f"Loader step failed: {e}") from e
        except Exception as e:
            self.logger.exception("Unexpected pipeline failure: %s", str(e))
            self.metadata["status"] = "failed_unexpected"
            raise PipelineExecutionError(f"Unexpected error: {e}") from e
        finally:
            # Always finalize metadata
            self.metadata["end_time"] = time.time()
            self.metadata["duration_sec"] = (
                self.metadata["end_time"] - self.metadata["start_time"]
            )
            self.logger.info("Pipeline finished with status: %s", self.metadata["status"])