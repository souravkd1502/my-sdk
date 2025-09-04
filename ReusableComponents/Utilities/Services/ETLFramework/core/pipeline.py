"""

"""

# Import necessary modules
from typing import Any, List
import logging

from ..extractors.base import BaseExtractor
from ..transformers.base import BaseTransformer
from ..loaders.base import BaseLoader

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Pipeline:
    def __init__(self, extractor: BaseExtractor, transformers: List[BaseTransformer], loader: BaseLoader):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Validate extractor
        if not isinstance(extractor, BaseExtractor):
            raise TypeError(f"Extractor must be a BaseExtractor, got {type(extractor).__name__}")
        self.extractor = extractor

        # Validate transformers
        if not isinstance(transformers, list):
            raise TypeError("Transformers must be a list")
        for t in transformers:
            if not isinstance(t, BaseTransformer):
                raise TypeError(f"All transformers must be BaseTransformer, got {type(t).__name__}")
        self.transformers = transformers

        # Validate loader
        if not isinstance(loader, BaseLoader):
            raise TypeError(f"Loader must be a BaseLoader, got {type(loader).__name__}")
        self.loader = loader

    def run(self):
        self.logger.info("Pipeline started.")

        df = self.extractor.extract()
        self.logger.info(f"Extracted {len(df)} records.")

        for transformer in self.transformers:
            df = transformer.transform(df)
            self.logger.info(f"Applied: {transformer.__class__.__name__}")

        self.loader.load(df)
        self.logger.info("Pipeline finished successfully.")