"""

"""

import pandas as pd
from abc import ABC, abstractmethod

class BaseTransformer(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms a DataFrame and returns it"""
        pass
