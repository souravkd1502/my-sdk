"""

"""
import pandas as pd
from abc import ABC, abstractmethod

class BaseExtractor(ABC):
    @abstractmethod
    def extract(self) -> pd.DataFrame:
        """Extracts data and returns a DataFrame"""
        pass
