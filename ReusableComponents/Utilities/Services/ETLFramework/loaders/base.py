"""

"""

import pandas as pd
from abc import ABC, abstractmethod

class BaseLoader(ABC):
    @abstractmethod
    def load(self, df: pd.DataFrame) -> None:
        """Loads a DataFrame into a destination"""
        pass
