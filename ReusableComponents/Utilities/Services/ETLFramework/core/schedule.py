"""

"""
import json
from croniter import croniter
from prefect import task, Flow


class ScheduleConfigLoader:
    """
    ScheduleConfigLoader is a class that loads the schedule configuration from a JSON file.
    """
    def __init__(self, config_path: str) -> None:
        """
        Initialize the ScheduleConfigLoader with the path to the schedule configuration file.
        Args:
            config_path (str): The path to the schedule configuration file.
        Returns:
            None
        """
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> dict:
        """
        Load the schedule configuration from a JSON file.
        Args:
            None
        Returns:
            dict: The schedule configuration.
        Raises:
            FileNotFoundError: If the schedule configuration file is not found.
        """
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Schedule configuration file not found at {self.config_path}")

    def _validate_config(self) -> None:
        """
        Validate the schedule configuration.
        Args:
            None
        Returns:
            None
        Raises:
            ValueError: If the schedule configuration is invalid.
        """
        