"""


"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Import dependencies
import os
import yaml
import wandb
import logging
import urllib.request
from ultralytics import YOLO

from typing import List, Any

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class YoloTrainer:
    """ """

    def __init__(self, config: dict):
        """
        Initializes the YoloTrainer with the given configuration.

        Args:
            config (dict): A dictionary containing the configuration parameters for the YOLO model.
        """
        self.config = config
        self._download_checkpoints(
            folder="checkpoints", checkpoint_urls=self.config["urls"]
        )
        self._validate_coco_yaml(yaml_path=self.config["coco_yaml"])

        if self.config["use_wandb"] is not None:
            self.wandb = wandb.init(
                project=self.config["wandb_project"], entity=self.config["wandb_entity"]
            )
        self.model = self._create_model(config)

    def _download_checkpoints(self, folder: str, checkpoint_urls: List[str]) -> None:
        """
        Downloads YOLOv8 checkpoint files from the given URLs into the specified folder.

        Args:
            folder (str): The folder where checkpoint files will be saved.
            checkpoint_urls (List[str]): A list of URLs pointing to the checkpoint files.

        Example:
            >>> download_checkpoints(
            ...     "checkpoints",
            ...     [
            ...         "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt",
            ...         "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8l-face.pt",
            ...         "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8m-face.pt",
            ...     ]
            ... )
        """
        os.makedirs(folder, exist_ok=True)  # Ensure the folder exists

        for url in checkpoint_urls:
            filename = os.path.basename(url)
            destination = os.path.join(folder, filename)

            if not os.path.exists(destination):
                _logger.info(f"Downloading {filename}...")
                try:
                    urllib.request.urlretrieve(url, destination)
                    _logger.info(f"Downloaded {filename} to {destination}")
                except Exception as e:
                    _logger.info(f"Failed to download {filename}: {e}")
            else:
                _logger.info(f"{filename} already exists, skipping download.")

    def _validate_coco_yaml(self, yaml_path: str) -> bool:
        """
        Validates the structure of a COCO-style YAML configuration.

        Args:
            yaml_path (str): Path to the YAML file to validate.

        Returns:
            bool: True if the YAML is valid, False otherwise.
        """
        required_fields = {"path": str, "train": str, "val": str, "names": dict}

        try:
            # Load YAML file
            with open(yaml_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)

            # Validate top-level fields
            for field, field_type in required_fields.items():
                if field not in config:
                    _logger.info(f"Error: Missing required field '{field}'.")
                    return False
                if not isinstance(config[field], field_type):
                    _logger.info(
                        f"Error: Field '{field}' should be of type {field_type.__name__}."
                    )
                    return False

            # Validate 'names' dictionary
            names = config["names"]
            if not all(
                isinstance(key, int) and isinstance(value, str)
                for key, value in names.items()
            ):
                _logger.info(
                    "Error: 'names' should be a dictionary with integer keys and string values."
                )
                return False

            # Optional fields
            if "test" in config and config["test"] is not None:
                if not isinstance(config["test"], str):
                    _logger.info("Error: 'test' should be of type str if provided.")
                    return False

            _logger.info("YAML is valid.")
            return True

        except yaml.YAMLError as e:
            _logger.info(f"Error parsing YAML file: {e}")
            return False
        except Exception as e:
            _logger.info(f"An unexpected error occurred: {e}")
            return False

    def _create_model(self, config: dict) -> YOLO:
        """
        Creates a YOLO model using the provided configuration.

        Args:
            config (dict): A dictionary containing the model configuration.

        Returns:
            YOLO: An instance of the YOLO class.
        """
        return YOLO(
            model=config["model"], task=config["device"], verbose=config["verbose"]
        )

    def train(self, yaml_path: str) -> Any | None:
        """
        Trains the YOLO model using the provided configuration.

        Args:
            yaml_path (str): Path to the YAML file containing the training configuration.

        Returns:
            Any | None: The result of the training process.
        """
        self.model.train(
            data=yaml_path,
            epochs=1,
            batch=16,
            imgsz=640,
            name="exp",
            exist_ok=False,
            plots=True,
            save=True,
            resume=False,
        )
