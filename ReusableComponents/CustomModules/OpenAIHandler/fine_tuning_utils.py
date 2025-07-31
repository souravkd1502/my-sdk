"""
text_handler.py
---------------
This module contains functions that handle chat completion using OpenAI Chat Completion API. It includes the following:
- chat_completion: This function takes a prompt and returns a completion of the prompt using OpenAI's Chat Completion API.
- Handles Conversational history: This function takes a prompt and a list of previous messages and returns a completion of the prompt using OpenAI's Chat Completion API.
- Handles Function calling: This function takes a prompt and a list of previous messages and returns a completion of the prompt using OpenAI's Chat Completion API.
- Handles Structured responses: This function takes a prompt and a list of previous messages and returns a completion of the prompt using OpenAI's Chat Completion API.

Requirements:
--------------
- openai==1.59.3
- python-dotenv==1.0.1

Author:
-------
Sourav Das

Date:
-----
2025-10-01
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Importing necessary libraries and modules

import os
import json
import logging
import tiktoken
import tempfile
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict

from streamlit.runtime.uploaded_file_manager import UploadedFile
from openai.types.fine_tuning.fine_tuning_job import FineTuningJob
from openai.pagination import SyncCursorPage

from typing import List, Dict, Any, Union, Optional

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Load environment variables
load_dotenv(override=True)

class FineTuneReport:
    """
    A class to process a dataset for fine-tuning, check for format errors,
    calculate token usage, and estimate training costs.
    """

    def __init__(self, model: str):
        """
        Initialize the FineTuneReport class with a specified model.

        Args:
            model (str): The model identifier to be used for token counting.
        """
        self.model = model

    @staticmethod
    def load_data(uploaded_file) -> Union[List[Dict[str, Any]], List]:
        """
        Load dataset from the uploaded JSONL file.

        Args:
            uploaded_file: A file-like object containing the JSONL data.

        Returns:
            A list of dictionaries representing the dataset or an empty list if there's an error.
        """
        try:
            # Load and decode each line in the uploaded file
            dataset = [json.loads(line.decode("utf-8")) for line in uploaded_file]
            _logger.info("Dataset loaded successfully")
            return dataset
        except json.JSONDecodeError as e:
            st.error(f"Error decoding JSON: {e}")
            _logger.error(f"Failed to load dataset: {e}")
            return []

    @staticmethod
    def check_message_format(
        message: Dict[str, Any],
        format_errors: Dict[str, int],
        allowed_keys: set,
        allowed_roles: set,
    ) -> None:
        """
        Check the format of a single message and log any format errors.

        Args:
            message: A dictionary representing a message.
            format_errors: A dictionary to record error counts.
            allowed_keys: A set of allowed keys in message dictionaries.
            allowed_roles: A set of allowed roles in message dictionaries.
        """
        if not all(key in message for key in ("role", "content")):
            format_errors["message_missing_key"] += 1
            _logger.warning("Message missing required keys: 'role' or 'content'.")

        if not set(message).issubset(allowed_keys):
            format_errors["message_unrecognized_key"] += 1
            _logger.warning(
                f"Message contains unrecognized keys: {set(message) - allowed_keys}"
            )

        if message.get("role") not in allowed_roles:
            format_errors["unrecognized_role"] += 1
            _logger.warning(
                f"Message contains unrecognized role: {message.get('role')}"
            )

        content = message.get("content")
        function_call = message.get("function_call")
        if (not content and not function_call) or not isinstance(content, str):
            format_errors["missing_content"] += 1
            _logger.warning(
                "Message is missing content or contains non-string content."
            )

    @staticmethod
    def check_example_format(
        ex: Dict[str, Any],
        format_errors: Dict[str, int],
        allowed_keys: set,
        allowed_roles: set,
    ) -> None:
        """
        Check the format of a single example and log any format errors.

        Args:
            ex: A dictionary representing a single example from the dataset.
            format_errors: A dictionary to record error counts.
            allowed_keys: A set of allowed keys in message dictionaries.
            allowed_roles: A set of allowed roles in message dictionaries.
        """
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            _logger.warning("Example is not a dictionary.")
            return

        messages = ex.get("messages")
        if messages is None:
            format_errors["missing_messages_list"] += 1
            _logger.warning("Example is missing 'messages' key.")
            return

        for message in messages:
            FineTuneReport.check_message_format(
                message, format_errors, allowed_keys, allowed_roles
            )

        if not any(msg.get("role") == "assistant" for msg in messages):
            format_errors["example_missing_assistant_message"] += 1
            _logger.warning("Example is missing an 'assistant' role message.")

    @staticmethod
    def format_errors(dataset: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Check for formatting errors in the entire dataset and return a report.

        Args:
            dataset: A list of dictionaries representing the dataset.

        Returns:
            A dictionary with counts of different formatting errors found.
        """
        format_errors = defaultdict(int)
        allowed_keys = {"role", "content", "name", "function_call", "weight"}
        allowed_roles = {"system", "user", "assistant", "function"}

        for ex in dataset:
            FineTuneReport.check_example_format(
                ex, format_errors, allowed_keys, allowed_roles
            )

        _logger.info("Format errors checked.")
        return format_errors

    @staticmethod
    def num_tokens_from_messages(messages: List[Dict[str, Any]], model: str) -> int:
        """
        Return the number of tokens used by a list of messages.

        Args:
            messages: A list of message dictionaries containing 'role', 'content', etc.
            model: The model identifier for token counting.

        Returns:
            Total token count for the provided messages.
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            _logger.warning(
                "Model not found. Using 'cl100k_base' encoding as fallback."
            )
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens_per_message = 3
        tokens_per_name = 1

        num_tokens = tokens_per_message * len(messages)
        for message in messages:
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name

        num_tokens += 3  # End of response tokens
        _logger.info(f"Total tokens calculated: {num_tokens}")
        return num_tokens

    @staticmethod
    def calculate_optimal_epochs(dataset: List[Dict[str, Any]]) -> int:
        """
        Calculate the optimal number of training epochs based on the dataset size.

        This method determines the number of training epochs based on the number of examples
        in the dataset. It adjusts the number of epochs to meet the requirements for the
        minimum and maximum number of training examples per epoch.

        Args:
            dataset (List[Dict[str, Any]]): A list of dictionaries representing the dataset.

        Raises:
            ValueError: If the dataset is empty or invalid.

        Returns:
            int: The optimal number of epochs for training.
        """
        # Pricing and default epochs estimate
        TARGET_EPOCHS = 3
        MIN_TARGET_EXAMPLES = 100
        MAX_TARGET_EXAMPLES = 25000
        MIN_DEFAULT_EPOCHS = 1
        MAX_DEFAULT_EPOCHS = 25

        n_train_examples = len(dataset)

        # Log the dataset size
        _logger.info(f"Dataset contains {n_train_examples} training examples.")

        # Ensure the dataset has at least one example, otherwise raise an error
        if n_train_examples == 0:
            _logger.error("Dataset is empty. Cannot calculate optimal epochs.")
            raise ValueError("Dataset is empty. Please provide a valid dataset.")

        n_epochs = TARGET_EPOCHS

        # Adjust epochs if the total training examples are too few or too many
        if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
            n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
            _logger.info(f"Adjusting epochs to {n_epochs} due to small dataset.")
        elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
            n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)
            _logger.info(f"Adjusting epochs to {n_epochs} due to large dataset.")

        # Log the final number of epochs
        _logger.info(f"Optimal number of epochs calculated: {n_epochs}")

        return n_epochs

    @staticmethod
    def calculate_training_cost(
        num_tokens: int,
        epochs: int,
        model: str,
    ) -> float:
        """
        Calculate the training cost based on the number of tokens and the specific model rates.

        This method takes into account the model-specific rates for training tokens and
        calculates the total cost based on the number of tokens, epochs, and the selected model.

        Args:
            num_tokens (int): The number of tokens used for training.
            model (str): The model identifier for the fine-tuning job.
            epochs (int): The number of training epochs.

        Returns:
            float: The total cost of training in USD.

        Raises:
            ValueError: If the model is not recognized.
        """
        # Define rates for different models per million training tokens
        model_rates = {"gpt-4o-2024-08-06": 25.000, "gpt-4o-mini-2024-07-18": 3.000}

        # Get the rate for the specified model
        rate_per_million_tokens = model_rates.get(model)

        # If the model is not recognized, raise an exception
        if rate_per_million_tokens is None:
            _logger.error(f"Model {model} not recognized for cost calculation.")
            raise ValueError(
                f"Model {model} not recognized. Please specify a valid model."
            )

        # Calculate the cost
        cost = (num_tokens / 1_000_000) * rate_per_million_tokens * epochs

        # Log the calculated cost
        _logger.info(
            f"Training cost for {model} with {num_tokens} tokens over {epochs} epochs: ${cost}"
        )

        return round(cost, 3)

    @classmethod
    def generate_report(cls, uploaded_file, model: str) -> Dict[str, Any]:
        """
        Generate a report including format errors, token count, and cost.

        Args:
            uploaded_file: The file uploaded by the user containing the dataset.
            model: The model identifier for token counting.

        Returns:
            A dictionary containing format error counts, total token count, and estimated training cost.
        """
        try:
            # Load data
            dataset = cls.load_data(uploaded_file)
            if not dataset:
                raise ValueError("Dataset could not be loaded or is empty.")

            # Check for format errors
            format_error_report = cls.format_errors(dataset)

            # Calculate token usage
            total_tokens = 0
            for example in dataset:
                messages = example.get("messages", [])
                total_tokens += cls.num_tokens_from_messages(messages, model)

            # Calculate optimal epochs
            n_epochs = cls.calculate_optimal_epochs(dataset)

            # Calculate cost
            cost = cls.calculate_training_cost(
                total_tokens, epochs=n_epochs, model=model
            )

            _logger.info("Report generation completed successfully.")
            # Return report as a dictionary
            return {
                "format_errors": format_error_report,
                "total_tokens": total_tokens,
                "total_training_tokens": total_tokens * n_epochs,
                "training_cost": f"${cost}",
                "optimal_epochs": n_epochs,
            }

        except Exception as e:
            _logger.error(f"Error generating report: {e}")
            return {"error": str(e)}


class FineTuneModel:
    """
        A class to manage the fine-tuning process of a model using OpenAI's API.

        Attributes:
            model (str): The model ID to fine-tune.
            training_data_file (str): Path to the JSONL training dataset.
            validation_data_file (Optional[str]): Path to the JSONL validation dataset.
            model_suffix (Optional[str]): Suffix for the fine-tuned model name.
            seed_input (Optional[int]): Random seed for reproducibility.
            batch_size_input (Optional[int]): Batch size for training.
            learning_rate_input (Optional[float]): Learning rate multiplier.
            epochs_input (Optional[int]): Number of epochs for training.
            wandb_integrations (bool): Flag to enable Weights & Biases integration.
            wandb_config (Dict[str, Any]): Configuration for Weights & Biases.

        Example Usage:
        ```python
        # Step 1: Import the FineTuneModel class
        from your_module import FineTuneModel  # Replace 'your_module' with the actual module name

        # Step 2: Set your API key (Ensure this is done before using the class)
        import os
        os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

        # Step 3: Upload your training file
        training_file_id = FineTuneModel.upload_file(
            file_path="path/to/training_data.jsonl",
            purpose="fine-tune"
        )

        # Step 4: Fine-tune the model using the uploaded training file
        fine_tuning_job = FineTuneModel.fine_tune_model(
            model="gpt-3.5-turbo",  # Specify the model ID you want to fine-tune
            training_data_file=training_file_id,  # Use the ID returned from upload_file
            validation_data_file=None,  # Optional: provide validation file ID if available
            model_suffix="custom-model",  # Optional: suffix for the fine-tuned model name
            seed_input=42,  # Optional: set a random seed for reproducibility
            batch_size_input=4,  # Optional: specify batch size
            learning_rate_input=0.01,  # Optional: specify learning rate multiplier
            epochs_input=5,  # Optional: specify number of epochs
            wandb_integrations=True,  # Optional: enable Weights & Biases integration
            wandb_config={"project": "fine-tuning"}  # Optional: Weights & Biases configuration
        )

        # Step 5: Check the fine-tuning job status (Optional)
        print(f"Fine-tuning job started with ID: {fine_tuning_job.id}")
    ```
    """

    def __init__(self):

        # Initialize the OpenAI client if it has not been initialized yet
        api_key = os.getenv("OPENAI_API_KEY")
        _logger.info("Initializing OpenAI client... With API key: " + api_key)
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

        self.client = OpenAI(api_key=api_key)
        _logger.info(f"OpenAI client: {self.client.__dict__}")

    def fine_tune_model(
        self,
        model: str,
        training_data_file: str,
        validation_data_file: Optional[str] = None,
        model_suffix: Optional[str] = None,
        seed_input: Optional[int] = None,
        batch_size_input: Optional[int] = None,
        learning_rate_input: Optional[float] = None,
        epochs_input: Optional[int] = None,
        wandb_integrations: bool = False,
        wandb_config: Optional[Dict[str, Any]] = None,
    ) -> "FineTuningJob":
        """
        Prepare the request body for creating a fine-tuning job.

        Args:
            model (str):
                Required. The name of the model to fine-tune.
                You can select one of the supported models.

            training_file (str):
                Required. The ID of an uploaded file that contains training data.
                The dataset must be formatted as a JSONL file, uploaded with the purpose 'fine-tune'.
                The contents must differ based on whether the model uses the chat or completions format.

            hyperparameters (Optional[Dict[str, Any]]):
                Optional. A dictionary of hyperparameters used for the fine-tuning job.

            suffix (Optional[str]):
                Optional. A string of up to 64 characters that will be added to the fine-tuned model name.
                For example, a suffix of "custom-model-name" would produce a model name like
                ft:gpt-4o-mini:openai:custom-model-name:7p4lURel.

            validation_file (Optional[str]):
                Optional. The ID of an uploaded file that contains validation data.
                If provided, this data will be used to generate validation metrics periodically during fine-tuning.
                The same data should not be present in both training and validation files.

            integrations (Optional[List[Dict[str, Any]]]):
                Optional. A list of integrations to enable for the fine-tuning job.

            seed (Optional[int]):
                Optional. An integer seed that controls the reproducibility of the job.
                Passing in the same seed and job parameters should yield the same results, but may differ in rare cases.
                If a seed is not specified, one will be generated automatically.

        Returns:
            FineTuningJob: The created fine-tuning job instance.

        Raises:
            ValueError: If the model name or training file ID is not provided.

        Example:
            job = FineTuneModel.fine_tune_model(
                model="gpt-3.5-turbo",
                training_data_file="path/to/training_data.jsonl",
                validation_data_file="path/to/validation_data.jsonl",
                model_suffix="my-custom-model",
                seed_input=42,
                batch_size_input=4,
                learning_rate_input=0.01,
                epochs_input=5,
                wandb_integrations=True,
                wandb_config={"project": "fine-tuning"}
            )
        """
        _logger.info("Starting fine-tuning job...")

        # Prepare job creation arguments
        job_args = {
            "model": model,
            "training_file": training_data_file,
            "validation_data": validation_data_file,
            "model_suffix": model_suffix,
            "seed": seed_input,
            "hyperparameters": {
                "batch_size": batch_size_input,
                "learning_rate_multiplier": learning_rate_input,
                "num_epochs": epochs_input,
            },
        }

        # Remove any None values from the job_args dictionary
        job_args = {k: v for k, v in job_args.items() if v is not None}

        # Add Weights & Biases integration if enabled
        if wandb_integrations:
            job_args["integrations"] = {
                "type": "wandb",
                "wandb": wandb_config,
            }
            _logger.info("Weights & Biases integration enabled.")

        try:
            # Create and return the fine-tuning job
            job = self.client.fine_tuning.jobs.create(
                model=model,
                training_file=training_data_file,
                hyperparameters={
                    "batch_size": batch_size_input,
                    "learning_rate_multiplier": learning_rate_input,
                    "n_epochs": epochs_input,
                },
                seed=seed_input,
                suffix=model_suffix,
                validation_file=validation_data_file,
            )
            _logger.info(f"Fine-tuning job created successfully: {job.id}")
            return job
        except Exception as e:
            _logger.error(f"Error creating fine-tuning job: {e}")
            raise

    def upload_file(self, file_path: str, purpose: str) -> str:
        """
        Upload a JSONL file to the OpenAI API for fine-tuning.

        Args:
            file_path (str): The path to the file to upload (must be a JSONL file).
            purpose (str): The purpose of the file upload (must be 'fine-tune').

        Returns:
            str: The ID of the uploaded file.

        Raises:
            ValueError: If the provided file is not a JSONL file.

        Example:
            file_id = FineTuneModel.upload_file(
                file_path="path/to/training_data.jsonl",
                purpose="fine-tune"
            )
        """
        try:
            # Upload the temporary file to OpenAI API
            with open(file_path, "rb") as f:
                response = self.client.files.create(file=f, purpose=purpose)
                file_id = response.id
                _logger.info(f"File uploaded successfully: {file_id}")
            return file_id
        except Exception as e:
            _logger.error(f"Error uploading file: {e}")
            raise

    def list_fine_tune_events(self, fine_tuning_job_id: str) -> SyncCursorPage[FineTuningJob]:
        return self.client.fine_tuning.jobs.list_events(
            fine_tuning_job_id=fine_tuning_job_id
        )
        
    def list_all_fine_tune_jobs(self) -> List[FineTuningJob]:
        jobs = self.client.fine_tuning.jobs.list()
        
        