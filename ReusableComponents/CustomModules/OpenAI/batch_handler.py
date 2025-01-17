"""
batch_handler.py
----------------
This module contains a class to handle batch requests to OpenAI API.
It includes the following features:
- Batch multiple requests into a single request.
- Handles batch files with multiple requests.
- Batch image requests into a single request.
- Handles batch image files with multiple requests.
- Retrieves batch responses and returns them as a list.
- Cancel batch requests if needed.
- Lists batch jobs and filters them.

Requirements:
-------------
- python-dotenv==1.0.1
- openai==1.59.3

Author:
-------
Sourav Das

Date:
-----
16.01.2025
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Importing necessary libraries and modules
import json
import logging
import tempfile
from openai import OpenAI
from dotenv import load_dotenv

from typing import List, Dict, Any, Literal, Optional

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Load environment variables
load_dotenv(override=True)


class BatchHandler:
    """
    A class to handle batch requests to OpenAI API.
    This class has the following features:
    - Batch multiple requests into a single request.
    - Handles batch files with multiple requests.
    - Retrieves batch responses and returns them as a list.
    - Cancel batch requests if needed.
    - Lists batch jobs and filters them.
    """

    def __init__(self, openai_key: str, model: str = "gpt-4o") -> None:
        """
        Initialize the BatchHandler class.

        Args:
        - openai_key (str): The OpenAI API key to use for authentication.
        - model (str): The model to use for the requests. Defaults to "gpt-4o".
        """

        # Store the provided API key and model
        self.openai_key = openai_key
        self.model = model
        _logger.info(
            f"Initializing BatchHandler with openai_key: {openai_key} and model: {model}"
        )

        # Initialize the OpenAI client with the API key
        self.client = OpenAI(api_key=self.openai_key)
        _logger.info(f"Initialized OpenAI client with API key: {openai_key}")

        # Mapping of endpoints to tasks
        self.ENDPOINT_TASK_MAPPING = {
            # Completion endpoint
            "completion": "/v1/completions",
            # Chat completion endpoint
            "chat_completion": "/v1/chat/completions",
            # Embeddings endpoint
            "embeddings": "/v1/embeddings",
        }
        _logger.info(f"Endpoint-task mapping: {self.ENDPOINT_TASK_MAPPING}")

    def _validate_request(self, request: Dict[str, Any]) -> bool:
        """
        Validate a single request object.

        Validate a single request object to ensure it contains all the required keys
        and the optional keys have valid default values.

        Args:
        - request (Dict[str, Any]): The request object to validate.

        Returns:
        - bool: True if the request is valid, False otherwise.
        """
        # Mandatory keys. These keys must be present in the request object.
        mandatory_keys = ["model", "user_prompt"]

        # Optional keys with default values. If these keys are not present in the
        # request object, the default values will be used.
        optional_keys = {
            "temperature": 0.7,  # The temperature controls the randomness of the output.
            "max_tokens": 300,  # The maximum number of tokens to generate.
            "top_p": 1.0,  # The threshold for the top tokens to consider.
            "frequency_penalty": 0.0,  # The penalty for repeating tokens.
            "presence_penalty": 0.0,  # The penalty for tokens that are not in the input.
        }

        # Check mandatory keys
        _logger.info("Validating mandatory keys...")
        for key in mandatory_keys:
            if key not in request:
                _logger.error(f"Request is missing mandatory field: '{key}'.")
                _logger.info(f"Validation failed: Missing mandatory field '{key}'")
                return False
            _logger.info(f"Mandatory field '{key}' is present.")

        # Set default values for optional keys if missing
        _logger.info("Setting default values for optional keys if missing...")
        for key, default_value in optional_keys.items():
            if key not in request:
                request[key] = default_value
                _logger.info(
                    f"Optional field '{key}' not provided. Using default: {default_value}."
                )
                _logger.info(
                    f"Optional field '{key}' not provided. Using default: {default_value}."
                )

        _logger.info("Request validation completed successfully.")
        return True
    
    def _validate_image_request(self, request: Dict[str, Any]) -> bool:
        """
        Validate a single request object.

        Validate a single request object to ensure it contains all the required keys
        and the optional keys have valid default values.

        Args:
        - request (Dict[str, Any]): The request object to validate.

        Returns:
        - bool: True if the request is valid, False otherwise.
        """
        # Mandatory keys. These keys must be present in the request object.
        mandatory_keys = ["model", "user_prompt", "image_url"]

        # Optional keys with default values. If these keys are not present in the
        # request object, the default values will be used.
        optional_keys = {
            "temperature": 0.7,  # The temperature controls the randomness of the output.
            "max_tokens": 300,  # The maximum number of tokens to generate.
            "top_p": 1.0,  # The threshold for the top tokens to consider.
            "frequency_penalty": 0.0,  # The penalty for repeating tokens.
            "presence_penalty": 0.0,  # The penalty for tokens that are not in the input.
        }

        # Check mandatory keys
        _logger.info("Validating mandatory keys...")
        for key in mandatory_keys:
            if key not in request:
                _logger.error(f"Request is missing mandatory field: '{key}'.")
                _logger.info(f"Validation failed: Missing mandatory field '{key}'")
                return False
            _logger.info(f"Mandatory field '{key}' is present.")

        # Set default values for optional keys if missing
        _logger.info("Setting default values for optional keys if missing...")
        for key, default_value in optional_keys.items():
            if key not in request:
                request[key] = default_value
                _logger.info(
                    f"Optional field '{key}' not provided. Using default: {default_value}."
                )
                _logger.info(
                    f"Optional field '{key}' not provided. Using default: {default_value}."
                )

        _logger.info("Request validation completed successfully.")
        return True

    def _format_batch_request(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format a list of requests into a batch request object.

        Args:
        - requests (List[Dict[str, Any]]): A list of request objects.

        Returns:
        - Dict[str, Any]: The file path of the temporary JSONL file containing batch requests.

        Raises:
        - ValueError: If a request object is invalid.
        """
        batch_requests = []

        # Iterate over each request, validate and format it
        for i, request in enumerate(requests):
            _logger.info(f"Validating request {i} of {len(requests)}...")
            if not self._validate_request(request):
                raise ValueError("Invalid request object.")
            _logger.info(f"Request {i} is valid.")
            batch_requests.append(
                {
                    "custom_id": f"request_{i}",
                    "method": "POST",
                    "url": request["url"],
                    "body": {
                        "model": request["model"],
                        "temperature": request["temperature"],
                        "max_tokens": request["max_tokens"],
                        "top_p": request["top_p"],
                        "frequency_penalty": request["frequency_penalty"],
                        "presence_penalty": request["presence_penalty"],
                        "response_format": {
                            "type": request["response_format"],
                        },
                        "messages": [
                            {
                                "role": "system",
                                "content": request["system_prompt"],
                            },
                            {
                                "role": "user",
                                "content": request["user_prompt"],
                            },
                        ],
                    },
                }
            )
            _logger.info(f"Formatted request {i}.")

        # Create a temporary file and save the batch requests as JSONL
        _logger.info("Creating a temporary file to save the batch requests as JSONL...")
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".jsonl", mode="w", encoding="utf-8"
        )
        for batch_request in batch_requests:
            _logger.info(
                f"Writing batch request {batch_request['custom_id']} to the temporary file..."
            )
            temp_file.write(json.dumps(batch_request) + "\n")
        _logger.info("Finished writing batch requests to the temporary file.")
        temp_file.close()

        # Return the name of the temporary file
        _logger.info(f"Returning the name of the temporary file: {temp_file.name}.")
        return temp_file.name

    def _format_image_batch_request(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format a list of requests into a batch request object.

        Args:
        - requests (List[Dict[str, Any]]): A list of request objects.

        Returns:
        - Dict[str, Any]: The file path of the temporary JSONL file containing batch requests.

        Raises:
        - ValueError: If a request object is invalid.
        """
        batch_requests = []

        # Iterate over each request, validate and format it
        for i, request in enumerate(requests):
            _logger.info(f"Validating request {i} of {len(requests)}...")
            if not self._validate_image_request(request):
                raise ValueError("Invalid request object.")
            _logger.info(f"Request {i} is valid.")
            batch_requests.append(
                {
                    "custom_id": f"request_{i}",
                    "method": "POST",
                    "url": request["url"],
                    "body": {
                        "model": request["model"],
                        "temperature": request["temperature"],
                        "max_tokens": request["max_tokens"],
                        "top_p": request["top_p"],
                        "frequency_penalty": request["frequency_penalty"],
                        "presence_penalty": request["presence_penalty"],
                        "response_format": {
                            "type": request["response_format"],
                        },
                        "messages": [
                            {
                                "role": "system",
                                "content": request["system_prompt"],
                            },
                            {
                                "role": "user",
                                "content": [
                                        {
                                            "type": "text",
                                            "content": request["user_prompt"],
                                        },
                                        {
                                            "type": "image",
                                            "content": request["image_path"],
                                        }
                                    ],
                            },
                        ],
                    },
                }
            )
            _logger.info(f"Formatted request {i}.")

        # Create a temporary file and save the batch requests as JSONL
        _logger.info("Creating a temporary file to save the batch requests as JSONL...")
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".jsonl", mode="w", encoding="utf-8"
        )
        for batch_request in batch_requests:
            _logger.info(
                f"Writing batch request {batch_request['custom_id']} to the temporary file..."
            )
            temp_file.write(json.dumps(batch_request) + "\n")
        _logger.info("Finished writing batch requests to the temporary file.")
        temp_file.close()

        # Return the name of the temporary file
        _logger.info(f"Returning the name of the temporary file: {temp_file.name}.")
        return temp_file.name

    def _save_batchfile_to_openai(self, batch_file_path: str) -> str:
        """
        Save a batch file to OpenAI.

        This method reads the batch file from the local file system and uploads it to OpenAI
        using the Files API. The purpose of the file is specified as "batch" to indicate that
        it is a batch job.

        Args:
        - batch_file_path (str): The path to the batch file.

        Returns:
        - str: The ID of the batch job.
        """
        _logger.info(f"Reading batch file from {batch_file_path}...")
        with open(batch_file_path, "rb") as file:
            batch_file = file.read()
        _logger.info("Finished reading batch file.")
        # Upload the batch file to OpenAI
        _logger.info("Uploading batch file to OpenAI...")
        file_id = self.client.files.create(file=batch_file, purpose="batch").id
        _logger.info(f"Finished uploading batch file. File ID: {file_id}.")
        return file_id

    def create_batch(
        self,
        task: Literal["completion", "chat_completion", "embeddings"],
        data: List[str] = None,
        data_file: str = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 300,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        response_format: str = "text",
        system_prompt: str = None,
        metadata: Dict[str, Any] = None,
        completion_window: str = "24h",
    ) -> Dict[str, Any]:
        """
        Create a batch request to the OpenAI API.

        This function formats and sends a batch request to the OpenAI API,
        using the specified task and data. It supports processing data from
        a list or a file, and provides options for customizing the request
        parameters.

        Args:
            task (Literal["completion", "chat_completion", "embeddings"]): The task type.
            data (List[str], optional): List of data strings for the batch.
            data_file (str, optional): File path to read data from.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling parameter.
            max_tokens (int): Maximum tokens for the response.
            frequency_penalty (float): Penalty for token frequency.
            presence_penalty (float): Penalty for token presence.
            response_format (str): Format of the response.
            system_prompt (str, optional): System-level prompt.
            metadata (Dict[str, Any], optional): Additional metadata for the request.
            completion_window (str): Time window for batch completion.

        Returns:
            Dict[str, Any]: Response from the OpenAI API.

        Raises:
            ValueError: If invalid arguments are provided.

        Example:
        --------
        ```python
        from batch_handler import BatchHandler

        # Create an instance of BatchHandler
        batch_handler = BatchHandler(openai_key="<YOUR_OPENAI_API_KEY>")

        # Create a batch request
        batch_response = batch_handler.create_batch(
            task="completion",
            data=["Hello", "World"],
            temperature=0.7,
            top_p=1.0,
            max_tokens=100,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            response_format="text",
            system_prompt="System: ",
            metadata={"customer_id": "user_123456789"},
            completion_window="24h"
        )
        ```
        """
        _logger.info("Starting create_batch function...")

        # Validate input arguments
        if data is None and data_file is None:
            _logger.info("Error: Neither 'data' nor 'data_file' provided!")
            raise ValueError("Either 'data' or 'data_file' must be provided.")

        if data is not None and data_file is not None:
            _logger.info("Error: Both 'data' and 'data_file' provided!")
            raise ValueError("Only one of 'data' or 'data_file' should be provided.")

        # Read data from file if provided
        if data_file is not None:
            _logger.info(f"Reading data from file: {data_file}")
            with open(data_file, "r") as file:
                data = file.readlines()

        # Validate response format
        if response_format not in ["text", "json_schema"]:
            _logger.info(f"Error: Invalid response_format '{response_format}'")
            raise ValueError("response_format must be either 'text' or 'json_schema'")

        # Validate task
        if task not in self.ENDPOINT_TASK_MAPPING.keys():
            _logger.info(f"Error: Invalid task '{task}'")
            raise ValueError(
                f"task must be one of {list(self.ENDPOINT_TASK_MAPPING.keys())}"
            )

        _logger.info("Formatting batch request...")
        # Format batch request
        formatted_requests = self._format_batch_request(
            requests=[
                {
                    "model": self.model,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "response_format": response_format,
                    "system_prompt": system_prompt,
                    "user_prompt": prompt,
                    "url": self.ENDPOINT_TASK_MAPPING[task],
                }
                for prompt in data
            ]
        )

        _logger.info("Saving batch request to OpenAI...")
        # Save batch request to OpenAI
        input_file_id = self._save_batchfile_to_openai(formatted_requests)

        _logger.info("Creating batch request with OpenAI...")
        # Create batch request with OpenAI
        batch_response = self.client.batches.create(
            completion_window=completion_window,
            input_file_id=input_file_id,
            endpoint=self.ENDPOINT_TASK_MAPPING[task],
            metadata=metadata,
        ).to_dict()

        _logger.info("Batch request created successfully.")
        return batch_response

    def create_image_batch(
        self,
        task: str = "chat_completion",
        data: List[str] = None,
        data_file: str = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 300,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        response_format: str = "text",
        system_prompt: str = None,
        metadata: Dict[str, Any] = None,
        completion_window: str = "24h",
    ) -> Dict[str, Any]:
        """
        Create a batch request for image generation.

        Args:
        - data (List[str]): List of prompts for image generation.
        - n (int): Number of images to generate for each prompt.
        - size (str): Size of the image in pixels.
        - background_color (str): Background color of the image.
        - file_format (str): Format of the generated image.
        - metadata (Dict[str, Any], optional): Additional metadata for the request.
        - completion_window (str): Time window for batch completion.

        Returns:
        - Dict[str, Any]: Response from the OpenAI API.

        Raises:
        - ValueError: If invalid arguments are provided.

        Example:
        --------
        ```python
        from batch_handler import BatchHandler

        # Create an instance of BatchHandler
        batch_handler = BatchHandler(openai_key="<YOUR_OPENAI_API_KEY>")

        # Create a batch request for image generation
        batch_response = batch_handler.create_image_batch(
            data=["A cat", "A dog"],
            n=1,
            size="1024x1024",
            background_color="white",
            file_format="png",
            metadata={"customer_id": "user_123456789"},
            completion_window="24h"
        )
        ```
        """
        _logger.info("Starting create_batch function...")

        # Validate input arguments
        if data is None and data_file is None:
            _logger.info("Error: Neither 'data' nor 'data_file' provided!")
            raise ValueError("Either 'data' or 'data_file' must be provided.")

        if data is not None and data_file is not None:
            _logger.info("Error: Both 'data' and 'data_file' provided!")
            raise ValueError("Only one of 'data' or 'data_file' should be provided.")

        # Read data from file if provided
        if data_file is not None:
            _logger.info(f"Reading data from file: {data_file}")
            with open(data_file, "r") as file:
                data = file.readlines()

        # Validate response format
        if response_format not in ["text", "json_schema"]:
            _logger.info(f"Error: Invalid response_format '{response_format}'")
            raise ValueError("response_format must be either 'text' or 'json_schema'")

        # Validate task for image QA
        if task != "chat_completion":
            _logger.info(f"Error: Invalid task '{task}'")
            raise ValueError("task must `chat_completion` for image based QA")

        _logger.info("Formatting batch request...")
        # Format batch request
        formatted_requests = self._format_batch_request(
            requests=[
                {
                    "model": self.model,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "response_format": response_format,
                    "system_prompt": system_prompt,
                    "user_prompt": prompt,
                    "url": self.ENDPOINT_TASK_MAPPING[task],
                }
                for prompt in data
            ]
        )

        _logger.info("Saving batch request to OpenAI...")
        # Save batch request to OpenAI
        input_file_id = self._save_batchfile_to_openai(formatted_requests)

        _logger.info("Creating batch request with OpenAI...")
        # Create batch request with OpenAI
        batch_response = self.client.batches.create(
            completion_window=completion_window,
            input_file_id=input_file_id,
            endpoint=self.ENDPOINT_TASK_MAPPING[task],
            metadata=metadata,
        ).to_dict()

        _logger.info("Batch request created successfully.")
        return batch_response

    def retrieve_batch(self, batch_id: str) -> Dict[str, Any]:
        """
        Retrieve a batch response from OpenAI.

        Args:
            batch_id (str): The ID of the batch job.

        Returns:
            Dict[str, Any]: The response from the API.

        Raises:
            ValueError: If the batch_id is invalid.

        Example:
        --------
        ```python
        from batch_handler import BatchHandler

        # Create an instance of BatchHandler
        batch_handler = BatchHandler(openai_key="<YOUR_OPENAI_API_KEY>")

        # Retrieve a batch response
        batch_response = batch_handler.retrieve_batch(batch_id="batch_123456789")
        ```
        """
        _logger.info(f"Attempting to retrieve batch with ID: {batch_id}")
        try:
            response = self.client.batches.retrieve(batch_id).to_dict()
            _logger.info(f"Successfully retrieved batch with ID: {batch_id}")
            return response
        except Exception as e:
            _logger.info(f"Error retrieving batch with ID: {batch_id}")
            raise ValueError(f"Error retrieving batch: {e}") from e

    def fetch_batch_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """
        Fetch the results of a batch job from OpenAI.

        This method retrieves the results of a batch job from OpenAI and
        returns them as a list of dictionaries.

        Args:
        - batch_id (str): The ID of the batch job.

        Returns:
        - List[Dict[str, Any]]: A list of responses from the API.

        Raises:
        - ValueError: If the batch_id is invalid.

        Example:
        --------
        ```python
        from batch_handler import BatchHandler

        # Create an instance of BatchHandler
        batch_handler = BatchHandler(openai_key="<YOUR_OPENAI_API_KEY>")

        # Fetch the results of a batch job
        batch_results = batch_handler.fetch_batch_results(batch_id="batch_123456789")
        ```
        """
        _logger.info(f"Fetching results for batch ID: {batch_id}")
        try:
            # Retrieve the batch results from OpenAI
            batch_results = self.retrieve_batch(batch_id)
            _logger.info(f"Retrieved batch results for ID: {batch_id}")

            # Get the content of the output file
            _logger.info("Fetching output file content...")
            output_file_content = self.client.files.content(
                batch_results["output_file_id"]
            ).content
            _logger.info("Output file content fetched successfully.")

            # Parse the content of the output file as JSON
            _logger.info("Parsing output file content as JSON...")
            batch_results_list = json.loads(output_file_content)
            _logger.info("Parsed output file content successfully.")

            # Return the list of results
            _logger.info("Returning batch results list.")
            return batch_results_list
        except Exception as e:
            _logger.info(f"Error fetching batch results for ID: {batch_id}: {e}")
            raise

    def cancel_batch_request(self, batch_id: str) -> Dict[str, Any]:
        """
        Cancel a batch request from OpenAI.

        Args:
            batch_id (str): The ID of the batch job.

        Returns:
            Dict[str, Any]: The response from the API.

        Raises:
            ValueError: If the batch_id is invalid.

        Example:
        --------
        ```python
        from batch_handler import BatchHandler

        # Create an instance of BatchHandler
        batch_handler = BatchHandler(openai_key="<YOUR_OPENAI_API_KEY>")

        # Cancel a batch request
        batch_response = batch_handler.cancel_batch_request(batch_id="batch_123456789")
        ```
        """
        _logger.info(f"Attempting to cancel batch with ID: {batch_id}")
        try:
            _logger.info("Calling batch.cancel() method...")
            response = self.client.batches.cancel(batch_id).to_dict()
            _logger.info(f"Received response from batch.cancel(): {response}")
            _logger.info(f"Successfully cancelled batch with ID: {batch_id}")
            return response
        except Exception as e:
            _logger.info(f"Error cancelling batch with ID: {batch_id}")
            _logger.info(f"Error message: {e}")
            raise ValueError(f"Error cancelling batch: {e}") from e

    def list_batch_jobs(
        self,
        after: Optional[str] = None,
        limit: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all batch jobs from OpenAI with dynamic filters.

        Args:
            after (Optional[str]): The ID of the batch job to start the list from.
            limit (int): The maximum number of batch jobs to return. Defaults to 20.
            filters (Optional[Dict[str, Any]]): A dictionary of filters to apply.
                Example:
                    {
                        "status": "completed",
                        "customer_id": "user_123456789",
                        "created_at_range": [start_timestamp, end_timestamp]
                    }

        Returns:
            List[Dict[str, Any]]: A filtered list of batch jobs.

        Raises:
            ValueError: If there is an error fetching the batch jobs.

        Example:
        --------
        ```python
        from batch_handler import BatchHandler

        # Create an instance of BatchHandler
        batch_handler = BatchHandler(openai_key="<YOUR_OPENAI_API_KEY>")

        # List all batch jobs
        batch_jobs = batch_handler.list_batch_jobs()
        ```
        """
        _logger.info(
            f"Fetching batch jobs with limit={limit}, after={after}, filters={filters}"
        )

        try:
            # Call the batch list API
            if after:
                response = self.client.batches.list(after=after, limit=limit)
            else:
                response = self.client.batches.list(limit=limit)

            _logger.info(f"API response: {response}")

            # Parse the response
            batch_jobs = response.to_dict().get("data", [])

            # Apply additional filtering
            if filters:
                _logger.info("Applying filters to batch jobs...")
                filtered_jobs = []
                for job in batch_jobs:
                    if "status" in filters and job.get("status") != filters["status"]:
                        continue
                    if (
                        "customer_id" in filters
                        and job.get("metadata", {}).get("customer_id")
                        != filters["customer_id"]
                    ):
                        continue
                    if "created_at_range" in filters:
                        created_at = job.get("created_at")
                        start, end = filters["created_at_range"]
                        if created_at < start or created_at > end:
                            continue
                    filtered_jobs.append(job)
                batch_jobs = filtered_jobs

            _logger.info(
                f"Successfully fetched and filtered batch jobs. Count: {len(batch_jobs)}"
            )
            return batch_jobs

        except Exception as e:
            _logger.error(f"Error fetching batch jobs: {e}")
            raise ValueError(f"Failed to fetch batch jobs. Details: {e}")
