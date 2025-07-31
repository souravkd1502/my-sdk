"""
admin_handler.py
----------------

Requirements:
-------------

Description:
------------

Functions:
----------

Environment Variables:
----------------------
- OPENAI_API_KEY: The API key for OpenAI API.

TODO:
-----
1. Add function for Users API.
2. Add function for Projects API.
3. Add function for API Keys API.
4. Add function for Audit log API.

FIXME:
------

Author:
-------
Sourav Das

Date:
-----
17.01.2025
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Importing necessary libraries and modules
import os
import logging
import requests
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

from typing import Literal, List, Any, Dict, Optional, Union


# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Load environment variables
load_dotenv(override=True)


class AdminHandler:
    """ """

    USAGE_MODEL_MAPPING = {
        "completions": "https://api.openai.com/v1/organization/usage/completions",
        "embeddings": "https://api.openai.com/v1/organization/usage/embeddings",
        "moderations": "https://api.openai.com/v1/organization/usage/moderations",
        "images": "https://api.openai.com/v1/organization/usage/images",
        "audio": "https://api.openai.com/v1/organization/usage/audio_speeches",
        "transcription": "https://api.openai.com/v1/organization/usage/audio_transcriptions",
        "vector_store": "https://api.openai.com/v1/organization/usage/vector_stores",
    }

    def __init__(self, openai_key: str):
        """
        Initialize the AdminHandler class with the provided API key.

        Args:
        - openai_key (str): The API key for OpenAI. If not provided, it defaults to the environment variable 'OPENAI_API_KEY'.
        """
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.openai_key)

    @staticmethod
    def _format_params(**kwargs: Any) -> Dict[str, Any]:
        """
        Format the parameters for the API request.

        Args:
            **kwargs: Arbitrary keyword arguments containing the parameters for the API request.

        Returns:
            Dict[str, Any]: A dictionary containing the formatted parameters.

        Notes:
            - `start_time` and `end_time` should be provided as ISO 8601 formatted strings (e.g., "2025-01-17T00:00:00Z").
            - These are converted to Unix seconds for API compatibility.
            - Parameters with `None` values are excluded from the result.
        """
        params = {}
        for key, value in kwargs.items():
            if value is not None:
                # Convert start_time and end_time to Unix seconds
                if key in ["start_time", "end_time"]:
                    try:
                        # Assuming the input format is ISO 8601
                        value = int(
                            datetime.fromisoformat(
                                value.replace("Z", "+00:00")
                            ).timestamp()
                        )
                    except ValueError as e:
                        raise ValueError(
                            f"Invalid format for {key}. Expected ISO 8601 format."
                        ) from e
                params[key] = value
        return params

    def get_usage_data(
        self,
        usage_model: Literal[
            "completions",
            "embeddings",
            "moderations",
            "images",
            "audio",
            "transcription",
            "vector_store",
        ],
        start_date: str,
        end_date: str = None,
        time_interval: Literal["minutely", "hourly", "daily"] = "daily",
        project_ids: List = None,
        api_key_ids: List = None,
        user_ids: List = None,
        model: str = None,
        batch: bool = False,
        group_by: List = None,
        limit: int = 10,
        page: str = None,
    ):
        """
        Get the usage data for the OpenAI API key.

        Args:
        - usage_model (str): The usage model for which the data is to be fetched. It can be one of the following:
            - 'completions'
            - 'embeddings'
            - 'moderations'
            - 'images'
            - 'audio'
            - 'transcription'
            - 'vector_store'
        - start_date (str): The start date for the usage data in the format 'YYYY-MM-DD'.
        - end_date (str): The end date for the usage data in the format 'YYYY-MM-DD'. If not provided, it defaults to the 'start_date'.
        - time_interval (str): The time interval for the usage data. It can be one of the following:
            - 'minutely'
            - 'hourly'
            - 'daily'
        - project_ids (list[Optional]): A list of project IDs to filter the usage data by.
        - api_key_ids (list[Optional]): A list of API key IDs to filter the usage data by.
        - user_ids (list[Optional]): A list of user IDs to filter the usage data by.
        - model (str): The model to filter the usage data by.
        - batch (bool): A boolean flag to indicate whether the usage data is to be fetched in batches.
        - group_by (list[Optional]): A list of fields to group the usage data by.
            Support fields include project_id, user_id, api_key_id, model, batch or any combination of them.
        - limit (int): The limit for the number of records to fetch.
                        Specifies the number of buckets to return.
                            - bucket_width=1d: default: 7, max: 31
                            - bucket_width=1h: default: 24, max: 168
                            - bucket_width=1m: default: 60, max: 1440
        - page (str): The page token to fetch the next page of records. A cursor for use in pagination.
                        Corresponding to the next_page field from the previous response.

        Returns:
        - dict: A dictionary containing the usage data.

        Raises:
        - ValueError: If the server could not understand the request due to invalid syntax. (Status Code - 400)
        - PermissionError: If access is denied due to invalid or insufficient credentials. (Status Code - 401)
        - RuntimeError: If too many requests are sent in a given amount of time. (Status Code - 429)
        - RuntimeError: if the server has encountered a situation it doesn't know how to handle. (Status Code - 500)
        - requests.HTTPError: If an unexpected error occurs while making the API request. (Status Code - Any other)

        Notes:
        - The `start_date` and `end_date` should be provided in the format 'YYYY-MM-DD'.

        Examples:
        ---------
        1. Get the usage data for completions model for the last 7 days.
        ```python
        admin_handler = AdminHandler(openai_key="<OPENAI_API_KEY>")
        usage_details = admin_handler.get_usage_data(
            usage_model="completions",
            start_date="2025-01-01",
            end_date="2025-01-17",
            time_interval="daily",
            model="gpt-4o",
            batch=False,
            group_by=["project_id", "user_id"],
            limit=10,
        )
        print(usage_details)
        ```
        """
        # Define the endpoint and headers
        url = self.USAGE_MODEL_MAPPING[usage_model]
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json",
        }

        # Define the query parameters
        kwargs = {
            "start_date": start_date,
            "end_date": end_date,
            "time_interval": time_interval,
            "project_ids": project_ids,
            "api_key_ids": api_key_ids,
            "user_ids": user_ids,
            "model": model,
            "batch": batch,
            "group_by": group_by,
            "limit": limit,
            "page": page,
        }
        params = self._format_params(**kwargs)

        # Make the API request
        response = requests.get(url, headers=headers, params=params)

        # Handle the response
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 400:
            _logger.error(f"Bad Request: {response.status_code}, {response.text}")
            raise ValueError(
                "Bad Request: The server could not understand the request due to invalid syntax."
            )
        elif response.status_code == 401:
            _logger.error(f"Unauthorized: {response.status_code}, {response.text}")
            raise PermissionError(
                "Unauthorized: Access is denied due to invalid or insufficient credentials."
            )
        elif response.status_code == 429:
            _logger.error(f"Too Many Requests: {response.status_code}, {response.text}")
            raise RuntimeError(
                "Too Many Requests: You have sent too many requests in a given amount of time."
            )
        elif response.status_code == 500:
            _logger.error(
                f"Internal Server Error: {response.status_code}, {response.text}"
            )
            raise RuntimeError(
                "Internal Server Error: The server has encountered a situation it doesn't know how to handle."
            )
        else:
            _logger.error(f"Unexpected Error: {response.status_code}, {response.text}")
            response.raise_for_status()

    def get_cost_data(
        self,
        start_date: str,
        end_date: str = None,
        project_ids: List = None,
        api_key_ids: List = None,
        user_ids: List = None,
        model: str = None,
        batch: bool = False,
        group_by: List = None,
        limit: int = 10,
        page: str = None,
    ):
        """
        Get the cost data for the OpenAI API key.

        Args:
        - start_date (str): The start date for the usage data in the format 'YYYY-MM-DD'.
        - end_date (str): The end date for the usage data in the format 'YYYY-MM-DD'. If not provided, it defaults to the 'start_date'.
        - project_ids (list[Optional]): A list of project IDs to filter the usage data by.
        - api_key_ids (list[Optional]): A list of API key IDs to filter the usage data by.
        - user_ids (list[Optional]): A list of user IDs to filter the usage data by.
        - model (str): The model to filter the usage data by.
        - batch (bool): A boolean flag to indicate whether the usage data is to be fetched in batches.
        - group_by (list[Optional]): A list of fields to group the usage data by.
                                    Support fields include `project_id`, `line_item` and any combination of them.
        - limit (int): A limit on the number of buckets to be returned. Limit can range between 1 and 180, and the default is 7.
        - page (str): The page token to fetch the next page of records. A cursor for use in pagination.
                        Corresponding to the next_page field from the previous response.

        Returns:
        - dict: A dictionary containing the usage data.

        Raises:
        - ValueError: If the server could not understand the request due to invalid syntax. (Status Code - 400)
        - PermissionError: If access is denied due to invalid or insufficient credentials. (Status Code - 401)
        - RuntimeError: If too many requests are sent in a given amount of time. (Status Code - 429)
        - RuntimeError: if the server has encountered a situation it doesn't know how to handle. (Status Code - 500)
        - requests.HTTPError: If an unexpected error occurs while making the API request. (Status Code - Any other)

        Notes:
        - The `start_date` and `end_date` should be provided in the format 'YYYY-MM-DD'.

        Examples:
        ---------
        1. Get the usage data for completions model for the last 7 days.
        ```python
        admin_handler = AdminHandler(openai_key="<OPENAI_API_KEY>")
        usage_details = admin_handler.get_usage_data(
            usage_model="completions",
            start_date="2025-01-01",
            end_date="2025-01-17",
            time_interval="daily",
            model="gpt-4o",
            batch=False,
            group_by=["project_id", "user_id"],
            limit=10,
        )
        print(usage_details)
        ```
        """
        # Define the endpoint and headers
        url = "https://api.openai.com/v1/organization/costs"
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json",
        }

        # Define the query parameters
        kwargs = {
            "start_date": start_date,
            "end_date": end_date,
            "project_ids": project_ids,
            "api_key_ids": api_key_ids,
            "user_ids": user_ids,
            "model": model,
            "batch": batch,
            "group_by": group_by,
            "limit": limit,
            "page": page,
        }
        params = self._format_params(**kwargs)

        # Make the API request
        response = requests.get(url, headers=headers, params=params)

        # Handle the response
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 400:
            _logger.error(f"Bad Request: {response.status_code}, {response.text}")
            raise ValueError(
                "Bad Request: The server could not understand the request due to invalid syntax."
            )
        elif response.status_code == 401:
            _logger.error(f"Unauthorized: {response.status_code}, {response.text}")
            raise PermissionError(
                "Unauthorized: Access is denied due to invalid or insufficient credentials."
            )
        elif response.status_code == 429:
            _logger.error(f"Too Many Requests: {response.status_code}, {response.text}")
            raise RuntimeError(
                "Too Many Requests: You have sent too many requests in a given amount of time."
            )
        elif response.status_code == 500:
            _logger.error(
                f"Internal Server Error: {response.status_code}, {response.text}"
            )
            raise RuntimeError(
                "Internal Server Error: The server has encountered a situation it doesn't know how to handle."
            )
        else:
            _logger.error(f"Unexpected Error: {response.status_code}, {response.text}")
            response.raise_for_status()


class ModelPricing:
    """
    A static class for managing and calculating costs of various models and services.
    """

    MODEL_PRICING = {
        "gpt-4o": {
            "gpt-4o-2024-08-06": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
            "gpt-4o-2024-11-20": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
            "gpt-4o-2024-05-13": {"input": 5.00, "cached_input": None, "output": 15.00},
        },
        "gpt-4o-audio-preview": {
            "gpt-4o-audio-preview-2024-12-17": {
                "input": 2.50,
                "cached_input": None,
                "output": 10.00,
            },
            "gpt-4o-audio-preview-2024-10-01": {
                "input": 2.50,
                "cached_input": None,
                "output": 10.00,
            },
        },
        "gpt-4o-realtime-preview": {
            "gpt-4o-realtime-preview-2024-12-17": {
                "input": 5.00,
                "cached_input": 2.50,
                "output": 20.00,
            },
            "gpt-4o-realtime-preview-2024-10-01": {
                "input": 5.00,
                "cached_input": 2.50,
                "output": 20.00,
            },
        },
        "gpt-4o-mini": {
            "gpt-4o-mini-2024-07-18": {
                "input": 0.15,
                "cached_input": 0.075,
                "output": 0.60,
            },
        },
        "gpt-4o-mini-audio-preview": {
            "gpt-4o-mini-audio-preview-2024-12-17": {
                "input": 0.15,
                "cached_input": None,
                "output": 0.60,
            },
        },
        "gpt-4o-mini-realtime-preview": {
            "gpt-4o-mini-realtime-preview-2024-12-17": {
                "input": 0.60,
                "cached_input": 0.30,
                "output": 2.40,
            },
        },
        "o1": {
            "o1-2024-12-17": {"input": 15.00, "cached_input": 7.50, "output": 60.00},
            "o1-preview-2024-09-12": {
                "input": 15.00,
                "cached_input": 7.50,
                "output": 60.00,
            },
        },
        "o1-mini": {
            "o1-mini-2024-09-12": {
                "input": 3.00,
                "cached_input": 1.50,
                "output": 12.00,
            },
        },
        "audio_tokens": {
            "gpt-4o-audio-preview": {
                "input": 40.00,
                "cached_input": None,
                "output": 80.00,
            },
            "gpt-4o-mini-audio-preview": {
                "input": 10.00,
                "cached_input": None,
                "output": 20.00,
            },
        },
        "fine_tuning": {
            "gpt-4o-2024-08-06": {
                "training": 25.00,
                "input": 3.75,
                "cached_input": 1.875,
                "output": 15.00,
            },
            "gpt-4o-mini-2024-07-18": {
                "training": 3.00,
                "input": 0.30,
                "cached_input": 0.15,
                "output": 1.20,
            },
            "gpt-3.5-turbo": {
                "training": 8.00,
                "input": 3.00,
                "cached_input": None,
                "output": 6.00,
            },
            "davinci-002": {
                "training": 6.00,
                "input": 12.00,
                "cached_input": None,
                "output": 12.00,
            },
            "babbage-002": {
                "training": 0.40,
                "input": 1.60,
                "cached_input": None,
                "output": 1.60,
            },
        },
        "assistants_api": {
            "code_interpreter": {"cost": 0.03},
            "file_search": {"cost": 0.10},
        },
        "transcription_and_speech_generation": {
            "whisper_transcription": {"cost_per_minute": 0.006},
            "tts_speech_generation": {"cost_per_1m_chars": 15.00},
            "tts_hd_speech_generation": {"cost_per_1m_chars": 30.00},
        },
        "image_generation": {
            "dalle_3_standard": {"1024x1024": 0.04, "1024x1792": 0.08},
            "dalle_3_hd": {"1024x1024": 0.08, "1024x1792": 0.12},
            "dalle_2": {"256x256": 0.016, "512x512": 0.018, "1024x1024": 0.02},
        },
        "embeddings": {
            "text-embedding-3-small": {"cost_per_1m_tokens": 0.02},
            "text-embedding-3-large": {"cost_per_1m_tokens": 0.13},
            "text-embedding-ada-002": {"cost_per_1m_tokens": 0.10},
        },
        "other_models": {
            "chatgpt-4o-latest": {"input": 5.00, "output": 15.00},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-4": {"input": 30.00, "output": 60.00},
            "gpt-4-32k": {"input": 60.00, "output": 120.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            "gpt-3.5-turbo-16k": {"input": 3.00, "output": 4.00},
        },
    }

    @staticmethod
    def filter_by_cost(
        pricing: Dict,
        cost_type: str,
        max_cost: Optional[float],
        min_cost: Optional[float],
    ) -> bool:
        """
        Helper function to filter pricing by cost range.

        Parameters:
            pricing (Dict): Pricing dictionary to filter.
            cost_type (str): Type of cost to filter by.
            max_cost (Optional[float]): Maximum allowable cost.
            min_cost (Optional[float]): Minimum allowable cost.

        Returns:
            bool: True if the pricing matches the cost range, False otherwise.
        """
        cost_value = pricing.get(cost_type, None)
        if cost_value is None:
            return False
        if max_cost is not None and cost_value > max_cost:
            return False
        if min_cost is not None and cost_value < min_cost:
            return False
        return True

    @staticmethod
    def get_pricing(
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        feature: Optional[str] = None,
        cost_type: Optional[str] = None,
        max_cost: Optional[float] = None,
        min_cost: Optional[float] = None,
    ) -> Union[Dict, List[Dict]]:
        """
        Retrieve and filter pricing information from the MODEL_PRICING dictionary.

        Parameters:
            model_type (Optional[str]): The category of the model (e.g., "gpt-4o", "gpt-4o-mini").
            model_name (Optional[str]): The specific model version (e.g., "gpt-4o-2024-08-06").
            feature (Optional[str]): The specific feature to retrieve (e.g., "input", "output", "training").
            cost_type (Optional[str]): The type of cost (e.g., "input", "output", "cached_input").
            max_cost (Optional[float]): The maximum allowable cost for filtering.
            min_cost (Optional[float]): The minimum allowable cost for filtering.

        Returns:
            Union[Dict, List[Dict]]: The filtered pricing information. A dictionary if a single model is specified,
                                    or a list of dictionaries for broader queries.

        Raises:
            ValueError: If the provided arguments do not match any data in the MODEL_PRICING dictionary.
        """
        result = []

        def process_category(category: str, models: Dict):
            """Process a single category of models."""
            for name, pricing in models.items():
                # Filter by model name if specified
                if model_name and name != model_name:
                    continue

                # Filter by feature if specified
                if feature and feature not in pricing:
                    continue

                # Filter by cost range if specified
                if cost_type and not ModelPricing.filter_by_cost(
                    pricing, cost_type, max_cost, min_cost
                ):
                    continue

                # Add the matching model to the results
                result.append({"category": category, "name": name, "pricing": pricing})

        # Iterate through the top-level keys in the dictionary
        for category, models in ModelPricing.MODEL_PRICING.items():
            # Filter by model type if specified
            if model_type and category != model_type:
                continue
            process_category(category, models)

        # Raise an error if no results were found
        if not result:
            raise ValueError(
                "No matching pricing data found with the provided filters."
            )

        # Return a single dictionary if only one model was matched
        if len(result) == 1:
            return result[0]

        # Return a list of results otherwise
        return result

    @staticmethod
    def calculate_cost(
        model_name: str,
        version: str,
        input_tokens: int,
        cached_input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> Dict[str, float]:
        """
        Calculate the total cost based on token usage.

        Args:
            model_name (str): The name of the model.
            version (str): The version of the model.
            input_tokens (int): Number of input tokens.
            cached_input_tokens (int, optional): Number of cached input tokens. Defaults to 0.
            output_tokens (int, optional): Number of output tokens. Defaults to 0.

        Returns:
            Dict[str, float]: A dictionary containing the cost breakdown.
        """
        pricing = ModelPricing.get_pricing(model_name, version)['pricing']
        _logger.info(f"Pricing found for model '{model_name}' and version '{version}': {pricing}")
        if not pricing:
            raise ValueError(
                f"Pricing not found for model '{model_name}' and version '{version}'."
            )

        input_cost = pricing.get("input", 0) * (input_tokens / 1_000_000)
        cached_input_cost = pricing.get("cached_input", 0) * (
            cached_input_tokens / 1_000_000
        )
        output_cost = pricing.get("output", 0) * (output_tokens / 1_000_000)
        
        _logger.info(f"Cost breakdown for {model_name}-{version}:")
        cost = {
            "input": input_cost,
            "cached_input": cached_input_cost,
            "output": output_cost,
            "total": input_cost + cached_input_cost + output_cost,
        }
        _logger.info(f"Cost: {cost}")


        return {
            "input": input_cost,
            "cached_input": cached_input_cost,
            "output": output_cost,
            "total": input_cost + cached_input_cost + output_cost,
        }
