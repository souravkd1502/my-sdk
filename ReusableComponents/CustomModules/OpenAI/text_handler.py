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
import logging
from openai import OpenAI
from dotenv import load_dotenv

from typing import List, Dict, Any, Union

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables
load_dotenv(override=True)


class OpenAIChatHandler:
    """ """

    def __init__(self, openai_key: str, model: str = "gpt-4o") -> None:
        """
        Initialize the OpenAIChatHandler class.

        Args:
        - openai_key (str): The OpenAI API key to use for authentication.
        - model (str): The name of the OpenAI model to use for chat. Defaults to "gpt-4o".
        """
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.openai = OpenAI(api_key=self.openai_key)
        
    def _validate_response_format(self, response_format: str) -> str:
        """
        This function validates the response format and returns the response format.

        Args:
        - response_format (str): The response format to validate.

        Returns:
        - str: The validated response format.
        
        Raises:
        - ValueError: If the response format is invalid.
        """
        if response_format not in ["json", "text", "srt", "verbose_json"]:
            raise ValueError(
                "Invalid response format. Please provide a valid response format."
            )
        return response_format
    
    def _validate_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        This function validates the messages and returns the messages.

        Args:
        - messages (List[Dict[str, Any]]): The messages to validate.

        Returns:
        - List[Dict[str, Any]]: The validated messages.
        
        Raises:
        - ValueError: If the messages are invalid.
        """
        raise NotImplementedError("This function is not implemented yet.")

    def chat_completion(
        self,
        prompt: str,
        system_prompt: str = None,
        messages: List[Dict[str, Any]] = None,
        response_format: str = "text",
        temperature: float = 0.7,
        max_tokens: int = 600,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        tools: List[Dict[str, Any]] = None,
        streaming: bool = False,
    ) -> Dict[str, Any]:
        """
        This function takes a prompt and returns a completion of the prompt using OpenAI's Chat Completion API.

        Args:
        - prompt (str): The prompt to generate a completion for.
        - messages (List[Dict[str, Any]]): The list of messages to include in the completion.
        - response_format (str): The format of the response. Defaults to "json".
        - temperature (float): The temperature to use for sampling. Defaults to 0.7.
        - max_tokens (int): The maximum number of tokens to generate. Defaults to 600.
        - top_p (float): The nucleus sampling parameter. Defaults to 1.0.
        - frequency_penalty (float): The frequency penalty to use. Defaults to 0.0.
        - presence_penalty (float): The presence penalty to use. Defaults to 0.0.
        - tools (List[Dict[str, Any]]): The list of tools to use in the completion.
        - streaming (bool): Whether to use streaming completion. Defaults to False.

        Returns:
        - Dict[str, Any]: The completion of the prompt.
        """
        return self.openai.chat.completions.create(
            model = self.model,
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt or "",
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ]
                }
            ],
            response_format={
                "type": response_format,
            },
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            tools=tools,
            stream=streaming,
        )
        
