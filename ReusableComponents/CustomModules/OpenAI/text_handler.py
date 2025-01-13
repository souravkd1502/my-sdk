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

from typing import List, Dict, Any, Optional

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables
load_dotenv(override=True)


class OpenAIChatHandler:
    """
    A class to handle chat interactions with OpenAI's Chat Completion API.

    Attributes:
        openai_key (str): The API key to authenticate with OpenAI.
        model (str): The OpenAI model to use for chat.
        openai (OpenAI): An instance of the OpenAI client.
        chat_history (List[Dict[str, Any]]): A list to store the history of chats.
    """

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
        self.conversation_history: List[Dict[str, Any]] = []

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
        if response_format not in ["json_object", "json_schema", "text"]:
            raise ValueError(
                "Invalid response format. Please provide a valid response format."
            )
        return response_format

    def chat_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        response_format: str = "text",
        temperature: float = 0.7,
        max_tokens: int = 600,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        streaming: bool = False,
        is_structured_output: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a chat completion with support for multi-turn conversations.

        Args:
            prompt (str): The user's input prompt.
            system_prompt (Optional[str]): Optional system-level prompt.
            messages (Optional[List[Dict[str, Any]]]): List of conversation history messages.
            response_format (str): The response format, either "text" or "json_schema". Defaults to "text".
            temperature (float): Sampling temperature, range [0, 1]. Defaults to 0.7.
            max_tokens (int): Maximum tokens for the response. Defaults to 600.
            top_p (float): Nucleus sampling parameter, range [0, 1]. Defaults to 1.0.
            frequency_penalty (float): Penalty for token frequency. Defaults to 0.0.
            presence_penalty (float): Penalty for token presence. Defaults to 0.0.
            tools (Optional[List[Dict[str, Any]]]): Optional tools for the completion.
            streaming (bool): Whether to use streaming responses. Defaults to False.
            is_structured_output (bool): Whether the output should be structured. Defaults to False.
            json_schema (Optional[Dict[str, Any]]): Schema for structured JSON output.

        Returns:
            Dict[str, Any]: The response from the API.

        Raises:
            ValueError: If invalid arguments are provided.
        """
        # Constants for roles
        ROLE_SYSTEM = "system"
        ROLE_USER = "user"
        ROLE_ASSISTANT = "assistant"

        # Validate arguments
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        if is_structured_output and not json_schema:
            raise ValueError("JSON schema must be provided for structured output.")

        # Initialize or extend the conversation history
        if messages:
            self.conversation_history = messages
        else:
            if system_prompt and not self.conversation_history:
                self.conversation_history.append(
                    {"role": ROLE_SYSTEM, "content": system_prompt}
                )
            self.conversation_history.append({"role": ROLE_USER, "content": prompt})

        # Response format configuration
        if not is_structured_output and response_format != "text":
            raise ValueError(
                "Response format must be 'text' when structured output is disabled."
            )
        response_config = {"type": response_format}
        if is_structured_output:
            response_config["json_schema"] = json_schema

        # Call the API
        try:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                response_format=response_config,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                tools=tools,
                stream=streaming,
            )
            response = response.to_dict()

            # Add the assistant's response to the conversation history
            self.conversation_history.append(
                {
                    "role": ROLE_ASSISTANT,
                    "content": response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", ""),
                }
            )
            _logger.info(
                f"Generated chat completion: {response['choices'][0]['message']['content']}"
            )
            _logger.info(f"Conversation history: {self.conversation_history}")

            return response
        except Exception as e:
            raise RuntimeError(f"Failed to generate chat completion: {e}")
