"""
llm_flow.py
---------------------
Design for a Unified LLM API Wrapper
1. Objectives
    - Provide a single interface to interact with multiple LLM providers.
    - Ensure seamless switching between providers based on cost, performance, or availability.
    - Support authentication, prompt engineering, model selection, and response handling uniformly.
    - Implement error handling, logging, and caching for efficiency and robustness.
    - Allow for scalability to integrate additional providers in the future.
    
2. Architecture Overview
    The wrapper will follow a modular architecture with a core interface and provider-specific implementations.
    Core Components
        1. Unified API Interface
            - Standardizes request and response formats.
            - Abstracts provider-specific details.
        2. Provider Handlers
            - Dedicated classes for each provider (DeepSeek, OpenAI, Anthropic, Azure, AWS).
            - Implements provider-specific API calls.
        3. Routing & Load Balancing
            - Selects the best provider based on latency, cost, or custom-defined logic.
            - Can be weighted or randomized.
        4. Authentication Module
            - Handles API keys, tokens, and authentication mechanisms.
        5. Request Orchestration
            - Prepares requests: prompt formatting, token limits, model selection.
            - Processes responses: extracts, normalizes, and validates.
        6. Error Handling & Logging
            - Unified exception handling.
            - Logs requests, responses, errors for debugging and monitoring.
        7. Caching & Rate Limiting
            - Uses in-memory or external cache (Redis) for frequent queries.
            - Implements rate limiting to avoid exceeding API quotas.
        8. Configuration Management
            -Reads API keys, model names, and settings from YAML or environment variables.

3. Key Functionalities
| Feature | Description |
|---|---|
| Model Agnostic Calls | Users can call `generate_text()` without specifying a provider explicitly. |
| Multi-Tenancy Support | Different clients can use different providers dynamically. |
| Streaming Support | Supports both batch and streaming responses where available. |
| Parallel Requests | Sends requests to multiple providers and returns the fastest response. |
| Response Ranking | Applies scoring logic to rank responses based on relevance. |
| Cost Optimization | Selects the cheapest provider for a given prompt size. |

4. Example API Flow
    1. User calls the wrapper → wrapper.generate_text(prompt, model="best")
    2. Routing engine selects provider based on latency, availability, or predefined priority.
    3. Provider handler formats request and sends it to the selected API.
    4. Response is received, normalized, and cached.
    5. Wrapper returns a unified response to the user.
    
    
5. Providers & Considerations
| Provider | API Type | Key Features | Challenges |
|---|---|---|---|
| DeepSeek | REST | Competitive cost, supports fine-tuning | Lesser-known ecosystem |
| OpenAI | REST | GPT-4, GPT-4-turbo, broad support | Costly, rate limits |
| Anthropic | REST | Claude models, large context window | Expensive |
| Azure OpenAI | REST | Enterprise security, same models as OpenAI | Complex quota management |
| AWS Bedrock | REST | Supports multiple models (Claude, Llama, etc.) | Latency varies |

Requirements
------------

Usage:
------

Dependencies:
-------------

Author:
--------
Sourav Das

Version:
--------
1.0

Reference:
----------

Date:
-----
29.01.2025
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Importing necessary libraries and modules
import logging
from dotenv import load_dotenv

from typing import Any, Dict, List, Literal

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Load Environment variables
load_dotenv(override=True)


class LLMFlow:
    """ """

    def __init__(
        self,
        provider: Literal["deepseek", "openai", "anthropic", "azure", "aws"],
        model_config: Dict[str, Any],
        auth_config: Dict[str, Any],
    ) -> None:
        """
        Initialize the LLMFlow object with provider, model configuration, and authentication configuration.

        Args:
            provider (str): The name of the LLM provider.
            model_config (dict): The configuration for the LLM model.
            auth_config (dict): The configuration for authentication with the LLM provider.
        """
        self.provider = provider
        self.model_config = model_config
        self.auth_config = auth_config

        # Validate the model and authentication configurations
        is_validated = self._validate_model_config()
        is_auth = self._validate_auth_config()

        if is_validated and is_auth:
            self._initialize_provider()

    def _validate_model_config(self) -> bool:
        """
        Validates the model configuration dictionary for supported LLM providers.

        Returns:
            bool: True if the configuration is valid.

        Raises:
            ValueError: If any required key is missing or any value is invalid.
        """

        # Define the set of required keys for model configuration
        required_keys = {"provider", "model_name", "temperature", "max_tokens"}

        # Check if all required fields are present in the configuration
        missing_keys = required_keys - self.model_config.keys()
        if missing_keys:
            raise ValueError(
                f"Missing required keys in model configuration: {missing_keys}"
            )

        # Retrieve the provider and convert to lowercase for uniformity
        provider = self.model_config["provider"].lower()
        supported_providers = {"openai", "anthropic", "deepseek", "azure", "aws", "huggingface"}

        # Validate if the provider is supported
        if provider not in supported_providers:
            raise ValueError(
                f"Unsupported provider: {provider}. Supported providers are: {supported_providers}"
            )

        # Validate the model name is a non-empty string
        if (
            not isinstance(self.model_config["model_name"], str)
            or not self.model_config["model_name"].strip()
        ):
            raise ValueError("model_name must be a non-empty string.")

        # Validate the temperature is a float between 0 and 1
        if not isinstance(self.model_config["temperature"], (float, int)) or not (
            0 <= self.model_config["temperature"] <= 1
        ):
            raise ValueError("temperature must be a float between 0 and 1.")

        # Validate max_tokens is a positive integer
        if (
            not isinstance(self.model_config["max_tokens"], int)
            or self.model_config["max_tokens"] <= 0
        ):
            raise ValueError("max_tokens must be a positive integer.")

        return True

    def _validate_auth_config(self) -> bool:
        """
        Validates the authentication configuration dictionary for supported LLM providers.

        Args:
            auth_config (dict): The authentication configuration to validate.

        Returns:
            bool: True if the authentication configuration is valid.

        Raises:
            ValueError: If any required key is missing or any value is invalid.
        """
        # Ensure 'provider' is present
        if "provider" not in self.auth_config or not isinstance(
            self.auth_config["provider"], str
        ):
            raise ValueError(
                "Missing or invalid 'provider'. It must be a non-empty string."
            )

        provider = self.auth_config["provider"].lower()
        required_keys_by_provider = {
            "openai": {"api_key"},
            "anthropic": {"api_key"},
            "deepseek": {"api_key"},
            "azure": {"api_key", "deployment_name", "api_version"},
            "aws": {"api_key", "region", "service"},
            "huggingface": {},
        }

        if provider not in required_keys_by_provider:
            raise ValueError(
                f"Unsupported provider: {provider}. Supported providers are: {set(required_keys_by_provider.keys())}"
            )

        required_keys = required_keys_by_provider[provider]

        # Check for missing keys
        missing_keys = required_keys - self.auth_config.keys()
        if missing_keys:
            raise ValueError(
                f"Missing required authentication keys for {provider}: {missing_keys}"
            )

        # Validate API key
        if "api_key" in self.auth_config and (
            not isinstance(self.auth_config["api_key"], str)
            or not self.auth_config["api_key"].strip()
        ):
            raise ValueError("api_key must be a non-empty string.")

        # Azure-specific validations
        if provider == "azure":
            if (
                not isinstance(self.auth_config["deployment_name"], str)
                or not self.auth_config["deployment_name"].strip()
            ):
                raise ValueError(
                    "deployment_name must be a non-empty string for Azure."
                )
            if (
                not isinstance(self.auth_config["api_version"], str)
                or not self.auth_config["api_version"].strip()
            ):
                raise ValueError("api_version must be a non-empty string for Azure.")

        # AWS-specific validations
        if provider == "aws":
            if (
                not isinstance(self.auth_config["region"], str)
                or not self.auth_config["region"].strip()
            ):
                raise ValueError("region must be a non-empty string for AWS.")
            if (
                not isinstance(self.auth_config["service"], str)
                or not self.auth_config["service"].strip()
            ):
                raise ValueError("service must be a non-empty string for AWS.")
            
        if provider == "huggingface":
            # No authentication required for Hugging Face models
            pass

        return True

    def _initialize_provider(self) -> None:
        """
        Initialize the provider handler based on the specified provider.

        This method will instantiate the provider's client object and store it
        in the `_client` instance variable.

        Raises:
            NotImplementedError: If the provider is not supported.

        Returns:
            None
        """
        provider = self.provider.lower()

        if provider == "openai":
            # Import the OpenAI client
            from openai import OpenAI

            # Initialize the client with the API key
            self._client = OpenAI(api_key=self.auth_config["api_key"])
        elif provider == "anthropic":
            # Import the Anthropic client
            from anthropic import Anthropic

            # Initialize the client with the API key
            self._client = Anthropic(api_key=self.auth_config["api_key"])
        elif provider == "deepseek":
            # Raise an exception if DeepSeek is not yet supported
            raise NotImplementedError("DeepSeek provider is not yet supported.")
        elif provider == "azure":
            # Raise an exception if Azure is not yet supported
            raise NotImplementedError("Azure provider is not yet supported.")
        elif provider == "aws":
            # Raise an exception if AWS is not yet supported
            raise NotImplementedError("AWS provider is not yet supported.")
        elif provider == "huggingface":
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config["model_name"], device_map="auto", torch_dtype="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config["model_name"]
            )
        else:
            # Raise an exception if the provider is not recognized
            raise ValueError(f"Unsupported provider: {provider}.")

    def _validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """
        Validate the messages for the LLM provider.

        Args:
            messages (List[Dict[str, str]]): The list of messages to validate.

        Raises:
            ValueError: If the messages are invalid for the LLM provider.
        """
        # Placeholder implementation for validation
        pass

    def _compile_messages(
        self,
        user_prompt: str,
        messages: List[Dict[str, str]] = None,
        system_prompt: str = None,
    ) -> List[Dict[str, str]] | str:
        """
        Compile the messages for the LLM provider.

        Args:
            user_prompt (str): The user's prompt.
            messages (List[Dict[str, str]]): The list of messages.
            system_prompt (str): The system prompt.

        Returns:
            List[Dict[str, str]]: Compiled messages for the LLM provider.
        """
        # Placeholder implementation for message compilation
        if self.provider == "anthropic":
            anthropic_messages = []
            if messages:
                anthropic_messages.extend(messages)
            anthropic_messages.append({"role": "user", "content": user_prompt})
            return anthropic_messages

        if self.provider == "openai":
            openai_messages = []
            if system_prompt:
                openai_messages.append({"role": "system", "content": system_prompt})
            if messages:
                openai_messages.extend(messages)
            openai_messages.append({"role": "user", "content": user_prompt})
            return openai_messages
        
        if self.provider == "huggingface":
            hf_messages = ""
            if system_prompt:
                hf_messages += f"<system>{system_prompt}</system>\n"
            if messages:
                for msg in messages:
                    hf_messages += f"<{msg['role']}>{msg['content']}</{msg['role']}>\n"
            hf_messages += f"<user>{user_prompt}</user>"
            return hf_messages
        
        

    def generate_text(
        self,
        user_prompt: str,
        system_prompt: str = None,
        messages: List[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate text using the LLM provider.

        Args:
            user_prompt (str): The user's prompt.
            system_prompt (str, optional): The system prompt. Defaults to None.
            messages (List[Dict[str, str]], optional): The list of messages. Defaults to None.

        Returns:
            Dict[str, Any]: Output with the generated text.
        """
        if self.provider == "anthropic":
            messages = self._compile_messages(user_prompt, messages, system_prompt)
            res = self._client.messages.create(
                model=self.model_config["model_name"],
                max_tokens=self.model_config["max_tokens"],
                temperature=self.model_config["temperature"],
                messages=messages,
                system=system_prompt,
            )

        if self.provider == "openai":
            messages = self._compile_messages(user_prompt, messages, system_prompt)
            res = self._client.chat.completions.create(
                model=self.model_config["model_name"],
                max_tokens=self.model_config["max_tokens"],
                temperature=self.model_config["temperature"],
                messages=messages,
            )

        if self.provider == "huggingface":
            messages = self._compile_messages(user_prompt, messages, system_prompt)
            inputs = self.tokenizer(messages, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_length=self.model_config["max_tokens"],
            )
            res = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return res


# Example Usage
import time

start = time.time()

llm_flow = LLMFlow(
    provider="huggingface",
    model_config={
        "provider": "huggingface",
        "model_name": "Qwen/Qwen2.5-3B",
        "temperature": 0.5,
        "max_tokens": 1000,
    },
    auth_config={
        "provider": "huggingface",
        },  # No authentication required for Hugging Face models
)

res = llm_flow.generate_text(
    system_prompt="Generate a text response.",
    user_prompt="Write an essay about AI?",
)



from pprint import pprint
print("-" * 50)
pprint(res)
print(type(res))
print("-" * 50)

end = time.time()
print(f"Time taken: {end - start} seconds")
