"""
Prompt Refinement Module
========================

This module provides functionality to refine user-provided prompts and system templates to achieve 
more precise and effective outputs from large language models (LLMs). It integrates features for 
enhancing prompt clarity, maintaining contextual coherence through history, and ensuring that the 
refinement process aligns with desired output specifications.

Main Features:
--------------
1. **Prompt Refinement**:
    - Refines user prompts for clarity, precision, and alignment with intended outputs.
    - Enhances system templates to improve flexibility and adaptability for dynamic input scenarios.
    - Integrates conversation history to maintain contextual continuity.

2. **Structured Outputs**:
    - Ensures that refined prompts and templates are returned in a JSON format with the following keys:
        - `"refined_prompt"`: The improved user prompt or `None`.
        - `"refined_template"`: The enhanced system template or `None`.
        - `"contextual_notes"`: Contextual insights derived from conversation history or `None`.

3. **Error Handling and Logging**:
    - Provides detailed logging for debugging and analysis.
    - Handles errors gracefully, ensuring robust performance in various scenarios.

4. **Token Usage Tracking**:
    - Tracks input and output tokens used during the refinement process.
    - Updates token usage statistics to monitor LLM interaction costs.

Classes:
--------
- **PromptRefinement**: A class for refining prompts and templates for effective model outputs.

Dependencies:
-------------
- `sys`: To modify the system path for importing custom modules.
- `json`: For handling JSON data and parsing outputs.
- `dotenv.load_dotenv`: To load environment variables from a `.env` file.
- `langchain.prompts.PromptTemplate`: To create and format prompt templates.
- `llm.models.Config`: Custom module for managing LLM configurations and invocations.
- `utils.common.update_tokens`: Custom utility for updating token usage statistics.
- `utils.logger.create_logger`: Custom utility for creating a logger instance.
- `typing`: For type hinting to ensure clarity in method definitions.

Example Usage:
--------------
```python
from prompt_refinement import PromptRefinement
from llm.models import Config

# Initialize PromptRefinement with an LLM instance
refiner = PromptRefinement(llm=Config.llm)

# Refine a prompt
refined_output = refiner.refine_prompt(
    prompt="Summarize the text: 'The quick brown fox jumps over the lazy dog.'",
    template="Summarize the text in less than 20 words.",
    history=[{"user": "Summarize this text", "assistant": "Sure, what kind of summary?"}]
)

# Parse the refined output
parsed_output = refiner._prompt_refinement_output_parser(refined_output)
print(parsed_output)
```

TODO:

FIXME:

Author:
-------
- Sourav Das

Date:
-----
- 2025-07-01
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Importing required libraries
import json
import logging
from dotenv import load_dotenv
from json import JSONDecodeError
from langchain.prompts import PromptTemplate

from llm.models import Config

from typing import Any, List, Dict, Tuple

# Loading environment variables
load_dotenv(override=True)

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Defining the class for prompt refinement
class PromptRefinement:
    """A class for refining prompts and templates for effective model outputs."""

    def __init__(self, llm: Any):
        """
        Initialize the PromptRefinement with a large language model instance.

        Args:
            llm (Any): The large language model to use for refinement of prompts and templates.
        """
        self.llm = llm

    def refine_prompt(
        self,
        prompt: str = None,
        template: str = None,
        history: List[Dict[str, str]] = None,
    ) -> Tuple[str, int, int]:
        """
        Refine the given prompt and/or template using the provided history.

        This method refines a user-provided prompt or system template by enhancing their clarity, precision,
        and contextual coherence. If a history is provided, it is used to maintain contextual continuity.
        The output is formatted as a JSON string with refined elements.

        Args:
            prompt (str, optional): The user prompt to refine. Defaults to None.
            template (str, optional): The system template to refine. Defaults to None.
            history (List[Dict[str, str]], optional): Contextual history for refinement. Defaults to None.

        Returns:
            str: A JSON-formatted string containing refined prompt, template, and contextual notes.

        Raises:
            ValueError: If neither prompt nor template is provided.
            Exception: If an error occurs during the refinement process.

        Example:
            ```python
                # Define the input prompt, template, and history
                prompt = "Hello, how can I help you today?"
                template = "You are a helpful assistant. Please provide the following information:"
                history = [{"role": "user", "content": "I need to book a flight."}, {"role": "assistant", "content": "Sure, what would you like to fly from?"}]

                # Create an instance of PromptRefinement
                prompt_refiner = PromptRefinement(llm)

                # Refine the prompt and template
                refined_output = prompt_refiner.refine_prompt(prompt=prompt, template=template, history=history)
                print(refined_output)
            ```
        """
        if not prompt and not template:
            _logger.error(
                "Please provide either prompt or template for prompt refinement."
            )
            raise ValueError(
                "Please provide either prompt or template for prompt refinement."
            )

        # Define the system prompt for the refinement process
        SYSTEM_PROMPT = """
        You are an expert prompt engineer specializing in refining prompts and templates to achieve precise and effective outputs from the model. Your task is to:  
        1. Refine the user's provided **Prompt** or **Template** to ensure clarity, conciseness, and alignment with the intended output.  
        2. Incorporate **History**, if provided, to maintain contextual continuity.  
        3. Return the output in JSON format with the following keys:
        - `refined_prompt`: The improved user prompt, or `null` if not provided.
        - `refined_template`: The enhanced system template, or `null` if not provided.
        - `contextual_notes`: Notes derived from **History**, if applicable, or `null` if not provided.

        Inputs:
        - **Prompt**: {prompt}  # Provided if a prompt is available
        - **Template**: {template}  # Provided if a template is available
        - **History**: {history}  # Provided if a conversation history is available

        Refinement Guidelines:
        - If only a **Prompt** is given, focus on improving its clarity, precision, and intent.
        - If only a **Template** is given, focus on enhancing its structure and adaptability for dynamic input scenarios.
        - If both are provided, refine them together to ensure they are synergistic and contextually coherent.

        Output Specification:
        The response must always be returned in valid JSON format with no additional text. The JSON structure should look like this:
        {{{{
            "refined_prompt": "<Insert refined user prompt here, or null>",
            "refined_template": "<Insert refined system template here, or null>",
            "contextual_notes": "<Insert any relevant notes from History, or null>"
        }}}}
        """

        try:
            # Define the input variables for the prompt template
            chain_input = {"prompt": prompt, "template": template, "history": history}
            prompt_template = PromptTemplate(
                template=SYSTEM_PROMPT,
                input_variables=["prompt", "template", "history"],
            )
            prompt = prompt_template.format(**chain_input)
            res = Config.llm.invoke(prompt).dict()

            _logger.info(
                f"Prompt refinement output tokens: {res['usage_metadata']['input_tokens']}"
            )
            _logger.info(
                f"Prompt refinement input tokens: {res['usage_metadata']['output_tokens']}"
            )

            return (
                res["content"],
                res["usage_metadata"]["input_tokens"],
                res["usage_metadata"]["output_tokens"],
            )
        except Exception as e:
            _logger.error(f"Error in refining prompt: {e}")
            raise e

    def _prompt_refinement_output_parser(self, text: str) -> Dict[str, str]:
        """
        Parse the output from the prompt refinement process.

        Args:
            text (str): The raw output from the language model.

        Returns:
            Dict[str, str]: A dictionary containing refined prompt, template, and contextual notes.

        Raises:
            ValueError: If the output is not in the expected format.
            JSONDecodeError: If the output cannot be parsed into a dictionary.

        Example:
            ```python
                # Define the input prompt, template, and history
                prompt = "Hello, how can I help you today?"
                template = "You are a helpful assistant. Please provide the following information:"
                history = [{"role": "user", "content": "I need to book a flight."}, {"role": "assistant", "content": "Sure, what would you like to fly from?"}]

                # Create an instance of PromptRefinement
                prompt_refiner = PromptRefinement(llm)

                # Refine the prompt and template
                refined_output = prompt_refiner.refine_prompt(prompt=prompt, template=template, history=history)

                # Parse the refined output
                parsed_output = prompt_refiner._prompt_refinement_output_parser(refined_output)
                print(parsed_output)
            ```
        """
        try:
            _logger.info(f"Received prompt refinement output: {text}")
            # Parse the output into a dictionary
            res = json.loads(text)
            if not isinstance(res, dict):
                raise ValueError("Invalid output format. Expected a dictionary.")
            _logger.info(f"Refined prompt: {res.get('refined_prompt')}")
            _logger.info(f"Refined template: {res.get('refined_template')}")
            _logger.info(f"Contextual notes: {res.get('contextual_notes')}")
            return res
        except JSONDecodeError as e:
            _logger.error(f"Error in parsing prompt refinement output: {e}")
            raise e
