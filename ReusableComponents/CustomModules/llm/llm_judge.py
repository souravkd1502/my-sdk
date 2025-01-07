"""
Module: LLMJudge

This module provides the `LLMJudge` class, designed to evaluate and select the best option from a list of provided choices using a language model. The module facilitates decision-making based on objective guidelines, evaluation criteria, and user-defined objectives.

Key Features:
- Dynamic prompt generation to guide the language model for fair and systematic evaluation.
- Logging integration for monitoring and debugging evaluation processes.
- Modular design for easy integration into larger applications.

Dependencies:
- `langchain.prompts.PromptTemplate` for prompt formatting.
- Custom modules located in the `llm.models` package.

Usage:
The `LLMJudge` class provides an `evaluate` method to input a query, guidelines, and options for analysis. The output includes a structured judgment based on the defined criteria.

Example:
```python
judge = LLMJudge()
result = judge.evaluate(
    query="Select the best strategy for project execution",
    history=[],
    objective="Maximize efficiency",
    guidelines="Use cost and time as primary factors",
    options=["Option A", "Option B", "Option C"],
    output_format="JSON"
)
print(result)
```

Author:
Sourav Das

Date:
2025-07-01
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Importing necessary libraries and modules
import json
import logging
from langchain.prompts import PromptTemplate

from llm.models import Config

from typing import List, Dict, Any

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class LLMJudge:
    """
    A class for evaluating options based on provided guidelines using a language model.

    Attributes:
        JUDGE_LLM_PROMPT_TEMPLATE (str): The prompt template for guiding the LLM evaluation.
    """

    JUDGE_LLM_PROMPT_TEMPLATE = """
        You are an impartial and objective judge tasked with evaluating and selecting the best option from a set of provided choices based on clear 
        criteria and guidelines. Your role is to analyze the options fairly, justify your selection, and explain how it 
        aligns with the given objectives. Follow these instructions:

        ### Input:
        1. **Options**: A list of options to evaluate.
        2. **Evaluation Criteria**: A set of objective guidelines or parameters to assess the options.
        3. **Preferred Outcome (if applicable)**: Any specific goal or desired result that should be prioritized during selection.

        ### Objective Guidelines:
        - Evaluate all options against the provided criteria, ensuring each is assessed fairly and systematically.
        - Justify your selection with logical reasoning and reference to the evaluation criteria.
        - If no option fully meets the criteria, recommend the closest match and suggest improvements or alternatives if applicable.
        - Remain neutral and do not incorporate personal biases into the evaluation.

        ### Preferred Outcome:
        - Clearly specify whether the preferred outcome is achievable based on the options provided.
        - If not achievable, suggest a recommendation or an approach to reach a feasible resolution.

        ### Output Format:
        Provide your judgment in the following format:
        1. **Problem Summary**:
            - Summarize the objective and the options provided.
        2. **Key Criteria**:
            - List the evaluation criteria and their importance.
        3. **Option Analysis**:
            - Evaluate each option against the criteria.
        4. **Selected Option**:
            - Clearly state the selected option and justify why it was chosen.
        5. **Recommendations** *(if applicable)*:
            - Suggest improvements or alternatives if no option fully meets the criteria.

        Here's the objective for your evaluation:
        User's Query: 
        {query}
        
        Conversational History:
        {history}
        
        Objective:
        {objective}
        
        Guidelines:
        {guidelines}
        
        Options:
        {options}
        
        Output Format:
        {output_format}
    """

    @classmethod
    def evaluate(
        cls,
        query: str = "",
        history: List[Dict[str, str]] = [],
        objective: str = "",
        guidelines: str = "",
        options: List[str] = [],
        output_format: str = "JSON",
    ) -> Dict[str, Any]:
        """
        Evaluates the input prompt based on provided guidelines.

        Args:
            query (str): The user's query or input prompt.
            history (List[Dict[str, str]]): The conversational history or context.
            objective (str): The objective or task to be evaluated.
            guidelines (str): The evaluation criteria or instructions.
            options (List[str]): A list of options to evaluate.
            output_format (str): The format in which the judgment should be presented.

        Returns:
            Dict[str, Any]: A dictionary containing the model's judgment and metadata.
        """
        try:
            _logger.info("Starting evaluation of the prompt.")

            # Define the input variables for the prompt template
            chain_input = {
                "query": query,
                "history": history,
                "objective": objective,
                "guidelines": guidelines,
                "options": options,
                "output_format": output_format,
            }

            prompt_template = PromptTemplate(
                template=cls.JUDGE_LLM_PROMPT_TEMPLATE,
                input_variables=[
                    "query",
                    "history",
                    "objective",
                    "guidelines",
                    "options",
                    "output_format",
                ],
            )

            # Generate the formatted prompt
            prompt = prompt_template.format(**chain_input)

            _logger.debug("Prompt generated: %s", prompt)

            # Call the language model and return the result
            result = Config.reasoning_llm.invoke(prompt).dict()

            _logger.info("Evaluation completed successfully.")
            return result

        except Exception as e:
            _logger.error("Error during evaluation: %s", str(e))
            return {"error": str(e)}

    @staticmethod
    def _llm_judge_output_parser(llm_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the LLMJudge output to extract structured information.

        Args:
            llm_output (Dict[str, Any]): The raw output from LLMJudge, containing the evaluation details.

        Returns:
            Dict[str, Any]: A dictionary with parsed information, including selected option details,
                            key criteria, and recommendations.
        """
        try:
            # Extract the 'content' key, which contains the JSON string
            content = llm_output.get("content", "")
            input_tokens = (llm_output["usage_metadata"]["input_tokens"],)
            output_tokens = (llm_output["usage_metadata"]["output_tokens"],)

            # Strip Markdown formatting and parse the JSON content
            if content.startswith("```json") and content.endswith("```"):
                content = content[7:-3].strip()

            parsed_content = json.loads(content)

            # Extract relevant sections
            selected_option = parsed_content.get("Selected Option", {})
            key_criteria = parsed_content.get("Key Criteria", {})
            recommendations = parsed_content.get("Recommendations", {}).get(
                "General", ""
            )

            return {
                "selected_option": selected_option,
                "key_criteria": key_criteria,
                "recommendations": recommendations,
                "input_tokens": input_tokens[0],
                "output_tokens": output_tokens[0],
                "entire_output": llm_output,
                "document_ids": selected_option.get("Document ID", []),
            }
        except (json.JSONDecodeError, AttributeError) as e:
            _logger.error("Error parsing LLMJudge output: %s", str(e))
            return {
                "error": "Failed to parse LLMJudge output.",
                "details": str(e),
            }

    @classmethod
    def generate_evaluation(
        cls,
        query: str = "",
        history: List[Dict[str, str]] = [],
        objective: str = "",
        guidelines: str = "",
        options: List[str] = [],
        output_format: str = "JSON",
        use_json_parser: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate an evaluation of the input prompt based on provided guidelines.
        It also parses the LLMJudge output to extract structured information.

        Args:
            query (str): The user's query or input prompt.
            history (List[Dict[str, str]]): The conversational history or context.
            objective (str): The objective or task to be evaluated.
            guidelines (str): The evaluation criteria or instructions.
            options (List[str]): A list of options to evaluate.
            output_format (str): The format in which the judgment should be presented.
            use_json_parser (bool): A flag to indicate whether to use the JSON parser for output.

        Returns:
            Dict[str, Any]: A dictionary containing the model's judgment and metadata.
        """

        llm_output = cls.evaluate(
            query=query,
            history=history,
            objective=objective,
            guidelines=guidelines,
            options=options,
            output_format=output_format,
        )

        if use_json_parser:
            return cls._llm_judge_output_parser(llm_output)
        return llm_output
