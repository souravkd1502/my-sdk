"""
Template Engine Module
======================

Provides utilities for rendering string templates with variable placeholders.
Designed for use in SMS messaging, email generation, and other text-based
automation workflows, including LLM-driven pipelines.

Features
--------
1. Placeholder replacement:
    - Supports {{key}} syntax to inject variable values into templates.
    - Handles whitespace within placeholders (e.g., {{ key }}).
    - Missing keys are replaced with empty strings to prevent runtime errors.

2. Efficient processing:
    - Pre-compiles regex patterns for faster repeated template rendering.
    - Minimal overhead for high-throughput systems like messaging servers.

3. Logging and debugging:
    - Warns when a template key is missing from the provided variables dictionary.
    - Integrates seamlessly with standard Python logging for traceability.

4. LLM-friendly:
    - Deterministic output for predictable automation.
    - Clear argument and return types for tool integration.

Usage Example
-------------
```python
from template_engine import apply_template

template = "Hello {{name}}, your OTP is {{code}}"
variables = {"name": "Alice", "code": 123456}
message = apply_template(template, variables)
# message => "Hello Alice, your OTP is 123456"
````

## Notes

* Extra keys in the variables dictionary are ignored.
* Intended for text templates where placeholders follow the double curly braces {{key}} convention.
* Can be integrated into larger automation systems such as SMS servers, MCP servers, or LLM tool pipelines.
"""

import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Prevents logging errors if no config

# Pre-compile regex for efficiency
TEMPLATE_PATTERN = re.compile(r"{{\s*(.*?)\s*}}")

def apply_template(template_str: str, variables: Dict[str, any]) -> str:
    """
    Replace placeholders in the form {{key}} in a template string with values from a dictionary.
    
    Args:
        template_str (str): The template string containing placeholders.
        variables (Dict[str, any]): Dictionary of key-value pairs to fill the template.

    Returns:
        str: The rendered template with placeholders replaced by values.

    Notes:
        - If a key in the template is missing from the variables dict, it is replaced with an empty string.
        - Extra keys in the variables dict are ignored.
    
    Example:
        >>> apply_template("Hello {{name}}, your code is {{code}}", {"name": "Alice", "code": 1234})
        "Hello Alice, your code is 1234"
    """
    def replace(match: re.Match) -> str:
        key = match.group(1).strip()
        value = variables.get(key)
        if value is None:
            logger.debug(f"Missing template variable '{key}', using empty string.")
            return ""
        return str(value)

    return TEMPLATE_PATTERN.sub(replace, template_str)
