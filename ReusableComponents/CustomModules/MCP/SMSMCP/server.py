"""
SMS MCP Server – LLM‑Optimized Documentation
===========================================

This module implements an MCP (Model Context Protocol) server that provides
structured, tool-callable SMS functions useful for AI agents and LLM workflows.
It exposes the following capabilities:

1. **send_sms_with_template**
    - Sends an SMS using a predefined template file.
    - Allows LLMs to supply variable dictionaries to fill placeholders.
    - Useful for transactional messages, automated workflows, and dynamic text generation.

2. **send_sms_raw**
    - Sends an SMS using a raw message string.
    - Useful for free‑form agent messaging.

3. **list_templates**
    - Returns all available template files.
    - Allows LLMs to explore available structured message formats.

4. **preview_template**
    - Renders a template with data variables without sending an SMS.
    - Useful for verifying correctness before calling `send_sms_with_template`.

Additional Features
-------------------
- **LLM‑Friendly I/O Schema**: All tools return JSON‑serializable dictionaries
    suitable for AI agents, LangChain tools, MCP clients, and autonomous workflows.
- **Robust Logging**: Every operation, error, and action is logged.
- **Exception Handling**: Errors are raised with clear messages for agent debugging.
- **Environment Validation**: Ensures required API keys are present, preventing silent failures.
- **Type Hints**: Enables static analysis and LLM reasoning about input/output shapes.

Intended Usage
--------------
This server is designed to be used by:
- MCP-compatible LLMs (ChatGPT, Claude, etc.)
- Orchestration frameworks executing tool calls
- Automation agents needing structured SMS capabilities

LLMs can safely call any exposed tool by providing arguments in the declared
schema. All responses are deterministic and easy for LLMs to interpret.
"""

import os
import logging
from typing import Dict, Any, Optional

from mcp.server.fastmcp import FastMCP
from utils.template_engine import apply_template
from utils.sms_provider import send_sms

# ---------------------------------
# Logging Configuration
# ---------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------
# MCP Server Setup
# ---------------------------------
server = FastMCP(name="sms-mcp-server", port=8000)

TEMPLATE_DIR = "templates"

# ---------------------------------
# Helpers
# ---------------------------------


def load_template(template_id: str) -> str:
    """
    Load an SMS template by ID.

    Args:
        template_id (str): Name of the template file (without .txt).

    Returns:
        str: Contents of the template file.

    Raises:
        FileNotFoundError: If template does not exist.
        Exception: If file cannot be read.
    """
    filepath = os.path.join(TEMPLATE_DIR, f"{template_id}.txt")
    logger.debug(f"Loading template from: {filepath}")

    if not os.path.exists(filepath):
        logger.error(f"Template '{template_id}' not found")
        raise FileNotFoundError(f"Template '{template_id}' does not exist")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.exception(f"Failed to read template '{template_id}'")
        raise e


# ---------------------------------
# TOOL: send_sms_with_template
# ---------------------------------
@server.tool()
def send_sms_with_template(
    phone_number: str,
    template_id: str,
    variables: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    send_sms_with_template
    Send an SMS using a stored template. LLMs can provide variables to fill placeholders
    inside the template (e.g., {{name}}, {{code}}, {{amount}}).

    Args:
        phone_number: Recipient phone number in international or local format.
        template_id: ID of a template located in /templates (e.g., "otp", "receipt").
        variables: Dictionary of keys used to populate template placeholders.
        meta: Optional metadata for orchestration or tracking.

    Returns:
        Dictionary containing:
            - status: "sent"
            - delivery_id: Provider-generated SMS transaction ID
            - message: Final rendered text that was sent

    Typical LLM usage example:
        {
            "tool": "send_sms_with_template",
            "arguments": {
                "phone_number": "+1234567890",
                "template_id": "otp",
                "variables": {"code": "123456", "expires": "5 minutes"}
            }
        }
    """
    try:
        logger.info(f"Sending SMS using template '{template_id}' to {phone_number}")

        raw = load_template(template_id)
        message = apply_template(raw, variables)
        sid = send_sms(phone_number, message)

        return {"status": "sent", "delivery_id": sid, "message": message}

    except Exception as e:
        logger.exception("Error sending SMS with template")
        raise e


# ---------------------------------
# TOOL: send_sms_raw
# ---------------------------------
@server.tool()
def send_sms_raw(phone_number: str, message: str) -> Dict[str, Any]:
    """
    send_sms_raw
    Send a plain SMS message without using templates. Useful for free-form agent
    messaging or dynamic instructions.

    Args:
        phone_number: Destination number.
        message: Exact text body to be delivered.

    Returns:
        Dictionary containing:
            - status: "sent"
            - delivery_id: Provider message ID

    Typical LLM usage example:
        {
            "tool": "send_sms_raw",
            "arguments": {
                "phone_number": "+1234567890",
                "message": "Your order has shipped!"
            }
        }
    """
    try:
        logger.info(f"Sending raw SMS to {phone_number}")
        sid = send_sms(phone_number, message)
        return {"status": "sent", "delivery_id": sid}
    except Exception as e:
        logger.exception("Failed to send raw SMS")
        raise e


# ---------------------------------
# TOOL: list_templates
# ---------------------------------
@server.tool()
def list_templates() -> Dict[str, Any]:
    """
    list_templates
    Returns a list of all available SMS templates in the server's template directory.
    LLMs can call this before attempting preview or send operations to discover IDs.

    Args:
        None

    Returns:
        Dictionary containing:
            - templates: List of objects with:
                - id: Template identifier (filename without extension)
                - filename: Actual template filename

    Typical LLM usage example:
        {
            "tool": "list_templates",
            "arguments": {}
        }
    """
    try:
        logger.info("Listing all templates")
        files = os.listdir(TEMPLATE_DIR)
        templates = [
            {"id": file.replace(".txt", ""), "filename": file}
            for file in files
            if file.endswith(".txt")
        ]
        return {"templates": templates}
    except Exception as e:
        logger.exception("Failed to list templates")
        raise e


# ---------------------------------
# TOOL: preview_template
# ---------------------------------
@server.tool()
def preview_template(template_id: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    """
    preview_template
    Render a template using provided variables without sending an SMS.
    Useful for validation or multi-step flows in LLM agents.

    Args:
        template_id: Template file identifier.
        variables: Dictionary of placeholder values to inject.

    Returns:
        Dictionary containing:
            - message: Rendered SMS message text

    Typical LLM usage example:
        {
            "tool": "preview_template",
            "arguments": {
                "template_id": "receipt",
                "variables": {"name": "John", "amount": "$29.40"}
            }
        }
    """
    try:
        logger.info(f"Previewing template '{template_id}'")
        raw = load_template(template_id)
        message = apply_template(raw, variables)
        return {"message": message}
    except Exception as e:
        logger.exception("Failed to preview template")
        raise e


# -------------------------------
# Run the server
# -------------------------------

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting SMS MCP Server...")
    server.run(transport="sse")
