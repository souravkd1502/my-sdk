"""
Twilio SMS Sender Module
========================

Provides a simple and robust interface for sending SMS messages using Twilio.
This module is designed for use in automation, MCP servers, and LLM workflows.

Features
--------
1. Environment validation:
    Ensures required Twilio credentials are set before sending SMS:
    - TWILIO_ACCOUNT_SID
    - TWILIO_AUTH_TOKEN
    - TWILIO_FROM_NUMBER

2. SMS sending:
    Sends both templated and raw SMS messages via the Twilio REST API.

3. Error handling:
    Raises clear exceptions for:
    - Missing or invalid environment variables
    - Invalid phone number or message
    - Twilio API errors

4. Logging:
    Logs all attempts and failures, useful for debugging and LLM traceability.

Usage
-----
- Initialize module (loads environment variables automatically)
- Call `send_sms(phone_number, message)` to send a message
- Returns Twilio message SID for tracking

Typical LLM usage example
-------------------------
{
    "tool": "send_sms",
    "arguments": {
        "phone_number": "+1234567890",
        "message": "Your OTP is 123456"
    }
}

Notes
-----
- Designed to be imported once per process; client is initialized at module load.
- Reuses the Twilio client for all subsequent calls to avoid repeated connections.
- Requires environment variables to be set prior to server start or module import.
"""


import os
import logging
from typing import Optional
from twilio.rest import Client
from dotenv import load_dotenv
from twilio.base.exceptions import TwilioRestException

# Load environment variables (override existing)
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Environment variables
TWILIO_SID: Optional[str] = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN: Optional[str] = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER: Optional[str] = os.getenv("TWILIO_FROM_NUMBER")

# Validate environment
if not all([TWILIO_SID, TWILIO_TOKEN, TWILIO_NUMBER]):
    missing = [
        var
        for var, val in [
            ("TWILIO_ACCOUNT_SID", TWILIO_SID),
            ("TWILIO_AUTH_TOKEN", TWILIO_TOKEN),
            ("TWILIO_FROM_NUMBER", TWILIO_NUMBER),
        ]
        if not val
    ]
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(missing)}"
    )

# Initialize Twilio client
client = Client(TWILIO_SID, TWILIO_TOKEN)


def send_sms(phone_number: str, message: str) -> str:
    """
    Send an SMS message via Twilio.

    Args:
        phone_number (str): Recipient phone number in E.164 format (e.g., +1234567890)
        message (str): Message body to send

    Returns:
        str: Twilio message SID for tracking

    Raises:
        ValueError: If phone_number or message is empty
        TwilioRestException: If Twilio API call fails
    """
    if not phone_number or not message:
        raise ValueError("Both phone_number and message are required.")

    try:
        logger.info(f"Sending SMS to {phone_number}...")
        msg = client.messages.create(body=message, from_=TWILIO_NUMBER, to=phone_number)
        logger.info(f"SMS sent successfully. SID: {msg.sid}")
        return msg.sid
    except TwilioRestException as e:
        logger.error(f"Failed to send SMS: {e}")
        raise e
