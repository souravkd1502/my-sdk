"""
SMS Service Module

Summary:
    A comprehensive SMS/MMS service built on Twilio API providing robust message 
    delivery capabilities with automatic retry mechanisms, bulk operations, and 
    both synchronous and asynchronous support for high-performance applications.

Usage:
    Basic SMS sending:
        >>> from sms_service import SMSService
        >>> service = SMSService()
        >>> sid = service.send_sms("+1234567890", "Hello World!")
    
    Bulk messaging:
        >>> recipients = ["+1234567890", "+0987654321"] 
        >>> sids = service.send_bulk_sms(recipients, "Bulk message!")
    
    Asynchronous operations:
        >>> import asyncio
        >>> async def send_async():
        ...     sid = await service.send_sms_async("+1234567890", "Async message!")
        >>> asyncio.run(send_async())
    
    Context manager usage:
        >>> with SMSService() as service:
        ...     sid = service.send_sms("+1234567890", "Hello!")

Dependencies:
    - twilio>=8.0.0: Official Twilio Python SDK for API communication
    - python-dotenv>=0.19.0: Environment variable management from .env files
    - asyncio: Built-in async/await support (Python 3.7+)
    - concurrent.futures: Thread pool execution for async operations
    - typing: Type hints support (Python 3.5+)

Features:
    - Automatic retry with exponential backoff and jitter
    - Bulk SMS/MMS sending with individual retry logic
    - Asynchronous operations using thread pool executors
    - MMS support with multiple media attachments
    - Comprehensive logging with structured output
    - Environment variable configuration support
    - Context manager for automatic resource cleanup
    - Production-ready error handling and validation
    - Thread-safe operations for concurrent usage
    - Rate limiting considerations for bulk operations

Author:
    Sourav Das
    Version: 1.0.0
    Created: 2025
"""

import asyncio
import os
import logging
import random
import time
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Any, Callable, List, Union, Optional, Awaitable, Tuple
from concurrent.futures import ThreadPoolExecutor

from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

# -------------------------
# Logging configuration
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger("SMSService")


@dataclass
class TwilioConfig:
    """
    Configuration dataclass for Twilio SMS service.

    This class holds all necessary credentials and settings required
    to authenticate and interact with the Twilio API.

    Attributes:
        account_sid (str): Twilio Account SID - unique identifier for your account
        auth_token (str): Twilio Auth Token - secret key for authentication
        from_number (str): Twilio phone number to send messages from (E.164 format)

    Note:
        All phone numbers should be in E.164 format (e.g., +1234567890)
    """

    account_sid: str
    auth_token: str
    from_number: str

    def __post_init__(self) -> None:
        """
        Validate configuration after initialization.

        Raises:
            ValueError: If any required configuration value is empty or invalid
        """
        if not self.account_sid:
            raise ValueError("Twilio Account SID is required")
        if not self.auth_token:
            raise ValueError("Twilio Auth Token is required")
        if not self.from_number:
            raise ValueError("Twilio From Number is required")

        # Log configuration initialization (without sensitive data)
        logger.info(f"TwilioConfig initialized with from_number: {self.from_number}")


class SMSService:
    """
    Comprehensive SMS/MMS service using Twilio API.

    This service provides both synchronous and asynchronous methods for sending
    SMS and MMS messages with built-in retry mechanisms, bulk sending capabilities,
    and comprehensive error handling.

    Features:
        - Automatic retry with exponential backoff
        - Bulk message sending (sync and async)
        - MMS support with media attachments
        - Thread pool for async operations
        - Comprehensive logging and error handling
        - Environment variable configuration support

    Attributes:
        config (TwilioConfig): Twilio configuration containing credentials
        client (Client): Twilio REST API client instance
        executor (ThreadPoolExecutor): Thread pool for async operations

    Example:
        Basic usage with environment variables:
        >>> service = SMSService()
        >>> message_sid = service.send_sms("+1234567890", "Hello World!")

        Custom configuration:
        >>> config = TwilioConfig(
        ...     account_sid="your_sid",
        ...     auth_token="your_token",
        ...     from_number="+1234567890"
        ... )
        >>> service = SMSService(config)
    """

    def __init__(self, config: Optional[TwilioConfig] = None) -> None:
        """
        Initialize the SMS service with Twilio configuration.

        Args:
            config (Optional[TwilioConfig]): Twilio configuration object.
                If None, configuration will be loaded from environment variables:
                - TWILIO_ACCOUNT_SID
                - TWILIO_AUTH_TOKEN
                - TWILIO_FROM_NUMBER

        Raises:
            ValueError: If configuration is missing or invalid
            TwilioRestException: If Twilio client initialization fails

        Note:
            Environment variables should be set in a .env file or system environment
        """
        logger.info("Initializing SMSService...")

        # Load configuration from environment if not provided
        if not config:
            logger.debug("Loading configuration from environment variables")
            load_dotenv()
            config = TwilioConfig(
                account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
                auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
                from_number=os.getenv("TWILIO_FROM_NUMBER", ""),
            )

        # Store configuration and initialize Twilio client
        self.config: TwilioConfig = config

        try:
            self.client: Client = Client(config.account_sid, config.auth_token)
            logger.info("Twilio client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Twilio client: {e}")
            raise

        # Initialize thread pool for async operations
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=5, thread_name_prefix="SMSService"
        )

        logger.info("SMSService initialization completed successfully")

    # -------------------------
    # Retry Helpers
    # -------------------------
    @staticmethod
    def _retry_with_backoff(
        func: Callable[..., Any],
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exceptions: Tuple[type, ...] = (Exception,),
        *args,
        **kwargs,
    ) -> Any:
        """
        Retry a synchronous function call with exponential backoff.

        This method implements an exponential backoff strategy with jitter
        to avoid thundering herd problems when multiple retries occur simultaneously.

        Args:
            func (Callable[..., Any]): The function to retry
            max_attempts (int): Maximum number of retry attempts (default: 5)
            base_delay (float): Base delay in seconds for exponential backoff (default: 1.0)
            max_delay (float): Maximum delay between retries in seconds (default: 30.0)
            exceptions (Tuple[type, ...]): Tuple of exception types to catch and retry on
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Any: The return value of the successful function call

        Raises:
            Exception: The last exception raised if all retry attempts fail

        Note:
            The delay calculation uses exponential backoff with jitter:
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay) * random_factor
            where random_factor is between 0.9 and 1.1 to add jitter
        """
        attempt = 0
        while attempt < max_attempts:
            try:
                logger.debug(
                    f"Attempting function call (attempt {attempt + 1}/{max_attempts})"
                )
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"Function succeeded after {attempt + 1} attempts")
                return result
            except exceptions as e:
                attempt += 1
                if attempt >= max_attempts:
                    logger.error(f"Function failed after {max_attempts} attempts: {e}")
                    raise

                # Calculate exponential backoff delay with jitter
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                jitter = (
                    0.9 + random.random() * 0.2
                )  # Random factor between 0.9 and 1.1
                delay = delay * jitter

                logger.warning(
                    f"[Retry {attempt}/{max_attempts}] {type(e).__name__}: {e} - "
                    f"retrying in {delay:.2f}s"
                )
                time.sleep(delay)

    @staticmethod
    async def _retry_with_backoff_async(
        func: Callable[..., Awaitable[Any]],
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exceptions: Tuple[type, ...] = (Exception,),
        *args,
        **kwargs,
    ) -> Any:
        """
        Retry an asynchronous function call with exponential backoff.

        This method implements an exponential backoff strategy with jitter
        for async operations, preventing thundering herd problems in concurrent scenarios.

        Args:
            func (Callable[..., Awaitable[Any]]): The async function to retry
            max_attempts (int): Maximum number of retry attempts (default: 5)
            base_delay (float): Base delay in seconds for exponential backoff (default: 1.0)
            max_delay (float): Maximum delay between retries in seconds (default: 30.0)
            exceptions (Tuple[type, ...]): Tuple of exception types to catch and retry on
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Any: The return value of the successful function call

        Raises:
            Exception: The last exception raised if all retry attempts fail

        Note:
            Uses asyncio.sleep() for non-blocking delays in async context.
            The delay calculation includes jitter to prevent synchronized retries.
        """
        attempt = 0
        while attempt < max_attempts:
            try:
                logger.debug(
                    f"Attempting async function call (attempt {attempt + 1}/{max_attempts})"
                )
                result = await func(*args, **kwargs)
                if attempt > 0:
                    logger.info(
                        f"Async function succeeded after {attempt + 1} attempts"
                    )
                return result
            except exceptions as e:
                attempt += 1
                if attempt >= max_attempts:
                    logger.error(
                        f"Async function failed after {max_attempts} attempts: {e}"
                    )
                    raise

                # Calculate exponential backoff delay with jitter
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                jitter = (
                    0.9 + random.random() * 0.2
                )  # Random factor between 0.9 and 1.1
                delay = delay * jitter

                logger.warning(
                    f"[Async Retry {attempt}/{max_attempts}] {type(e).__name__}: {e} - "
                    f"retrying in {delay:.2f}s"
                )
                await asyncio.sleep(delay)

    # -------------------------
    # Synchronous Methods
    # -------------------------
    def send_sms(self, to: str, body: str) -> Optional[str]:
        """
        Send an SMS message to a single recipient with automatic retry.

        This method sends a text message using the Twilio API with built-in
        retry logic to handle transient failures gracefully.

        Args:
            to (str): Recipient's phone number in E.164 format (e.g., +1234567890)
            body (str): Text content of the SMS message (max 1600 characters)

        Returns:
            Optional[str]: Twilio message SID if successful, None if failed after retries

        Raises:
            No exceptions are raised - all errors are caught and logged

        Example:
            >>> service = SMSService()
            >>> sid = service.send_sms("+1234567890", "Hello World!")
            >>> if sid:
            ...     print(f"Message sent successfully: {sid}")

        Note:
            - Uses automatic retry with exponential backoff (max 4 attempts)
            - Logs success and failure events for monitoring
            - Returns None on failure to allow graceful error handling
        """
        logger.debug(f"Sending SMS to {to} with message length: {len(body)}")

        def _send() -> str:
            """Internal function to send SMS message."""
            message = self.client.messages.create(
                to=to, from_=self.config.from_number, body=body
            )
            logger.info(f"SMS sent successfully to {to} | SID: {message.sid}")
            return message.sid

        try:
            return self._retry_with_backoff(
                _send, exceptions=(TwilioRestException,), max_attempts=4
            )
        except Exception as e:
            logger.error(f"Failed to send SMS to {to} after retries: {e}")
            return None

    def send_bulk_sms(self, recipients: List[str], body: str) -> List[str]:
        """
        Send the same SMS message to multiple recipients.

        This method sends identical SMS messages to a list of recipients,
        with individual retry logic for each recipient to maximize delivery success.

        Args:
            recipients (List[str]): List of phone numbers in E.164 format
            body (str): Text content of the SMS message

        Returns:
            List[str]: List of successful message SIDs (may be shorter than input list)

        Example:
            >>> service = SMSService()
            >>> recipients = ["+1234567890", "+0987654321"]
            >>> sids = service.send_bulk_sms(recipients, "Bulk message!")
            >>> print(f"Sent {len(sids)} out of {len(recipients)} messages")

        Note:
            - Each message is sent individually with retry logic
            - Failed messages are logged but don't stop the bulk operation
            - Returns only the SIDs of successfully sent messages
            - Consider rate limiting for large recipient lists
        """
        logger.info(f"Starting bulk SMS send to {len(recipients)} recipients")
        sent_sids = []

        for i, recipient in enumerate(recipients, 1):
            logger.debug(f"Sending bulk SMS {i}/{len(recipients)} to {recipient}")
            if sid := self.send_sms(recipient, body):
                sent_sids.append(sid)

        logger.info(
            f"Bulk SMS completed: {len(sent_sids)}/{len(recipients)} successful"
        )
        return sent_sids

    def send_sms_with_media(
        self, to: str, body: str, media_url: Union[List[str], str]
    ) -> Optional[str]:
        """
        Send an MMS message with media attachments and automatic retry.

        This method sends a multimedia message (MMS) containing text and media
        attachments using the Twilio API with built-in retry logic.

        Args:
            to (str): Recipient's phone number in E.164 format
            body (str): Text content of the MMS message
            media_url (Union[List[str], str]): URL(s) of media to attach
                - Can be a single URL string or list of URLs
                - Supported formats: JPEG, PNG, GIF (up to 5MB total)
                - Max 10 media attachments per message

        Returns:
            Optional[str]: Twilio message SID if successful, None if failed after retries

        Example:
            >>> service = SMSService()
            >>> media_urls = ["https://example.com/image1.jpg", "https://example.com/image2.png"]
            >>> sid = service.send_sms_with_media(
            ...     "+1234567890",
            ...     "Check out these images!",
            ...     media_urls
            ... )

        Note:
            - Media files must be publicly accessible via HTTPS
            - Carrier fees may apply for MMS messages
            - Some carriers have restrictions on MMS delivery
        """
        # Normalize media_url to list format
        if isinstance(media_url, str):
            media_url = [media_url]

        logger.debug(f"Sending MMS to {to} with {len(media_url)} media attachments")

        def _send() -> str:
            """Internal function to send MMS message."""
            message = self.client.messages.create(
                to=to, from_=self.config.from_number, body=body, media_url=media_url
            )
            logger.info(f"MMS sent successfully to {to} | SID: {message.sid}")
            return message.sid

        try:
            return self._retry_with_backoff(
                _send, exceptions=(TwilioRestException,), max_attempts=4
            )
        except Exception as e:
            logger.error(f"Failed to send MMS to {to} after retries: {e}")
            return None

    # -------------------------
    # Asynchronous Methods
    # -------------------------
    async def send_sms_async(self, to: str, body: str) -> Optional[str]:
        """
        Asynchronously send an SMS message with automatic retry.

        This method provides non-blocking SMS sending using a thread pool executor
        to avoid blocking the event loop while maintaining retry capabilities.

        Args:
            to (str): Recipient's phone number in E.164 format (e.g., +1234567890)
            body (str): Text content of the SMS message (max 1600 characters)

        Returns:
            Optional[str]: Twilio message SID if successful, None if failed after retries

        Raises:
            No exceptions are raised - all errors are caught and logged

        Example:
            >>> service = SMSService()
            >>> async def send_message():
            ...     sid = await service.send_sms_async("+1234567890", "Hello Async!")
            ...     if sid:
            ...         print(f"Async message sent: {sid}")
            >>> asyncio.run(send_message())

        Note:
            - Uses thread pool executor to prevent blocking the event loop
            - Implements async retry with exponential backoff (max 4 attempts)
            - Ideal for applications requiring high concurrency
        """
        logger.debug(f"Sending async SMS to {to} with message length: {len(body)}")
        loop = asyncio.get_event_loop()

        async def _send():
            """Internal async function to send SMS message via thread pool."""
            return await loop.run_in_executor(
                self.executor,
                lambda: self.client.messages.create(
                    to=to, from_=self.config.from_number, body=body
                ),
            )

        try:
            message = await self._retry_with_backoff_async(
                _send, exceptions=(TwilioRestException,), max_attempts=4
            )
            logger.info(f"Async SMS sent successfully to {to} | SID: {message.sid}")
            return message.sid
        except Exception as e:
            logger.error(f"Async SMS failed to {to} after retries: {e}")
            return None

    async def send_bulk_sms_async(self, recipients: List[str], body: str) -> List[str]:
        """
        Asynchronously send the same SMS message to multiple recipients concurrently.

        This method leverages asyncio to send messages to multiple recipients
        concurrently, significantly improving performance for bulk operations.

        Args:
            recipients (List[str]): List of phone numbers in E.164 format
            body (str): Text content of the SMS message

        Returns:
            List[str]: List of successful message SIDs (excludes failed sends)

        Example:
            >>> service = SMSService()
            >>> async def bulk_send():
            ...     recipients = ["+1234567890", "+0987654321", "+1122334455"]
            ...     sids = await service.send_bulk_sms_async(recipients, "Bulk async message!")
            ...     print(f"Successfully sent {len(sids)} out of {len(recipients)} messages")
            >>> asyncio.run(bulk_send())

        Note:
            - All messages are sent concurrently for maximum performance
            - Each message has independent retry logic
            - Failed sends don't affect successful ones
            - Returns empty list if all sends fail
            - Consider rate limiting for very large recipient lists
        """
        logger.info(f"Starting async bulk SMS send to {len(recipients)} recipients")

        # Create concurrent tasks for all recipients
        tasks = [self.send_sms_async(to, body) for to in recipients]

        # Execute all tasks concurrently and handle exceptions
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results (SID strings) and exclude None/exceptions
        successful_sids = [sid for sid in results if isinstance(sid, str)]

        logger.info(
            f"Async bulk SMS completed: {len(successful_sids)}/{len(recipients)} successful"
        )
        return successful_sids

    async def send_sms_with_media_async(
        self, to: str, body: str, media_url: Union[List[str], str]
    ) -> Optional[str]:
        """
        Asynchronously send an MMS message with media attachments and automatic retry.

        This method provides non-blocking MMS sending with media attachments
        using a thread pool executor to maintain event loop responsiveness.

        Args:
            to (str): Recipient's phone number in E.164 format
            body (str): Text content of the MMS message
            media_url (Union[List[str], str]): URL(s) of media to attach
                - Can be a single URL string or list of URLs
                - Supported formats: JPEG, PNG, GIF (up to 5MB total)
                - Max 10 media attachments per message

        Returns:
            Optional[str]: Twilio message SID if successful, None if failed after retries

        Example:
            >>> service = SMSService()
            >>> async def send_mms():
            ...     media_urls = ["https://example.com/image.jpg"]
            ...     sid = await service.send_sms_with_media_async(
            ...         "+1234567890",
            ...         "Check this out!",
            ...         media_urls
            ...     )
            ...     if sid:
            ...         print(f"Async MMS sent: {sid}")
            >>> asyncio.run(send_mms())

        Note:
            - Media files must be publicly accessible via HTTPS
            - Uses thread pool executor for non-blocking operation
            - Implements async retry with exponential backoff
            - Carrier fees and restrictions may apply for MMS
        """
        # Normalize media_url to list format
        if isinstance(media_url, str):
            media_url = [media_url]

        logger.debug(
            f"Sending async MMS to {to} with {len(media_url)} media attachments"
        )
        loop = asyncio.get_event_loop()

        async def _send():
            """Internal async function to send MMS message via thread pool."""
            return await loop.run_in_executor(
                self.executor,
                lambda: self.client.messages.create(
                    to=to, from_=self.config.from_number, body=body, media_url=media_url
                ),
            )

        try:
            message = await self._retry_with_backoff_async(
                _send, exceptions=(TwilioRestException,), max_attempts=4
            )
            logger.info(f"Async MMS sent successfully to {to} | SID: {message.sid}")
            return message.sid
        except Exception as e:
            logger.error(f"Async MMS failed to {to} after retries: {e}")
            return None

    def close(self) -> None:
        """
        Clean up resources used by the SMS service.

        This method shuts down the thread pool executor and releases
        any associated resources. Should be called when the service
        is no longer needed, especially in long-running applications.

        Example:
            >>> service = SMSService()
            >>> # ... use the service ...
            >>> service.close()  # Clean up when done

        Note:
            - This method should be called before application shutdown
            - After calling close(), the service should not be used
            - In context managers, this is called automatically
        """
        logger.info("Shutting down SMSService...")

        try:
            self.executor.shutdown(wait=True)
            logger.info("Thread pool executor shut down successfully")
        except Exception as e:
            logger.error(f"Error during SMSService shutdown: {e}")

    def __enter__(self) -> "SMSService":
        """
        Context manager entry point.

        Returns:
            SMSService: This service instance for use in with statement
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[Exception],
        exc_tb: Optional[object],
    ) -> None:
        """
        Context manager exit point.

        Automatically cleans up resources when exiting the context.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        self.close()


# -------------------------
# Example usage and demonstrations
# -------------------------
if __name__ == "__main__":
    """
    Example usage demonstrating various SMS service capabilities.

    This section shows how to use the SMS service in different scenarios:
    - Basic SMS sending
    - Bulk SMS operations
    - MMS with media attachments
    - Asynchronous operations
    - Error handling and logging
    """

    # Initialize service (loads config from environment variables)
    logger.info("=== SMS Service Demo Starting ===")

    # Example phone numbers (replace with real numbers for testing)
    test_number = "+1234567890"
    bulk_recipients = ["+1234567890", "+0987654321", "+1122334455"]

    # Use context manager for automatic cleanup
    with SMSService() as sms_service:
        logger.info("SMSService initialized successfully")

        # -------------------------
        # Synchronous Examples
        # -------------------------
        logger.info("--- Synchronous SMS Examples ---")

        # Basic SMS
        sid = sms_service.send_sms(test_number, "Hello from enhanced SMS Service!")
        if sid:
            logger.info(f"✅ SMS sent successfully: {sid}")
        else:
            logger.error("❌ SMS failed to send")

        # Bulk SMS
        bulk_sids = sms_service.send_bulk_sms(
            bulk_recipients, "Bulk message from enhanced SMS Service!"
        )
        logger.info(
            f"✅ Bulk SMS: {len(bulk_sids)}/{len(bulk_recipients)} sent successfully"
        )

        # MMS with media
        media_urls = ["https://example.com/sample-image.jpg"]
        mms_sid = sms_service.send_sms_with_media(
            test_number, "Check out this image!", media_urls
        )
        if mms_sid:
            logger.info(f"✅ MMS sent successfully: {mms_sid}")

        # -------------------------
        # Asynchronous Examples
        # -------------------------
        async def async_examples():
            """Demonstrate asynchronous SMS operations."""
            logger.info("--- Asynchronous SMS Examples ---")

            # Async SMS
            async_sid = await sms_service.send_sms_async(
                test_number, "Hello from async SMS Service!"
            )
            if async_sid:
                logger.info(f"✅ Async SMS sent successfully: {async_sid}")

            # Async bulk SMS (concurrent sending)
            async_bulk_sids = await sms_service.send_bulk_sms_async(
                bulk_recipients, "Async bulk message from SMS Service!"
            )
            logger.info(
                f"✅ Async bulk SMS: {len(async_bulk_sids)}/{len(bulk_recipients)} sent"
            )

            # Async MMS
            async_mms_sid = await sms_service.send_sms_with_media_async(
                test_number, "Async MMS with image!", media_urls
            )
            if async_mms_sid:
                logger.info(f"✅ Async MMS sent successfully: {async_mms_sid}")

        # Run async examples
        try:
            asyncio.run(async_examples())
        except Exception as e:
            logger.error(f"Error in async examples: {e}")

    logger.info("=== SMS Service Demo Completed ===")
    logger.info("Resources cleaned up automatically via context manager")
