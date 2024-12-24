"""
azure_service_bus.py
--------------------

A module to manage the connection to an Azure Service Bus queue using either Azure Active Directory credentials
or a namespace connection string. This module abstracts the initialization of the `ServiceBusClient` and provides
error handling and logging.

This module provides the following classes:
- AzureServiceBus: A class to manage the connection to an Azure Service Bus queue.

Usage:
    To use the AzureServiceBus class, import it into your Python script and initialize an instance with the required
    parameters. You can then send messages to the queue using the `send_single_message`, `send_list_of_messages`,
    or `send_batch_of_messages` methods. To receive messages from the queue, use the `receive_messages` method.

Example:
    # Initialize AzureServiceBus instance with Azure credentials
    service_bus = AzureServiceBus(
        fully_qualified_namespace="your-namespace.servicebus.windows.net",
        queue_name="your-queue-name",
        tenant_id="your-tenant-id",
        client_id="your-client-id",
        client_secret="your-client-secret",
    )

    # Initialize AzureServiceBus instance with namespace connection string
    service_bus = AzureServiceBus(
        queue_name="your-queue-name",
        namespace_connection_str="your-namespace-connection-string",
    )
    
    # Send a single message to the queue
    service_bus.send_single_message("Hello, Azure Service Bus!")
    
    # Receive messages from the queue
    messages = service_bus.receive_messages(max_messages=5, max_wait_time=10)
    
Requirements:
--------------
- azure-core==1.18.0
- azure-identity==1.6.0
- azure-servicebus==7.0.0

Author:
-------------
Sourav Das

Date:
-------------
2024-12-24
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Import dependencies
import logging

from azure.core.exceptions import AzureError
from azure.identity import ClientSecretCredential
from azure.servicebus import ServiceBusClient, ServiceBusMessage


from typing import Optional, List

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class AzureServiceBus:
    """
    A class to manage the connection to an Azure Service Bus queue using either Azure Active Directory credentials
    or a namespace connection string.

    This class abstracts the initialization of the `ServiceBusClient` and provides error handling and logging.
    """

    def __init__(
        self,
        fully_qualified_namespace: Optional[str] = None,
        queue_name: Optional[str] = None,
        namespace_connection_str: Optional[str] = None,
        tenant_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        logging_status: bool = True,
    ) -> None:
        """
        Initializes the AzureServiceBus instance with credentials and configuration.

        Args:
            fully_qualified_namespace (Optional[str]): The fully qualified namespace of the Azure Service Bus.
            queue_name (Optional[str]): The name of the queue to connect to. This is required.
            namespace_connection_str (Optional[str]): The connection string for the Azure Service Bus namespace.
            tenant_id (Optional[str]): Azure Active Directory tenant ID for authentication.
            client_id (Optional[str]): Azure Active Directory client ID for authentication.
            client_secret (Optional[str]): Azure Active Directory client secret for authentication.
            logging_status (bool): Whether to enable logging for the Service Bus Client. Defaults to True.

        Raises:
            ValueError: If queue_name is missing or invalid arguments are provided.
            AzureError: If initialization of credentials or Service Bus client fails.
        """
        self.fully_qualified_namespace = fully_qualified_namespace
        self.queue_name = queue_name
        self.namespace_connection_str = namespace_connection_str
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.logging_status = logging_status
        self.credential = (
            None  # Will store ClientSecretCredential if Azure credentials are used
        )
        self.client = None  # Will store ServiceBusClient instance
        self.sender = None  # Will store ServiceBusSender instance

        # Input validation
        self._validate_inputs()

        # Initialize credential or client based on inputs
        self._initialize_client()

        # Get the ServiceBusSender instance
        self.sender = self._initialize_sender()

    def _validate_inputs(self) -> None:
        """
        Validates the initialization inputs to ensure correct configuration.

        Raises:
            ValueError: If invalid input combinations are provided.
        """
        if not self.queue_name or not self.queue_name.strip():
            _logger.error("The `queue_name` is required but not provided.")
            raise ValueError(
                "The `queue_name` parameter is required and cannot be empty."
            )

        if self.fully_qualified_namespace and self.namespace_connection_str:
            _logger.error(
                "Both `fully_qualified_namespace` and `namespace_connection_str` are provided. Provide only one."
            )
            raise ValueError(
                "You cannot provide both `fully_qualified_namespace` and `namespace_connection_str`."
            )

        if self.namespace_connection_str and any(
            [self.tenant_id, self.client_id, self.client_secret]
        ):
            _logger.error(
                "Both `namespace_connection_str` and Azure credentials are provided. Provide only one."
            )
            raise ValueError(
                "You cannot provide both `namespace_connection_str` and Azure credentials."
            )

        if self.fully_qualified_namespace and not all(
            [self.tenant_id, self.client_id, self.client_secret]
        ):
            _logger.error(
                "Incomplete Azure credentials provided for `fully_qualified_namespace`."
            )
            raise ValueError(
                "Azure credentials (tenant_id, client_id, client_secret) are required with `fully_qualified_namespace`."
            )

    def _initialize_client(self) -> None:
        """
        Initializes the Service Bus Client based on the provided inputs.

        Raises:
            AzureError: If an error occurs during client or credential initialization.
        """
        try:
            # Initialize Azure credentials if provided
            if self.tenant_id and self.client_id and self.client_secret:
                _logger.info(
                    "Initializing `ClientSecretCredential` with Azure credentials."
                )
                self.credential = ClientSecretCredential(
                    tenant_id=self.tenant_id,
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                )
                _logger.info("Azure credentials initialized successfully.")

            # Initialize Service Bus Client
            if self.fully_qualified_namespace:
                _logger.info(
                    "Initializing `ServiceBusClient` with fully qualified namespace."
                )
                self.client = ServiceBusClient(
                    fully_qualified_namespace=self.fully_qualified_namespace,
                    credential=self.credential,
                    logging_enable=self.logging_status,
                )
            elif self.namespace_connection_str:
                _logger.info("Initializing `ServiceBusClient` with connection string.")
                self.client = ServiceBusClient.from_connection_string(
                    conn_str=self.namespace_connection_str,
                    logging_enable=self.logging_status,
                )

            _logger.info("ServiceBusClient initialized successfully.")
        except Exception as e:
            _logger.error(f"Error initializing ServiceBusClient: {e}")
            raise AzureError(f"Error initializing ServiceBusClient: {e}")

    def _initialize_sender(self) -> None:
        """
        Initializes the Service Bus Sender using Service Bus Client.
        Returns:
            ServiceBusClient: The initialized ServiceBusSender instance.

        Raises:
            AzureError: If the Sender is not initialized.
        """
        if not self.client:
            _logger.error(
                "ServiceBusClient is not initialized. Check _initialize_client()."
            )
            raise AzureError(
                "ServiceBusClient is not initialized. Check your configuration."
            )
        try:
            self.sender = self.client.get_queue_sender(queue_name=self.queue_name)
        except Exception as e:
            _logger.error(f"Error getting ServiceBusClient: {e}")
            raise AzureError(f"Error getting ServiceBusClient: {e}")

    def _initialize_receiver(self) -> None:
        """
        Initializes the Service Bus Receiver using Service Bus Client.
        Returns:
            ServiceBusClient: The initialized ServiceBusReceiver instance.

        Raises:
            AzureError: If the Receiver is not initialized.
        """
        if not self.client:
            _logger.error("ServiceBusClient is not initialized.")
            raise AzureError(
                "ServiceBusClient is not initialized. Check your configuration."
            )
        try:
            self.receiver = self.client.get_queue_receiver(queue_name=self.queue_name)
        except Exception as e:
            _logger.error(f"Error getting ServiceBusClient: {e}")
            raise AzureError(f"Error getting ServiceBusClient: {e}")

    def _close_client(self) -> None:
        """
        Closes the ServiceBusClient connection.

        Ensures that resources are cleaned up after usage.
        """
        if self.client:
            _logger.info("Closing ServiceBusClient connection.")
            self.client.close()
            _logger.info("ServiceBusClient connection closed.")

    def send_single_message(self, message: str) -> None:
        """
        Sends a single message to the Azure Service Bus queue.

        Args:
            message (str): The message to be sent.

        Raises:
            AzureError: If an error occurs during message sending.
        """
        try:
            self._initialize_sender()
            _logger.info("Sending message to Azure Service Bus queue.")
            message = ServiceBusMessage(message)
            self.sender.send_messages(message)
            _logger.info("Message sent to Azure Service Bus queue.")
        except Exception as e:
            _logger.error(f"Error sending message: {e}")
            raise AzureError(f"Error sending message: {e}")
        finally:
            self.sender.close()
            self._close_client()

    def send_list_of_messages(self, messages: list) -> None:
        """
        Sends a list of messages to the Azure Service Bus queue.

        Args:
            messages (list): A list of messages to be sent.

        Raises:
            AzureError: If an error occurs during message sending.
        """
        try:
            self._initialize_sender()
            _logger.info("Sending list of messages to Azure Service Bus queue.")
            messages = [ServiceBusMessage(msg) for msg in messages]
            self.sender.send_messages(messages)
            _logger.info("List of messages sent to Azure Service Bus queue.")
        except Exception as e:
            _logger.error(f"Error sending list of messages: {e}")
            raise AzureError(f"Error sending list of messages: {e}")
        finally:
            self._close_client()

    def send_batch_of_messages(self, messages: list) -> None:
        """
        Sends a batch of messages to the Azure Service Bus queue.

        Args:
            messages (list): A list of messages to be sent.

        Raises:
            AzureError: If an error occurs during message sending.
        """
        try:
            batch_message = self.sender.create_message_batch()
            for message in messages:
                batch_message.add_message(ServiceBusMessage(message))
            _logger.info("Sending batch of messages to Azure Service Bus queue.")
            self.sender.send_messages(batch_message)
            _logger.info("Batch of messages sent to Azure Service Bus queue.")
        except Exception as e:
            _logger.error(f"Error sending batch of messages: {e}")
            raise AzureError(f"Error sending batch of messages: {e}")
        finally:
            self._close_client()

    def receive_messages(self, max_messages: int, max_wait_time: int) -> List[str]:
        """
        Receives messages from the Azure Service Bus queue.

        This method connects to the queue receiver, retrieves messages,
        completes them (removing them from the queue), and returns the message content.

        Args:
            max_messages (int): The maximum number of messages to receive in one call.
            max_wait_time (int): The maximum time (in seconds) to wait for receiving messages.

        Returns:
            List[str]: A list of message contents as strings.

        Raises:
            ValueError: If `max_messages` or `max_wait_time` is invalid.
            AzureError: If an error occurs during message receiving.
        """
        # Input validation
        if max_messages <= 0:
            _logger.error("`max_messages` must be a positive integer.")
            raise ValueError("`max_messages` must be a positive integer.")
        if max_wait_time <= 0:
            _logger.error("`max_wait_time` must be a positive integer.")
            raise ValueError("`max_wait_time` must be a positive integer.")

        _logger.info("Initializing receiver for the Azure Service Bus queue.")
        try:
            # Ensure the client is initialized
            if not self.client:
                _logger.error(
                    "ServiceBusClient is not initialized. Cannot receive messages."
                )
                raise AzureError("ServiceBusClient is not initialized.")

            # Create a receiver for the specified queue
            receiver = self.client.get_queue_receiver(queue_name=self.queue_name)
            _logger.info(
                f"Receiving up to {max_messages} messages with a max wait time of {max_wait_time} seconds."
            )

            # Receive messages from the queue
            received_messages = receiver.receive_messages(
                max_message_count=max_messages,
                max_wait_time=max_wait_time,
            )

            # Process and complete messages
            messages = []
            for message in received_messages:
                _logger.debug(f"Processing message: {message.body}")
                messages.append(str(message))
                receiver.complete_message(message)
                _logger.debug("Message completed successfully.")

            _logger.info(
                f"Successfully received {len(messages)} message(s) from the queue."
            )
            return messages

        except Exception as e:
            _logger.error(f"ServiceBusError occurred while receiving messages: {e}")
            raise AzureError(f"ServiceBusError occurred: {e}")

        finally:
            self._close_client()
