"""
RabbitMQ Wrapper (Python)
----------------------------------

A batteries-included single-file wrapper around RabbitMQ using `pika` that provides:

- Connection management with automatic reconnect & exponential backoff
- Publisher confirms with retry-on-nack
- JSON (or bytes) message publish helper with persistent delivery
- Queue/exchange/Binding declaration helpers (idempotent)
- Consumer helper with graceful shutdown, QoS (prefetch), and dead-letter support
- Simple RPC helper using `amq.rabbitmq.reply-to`
- Structured logging hooks

Requirements:
    pip install pika

Environment:
    - RABBITMQ_URL (optional): amqp(s)://user:pass@host:port/vhost

Usage example (see __main__ at bottom):
    python rabbitmq_service.py

This file is meant to be copied into your project as-is. No external imports besides `pika`.

Author: Sourav Das
Email: sourav.bt.kt@gmail.com
Date:05-09-2025
Version: 1.0.0
"""

from __future__ import annotations

import json
import threading
import time
import uuid
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union

try:
    import pika
    from pika.adapters.blocking_connection import BlockingChannel
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "This wrapper requires the 'pika' package. Install with: pip install pika"
    )


# --------------------------------------------------------------------------------------
# Configuration dataclasses
# --------------------------------------------------------------------------------------


@dataclass
class ConnectionConfig:
    """Configuration for RabbitMQ connection parameters.

    Attributes:
        url: AMQP connection URL. Defaults to RABBITMQ_URL env var or localhost.
        heartbeat: Heartbeat interval in seconds for connection health monitoring.
        blocked_connection_timeout: Timeout in seconds for blocked connections.
        connection_attempts: Number of initial connection attempts (we handle retries separately).

    Example:
        >>> config = ConnectionConfig(
        ...     url="amqp://user:pass@localhost:5672/vhost",
        ...     heartbeat=60,
        ...     blocked_connection_timeout=120
        ... )
    """

    url: str = field(
        default_factory=lambda: os.getenv(
            "RABBITMQ_URL", "amqp://guest:guest@localhost:5672/%2F"
        )
    )
    heartbeat: int = 30
    blocked_connection_timeout: int = 60
    connection_attempts: int = 1  # we handle our own retries/backoff


@dataclass
class PublishOptions:
    """Configuration options for publishing messages to RabbitMQ.

    Attributes:
        exchange: Target exchange name for the message.
        routing_key: Routing key to determine which queues receive the message.
        mandatory: If True, message must be routed to at least one queue or an error is raised.
        persistent: If True, message survives broker restarts (delivery_mode=2).
        headers: Optional custom headers for the message.
        content_type: MIME type of the message content (auto-set for JSON).
        content_encoding: Character encoding of the message content.
        expiration_ms: Time-to-live for the message in milliseconds.
        correlation_id: Unique identifier for correlating request/response (used in RPC).
        reply_to: Queue name where replies should be sent (used in RPC).

    Example:
        >>> opts = PublishOptions(
        ...     exchange="my.exchange",
        ...     routing_key="task.process",
        ...     persistent=True,
        ...     headers={"priority": "high"}
        ... )
    """

    exchange: str
    routing_key: str
    mandatory: bool = True
    persistent: bool = True
    headers: Optional[Dict[str, Any]] = None
    content_type: Optional[str] = None  # auto-set for JSON
    content_encoding: Optional[str] = None
    expiration_ms: Optional[int] = None  # time-to-live
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None


@dataclass
class ConsumeOptions:
    """Configuration options for consuming messages from RabbitMQ.

    Attributes:
        queue: Name of the queue to consume from.
        prefetch: Number of unacknowledged messages that can be outstanding (QoS).
        auto_ack: If True, messages are automatically acknowledged upon delivery.
        consumer_tag: Unique identifier for the consumer (auto-generated if None).

    Example:
        >>> opts = ConsumeOptions(
        ...     queue="task.queue",
        ...     prefetch=10,
        ...     auto_ack=False,
        ...     consumer_tag="worker-1"
        ... )
    """

    queue: str
    prefetch: int = 50
    auto_ack: bool = False
    consumer_tag: Optional[str] = None


# --------------------------------------------------------------------------------------
# Core client
# --------------------------------------------------------------------------------------


class RabbitMQClient:
    """Resilient RabbitMQ client with simple APIs for publish/consume/RPC.

    This client provides a high-level interface for RabbitMQ operations with built-in
    resilience features including automatic reconnection, exponential backoff, publisher
    confirms, and comprehensive error handling.

    Features:
        - Automatic connection management with reconnect on failures
        - Publisher confirms with retry logic for guaranteed delivery
        - Consumer management with QoS and graceful shutdown
        - RPC pattern support using direct reply-to queues
        - Dead letter queue setup helpers
        - Topology persistence across reconnection (exchanges, queues, bindings)

    Thread Safety:
        Channels are not thread-safe. This client maintains separate publishing
        and consuming channels per instance. Use separate client instances for
        different threads if concurrent access is needed.

    Example:
        >>> config = ConnectionConfig(url="amqp://user:pass@localhost:5672/")
        >>> client = RabbitMQClient(config)
        >>> client.connect()
        >>> client.declare_exchange("my.exchange")
        >>> client.declare_queue("my.queue")
        >>> client.bind_queue("my.queue", "my.exchange", "my.routing.key")
        >>> client.close()
    """

    def __init__(
        self,
        conn: Optional[ConnectionConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the RabbitMQ client.

        Args:
            conn: Connection configuration. If None, uses default configuration.
            logger: Custom logger instance. If None, creates a default logger.

        Note:
            If no logger is provided, a default console logger is configured
            with INFO level and timestamp formatting.
        """
        self.conn_cfg = conn or ConnectionConfig()
        self.logger = logger or logging.getLogger("RabbitMQ")
        if not self.logger.handlers:
            # Sensible default logging
            handler = logging.StreamHandler()
            fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            handler.setFormatter(logging.Formatter(fmt))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        self._connection: Optional[pika.BlockingConnection] = None
        self._pub_channel: Optional[BlockingChannel] = None
        self._sub_channel: Optional[BlockingChannel] = None
        self._closing = False

        # Keep a registry of topology to re-declare after reconnect
        self._exchanges: Dict[str, Dict[str, Any]] = {}
        self._queues: Dict[str, Dict[str, Any]] = {}
        self._bindings: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        # RPC support
        self._rpc_lock = threading.Lock()
        self._rpc_consumer_started = False
        self._rpc_responses: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Establish connection and channels with exponential backoff retry.

        Creates a blocking connection to RabbitMQ server and sets up separate channels
        for publishing and consuming. Enables publisher confirms for reliable delivery.

        The method implements exponential backoff starting at 1 second and maxing out
        at 30 seconds between retry attempts. It will retry indefinitely until a
        successful connection is established.

        After successful connection, it automatically re-declares any previously
        registered topology (exchanges, queues, bindings).

        Raises:
            SystemExit: If pika package is not installed (caught at import time).

        Note:
            This method blocks until connection is successful. Use in a separate
            thread if non-blocking behavior is needed.
        """
        backoff = 1.0
        while True:
            try:
                # Configure connection parameters from the config
                params = pika.URLParameters(self.conn_cfg.url)
                params.heartbeat = self.conn_cfg.heartbeat
                params.blocked_connection_timeout = (
                    self.conn_cfg.blocked_connection_timeout
                )
                params.connection_attempts = self.conn_cfg.connection_attempts

                # Establish connection and create separate channels for pub/sub
                self._connection = pika.BlockingConnection(params)
                self._pub_channel = self._connection.channel()
                self._sub_channel = self._connection.channel()

                # Enable publisher confirms for reliable delivery
                self._pub_channel.confirm_delivery()

                self.logger.info(
                    "Connected to RabbitMQ at %s", sanitize_url(self.conn_cfg.url)
                )

                # Re-declare any previously registered topology
                self._redeclare_topology()
                return

            except Exception as e:  # pragma: no cover - network errors
                self.logger.warning(
                    "Connect failed (%s). Retrying in %.1fs", e, backoff
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)  # Cap backoff at 30 seconds

    def close(self) -> None:
        """Gracefully close all channels and connection.

        Sets the closing flag to prevent reconnection attempts, then closes
        the publishing channel, consuming channel, and main connection in sequence.
        All exceptions during closure are suppressed to ensure cleanup completes.

        This method is safe to call multiple times and will not raise exceptions
        even if channels/connection are already closed.

        Note:
            After calling this method, the client instance should not be reused.
            Create a new instance if you need to reconnect.
        """
        self._closing = True

        # Close publishing channel
        try:
            if self._pub_channel and self._pub_channel.is_open:
                self._pub_channel.close()
        except Exception:
            pass  # Suppress exceptions during cleanup

        # Close consuming channel
        try:
            if self._sub_channel and self._sub_channel.is_open:
                self._sub_channel.close()
        except Exception:
            pass  # Suppress exceptions during cleanup

        # Close main connection
        try:
            if self._connection and self._connection.is_open:
                self._connection.close()
        except Exception:
            pass  # Suppress exceptions during cleanup

        self.logger.info("RabbitMQ connection closed")

    # ---------------------- Topology helpers --------------------------
    def declare_exchange(
        self,
        name: str,
        ex_type: str = "topic",
        durable: bool = True,
        auto_delete: bool = False,
        **kwargs,
    ) -> str:
        """Declare an exchange on the RabbitMQ server.

        This method is idempotent - it will not fail if the exchange already exists
        with the same parameters. The exchange configuration is stored internally
        for automatic re-declaration after reconnects.

        Args:
            name: Name of the exchange to declare.
            ex_type: Exchange type (topic, direct, fanout, headers). Default is "topic".
            durable: If True, exchange survives broker restarts. Default is True.
            auto_delete: If True, exchange is deleted when no longer in use. Default is False.
            **kwargs: Additional exchange arguments (e.g., alternate-exchange).

        Returns:
            The exchange name for method chaining.

        Example:
            >>> client.declare_exchange("logs", "fanout", durable=True)
            >>> client.declare_exchange("tasks", "topic", arguments={"alternate-exchange": "alt"})
        """
        # Store exchange config for re-declaration after reconnect
        self._exchanges[name] = {
            "type": ex_type,
            "durable": durable,
            "auto_delete": auto_delete,
            "args": kwargs,
        }
        self._ensure_connection()
        self._pub_channel.exchange_declare(
            exchange=name,
            exchange_type=ex_type,
            durable=durable,
            auto_delete=auto_delete,
            arguments=kwargs,
        )
        return name

    def declare_queue(
        self,
        name: str,
        durable: bool = True,
        exclusive: bool = False,
        auto_delete: bool = False,
        **kwargs,
    ) -> str:
        """Declare a queue on the RabbitMQ server.

        This method is idempotent - it will not fail if the queue already exists
        with the same parameters. The queue configuration is stored internally
        for automatic re-declaration after reconnects.

        Args:
            name: Name of the queue to declare.
            durable: If True, queue survives broker restarts. Default is True.
            exclusive: If True, queue can only be used by this connection. Default is False.
            auto_delete: If True, queue is deleted when no longer in use. Default is False.
            **kwargs: Additional queue arguments (e.g., x-message-ttl, x-dead-letter-exchange).

        Returns:
            The queue name for method chaining.

        Example:
            >>> client.declare_queue("tasks", durable=True)
            >>> client.declare_queue("temp", exclusive=True, auto_delete=True)
            >>> client.declare_queue("work", arguments={"x-dead-letter-exchange": "dlx"})
        """
        # Store queue config for re-declaration after reconnect
        self._queues[name] = {
            "durable": durable,
            "exclusive": exclusive,
            "auto_delete": auto_delete,
            "args": kwargs,
        }
        self._ensure_connection()
        self._pub_channel.queue_declare(
            queue=name,
            durable=durable,
            exclusive=exclusive,
            auto_delete=auto_delete,
            arguments=kwargs,
        )
        return name

    def bind_queue(
        self, queue: str, exchange: str, routing_key: str = "#", **kwargs
    ) -> None:
        """Bind a queue to an exchange with a routing key.

        Creates a binding that determines which messages from the exchange
        will be routed to the queue. The binding configuration is stored
        internally for automatic re-declaration after reconnects.

        Args:
            queue: Name of the queue to bind.
            exchange: Name of the exchange to bind to.
            routing_key: Routing key pattern for message filtering. Default is "#" (all messages).
            **kwargs: Additional binding arguments.

        Example:
            >>> client.bind_queue("error_logs", "logs", "error.*")
            >>> client.bind_queue("all_logs", "logs", "#")  # Catch all messages
            >>> client.bind_queue("priority_tasks", "tasks", "high.priority")
        """
        # Store binding config for re-declaration after reconnect
        key = (queue, exchange, routing_key)
        self._bindings[key] = {"args": kwargs}
        self._ensure_connection()
        self._pub_channel.queue_bind(
            queue=queue, exchange=exchange, routing_key=routing_key, arguments=kwargs
        )

    def setup_dead_letter(
        self,
        base_queue: str,
        dlx_exchange: Optional[str] = None,
        dlq_name: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Create a dead-letter exchange and queue for handling failed messages.

        Sets up a complete dead letter infrastructure including:
        - A fanout exchange for dead letters
        - A queue to collect dead letters
        - Binding between the exchange and queue

        Args:
            base_queue: Name of the base queue (used to generate DLX/DLQ names if not provided).
            dlx_exchange: Name for the dead letter exchange. If None, uses "{base_queue}.dlx".
            dlq_name: Name for the dead letter queue. If None, uses "{base_queue}.dlq".

        Returns:
            Tuple of (dead_letter_exchange_name, dead_letter_queue_name).

        Example:
            >>> dlx, dlq = client.setup_dead_letter("tasks")
            >>> # Creates "tasks.dlx" exchange and "tasks.dlq" queue
            >>>
            >>> # Then configure your main queue to use the DLX:
            >>> client.declare_queue("tasks", arguments={"x-dead-letter-exchange": dlx})
        """
        dlx = dlx_exchange or f"{base_queue}.dlx"
        dlq = dlq_name or f"{base_queue}.dlq"

        # Create fanout exchange for dead letters (broadcasts to all bound queues)
        self.declare_exchange(dlx, "fanout")

        # Create queue to collect dead letters
        self.declare_queue(dlq)

        # Bind DLQ to DLX with empty routing key (fanout ignores routing keys)
        self.bind_queue(dlq, dlx, routing_key="")

        return dlx, dlq

    # ---------------------- Publish helpers --------------------------
    def publish(
        self,
        body: Union[bytes, bytearray, Any],
        opts: PublishOptions,
        retries: int = 3,
        retry_backoff: float = 0.5,
    ) -> bool:
        """Publish a message with publisher confirms and automatic retry on failures.

        This method handles both binary and JSON payloads with automatic serialization.
        It uses publisher confirms to ensure reliable delivery and implements retry
        logic with exponential backoff for transient network errors.

        Args:
            body: Message payload. Can be bytes/bytearray for binary data, or any
                JSON-serializable object (dict, list, str, int, etc.) for JSON messages.
            opts: Publishing options including exchange, routing key, and message properties.
            retries: Maximum number of retry attempts for transient failures. Default is 3.
            retry_backoff: Base delay in seconds between retries (multiplied by attempt number). Default is 0.5.

        Returns:
            True if the message was confirmed by the broker, False if NACKed.

        Raises:
            pika.exceptions.UnroutableError: If mandatory=True and message cannot be routed to any queue.
            ConnectionError: If connection fails after all retry attempts.
            json.JSONEncodeError: If body is not JSON-serializable when not bytes.

        Example:
            >>> # Publish JSON message
            >>> opts = PublishOptions(exchange="events", routing_key="user.created")
            >>> success = client.publish({"user_id": 123, "email": "user@example.com"}, opts)
            >>>
            >>> # Publish binary message
            >>> binary_data = b"\\x00\\x01\\x02\\x03"
            >>> opts = PublishOptions(exchange="files", routing_key="upload", content_type="application/octet-stream")
            >>> success = client.publish(binary_data, opts)
        """
        props = pika.BasicProperties()
        headers = opts.headers or {}

        # Handle binary vs JSON message payloads
        if isinstance(body, (bytes, bytearray)):
            payload = bytes(body)
            # Set content type for binary data if not explicitly provided
            if not opts.content_type:
                props.content_type = "application/octet-stream"
        else:
            # Serialize JSON and set appropriate content headers
            payload = json.dumps(body).encode("utf-8")
            props.content_type = opts.content_type or "application/json"
            props.content_encoding = opts.content_encoding or "utf-8"

        # Configure message persistence (survives broker restarts)
        if opts.persistent:
            # 2 = persistent, 1 = transient
            props.delivery_mode = 2

        # Set message expiration (time-to-live)
        if opts.expiration_ms is not None:
            props.expiration = str(int(opts.expiration_ms))

        # Add custom headers
        if headers:
            props.headers = headers

        # Set correlation ID for request/response patterns
        if opts.correlation_id:
            props.correlation_id = opts.correlation_id

        # Set reply-to queue for RPC patterns
        if opts.reply_to:
            props.reply_to = opts.reply_to

        # Retry loop with exponential backoff
        attempt = 0
        while True:
            attempt += 1
            try:
                self._ensure_connection()
                confirmed = self._pub_channel.basic_publish(
                    exchange=opts.exchange,
                    routing_key=opts.routing_key,
                    body=payload,
                    properties=props,
                    mandatory=opts.mandatory,
                )

                # In confirm mode, basic_publish returns True if the message reached the broker
                if not confirmed:
                    self.logger.warning(
                        "Publish returned NACK for %s:%s",
                        opts.exchange,
                        opts.routing_key,
                    )
                return bool(confirmed)

            except pika.exceptions.UnroutableError:
                # Mandatory routing failed - message couldn't be routed to any queue
                self.logger.error(
                    "Message was unroutable (mandatory). Exchange=%s routing_key=%s",
                    opts.exchange,
                    opts.routing_key,
                )
                raise

            except Exception as e:  # pragma: no cover - network
                if attempt > retries:
                    self.logger.exception(
                        "Publish failed after %d attempts: %s", attempt - 1, e
                    )
                    raise

                # Calculate exponential backoff delay
                sleep = retry_backoff * attempt
                self.logger.warning(
                    "Publish error (%s). Retrying in %.1fs (attempt %d/%d)",
                    e,
                    sleep,
                    attempt,
                    retries,
                )
                time.sleep(sleep)
                self._reconnect()

    # ---------------------- Consume helpers --------------------------
    def consume(
        self,
        opts: ConsumeOptions,
        handler: Callable[
            [
                BlockingChannel,
                pika.spec.Basic.Deliver,
                pika.spec.BasicProperties,
                bytes,
            ],
            None,
        ],
    ) -> Callable[[], None]:
        """Start consuming messages from a queue with the provided handler.

        This method sets up a consumer that processes messages from the specified queue
        using the provided handler function. It implements Quality of Service (QoS)
        controls via prefetch and handles message acknowledgments automatically.

        The consumer runs in the current thread with a blocking event loop. If the
        handler raises an exception, the message will be NACKed (negative acknowledgment)
        and not requeued, which typically sends it to a dead letter queue if configured.

        Args:
            opts: Consumer configuration options including queue name, prefetch count,
                acknowledgment mode, and optional consumer tag.
            handler: Callback function to process received messages. Should accept
                    (channel, method, properties, body) parameters where:
                    - channel: The channel object for manual ack/nack if needed
                    - method: Delivery information including routing key and delivery tag
                    - properties: Message properties including headers and content type
                    - body: Raw message payload as bytes

        Returns:
            A stop function that can be called to cancel the consumer and stop consuming.
            This function is also called automatically during cleanup.

        Raises:
            AssertionError: If the consuming channel is not available.
            ConnectionError: If connection to RabbitMQ fails.

        Example:
            >>> def process_message(channel, method, properties, body):
            ...     try:
            ...         # Decode JSON message
            ...         if properties.content_type == "application/json":
            ...             data = json.loads(body.decode('utf-8'))
            ...             print(f"Processing: {data}")
            ...         else:
            ...             print(f"Binary data: {len(body)} bytes")
            ...     except Exception as e:
            ...         print(f"Error processing message: {e}")
            ...         raise  # Re-raise to trigger NACK
            >>>
            >>> opts = ConsumeOptions(queue="tasks", prefetch=10, auto_ack=False)
            >>> stop_func = client.consume(opts, process_message)
            >>> # Consumer runs until interrupted or stop_func() is called

        Note:
            This method blocks the current thread. Use threading or multiprocessing
            if you need concurrent processing or non-blocking behavior.
        """
        self._ensure_connection()
        ch = self._sub_channel
        assert ch is not None, "Consuming channel is not available"

        # Configure QoS (Quality of Service) - limits unacknowledged messages
        ch.basic_qos(prefetch_count=max(1, int(opts.prefetch)))

        # Generate unique consumer tag if not provided
        consumer_tag = opts.consumer_tag or f"c.{uuid.uuid4().hex[:8]}"

        def _on_message(
            channel: BlockingChannel,
            method: pika.spec.Basic.Deliver,
            properties: pika.spec.BasicProperties,
            body: bytes,
        ) -> None:
            """Internal callback wrapper for message handling with error management."""
            try:
                # Call user-provided handler
                handler(channel, method, properties, body)
            except Exception:
                # Log the exception with full traceback
                self.logger.exception("Handler raised; message will be NACKed")
                if not opts.auto_ack:
                    # NACK the message without requeue (sends to DLQ if configured)
                    channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                return

            # If handler succeeded and auto_ack is disabled, manually ACK the message
            if not opts.auto_ack:
                channel.basic_ack(delivery_tag=method.delivery_tag)

        # Register the consumer with RabbitMQ
        ch.basic_consume(
            queue=opts.queue,
            on_message_callback=_on_message,
            auto_ack=opts.auto_ack,
            consumer_tag=consumer_tag,
        )
        self.logger.info(
            "Consuming from %s (tag=%s, prefetch=%d)",
            opts.queue,
            consumer_tag,
            opts.prefetch,
        )

        def _stop() -> None:
            """Stop function to cancel the consumer gracefully."""
            try:
                if ch.is_open:
                    ch.basic_cancel(consumer_tag)
            except Exception:
                pass  # Suppress exceptions during cleanup

        # Start the blocking consume loop
        try:
            ch.start_consuming()
        except KeyboardInterrupt:  # pragma: no cover
            self.logger.info("Interrupted; stopping consumer")
        finally:
            # Ensure cleanup happens even if an exception occurs
            _stop()
        return _stop

    # ---------------------- RPC helper -------------------------------
    def rpc_call(self, queue: str, message: Any, timeout: float = 10.0) -> Any:
        """Send an RPC (Remote Procedure Call) request and wait for the response.

        This method implements a synchronous request-response pattern using RabbitMQ's
        direct reply-to feature. It sends a message to the specified queue and waits
        for a response, using correlation IDs to match requests with responses.

        The RPC consumer is automatically set up on first use and uses the special
        'amq.rabbitmq.reply-to' queue for receiving responses.

        Args:
            queue: Name of the queue where the RPC server is listening.
            message: Request payload. Can be any JSON-serializable object.
            timeout: Maximum time to wait for a response in seconds. Default is 10.0.

        Returns:
            The response data. If the response content-type is "application/json",
            it will be automatically deserialized. Otherwise, returns raw bytes.

        Raises:
            TimeoutError: If no response is received within the timeout period.
            ConnectionError: If connection to RabbitMQ fails.
            json.JSONEncodeError: If the message is not JSON-serializable.

        Example:
            >>> # Simple RPC call
            >>> response = client.rpc_call("math.add", {"a": 5, "b": 3})
            >>> print(response)  # {"result": 8}
            >>>
            >>> # RPC call with custom timeout
            >>> try:
            ...     response = client.rpc_call("slow.operation", {"data": "..."}, timeout=30.0)
            ... except TimeoutError:
            ...     print("RPC call timed out")

        Note:
            This method blocks until a response is received or timeout occurs.
            For non-blocking RPC, consider using a separate thread or async framework.
        """
        # Generate unique correlation ID to match request with response
        correlation_id = uuid.uuid4().hex
        self._ensure_connection()

        # Set up the RPC reply consumer if not already started
        self._ensure_rpc_consumer()

        # Configure publishing options for RPC request
        opts = PublishOptions(
            exchange="",  # Use default exchange (direct to queue)
            routing_key=queue,
            persistent=False,  # RPC messages don't need persistence
            correlation_id=correlation_id,
            reply_to="amq.rabbitmq.reply-to",  # RabbitMQ's direct reply-to feature
        )

        # Send the RPC request
        self.publish(message, opts)

        # Wait for response with timeout
        deadline = time.time() + timeout
        while time.time() < deadline:
            # Check if response has arrived
            if correlation_id in self._rpc_responses:
                return self._rpc_responses.pop(correlation_id)

            # Process incoming data events to drive the RPC consumer
            try:
                self._connection.process_data_events(time_limit=0.1)  # type: ignore
            except Exception:
                # Suppress connection errors during data processing
                pass

        # Timeout reached without receiving response
        raise TimeoutError(f"RPC timeout after {timeout:.1f}s")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_connection(self) -> None:
        """Ensure that connection and channels are open, reconnect if needed.

        Checks if the main connection and both publishing/consuming channels are
        open and available. If any are closed or None, triggers a reconnection.

        This method is called before any operation that requires a connection
        to ensure the client is ready to perform RabbitMQ operations.
        """
        # Check if connection and both channels are open and available
        if (
            self._connection
            and self._connection.is_open
            and self._pub_channel
            and self._pub_channel.is_open
            and self._sub_channel
            and self._sub_channel.is_open
        ):
            return

        # One or more components are not available, trigger reconnection
        self._reconnect()

    def _reconnect(self) -> None:
        """Reconnect to RabbitMQ server after connection failure.

        Closes the existing connection if still open and establishes a new
        connection using the connect() method. This method respects the
        closing flag to prevent reconnection during shutdown.
        """
        # Don't reconnect if we're in the process of closing
        if self._closing:
            return

        self.logger.info("Reconnecting to RabbitMQ...")

        # Close existing connection if still open
        try:
            if self._connection and self._connection.is_open:
                self._connection.close()
        except Exception:
            pass  # Suppress exceptions during cleanup

        # Establish new connection (will retry with exponential backoff)
        self.connect()

    def _redeclare_topology(self) -> None:
        """Re-declare all registered topology after reconnection.

        Iterates through the stored exchange, queue, and binding configurations
        and re-declares them on the server. This ensures that the topology
        is consistent after reconnection.

        The declarations are idempotent, so they won't fail if the topology
        already exists with the same parameters.
        """
        # Can't declare topology without a publishing channel
        if not self._pub_channel:
            return

        # Re-declare all exchanges
        for exchange_name, config in self._exchanges.items():
            self._pub_channel.exchange_declare(
                exchange=exchange_name,
                exchange_type=config["type"],
                durable=config["durable"],
                auto_delete=config["auto_delete"],
                arguments=config["args"],
            )

        # Re-declare all queues
        for queue_name, config in self._queues.items():
            self._pub_channel.queue_declare(
                queue=queue_name,
                durable=config["durable"],
                exclusive=config["exclusive"],
                auto_delete=config["auto_delete"],
                arguments=config["args"],
            )

        # Re-create all bindings
        for (queue_name, exchange_name, routing_key), config in self._bindings.items():
            self._pub_channel.queue_bind(
                queue=queue_name,
                exchange=exchange_name,
                routing_key=routing_key,
                arguments=config["args"],
            )

    def _ensure_rpc_consumer(self) -> None:
        """Set up the RPC reply consumer if not already started.

        Creates a consumer on the special 'amq.rabbitmq.reply-to' queue to
        receive RPC responses. This consumer automatically deserializes JSON
        responses and stores them in the _rpc_responses dict using correlation
        IDs for matching with requests.

        This method is called automatically when the first RPC call is made.
        """
        # RPC consumer already running, nothing to do
        if self._rpc_consumer_started:
            return

        self._ensure_connection()
        assert self._sub_channel is not None, "Consuming channel not available for RPC"

        def _on_rpc_reply(
            ch: BlockingChannel,
            method: pika.spec.Basic.Deliver,
            properties: pika.spec.BasicProperties,
            body: bytes,
        ) -> None:
            """Handle RPC reply messages from the direct reply-to queue."""
            try:
                # Default to raw bytes
                data: Union[bytes, Any] = body

                # Attempt JSON deserialization if content-type indicates JSON
                if properties.content_type == "application/json":
                    encoding = properties.content_encoding or "utf-8"
                    data = json.loads(body.decode(encoding))
            except Exception:
                # Fall back to raw bytes if JSON parsing fails
                data = body

            # Store response using correlation ID for request matching
            if properties.correlation_id:
                self._rpc_responses[properties.correlation_id] = data

        # Set up consumer on the direct reply-to queue with auto-ack
        self._sub_channel.basic_consume(
            queue="amq.rabbitmq.reply-to",
            on_message_callback=_on_rpc_reply,
            auto_ack=True,
        )
        self._rpc_consumer_started = True


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


def sanitize_url(url: str) -> str:
    """Sanitize AMQP URL for safe logging by hiding password credentials.

    Takes an AMQP connection URL and replaces the password portion with asterisks
    to prevent sensitive credentials from appearing in log files.

    Args:
        url: The AMQP URL to sanitize. Expected format: amqp://user:pass@host:port/vhost

    Returns:
        Sanitized URL with password replaced by "***". If parsing fails,
        returns the original URL unchanged.

    Example:
        >>> sanitize_url("amqp://user:secret@localhost:5672/vhost")
        'amqp://user:***@localhost:5672/vhost'
        >>> sanitize_url("amqp://localhost:5672/")  # No credentials
        'amqp://localhost:5672/'
    """
    try:
        # Parse URL format: amqp://user:pass@host:port/vhost
        prefix, rest = url.split("//", 1)

        # Check if URL contains credentials (has @ symbol)
        if "@" not in rest:
            return url  # No credentials to sanitize

        credentials, host = rest.split("@", 1)

        # Check if credentials contain password (has : symbol)
        if ":" in credentials:
            user, _ = credentials.split(":", 1)
            return f"{prefix}//{user}:***@{host}"
        else:
            # Only username, no password
            return url

    except Exception:
        # Return original URL if parsing fails
        return url


# --------------------------------------------------------------------------------------
# Example usage
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    client = RabbitMQClient()
    client.connect()

    # Topology: exchange, queue, binding, and DLQ
    client.declare_exchange("demo.events", ex_type="topic")
    client.declare_queue(
        "demo.worker",
        durable=True,
        arguments={
            "x-dead-letter-exchange": "demo.worker.dlx",
        },
    )
    dlx, dlq = client.setup_dead_letter("demo.worker")
    client.bind_queue("demo.worker", "demo.events", routing_key="*.task")

    # Publish JSON
    client.publish(
        {"name": "mayami", "n": 1},
        PublishOptions(exchange="demo.events", routing_key="foo.task"),
    )

    # RPC demo (requires a server listening on queue 'rpc.echo')
    try:
        rsp = client.rpc_call("rpc.echo", {"ping": True})
        print("RPC response:", rsp)
    except Exception as e:
        print("RPC demo skipped:", e)

    # Consumer demo (Ctrl+C to stop)
    def handle_message(ch, method, props, body: bytes):
        try:
            if props.content_type == "application/json":
                data = json.loads(body.decode(props.content_encoding or "utf-8"))
            else:
                data = body
            print("Received:", data)
        except Exception:
            print("Malformed message:", body)
            raise  # will be NACKed and dead-lettered

    try:
        client.consume(
            ConsumeOptions(queue="demo.worker", prefetch=100, auto_ack=False),
            handle_message,
        )
    except KeyboardInterrupt:
        pass
    finally:
        client.close()
