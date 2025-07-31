"""
light_error_reporter.py

====================================
Summary:
--------
A lightweight, single-file, high-performance error reporting module for Python applications.
Designed for minimal interference with application logic and scalable handling of high error volumes.

This module captures exceptions using decorators, context managers, or global hooks,
and reports them asynchronously to multiple output sinks (e.g., log files, webhooks).
It is safe to use in production environments where performance and reliability are critical.

------------------------------------

Usage:
------
1. Initialize the reporter with one or more sinks:
    ```python
    config = ErrorReporterConfig(
        sinks=[
            FileSink("errors.log"),
            WebhookSink("https://hooks.example.com/error"),
        ]
    )
    error_reporter = ErrorReporter(config)
    install_global_hook(error_reporter)
    ```

2. Capture errors using decorators:
    ```python
    @error_reporter.capture
    def risky_function():
        ...
    ```

3. Or use context managers:
    ```python
    with error_reporter.capture_block(context={"job": "data_import"}):
        ...
    ```

4. Or capture manually:
    ```python
    try:
        ...
    except Exception as e:
        error_reporter.capture_exception(e, context={"module": "main"})
    ```

------------------------------------

Dependencies:
-------------
- Python Standard Library
    - `traceback`, `queue`, `threading`, `time`, `json`, `hashlib`, `smtplib`, `logging`
- Third-party (optional but recommended)
    - `requests` (for webhook sending)

To install:
    pip install requests

------------------------------------

Features:
---------
- Non-blocking exception reporting via background thread
- Supports multiple pluggable sinks: file, webhook, email, etc.
- Configurable queue size, deduplication, and rate limiting
- Safe fallback if sinks fail
- Global exception hook support
- No external dependencies for basic usage (file sink only)

------------------------------------

Author:
-------
Sourav Das (2025)
"""
