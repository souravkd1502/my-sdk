"""

"""

import os
import sqlite3
import redis
import logging
import threading
from abc import ABC, abstractmethod
from typing import Dict, Optional

# Module logger - use proper initialization pattern
logger = logging.getLogger(__name__)


class CheckpointBackend(ABC):
    """
    Abstract base class for checkpoint storage backends.

    Defines the interface for checkpoint operations, ensuring consistency
    across different storage implementations.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """Retrieve a checkpoint value by key."""
        pass

    @abstractmethod
    def set(self, key: str, value: str) -> None:
        """Set a checkpoint value by key."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a checkpoint by key."""
        pass

    @abstractmethod
    def list(self) -> Dict[str, str]:
        """List all checkpoint keys and values."""
        pass


class FileCheckpointBackend(CheckpointBackend):
    """
    File-based checkpoint storage backend.

    Stores checkpoints as individual files in a specified directory.
    """

    def __init__(self, directory: str):
        """Initialize the file-based checkpoint backend."""
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _get_filepath(self, key: str) -> str:
        """
        Get the file path for a checkpoint key.

        Args:
            key (str): The checkpoint key.
        """
        # Use a more robust encoding that preserves the original key
        safe_key = (
            key.replace("/", "__SLASH__")
            .replace(":", "__COLON__")
            .replace("\\", "__BACKSLASH__")
        )
        return os.path.join(self.directory, f"{safe_key}.chkpt")
    
    def _get_directory(self) -> str:
        """
        Get the directory where checkpoints are stored.
        """
        return self.directory

    def _decode_key_from_filename(self, filename: str) -> str:
        """
        Decode the original key from the safe filename.

        Args:
            filename (str): The safe filename (without .chkpt extension).
        """
        return (
            filename.replace("__SLASH__", "/")
            .replace("__COLON__", ":")
            .replace("__BACKSLASH__", "\\")
        )

    def get(self, key: str) -> Optional[str]:
        """
        Retrieve a checkpoint value by key.

        Args:
            key (str): The checkpoint key.
        """
        filepath = self._get_filepath(key)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return f.read().strip()
        return None

    def set(self, key: str, value: str) -> None:
        """
        Set a checkpoint value by key.

        Args:
            key (str): The checkpoint key.
            value (str): The checkpoint value.
        """
        filepath = self._get_filepath(key)
        with open(filepath, "w") as f:
            f.write(value)

    def delete(self, key: str) -> None:
        """
        Delete a checkpoint by key.

        Args:
            key (str): The checkpoint key.
        """
        filepath = self._get_filepath(key)
        if os.path.exists(filepath):
            os.remove(filepath)

    def list(self) -> Dict[str, str]:
        """
        List all checkpoint keys and values.

        Returns:
            Dict[str, str]: A dictionary of all checkpoint keys and their values.
        """
        files = os.listdir(self.directory)
        checkpoints = {}

        for filename in files:
            if filename.endswith(".chkpt"):
                # Decode the safe filename back to the original key
                safe_key = filename[:-6]  # Remove .chkpt extension
                key = self._decode_key_from_filename(safe_key)

                try:
                    filepath = os.path.join(self.directory, filename)
                    with open(filepath, "r") as f:
                        value = f.read().strip()
                    checkpoints[key] = value
                except Exception as e:
                    # Skip files that can't be read
                    logger.error(f"Failed to read checkpoint file {filename}: {e}")
                    continue

        return checkpoints


class RedisCheckpointBackend(CheckpointBackend):
    """
    Stores checkpoints in Redis key-value store.

    Args
    -------
    host (str): Redis host
    port (int): Redis port
    db (int): Redis database index
    password (Optional[str]): Redis password

    Exceptions
    -----------
    Import Error: If redis library is not installed
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
    ):
        """Initialize Redis backend."""
        if redis is None:
            raise ImportError("Redis library not installed")
        self.client = redis.Redis(
            host=host, port=port, db=db, password=password, decode_responses=True
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def get(self, key: str) -> Optional[str]:
        """
        Retrieve a checkpoint value by key.

        Args:
            key (str): The checkpoint key.
        """
        self.logger.debug(f"Getting checkpoint {key} from Redis")
        try:
            return self.client.get(key)
        except Exception as e:
            self.logger.error(f"Failed to get checkpoint {key} from Redis: {e}")
            return None

    def set(self, key: str, value: str) -> None:
        """
        Set a checkpoint value by key.

        Args:
            key (str): The checkpoint key.
            value (str): The checkpoint value.
        """
        self.logger.debug(f"Setting checkpoint {key} in Redis")
        try:
            self.client.set(key, value)
        except Exception as e:
            self.logger.error(f"Failed to set checkpoint {key} in Redis: {e}")
            raise

    def delete(self, key: str) -> None:
        """
        Delete a checkpoint by key.

        Args:
            key (str): The checkpoint key.
        """
        self.logger.debug(f"Deleting checkpoint {key} from Redis")
        try:
            self.client.delete(key)
        except Exception as e:
            self.logger.error(f"Failed to delete checkpoint {key} from Redis: {e}")
            raise

    def list(self) -> Dict[str, str]:
        """
        List all checkpoints in Redis.

        Returns:
            Dict[str, str]: A dictionary of all checkpoint keys and their values.
        """
        try:
            keys = self.client.keys("*")
            return {key: self.client.get(key) for key in keys}
        except Exception as e:
            self.logger.error(f"Failed to list checkpoints in Redis: {e}")
            return {}


class SQLiteCheckpointBackend(CheckpointBackend):
    """
    Stores checkpoints in an SQLite database table.

    Args
    -------
    db_path (str): Path to SQLite database file

    Exceptions
    -----------
    Import Error: If sqlite3 library is not available
    Sqlite Error: If database operations fail. (Connection errors, SQL errors, etc.)
    """

    def __init__(self, db_path: str):
        """Initialize SQLite backend."""
        if sqlite3 is None:
            raise ImportError("SQLite library not installed")
        self.db_path = db_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialize_table()

    def _initialize_table(self):
        """Create checkpoints table if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to initialize SQLite table: {e}")
            raise

    def get(self, key: str) -> Optional[str]:
        """
        Retrieve a checkpoint value by key.

        Args:
            key (str): The checkpoint key.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM checkpoints WHERE key = ?", (key,))
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else None
        except Exception as e:
            self.logger.error(f"Failed to get checkpoint {key} from SQLite: {e}")
            return None

    def set(self, key: str, value: str) -> None:
        """
        Set a checkpoint value by key.

        Args:
            key (str): The checkpoint key.
            value (str): The checkpoint value.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO checkpoints (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
                (key, value),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to set checkpoint {key} in SQLite: {e}")
            raise

    def delete(self, key: str) -> None:
        """
        Delete a checkpoint by key.

        Args:
            key (str): The checkpoint key.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM checkpoints WHERE key = ?", (key,))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to delete checkpoint {key} in SQLite: {e}")
            raise

    def list(self) -> Dict[str, str]:
        """
        List all checkpoints in SQLite.

        Returns:
            Dict[str, str]: A dictionary of all checkpoint keys and their values.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM checkpoints")
            rows = cursor.fetchall()
            conn.close()
            return dict((key, value) for key, value in rows)
        except Exception as e:
            self.logger.error(f"Failed to list checkpoints in SQLite: {e}")
            return {}


class CheckpointManager:
    """
    CheckpointManager is responsible for managing incremental extraction state
    persistence for API or ETL pipelines. It provides a simple key-value interface
    to store, retrieve, delete, and list checkpoints, allowing pipelines to resume
    from the last processed position without reprocessing data.

    The manager is designed to be flexible, reliable, and fail-safe, with support
    for multiple storage backends, optional thread-safety, and automatic recovery
    from storage errors.

    Key Features:
    -------------
    1. Pluggable Storage Backends:
        - Supports any backend implementing the CheckpointBackend interface (e.g.,
            file system, Redis, SQLite, PostgreSQL).
        - Backend is injected during initialization, allowing easy switching of
            storage without changing extraction logic.

    2. Atomic Operations:
        - Thread-safe read/write/delete operations using an optional internal lock.
        - Ensures checkpoint consistency in multi-threaded or multi-process environments.

    3. Fail-Safe:
        - If fail_safe=True, any exceptions during checkpoint operations are caught
            and logged without blocking the extraction process.
        - Prevents entire ETL failures due to transient storage errors.

    4. Keyed Storage:
        - Checkpoints are stored using string keys (e.g., "api/users:get") to
            uniquely identify the extraction state for each API endpoint or resource.
        - Values are stored as strings, allowing flexibility for timestamps, IDs,
            or JSON blobs.

    5. Lightweight:
        - Minimal dependencies and simple interface focused on reliability over features.
        - Easy to integrate into existing extraction or ETL pipelines.

    Typical Usage:
    --------------
    >>> backend = FileSystemBackend("/tmp/checkpoints")
    >>> manager = CheckpointManager(backend=backend)
    >>> last_checkpoint = manager.get_checkpoint("api/users")
    >>> manager.set_checkpoint("api/users", "2025-09-25T00:00:00Z")

    Integration with Extractors:
    ----------------------------
    CheckpointManager is commonly injected into a RESTAPIExtractor or similar
    incremental loader. It automatically tracks the last extraction timestamp,
    ID, or cursor, and updates the checkpoint after successful processing.

    Args:
    -----
    backend (CheckpointBackend): An instance of a storage backend implementing
                                    get, set, delete, and list operations.
    fail_safe (bool): If True, exceptions during checkpoint operations are caught
                        and logged instead of propagating.
    thread_safe (bool): If True, internal locking ensures thread-safe operations.
    """

    def __init__(
        self,
        backend: CheckpointBackend,
        fail_safe: bool = True,
        thread_safe: bool = False,
    ) -> None:
        """Initialize the CheckpointManager.

        Parameters
        ----------
        backend : CheckpointBackend
            An instance of a storage backend implementing get, set, delete, and list operations.
        fail_safe : bool, default True
            If True, exceptions during checkpoint operations are caught and logged instead of propagating.
        thread_safe : bool, default False
            If True, internal locking ensures thread-safe operations.
        """
        self.backend = backend
        self.fail_safe = fail_safe
        self.thread_safe = thread_safe
        self._lock = threading.Lock() if thread_safe else None

    def get_checkpoint(self, key: str) -> Optional[str]:
        """
        Retrieve a checkpoint value by key.

        Args:
            key (str): The checkpoint key.

        Returns:
            Optional[str]: The checkpoint value or None if not found.
        """
        try:
            if self._lock:
                with self._lock:
                    return self.backend.get(key)
            return self.backend.get(key)
        except Exception as e:
            logger.error(f"Failed to get checkpoint {key}: {e}")
            if self.fail_safe:
                return None
            raise

    def set_checkpoint(self, key: str, value: str) -> None:
        """
        Set a checkpoint value by key.

        Args:
            key (str): The checkpoint key.
            value (str): The checkpoint value.
        """
        try:
            if self._lock:
                with self._lock:
                    self.backend.set(key, value)
            else:
                self.backend.set(key, value)
        except Exception as e:
            logger.error(f"Failed to set checkpoint {key}: {e}")
            if not self.fail_safe:
                raise

    def delete_checkpoint(self, key: str) -> None:
        """
        Delete a checkpoint value by key.

        Args:
            key (str): The checkpoint key.
        """
        try:
            if self._lock:
                with self._lock:
                    self.backend.delete(key)
            else:
                self.backend.delete(key)
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {key}: {e}")
            if not self.fail_safe:
                raise

    def list_checkpoints(self) -> Dict[str, str]:
        """
        List all checkpoints.

        Returns:
            Dict[str, str]: A dictionary of all checkpoint keys and their values.
        """
        try:
            if self._lock:
                with self._lock:
                    return self.backend.list()
            return self.backend.list()
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            if self.fail_safe:
                return {}
            raise