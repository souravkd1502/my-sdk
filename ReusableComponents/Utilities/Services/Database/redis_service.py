"""
redis_service.py
----------------
This module contains the implementation of the RedisService class, which is responsible for handling all the Redis operations.

Classes:
--------

Features:
---------
- RedisService: This class is responsible for handling all the Redis operations.

TODO:
-----

FIXME:
------

Author:
-------
Sourav Das

Date:
-----
09-01-2025
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Import dependencies
import os
import redis
import logging
from dotenv import load_dotenv

from typing import Any, Optional, Dict, List, Tuple

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Load Environment variables
load_dotenv(override=True)


class RedisManager:
    """
    A class to manage the Redis connection and operations.
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = None,
        ssl: bool = False,
        password: str = None,
        from_connection_url: bool = False,
        connection_url: str = None,
        use_pool: bool = False,
    ) -> None:
        """
        Initialize a RedisManager instance for managing Redis connections.

        This constructor sets up the RedisManager class with the provided configuration parameters.
        It allows for connection setup using individual host details or a Redis connection URL.

        Args:
        -----
        host (str, optional):
            The hostname or IP address of the Redis server.
            Required if `from_connection_url` is False and `connection_url` is not provided.
        port (int, optional):
            The port number for the Redis server. Defaults to 6379.
            Required if `from_connection_url` is False and `connection_url` is not provided.
        db (int, optional):
            The Redis database index to connect to. Defaults to 0.
        ssl (bool, optional):
            Whether to use SSL for the Redis connection. Defaults to False.
        password (str, optional):
            The password for authenticating with the Redis server. Defaults to None.
        from_connection_url (bool, optional):
            Whether to initialize the connection using a connection URL. Defaults to False.
        connection_url (str, optional):
            The Redis connection URL, used if `from_connection_url` is True.
        use_pool (bool, optional):
            Whether to use a connection pool for Redis connections. Defaults to False.

        Returns:
        --------
        None

        Raises:
        -------
        ValueError: If `connection_url` is invalid or improperly formatted.
        ValueError: If `from_connection_url` is False but only one of `host` or `connection_url` is provided.
        ValueError: If neither `host`, `port`, and `db` nor a valid `connection_url` is provided.

        Examples:
        ---------
        >>> # Initialize using host and port
        >>> RedisManager(host="localhost", port=6379, db=0)

        >>> # Initialize using a connection URL
        >>> RedisManager(from_connection_url=True, connection_url="redis://localhost:6379/0")
        """

        # Initialize the RedisManager attributes
        self.host = host or os.getenv("REDIS_HOST")
        self.port = port or os.getenv("REDIS_PORT")
        self.db = db or os.getenv("REDIS_DB")
        self.ssl = ssl or os.getenv("REDIS_SSL")
        self.password = password or os.getenv("REDIS_PASSWORD")
        self.from_connection_url = from_connection_url or os.getenv(
            "REDIS_FROM_CONNECTION_URL"
        )
        self.connection_url = connection_url or os.getenv("REDIS_CONNECTION_URL")
        self.use_pool = use_pool or os.getenv("REDIS_USE_POOL")

        # Check if all parameters (host, port, db) are provided and non-empty
        if not self.from_connection_url and not all(
            [
                self.host,
                self.port,
                self.db,
            ]
        ):
            _logger.error(
                "All parameters (host, port, db) must be provided and non-empty."
            )
            raise ValueError(
                "All parameters (host, port, db) must be provided and non-empty."
            )

        # Check if only one of the parameters (host, connection_url) is provided
        if all([self.connection_url, self.host]):
            _logger.error(
                "Only one of the parameters (host, connection_url) must be provided."
            )
            raise ValueError(
                "Only one of the parameters (host, connection_url) must be provided."
            )

        # Check if the connection_url parameter is a valid Redis connection URL
        if self.from_connection_url and (
            not self.connection_url
            or not (
                self.connection_url.startswith("redis://")
                or self.connection_url.startswith("rediss://")
            )
        ):
            _logger.error(
                "The connection_url parameter must be a valid Redis connection URL."
            )
            raise ValueError(
                "The connection_url parameter must be a valid Redis connection URL."
            )

    # Connection Management
    def _connect(self) -> None:
        """
        Establish a connection to the Redis server.
        Sets up the connection to the Redis server using the provided configuration parameters.
        Sets up the connection using the connection URL if `from_connection_url` is True.

        Supports connection pooling if `use_pool` is True.

        Returns:
        --------
        None

        Raises:
        -------
        redis.exceptions.ConnectionError: If there is an error connecting to the Redis server.

        Examples:
        ---------
        >>> # Establish a connection to the Redis server
        >>> redis_manager = RedisManager(host="localhost", port=6379, db=0)
        >>> redis_manager._connect()

        >>> # Establish a connection to the Redis server using a connection URL
        >>> redis_manager = RedisManager(from_connection_url=True, connection_url="redis://localhost:6379/0", use_pool=True)
        >>> redis_manager._connect()
        """
        try:
            if self.from_connection_url:
                if self.use_pool:
                    self.pool = redis.ConnectionPool.from_url(
                        url=self.connection_url,
                    )
                    self.connection = redis.StrictRedis(connection_pool=self.pool)
                else:
                    self.connection = redis.StrictRedis.from_url(
                        url=self.connection_url
                    )
                    self.pool = None
            else:
                if self.use_pool:
                    self.pool = redis.ConnectionPool(
                        host=self.host,
                        port=self.port,
                        db=self.db,
                        password=self.password,
                        ssl=self.ssl,
                    )
                    self.connection = redis.StrictRedis(connection_pool=self.pool)
                else:
                    self.connection = redis.StrictRedis(
                        host=self.host,
                        port=self.port,
                        db=self.db,
                        password=self.password,
                        ssl=self.ssl,
                    )
                    self.pool = None
        except redis.exceptions.ConnectionError as e:
            _logger.error(f"Error connecting to Redis: {e}")
            raise e
        

    def get_redis_connection_and_pool_info(self) -> Dict[str, Optional[Any]]:
        """
        Retrieves information about the Redis connection and connection pool. 
        Returns only pool info if the pool exists.

        Returns:
            Dict[str, Optional[Any]]: A dictionary containing information about the connection 
            and connection pool (if pool exists).
        """
        try:
            # Extract connection information
            connection_info = {
                "host": getattr(self.connection, "host", None),
                "port": getattr(self.connection, "port", None),
                "db": getattr(self.connection, "db", None),
                "connection_class": type(self.connection).__name__ if self.connection else None,
                "use_url_for_connection": self.from_connection_url,
            }

            # Initialize pool_info as None
            pool_info = None
            
            # Extract connection pool information if pool exists
            if self.pool is not None:
                pool_info = {
                    "connection_pool_size": getattr(self.pool, "_created_connections", None),
                    "in_use_connections": len(getattr(self.pool, "_in_use_connections", [])),
                    "available_connections": (
                        getattr(self.pool, "_created_connections", 0) -
                        len(getattr(self.pool, "_in_use_connections", []))
                    ),
                    "max_connections": getattr(self.pool, "max_connections", None),
                    "pool_class": type(self.pool).__name__,
                }

            # Combine and return the details, including pool info only if available
            return {
                "connection_info": connection_info,
                "pool_info": pool_info,  # pool_info will be None if pool does not exist
            }

        except Exception as e:
            # Handle exceptions
            raise RuntimeError(f"Failed to retrieve Redis connection and pool info: {str(e)}")

    def _disconnect(self) -> None:
        """
        Closes the connection to the Redis server.

        Returns:
        --------
        None

        Raises:
        -------
        None

        Examples:
        ---------
        >>> # Close the connection to the Redis server
        >>> redis_manager = RedisManager(host="localhost", port=6379, db=0)
        >>> redis_manager._connect()
        >>> redis_manager._disconnect()
        """
        self.connection.close()

    def _flush_pool(self) -> None:
        """
        Flushes the Redis connection pool.

        Returns:
        --------
        None

        Raises:
        -------
        None

        Examples:
        ---------
        >>> # Flush the Redis connection pool
        >>> redis_manager = RedisManager(host="localhost", port=6379, db=0)
        >>> redis_manager._connect()
        >>> redis_manager.flush_pool()
        """
        self.pool.close()

    # Utility Methods
    def _flush_all(self) -> None:
        """
        Flushes all data from the Redis server.

        Returns:
        --------
        None

        Raises:
        -------
        None

        Examples:
        ---------
        >>> # Flush all data from the Redis server
        >>> redis_manager = RedisManager(host="localhost", port=6379, db=0)
        >>> redis_manager._connect()
        >>> redis_manager._flush_all()
        """
        self.connection.flushall()

    def _flush_db(self) -> None:
        """
        Flushes the Redis database.

        Returns:
        --------
        None

        Raises:
        -------
        None

        Examples:
        ---------
        >>> # Flush the Redis database
        >>> redis_manager = RedisManager(host="localhost", port=6379, db=0)
        >>> redis_manager._connect()
        >>> redis_manager.flush_db()
        """
        self.connection.flushdb()

    def _get_info(self) -> dict:
        """
        Get the Redis server information.

        Returns:
        --------
        dict
            A dictionary containing Redis server information.

        Raises:
        -------
        None

        Examples:
        ---------
        >>> # Get the Redis server information
        >>> redis_manager = RedisManager(host="localhost", port=6379, db=0)
        >>> redis_manager._connect()
        >>> redis_manager._get_info()
        """
        return self.connection.info()

    def scan_keys(self, pattern="*"):
        """
        Scan keys in the Redis database.

        Args:
        -----
        pattern (str, optional):
            The pattern to match keys against. Defaults to "*".

        Returns:
        --------
        Iterable
            An iterable of keys that match the provided pattern.

        Raises:
        -------
        None

        Examples:
        ---------
        >>> # Scan keys in the Redis database
        >>> redis_manager = RedisManager(host="localhost", port=6379, db=0)
        >>> redis_manager._connect()
        >>> for key in redis_manager.scan_keys():
        ...     print(key)
        """
        return self.connection.scan_iter(match=pattern)

    # Key-Value Operations
    def get(self, key):
        """
        Retrieve a value from the Redis database.

        Args:
        -----
        key (str):
            The key to retrieve the value for.

        Returns:
        --------
        str
            The value associated with the provided key.

        Raises:
        -------
        None

        Examples:
        ---------
        >>> # Retrieve a value from the Redis database
        >>> redis_manager = RedisManager(host="localhost", port=6379, db=0)
        >>> redis_manager._connect()
        >>> redis_manager.get("my_key")
        """
        return self.connection.get(key)

    def set_key(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Store a value associated with a key, with an optional TTL.

        Args:
            key (str): The key to store the value under.
            value (Any): The value to store.
            ttl (Optional[int], optional): The TTL (time to live) for the key in seconds. Defaults to None.

        Returns:
            bool: True if the key was set successfully, False otherwise.

        Raises:
            RuntimeError: If there is an error setting the key.
        """
        try:
            if ttl:
                # Set the key with a TTL
                return self.connection.setex(key, ttl, value)
            else:
                # Set the key without a TTL
                return self.connection.set(key, value)
        except Exception as e:
            raise RuntimeError(f"Failed to set key {key}: {e}")

    def get_key(self, key: str) -> Optional[Any]:
        """
        Retrieve the value associated with a key.

        Args:
            key (str): The key to retrieve the value for.

        Returns:
            Optional[Any]: The value associated with the provided key, or None if the key is missing.

        Raises:
            RuntimeError: If there is an error retrieving the key.
        """
        try:
            return self.connection.get(key)
        except Exception as e:
            raise RuntimeError(f"Failed to get key {key}: {e}")

    def delete_key(self, key: str) -> int:
        """
        Delete a specific key.

        Args:
            key (str): The key to delete.

        Returns:
            int: The number of keys deleted.

        Raises:
            RuntimeError: If there is an error deleting the key.
        """
        try:
            # Delete the key
            return self.connection.delete(key)
        except Exception as e:
            raise RuntimeError(f"Failed to delete key {key}: {e}")

    def key_exists(self, key: str) -> bool:
        """
        Check if a key exists.

        Args:
            key (str): The key to check.

        Returns:
            bool: True if the key exists, False otherwise.

        Raises:
            RuntimeError: If there is an error checking if the key exists.
        """
        try:
            # Use the EXISTS command to check if a key exists
            return self.connection.exists(key) > 0
        except Exception as e:
            raise RuntimeError(f"Failed to check if key {key} exists: {e}")

    def expire_key(self, key: str, ttl: int) -> bool:
        """
        Set an expiration time for a key.

        Args:
            key (str): The key to set the expiration time for.
            ttl (int): The time to live for the key in seconds.

        Returns:
            bool: True if the expiration time was set successfully, False otherwise.

        Raises:
            RuntimeError: If there is an error setting the expiration time.
        """
        try:
            # Use the EXPIRE command to set an expiration time for a key
            return self.connection.expire(key, ttl)
        except Exception as e:
            raise RuntimeError(f"Failed to set expiration for key {key}: {e}")

    def get_ttl(self, key: str) -> int:
        """
        Get the remaining Time To Live (TTL) for a key.

        Args:
            key (str): The key to get the TTL for.

        Returns:
            int: The remaining TTL in seconds for the key.
                    Returns -1 if the key exists but has no associated expiration time.
                    Returns -2 if the key does not exist.

        Raises:
            RuntimeError: If there is an error retrieving the TTL.
        """
        try:
            # Use the TTL command to get the remaining TTL for a key
            return self.connection.ttl(key)
        except Exception as e:
            # Handle any exceptions that occur while retrieving the TTL
            raise RuntimeError(f"Failed to get TTL for key {key}: {e}")

    # Hash Operations
    def set_hash(self, name: str, field: str, value: Any) -> bool:
        """
        Set a field-value pair in a hash.

        This method sets a specified field in a hash stored in Redis with the provided value.

        Args:
            name (str): The name of the hash.
            field (str): The field within the hash to set.
            value (Any): The value to set for the specified field.

        Returns:
            bool: True if the field was added to the hash, False if it was updated.

        Raises:
            RuntimeError: If there is an error setting the field in the hash.
        """
        try:
            # Set the field-value pair in the hash
            return self.connection.hset(name, field, value)
        except Exception as e:
            # Raise a RuntimeError if setting the field fails
            raise RuntimeError(f"Failed to set hash field {field}: {e}")

    def get_hash(self, name: str, field: str) -> Optional[Any]:
        """
        Retrieve a value from a hash by field.

        Args:
            name (str): The name of the hash.
            field (str): The field to retrieve from the hash.

        Returns:
            Optional[Any]: The value associated with the specified field in the hash, or None if the field does not exist.

        Raises:
            RuntimeError: If there is an error retrieving the field from the hash.
        """
        try:
            # Use the HGET command to retrieve the value associated with the specified field in the hash
            value = self.connection.hget(name, field)
            if value is not None:
                # Return the value as a string
                return value.decode()
            else:
                # Return None if the field does not exist
                return None
        except Exception as e:
            # Raise a RuntimeError if retrieving the field fails
            raise RuntimeError(f"Failed to get hash field {field}: {e}")

    def delete_hash(self, name: str, field: str) -> int:
        """
        Remove a field from a hash.

        Args:
            name (str): The name of the hash.
            field (str): The field to remove from the hash.

        Returns:
            int: The number of fields removed from the hash.

        Raises:
            RuntimeError: If there is an error removing the field from the hash.
        """
        try:
            # Use the HDEL command to delete the specified field from the hash
            return self.connection.hdel(name, field)
        except Exception as e:
            # Raise a RuntimeError if deleting the field fails
            raise RuntimeError(f"Failed to delete hash field {field}: {e}")

    def get_all_hash(self, name: str) -> Dict[str, Any]:
        """
        Retrieve all field-value pairs in a hash.

        Args:
            name (str): The name of the hash.

        Returns:
            Dict[str, Any]: A dictionary containing all the field-value pairs in the hash.

        Raises:
            RuntimeError: If there is an error retrieving the fields from the hash.
        """
        try:
            # Use the HGETALL command to retrieve all the field-value pairs in the hash
            return {k.decode(): v for k, v in self.connection.hgetall(name).items()}
        except Exception as e:
            # Raise a RuntimeError if retrieving the fields fails
            raise RuntimeError(f"Failed to get all fields in hash {name}: {e}")

    # List Operations
    def push_list(self, name: str, value: Any, from_left: bool = True) -> int:
        """
        Add an element to a list in Redis.

        This method adds an element to either the beginning or the end of the specified list.
        
        Args:
            name (str): The name of the list.
            value (Any): The value to be added to the list.
            from_left (bool): If True, the value is added to the beginning of the list (LPUSH).
                                If False, the value is added to the end of the list (RPUSH).

        Returns:
            int: The length of the list after the push operation.

        Raises:
            RuntimeError: If there is an error pushing the value to the list.
        """
        try:
            # Use LPUSH to add the value to the beginning of the list
            if from_left:
                return self.connection.lpush(name, value)
            # Use RPUSH to add the value to the end of the list
            return self.connection.rpush(name, value)
        except Exception as e:
            # Raise a RuntimeError if the push operation fails
            raise RuntimeError(f"Failed to push value to list {name}: {e}")

    def pop_list(self, name: str, from_left: bool = True) -> Optional[Any]:
        """
        Remove and return an element from the beginning (LPOP) or end (RPOP) of a list.

        Args:
            name (str): The name of the list.
            from_left (bool): If True, the value is popped from the beginning of the list (LPOP).
                                If False, the value is popped from the end of the list (RPOP).

        Returns:
            Optional[Any]: The popped value if the list is not empty, otherwise None.

        Raises:
            RuntimeError: If there is an error popping the value from the list.
        """
        try:
            if from_left:
                return self.connection.lpop(name)
            return self.connection.rpop(name)
        except Exception as e:
            raise RuntimeError(f"Failed to pop value from list {name}: {e}")

    def get_list_range(self, name: str, start: int, end: int) -> List[Any]:
        """
        Retrieve elements from a specific range in a list.

        Args:
            name (str): The name of the list.
            start (int): The index of the first element to retrieve.
            end (int): The index of the last element to retrieve.

        Returns:
            List[Any]: A list of elements in the specified range of the list.

        Raises:
            RuntimeError: If there is an error retrieving the range from the list.
        """
        try:
            # Use the LRANGE command to retrieve the specified range of elements from the list
            return self.connection.lrange(name, start, end)
        except Exception as e:
            # Raise a RuntimeError if retrieving the range fails
            raise RuntimeError(f"Failed to get range from list {name}: {e}")

    # Set Operations
    def add_to_set(self, name: str, *values: Any) -> int:
        """
        Add one or more members to a set.

        Args:
            name (str): The name of the set.
            *values (Any): One or more values to add to the set.

        Returns:
            int: The number of elements added to the set.

        Raises:
            RuntimeError: If there is an error adding the values to the set.
        """
        try:
            return self.connection.sadd(name, *values)
        except Exception as e:
            raise RuntimeError(f"Failed to add values to set {name}: {e}")

    def remove_from_set(self, name: str, *values: Any) -> int:
        """
        Remove one or more members from a set.

        Args:
            name (str): The name of the set.
            *values (Any): One or more values to remove from the set.

        Returns:
            int: The number of elements removed from the set.

        Raises:
            RuntimeError: If there is an error removing the values from the set.
        """
        try:
            # Use the SREM command to remove the specified values from the set
            return self.connection.srem(name, *values)
        except Exception as e:
            # Raise a RuntimeError if removing the values fails
            raise RuntimeError(f"Failed to remove values from set {name}: {e}")

    def is_member(self, name: str, value: Any) -> bool:
        """
        Check if a value is a member of a set.

        Args:
            name (str): The name of the set.
            value (Any): The value to check for membership.

        Returns:
            bool: True if the value is a member of the set, False otherwise.

        Raises:
            RuntimeError: If there is an error checking the membership.
        """
        try:
            # Use the SISMEMBER command to check if the value is in the set
            return self.connection.sismember(name, value)
        except Exception as e:
            # Raise a RuntimeError if the membership check fails
            raise RuntimeError(f"Failed to check membership in set {name}: {e}")

    def get_all_set_members(self, name: str) -> List[Any]:
        """
        Retrieve all members of a set.

        Args:
            name (str): The name of the set to retrieve members from.

        Returns:
            List[Any]: A list of all members in the specified set.

        Raises:
            RuntimeError: If there is an error retrieving the members from the set.
        """
        try:
            # Use the SMEMBERS command to retrieve all members of the set
            return list(self.connection.smembers(name))
        except Exception as e:
            # Raise a RuntimeError if retrieving the members fails
            raise RuntimeError(f"Failed to get all members of set {name}: {e}")

    # Sorted Set Operations
    def add_to_sorted_set(self, name: str, score: float, member: Any) -> int:
        """
        Add a member to a sorted set with a score.

        Args:
            name (str): The name of the sorted set.
            score (float): The score to associate with the member.
            member (Any): The member to add to the sorted set.

        Returns:
            int: The number of elements added to the sorted set.

        Raises:
            RuntimeError: If there is an error adding the member to the sorted set.
        """
        try:
            # Use the ZADD command to add the member to the sorted set
            return self.connection.zadd(name, {member: score})
        except Exception as e:
            # Raise a RuntimeError if adding the member fails
            raise RuntimeError(f"Failed to add member to sorted set {name}: {e}")

    def remove_from_sorted_set(self, name: str, *members: Any) -> int:
        """
        Remove a member from a sorted set.

        Args:
            name (str): The name of the sorted set.
            *members (Any): One or more members to remove from the sorted set.

        Returns:
            int: The number of elements removed from the sorted set.

        Raises:
            RuntimeError: If there is an error removing the members from the sorted set.
        """
        try:
            # Use the ZREM command to remove the specified members from the sorted set
            return self.connection.zrem(name, *members)
        except Exception as e:
            # Raise a RuntimeError if removing the members fails
            raise RuntimeError(f"Failed to remove members from sorted set {name}: {e}")

    def get_sorted_set_range(
        self, name: str, start: int, end: int, with_scores: bool = False
    ) -> List[Tuple[Any, float]]:
        """
        Retrieve members in a specific rank range from a sorted set.

        This method uses the ZRANGE command to get members from the sorted set
        specified by 'name', within the range from 'start' to 'end'. If 'with_scores'
        is True, the scores of the members will also be returned.

        Args:
            name (str): The name of the sorted set.
            start (int): The starting rank index.
            end (int): The ending rank index.
            with_scores (bool, optional): Whether to include scores in the result. Defaults to False.

        Returns:
            List[Tuple[Any, float]]: A list of tuples containing the members and their scores
            if 'with_scores' is True. Otherwise, a list of members.

        Raises:
            RuntimeError: If there is an error retrieving the range from the sorted set.
        """
        try:
            # Retrieve the specified range of members from the sorted set
            return self.connection.zrange(name, start, end, withscores=with_scores)
        except Exception as e:
            # Raise a RuntimeError if the operation fails
            raise RuntimeError(f"Failed to get range from sorted set {name}: {e}")
