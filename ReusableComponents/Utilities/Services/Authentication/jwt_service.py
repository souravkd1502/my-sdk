"""
jwt_service.py
----------------
This module contains the implementation of the JWT service. This service is responsible for generating and validating JWT tokens.


TODO:
-----

FIXME:
------

Author:
-------
Sourav Das

Date:
-----
2025-01-08
"""

# Adding directories to system path to allow importing custom modules
import sys

sys.path.append("./")
sys.path.append("../")

# Import dependencies
import os
import jwt
import logging
from jwt import PyJWTError
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

from typing import Dict, Any, Optional

# Set up logging
_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - line: %(lineno)d",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Load Environment variables
load_dotenv(override=True)


class TokenManager:
    """
    This class is responsible for generating and validating JWT tokens.
    """

    def __init__(
        self,
        secret_key: str = None,
        issuer: str = None,
        audience: str = None,
        duration: int = None,
    ) -> None:
        """
        Constructor for TokenManager class.

        Args:
            secret_key (str): The secret key to be used for generating and validating JWT tokens.
            issuer (str): The issuer of the JWT token.
            audience (str): The audience of the JWT token.
            duration (int): The duration for which the JWT token is valid.

        Raises:
            ValueError: If the secret key is not provided.
        """
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY")
        if not self.secret_key:
            _logger.error("Secret key not found. Cannot generate JWT token.")
            raise ValueError("Secret key not found. Cannot generate JWT token.")

        self.issuer = issuer or os.getenv("JWT_ISSUER", "default_issuer")
        self.audience = audience or os.getenv("JWT_AUDIENCE", "default_audience")
        self.duration = duration or int(
            os.getenv("JWT_DURATION", 30)
        )  # Default duration is 30 seconds

    def _generate_token(self, payload: Dict[str, Any]) -> str:
        """
        Generate a JWT token with the provided payload.

        Args:
            payload (Dict[str, Any]): The payload to encode in the JWT token.

        Returns:
            str: The generated JWT token.

        Raises:
            ValueError: If payload is empty.
        """
        if not payload:
            raise ValueError("Payload cannot be empty.")

        # Add standard claims
        current_time = datetime.now(timezone.utc)
        payload.update(
            {
                "iss": self.issuer,
                "aud": self.audience,
                "iat": current_time,
                "exp": current_time + timedelta(seconds=self.duration),
            }
        )

        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        _logger.info("Token generated successfully.")
        return token

    def _validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate a JWT token.

        Args:
            token (str): The JWT token to validate.

        Returns:
            Optional[Dict[str, Any]]: The decoded payload if the token is valid, None otherwise.

        Raises:
            JWTError: If the token is invalid or expired.
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=["HS256"],
                audience=self.audience,
                issuer=self.issuer,
            )

            # Manually check expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(
                timezone.utc
            ):
                _logger.error("Token expired.")
                raise ValueError("Token expired.")

            _logger.info("Token validated successfully.")
            return payload
        except PyJWTError as e:
            _logger.error(f"Token validation failed: {str(e)}")
            raise

    def _decode_token(self, token: str) -> Dict[str, Any]:
        """
        Decode a JWT token without validation.

        Args:
            token (str): The JWT token to decode.

        Returns:
            Dict[str, Any]: The decoded payload.

        Raises:
            JWTError: If the token cannot be decoded.
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            _logger.info("Token decoded successfully.")
            return payload
        except PyJWTError as e:
            _logger.error(f"Token decoding failed: {str(e)}")
            raise

    @classmethod
    def generate_token(
        cls,
        secret_key: str,
        issuer: str,
        audience: str,
        duration: int,
        payload: Dict[str, Any],
    ) -> str:
        """
        Generate a JWT token with the provided payload.

        Args:
            secret_key (str): The secret key to be used for generating and validating JWT tokens.
            issuer (str): The issuer of the JWT token.
            audience (str): The audience of the JWT token.
            duration (int): The duration for which the JWT token is valid.
            payload (Dict[str, Any]): The payload to encode in the JWT token.

        Returns:
            str: The generated JWT token.

        Raises:
            ValueError: If payload is empty.
        """
        if not payload:
            raise ValueError("Payload cannot be empty.")

        # Add standard claims
        current_time = datetime.now(timezone.utc)
        payload.update(
            {
                "iss": issuer,
                "aud": audience,
                "iat": current_time,
                "exp": current_time + timedelta(seconds=duration),
            }
        )

        token = jwt.encode(payload, secret_key, algorithm="HS256")
        _logger.info("Token generated successfully.")
        return token

    @classmethod
    def validate_token(
        cls, secret_key: str, issuer: str, audience: str, token: str
    ) -> Optional[Dict[str, Any]]:
        """
        Validate a JWT token, including expiration.

        Args:
            secret_key (str): The secret key to be used for generating and validating JWT tokens.
            issuer (str): The issuer of the JWT token.
            audience (str): The audience of the JWT token.
            token (str): The JWT token to validate.

        Returns:
            Optional[Dict[str, Any]]: The decoded payload if the token is valid, None otherwise.

        Raises:
            ValueError: If the token is expired.
            JWTError: If the token is invalid.
        """
        try:
            payload = jwt.decode(
                token,
                secret_key,
                algorithms=["HS256"],
                audience=audience,
                issuer=issuer,
            )

            # Manually check expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(
                timezone.utc
            ):
                _logger.error("Token has expired.")
                raise ValueError("Token has expired.")

            _logger.info("Token validated successfully.")
            return payload

        except PyJWTError as e:
            _logger.error(f"Token validation failed: {str(e)}")
            raise

    @classmethod
    def decode_token(cls, secret_key: str, token: str) -> Dict[str, Any]:
        """
        Decode a JWT token without validation.

        Args:
            secret_key (str): The secret key to be used for generating and validating JWT tokens.
            token (str): The JWT token to decode.

        Returns:
            Dict[str, Any]: The decoded payload.

        Raises:
            JWTError: If the token cannot be decoded.
        """
        try:
            payload = jwt.decode(
                token,
                secret_key,
                algorithms=["HS256"],
            )
            _logger.info("Token decoded successfully.")
            return payload
        except PyJWTError as e:
            _logger.error(f"Token decoding failed: {str(e)}")
            raise