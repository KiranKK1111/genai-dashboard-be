"""
JWT authentication helpers.

This module provides simple functions for encoding and decoding JSON
Web Tokens (JWT). Tokens are used to identify users without persisting
session state on the server. In a production environment you would
likely integrate with a proper identity provider and implement
refresh tokens; here we use a straightforward bearer token scheme.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import jwt
from passlib.context import CryptContext

from .config import settings

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a password using pbkdf2_sha256."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[int] = None) -> str:
    """Create a signed JWT containing the provided payload.

    Args:
        data: Claims to include in the token payload.
        expires_delta: Optional expiration in minutes. If omitted the
            ``JWT_EXPIRE_MIN`` setting is used.

    Returns:
        A signed JWT as a string.
    """
    to_encode = data.copy()
    expire_minutes = expires_delta or settings.jwt_expire_min
    expire = datetime.utcnow() + timedelta(minutes=expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_alg)
    return encoded_jwt


def verify_access_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT.

    Args:
        token: The bearer token to decode.

    Returns:
        The decoded payload if verification succeeds.

    Raises:
        jwt.PyJWTError: if the token is invalid or expired.
    """
    payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_alg])
    return payload


def create_access_token(data: Dict[str, Any], expires_delta: Optional[int] = None) -> str:
    """Create a signed JWT containing the provided payload.

    Args:
        data: Claims to include in the token payload.
        expires_delta: Optional expiration in minutes. If omitted the
            ``JWT_EXPIRE_MIN`` setting is used.

    Returns:
        A signed JWT as a string.
    """
    to_encode = data.copy()
    expire_minutes = expires_delta or settings.jwt_expire_min
    expire = datetime.utcnow() + timedelta(minutes=expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_alg)
    return encoded_jwt


def verify_access_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT.

    Args:
        token: The bearer token to decode.

    Returns:
        The decoded payload if verification succeeds.

    Raises:
        jwt.PyJWTError: if the token is invalid or expired.
    """
    payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_alg])
    return payload