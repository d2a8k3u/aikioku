"""JWT authentication helpers for Aikioku."""
from __future__ import annotations

import bcrypt
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from src.config import settings
from src import runtime_config

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


def get_secret_key() -> str:
    """Return the persisted JWT signing secret (survives restarts)."""
    return runtime_config.jwt_secret()


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class UserInDB(BaseModel):
    username: str
    email: str
    hashed_password: str


# --- persisted user store -------------------------------------------------------

_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    email TEXT NOT NULL,
    hashed_password TEXT NOT NULL,
    created TEXT NOT NULL
)
"""


def _ensure_users_table() -> None:
    Path(settings.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        conn.execute(_USERS_TABLE)


def _safe_password(password: str) -> bytes:
    """Return UTF-8 encoded password truncated to bcrypt's 72-byte limit."""
    return password.encode("utf-8")[:72]


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(_safe_password(plain_password), hashed_password.encode("utf-8"))


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(_safe_password(password), bcrypt.gensalt()).decode("utf-8")


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, get_secret_key(), algorithm=ALGORITHM)


def token_username(token: str) -> str | None:
    """Return the subject of a valid JWT, or None. Used by the auth middleware."""
    try:
        payload = jwt.decode(token, get_secret_key(), algorithms=[ALGORITHM])
    except JWTError:
        return None
    return payload.get("sub")


def _get_user(username: str) -> UserInDB | None:
    _ensure_users_table()
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        row = conn.execute(
            "SELECT username, email, hashed_password FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if row is None:
        return None
    return UserInDB(username=row[0], email=row[1], hashed_password=row[2])


def user_count() -> int:
    _ensure_users_table()
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def register_user(username: str, email: str, password: str) -> UserInDB:
    if _get_user(username) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already registered",
        )
    user = UserInDB(
        username=username,
        email=email,
        hashed_password=get_password_hash(password),
    )
    _ensure_users_table()
    with sqlite3.connect(settings.sqlite_db_path) as conn:
        conn.execute(
            "INSERT INTO users (username, email, hashed_password, created) "
            "VALUES (?, ?, ?, ?)",
            (user.username, user.email, user.hashed_password,
             datetime.now(timezone.utc).isoformat()),
        )
    return user


async def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)],
) -> UserInDB | None:
    """Dependency that returns the current user or None when no token is provided.

    Use `require_auth` (below) for endpoints that mandate authentication.
    """
    if token is None:
        return None
    try:
        payload = jwt.decode(token, get_secret_key(), algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        if username is None:
            return None
    except JWTError:
        return None
    return _get_user(username)


async def require_auth(
    user: Annotated[UserInDB | None, Depends(get_current_user)],
) -> UserInDB:
    """Dependency implementing the LOCAL-TRUST optional-auth model.

    Behaviour is controlled by the DB-backed ``auth_required`` flag (set by the
    setup wizard):

    - ``auth_required=False``: when a valid token is provided the real user is
      returned; otherwise an anonymous local user is returned so the app works
      without a login wall.
    - ``auth_required=True``: a valid token is mandatory. Missing or invalid
      credentials raise ``401``.
    """
    if user is not None:
        return user
    if runtime_config.auth_required():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return UserInDB(
        username="anonymous",
        email="anonymous@local",
        hashed_password="",
    )
