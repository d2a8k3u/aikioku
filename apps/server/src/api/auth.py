"""Authentication API endpoints."""
from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from src.auth import (
    Token,
    create_access_token,
    register_user,
    require_auth,
    verify_password,
    _get_user,
    UserInDB,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=Token)
async def auth_register(form_data: OAuth2PasswordRequestForm = Depends()) -> Token:
    """Register a new user and return a JWT access token."""
    if _get_user(form_data.username) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already registered",
        )
    user = register_user(
        username=form_data.username,
        email=form_data.username,  # MVP: reuse username as email
        password=form_data.password,
    )
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(days=7),
    )
    return Token(access_token=access_token, token_type="bearer")


@router.post("/login", response_model=Token)
async def auth_login(form_data: OAuth2PasswordRequestForm = Depends()) -> Token:
    """Authenticate an existing user and return a JWT access token."""
    user = _get_user(form_data.username)
    if user is None or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(days=7),
    )
    return Token(access_token=access_token, token_type="bearer")


@router.get("/me")
async def auth_me(user: UserInDB = Depends(require_auth)) -> dict:
    """Return the currently authenticated user's profile."""
    return {"username": user.username, "email": user.email}
