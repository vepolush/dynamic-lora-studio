"""Simple auth: register, login, JWT."""

from __future__ import annotations

import os
import re
import secrets
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import bcrypt
from jose import JWTError, jwt

from db import UserModel, session_scope

SECRET_KEY = os.getenv("JWT_SECRET", secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30

security = HTTPBearer(auto_error=False)


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def _create_token(user_id: str, email: str) -> str:
    expire = datetime.utcnow()
    from datetime import timedelta
    expire = expire + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    payload = {"sub": user_id, "email": email, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


def _valid_email(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email.strip()))


def register(email: str, password: str) -> dict:
    """Register new user. Returns {user_id, email, token} or raises HTTPException."""
    email = email.strip().lower()
    if not _valid_email(email):
        raise HTTPException(status_code=400, detail="Invalid email format")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    with session_scope() as session:
        existing = session.query(UserModel).filter(UserModel.email == email).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")

        user_id = f"user_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        user = UserModel(
            id=user_id,
            email=email,
            password_hash=_hash_password(password),
            created_at=now,
        )
        session.add(user)

    token = _create_token(user_id, email)
    return {"user_id": user_id, "email": email, "token": token}


def login(email: str, password: str) -> dict:
    """Login user. Returns {user_id, email, token} or raises HTTPException."""
    email = email.strip().lower()

    with session_scope() as session:
        user = session.query(UserModel).filter(UserModel.email == email).first()
        if not user or not _verify_password(password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid email or password")

    token = _create_token(user.id, user.email)
    return {"user_id": user.id, "email": user.email, "token": token}


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
) -> dict | None:
    """Extract user from Bearer token. Returns {user_id, email} or None if not authenticated."""
    if not credentials or credentials.credentials is None:
        return None
    payload = _decode_token(credentials.credentials)
    if not payload:
        return None
    user_id = payload.get("sub")
    email = payload.get("email")
    if not user_id or not email:
        return None
    return {"user_id": user_id, "email": email}


def require_auth(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
) -> dict:
    """Require authenticated user. Raises 401 if not."""
    user = get_current_user(credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
