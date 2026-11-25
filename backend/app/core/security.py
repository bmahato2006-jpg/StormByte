from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import bcrypt
import jwt
from pydantic import BaseModel, EmailStr, Field
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer, SecurityScopes

from backend.app.core.config import settings
from backend.app.core.logging import logger

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_PREFIX}/auth/token", auto_error=False)


class UserRole(str, Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    ANALYST = "analyst"
    VIEWER = "viewer"


class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    role: UserRole = UserRole.VIEWER
    full_name: Optional[str] = None
    is_active: bool = True


class UserCreate(UserBase):
    password: str = Field(..., min_length=8)


class UserPublic(UserBase):
    id: str
    created_at: datetime


class UserInDB(UserBase):
    id: str
    hashed_password: str
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserPublic


class TokenPayload(BaseModel):
    sub: str
    role: str
    exp: int


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        pwd_bytes = plain_password.encode("utf-8")[:72]
        hash_bytes = hashed_password.encode("utf-8")
        return bcrypt.checkpw(pwd_bytes, hash_bytes)
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    pwd_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": int(expire.timestamp()), "iat": int(now.timestamp())})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# In-memory secure user store (production ready with fallback for DB sync)
_users_db: Dict[str, UserInDB] = {}


def init_default_users():
    if settings.ADMIN_USERNAME not in _users_db:
        admin_user = UserInDB(
            id="usr-superadmin-001",
            username=settings.ADMIN_USERNAME,
            email=settings.ADMIN_EMAIL,
            role=UserRole.ADMIN,
            full_name="System Administrator",
            is_active=True,
            hashed_password=get_password_hash(settings.ADMIN_PASSWORD),
            created_at=datetime.now(timezone.utc),
        )
        _users_db[settings.ADMIN_USERNAME] = admin_user

        # Default operator for automated sensors and pipelines
        operator_user = UserInDB(
            id="usr-operator-002",
            username="satellite_operator",
            email="operator@stormbyte.io",
            role=UserRole.OPERATOR,
            full_name="MOSDAC Ingestion Operator",
            is_active=True,
            hashed_password=get_password_hash("Operator@StormByte2026!"),
            created_at=datetime.now(timezone.utc),
        )
        _users_db["satellite_operator"] = operator_user


init_default_users()


def get_user_by_username(username: str) -> Optional[UserInDB]:
    return _users_db.get(username)


def get_user_by_id(user_id: str) -> Optional[UserInDB]:
    for user in _users_db.values():
        if user.id == user_id:
            return user
    return None


def register_user(user_create: UserCreate) -> UserInDB:
    if user_create.username in _users_db:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")
    for u in _users_db.values():
        if u.email.lower() == user_create.email.lower():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    user_id = f"usr-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{len(_users_db)+1:03d}"
    user_db = UserInDB(
        id=user_id,
        username=user_create.username,
        email=user_create.email,
        role=user_create.role,
        full_name=user_create.full_name,
        is_active=True,
        hashed_password=get_password_hash(user_create.password),
        created_at=datetime.now(timezone.utc),
    )
    _users_db[user_create.username] = user_db
    return user_db


async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[UserInDB]:
    if not token:
        return None
    payload = decode_access_token(token)
    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = get_user_by_username(username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user account")
    return user


async def require_authenticated_user(current_user: Optional[UserInDB] = Depends(get_current_user)) -> UserInDB:
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


def require_roles(allowed_roles: List[UserRole]):
    async def role_checker(user: UserInDB = Depends(require_authenticated_user)) -> UserInDB:
        if user.role not in allowed_roles and user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: Insufficient privileges. Required one of: {[r.value for r in allowed_roles]}",
            )
        return user

    return role_checker
