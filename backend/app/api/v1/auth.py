from datetime import timedelta
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from backend.app.core.config import settings
from backend.app.core.limiter import limiter
from backend.app.core.security import (
    Token,
    UserCreate,
    UserPublic,
    UserRole,
    create_access_token,
    get_user_by_id,
    get_user_by_username,
    register_user,
    require_authenticated_user,
    require_roles,
    verify_password,
    UserInDB,
    _users_db,
)

router = APIRouter(prefix="/auth", tags=["Authentication & Access Control"])


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login", response_model=Token, summary="Authenticate user and issue JWT token")
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def login(request: Request, credentials: LoginRequest):
    user = get_user_by_username(credentials.username)
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive account")

    expires_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(
        data={"sub": user.username, "role": user.role.value, "uid": user.id},
        expires_delta=expires_delta,
    )
    return Token(
        access_token=token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserPublic(
            id=user.id,
            username=user.username,
            email=user.email,
            role=user.role,
            full_name=user.full_name,
            is_active=user.is_active,
            created_at=user.created_at,
        ),
    )


@router.post("/token", response_model=Dict[str, str], summary="OAuth2 compatible token endpoint for Swagger docs")
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def login_oauth2(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user_by_username(form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(
        data={"sub": user.username, "role": user.role.value, "uid": user.id},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": token, "token_type": "bearer"}


@router.post("/register", response_model=UserPublic, status_code=status.HTTP_201_CREATED, summary="Register user account")
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def register(request: Request, user_in: UserCreate):
    # Enforce safe default role: public self-registration defaults to VIEWER
    if user_in.role in (UserRole.ADMIN, UserRole.OPERATOR):
        user_in.role = UserRole.VIEWER

    user_db = register_user(user_in)
    return UserPublic(
        id=user_db.id,
        username=user_db.username,
        email=user_db.email,
        role=user_db.role,
        full_name=user_db.full_name,
        is_active=user_db.is_active,
        created_at=user_db.created_at,
    )


@router.get("/me", response_model=UserPublic, summary="Get current authenticated user profile")
async def get_me(current_user: UserInDB = Depends(require_authenticated_user)):
    return UserPublic(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
    )


@router.get("/users/{user_id}", response_model=UserPublic, summary="Get user profile by ID with strict IDOR prevention")
async def get_user_profile(
    user_id: str,
    current_user: UserInDB = Depends(require_authenticated_user),
):
    # Strict IDOR prevention: only allow users to inspect their own profile, unless Admin
    if current_user.role != UserRole.ADMIN and current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You cannot view other user profiles",
        )

    target_user = get_user_by_id(user_id)
    if not target_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return UserPublic(
        id=target_user.id,
        username=target_user.username,
        email=target_user.email,
        role=target_user.role,
        full_name=target_user.full_name,
        is_active=target_user.is_active,
        created_at=target_user.created_at,
    )


@router.get("/users", response_model=List[UserPublic], summary="List all registered users (Admin only)")
async def list_users(admin: UserInDB = Depends(require_roles([UserRole.ADMIN]))):
    return [
        UserPublic(
            id=u.id,
            username=u.username,
            email=u.email,
            role=u.role,
            full_name=u.full_name,
            is_active=u.is_active,
            created_at=u.created_at,
        )
        for u in _users_db.values()
    ]
