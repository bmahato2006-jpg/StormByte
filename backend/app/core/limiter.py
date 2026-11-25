from slowapi import Limiter
from slowapi.util import get_remote_address
from backend.app.core.config import settings

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.RATE_LIMIT_DEFAULT],
    headers_enabled=False,
    storage_uri="memory://",
)
