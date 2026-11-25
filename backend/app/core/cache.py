import asyncio
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
from threading import RLock
import time
from typing import Any, Callable, Dict, Optional, Tuple


class TTLCache:
    """Thread-safe in-memory cache with time-to-live expiration."""

    def __init__(self, default_ttl_seconds: int = 60, max_entries: int = 1000):
        self._default_ttl = default_ttl_seconds
        self._max_entries = max_entries
        self._store: Dict[str, Tuple[Any, float]] = {}
        self._lock = RLock()
        self._hits = 0
        self._misses = 0

    def _purge_expired(self, now: float) -> None:
        expired_keys = [k for k, (_, exp) in self._store.items() if exp <= now]
        for k in expired_keys:
            del self._store[k]

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            if key in self._store:
                val, exp = self._store[key]
                if exp > now:
                    self._hits += 1
                    return val
                else:
                    del self._store[key]
            self._misses += 1
            return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        now = time.time()
        with self._lock:
            if len(self._store) >= self._max_entries:
                self._purge_expired(now)
                # If still at capacity, evict the earliest expiring entry
                if len(self._store) >= self._max_entries:
                    earliest_key = min(self._store.keys(), key=lambda k: self._store[k][1])
                    del self._store[earliest_key]
            self._store[key] = (value, now + ttl)

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_ratio = (self._hits / total) if total > 0 else 0.0
            return {
                "entries": len(self._store),
                "max_entries": self._max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": round(hit_ratio, 3),
            }


# Global singleton cache instance for application-wide heavy lookups
memory_cache = TTLCache(default_ttl_seconds=120, max_entries=2000)


def cached_result(ttl_seconds: int = 60, prefix: str = "cache"):
    """Decorator to cache sync or async function results based on arguments."""

    def decorator(fn: Callable):
        if asyncio.iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapper(*args, **kwargs):
                raw_key = f"{prefix}:{fn.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
                key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
                cached_val = memory_cache.get(key_hash)
                if cached_val is not None:
                    return cached_val
                result = await fn(*args, **kwargs)
                memory_cache.set(key_hash, result, ttl_seconds=ttl_seconds)
                return result

            return async_wrapper
        else:

            @wraps(fn)
            def sync_wrapper(*args, **kwargs):
                raw_key = f"{prefix}:{fn.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
                key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
                cached_val = memory_cache.get(key_hash)
                if cached_val is not None:
                    return cached_val
                result = fn(*args, **kwargs)
                memory_cache.set(key_hash, result, ttl_seconds=ttl_seconds)
                return result

            return sync_wrapper

    return decorator
