import html
import re
from typing import Optional


# Match HTML tags and script/iframe injection vectors
TAG_PATTERN = re.compile(r"<[^>]+>", re.IGNORECASE)
SCRIPT_PATTERN = re.compile(r"(script|iframe|object|embed|style|svg)", re.IGNORECASE)
EVENT_PATTERN = re.compile(r"\bon[a-z]+\s*=", re.IGNORECASE)
JS_PROTOCOL_PATTERN = re.compile(r"javascript\s*:", re.IGNORECASE)
SAFE_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.\:\s]+$")


def sanitize_string(val: Optional[str]) -> Optional[str]:
    """Sanitizes user input by stripping HTML tags and neutralization of injection vectors."""
    if val is None:
        return None
    if not isinstance(val, str):
        val = str(val)

    # 1. Strip raw HTML tags
    cleaned = TAG_PATTERN.sub("", val)

    # 2. Neutralize event attributes and protocols
    cleaned = EVENT_PATTERN.sub("", cleaned)
    cleaned = JS_PROTOCOL_PATTERN.sub("", cleaned)

    # 3. HTML-escape remaining content for safe browser rendering
    cleaned = html.escape(cleaned.strip())
    return cleaned


def validate_safe_identifier(val: str, field_name: str = "identifier") -> str:
    """Validates that a string contains only alphanumeric, dash, underscore, colon, and period."""
    if not val or not isinstance(val, str):
        raise ValueError(f"{field_name} must be a non-empty string")
    clean_val = val.strip()
    if not SAFE_IDENTIFIER_PATTERN.match(clean_val):
        raise ValueError(f"{field_name} contains disallowed characters")
    return clean_val
