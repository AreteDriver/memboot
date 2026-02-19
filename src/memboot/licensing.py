"""Licensing and tier management for memboot."""

from __future__ import annotations

import hashlib
import logging
import os
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_KEY_SALT = "memboot-v1"

_LICENSE_LOCATIONS: list[str] = [
    ".memboot-license",
    "~/.config/memboot/license",
    "~/.memboot/license",
    "~/.memboot-license",
]

_ENV_LICENSE_KEY = "MEMBOOT_LICENSE"


class Tier(StrEnum):
    """Product tier levels."""

    FREE = "free"
    PRO = "pro"


class TierConfig(BaseModel):
    """Configuration for a product tier."""

    name: str
    price_label: str
    features: list[str]


TIER_DEFINITIONS: dict[Tier, TierConfig] = {
    Tier.FREE: TierConfig(
        name="Free",
        price_label="Free forever",
        features=[
            "init",
            "query",
            "remember",
            "context",
            "reset",
            "status",
            "ingest_files",
        ],
    ),
    Tier.PRO: TierConfig(
        name="Pro",
        price_label="$8/mo",
        features=[
            "init",
            "query",
            "remember",
            "context",
            "reset",
            "status",
            "ingest_files",
            "serve",
            "ingest_pdf",
            "ingest_web",
            "multi_project",
            "watch",
        ],
    ),
}

PRO_FEATURES: frozenset[str] = frozenset(
    {
        "serve",
        "ingest_pdf",
        "ingest_web",
        "multi_project",
        "watch",
    }
)


class LicenseInfo(BaseModel):
    """Validated license information."""

    tier: Tier = Tier.FREE
    license_key: str | None = None
    valid: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


def _validate_key_format(key: str) -> bool:
    """Check if a license key matches ``MMBT-XXXX-XXXX-XXXX``."""
    key = key.strip()
    if not key.startswith("MMBT-"):
        return False
    parts = key.split("-")
    if len(parts) != 4:
        return False
    for part in parts[1:]:
        if len(part) != 4 or not part.isalnum() or not part.isupper():
            return False
    return True


def _compute_check_segment(body: str) -> str:
    """Derive the check segment from the key body."""
    digest = hashlib.sha256(f"{_KEY_SALT}:{body}".encode()).hexdigest()
    return digest[:4].upper()


def _validate_key_checksum(key: str) -> bool:
    """Verify the last segment matches the HMAC-derived value."""
    parts = key.strip().split("-")
    if len(parts) != 4:
        return False
    body = f"{parts[1]}-{parts[2]}"
    expected = _compute_check_segment(body)
    return parts[3] == expected


def _find_license_key() -> str | None:
    """Search for a license key in environment and filesystem."""
    env_key = os.environ.get(_ENV_LICENSE_KEY)
    if env_key and env_key.strip():
        return env_key.strip()

    for location in _LICENSE_LOCATIONS:
        path = Path(location).expanduser()
        if path.is_file():
            try:
                content = path.read_text().strip()
                if content:
                    return content
            except OSError:
                continue

    return None


def get_license_info() -> LicenseInfo:
    """Detect and validate the current license."""
    key = _find_license_key()

    if key is None:
        return LicenseInfo(tier=Tier.FREE)

    if not _validate_key_format(key):
        logger.warning("Invalid license key format")
        return LicenseInfo(tier=Tier.FREE, license_key=key, valid=False)

    if not _validate_key_checksum(key):
        logger.warning("License key checksum mismatch")
        return LicenseInfo(tier=Tier.FREE, license_key=key, valid=False)

    return LicenseInfo(tier=Tier.PRO, license_key=key, valid=True)


def has_feature(feature: str) -> bool:
    """Check if the current license grants access to a feature."""
    info = get_license_info()
    tier_config = TIER_DEFINITIONS[info.tier]
    return feature in tier_config.features


def is_pro() -> bool:
    """Check if the current license is Pro tier."""
    return get_license_info().tier == Tier.PRO


def get_upgrade_message(feature: str) -> str:
    """Return a user-facing upgrade prompt for a gated feature."""
    pro_config = TIER_DEFINITIONS[Tier.PRO]
    return (
        f"'{feature}' requires memboot Pro ({pro_config.price_label}).\n"
        f"Set your key via: export {_ENV_LICENSE_KEY}=MMBT-XXXX-XXXX-XXXX"
    )
