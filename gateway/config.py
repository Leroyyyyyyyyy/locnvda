"""Gateway-only environment settings; independent of GPU/model deployment."""
from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math
import os
from types import MappingProxyType
from urllib.parse import urlsplit

from gateway.auth import validate_api_keys, valid_token


def load_api_keys(raw):
    def unique_object(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("GATEWAY_API_KEYS contains duplicate caller IDs")
            result[name] = value
        return result

    if raw is None:
        raise ValueError("GATEWAY_API_KEYS is required; anonymous API access is disabled")
    try:
        keys = json.loads(raw, object_pairs_hook=unique_object)
    except json.JSONDecodeError:
        raise ValueError("GATEWAY_API_KEYS must be a JSON object mapping caller IDs to API keys") from None
    validate_api_keys(keys)
    return keys


@dataclass(frozen=True)
class Settings:
    api_keys: Mapping[str, str] = field(repr=False)
    upstream_api_key: str | None = field(default=None, repr=False)
    upstream_url: str = "http://127.0.0.1:8000"
    connect_timeout: float = 5.0
    read_timeout: float = 120.0
    write_timeout: float = 30.0
    pool_timeout: float = 5.0
    max_connections: int = 100
    max_keepalive_connections: int = 20

    def __post_init__(self):
        validate_api_keys(self.api_keys)
        # Freeze a copy, not the caller's mutable registry. Never show keys in repr.
        object.__setattr__(self, "api_keys", MappingProxyType(dict(self.api_keys)))
        if self.upstream_api_key is not None and not valid_token(self.upstream_api_key):
            raise ValueError("GATEWAY_UPSTREAM_API_KEY must be a non-empty ASCII bearer token")
        url = urlsplit(self.upstream_url)
        # One trusted upstream, not an arbitrary URL supplied by the caller.
        if (url.scheme not in {"http", "https"} or not url.hostname
                or url.username is not None or url.password is not None
                or url.query or url.fragment):
            raise ValueError("GATEWAY_UPSTREAM_URL must be an HTTP(S) URL without credentials/query/fragment")
        _ = url.port  # Also reject invalid port numbers.
        for name in ("connect_timeout", "read_timeout", "write_timeout", "pool_timeout"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_connections <= 0:
            raise ValueError("max_connections must be positive")
        if not 0 <= self.max_keepalive_connections <= self.max_connections:
            raise ValueError("max_keepalive_connections must be between 0 and max_connections")

    @classmethod
    def from_env(cls):
        defaults = cls(api_keys=load_api_keys(os.environ.get("GATEWAY_API_KEYS")))
        values = {
            "api_keys": defaults.api_keys,
            "upstream_api_key": os.environ.get("GATEWAY_UPSTREAM_API_KEY"),
            "upstream_url": os.environ.get("GATEWAY_UPSTREAM_URL", defaults.upstream_url),
        }
        for name in ("connect_timeout", "read_timeout", "write_timeout", "pool_timeout"):
            values[name] = float(os.environ.get(f"GATEWAY_{name.upper()}", getattr(defaults, name)))
        for name in ("max_connections", "max_keepalive_connections"):
            values[name] = int(os.environ.get(f"GATEWAY_{name.upper()}", getattr(defaults, name)))
        return cls(**values)
