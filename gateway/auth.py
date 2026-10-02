"""Static caller credentials for stage 4b; no raw keys in request state/logs."""
from collections.abc import Mapping
import hashlib
import hmac
import re

# RFC 6750 bearer token syntax; bounded ASCII also avoids header injection.
TOKEN = re.compile(r"[A-Za-z0-9._~+/-]+=*", re.ASCII)
CALLER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", re.ASCII)
BEARER = re.compile(r"(?i:Bearer) +([A-Za-z0-9._~+/-]+=*)", re.ASCII)


def valid_token(value):
    return isinstance(value, str) and 0 < len(value) <= 512 and TOKEN.fullmatch(value) is not None


def validate_api_keys(api_keys):
    if not isinstance(api_keys, Mapping) or not api_keys:
        raise ValueError("GATEWAY_API_KEYS must be a non-empty object mapping caller IDs to unique API keys")
    seen = set()
    for caller_id, key in api_keys.items():
        if (not isinstance(caller_id, str) or CALLER_ID.fullmatch(caller_id) is None
                or not valid_token(key) or key in seen):
            # Never include the offending credential in configuration errors.
            raise ValueError("GATEWAY_API_KEYS contains an invalid caller ID, invalid key or duplicate key")
        seen.add(key)


class Authenticator:
    def __init__(self, api_keys):
        validate_api_keys(api_keys)
        self._callers = tuple(
            (caller_id, hashlib.sha256(key.encode("ascii")).digest())
            for caller_id, key in api_keys.items()
        )

    def authenticate(self, authorization_headers):
        # Reject ambiguous duplicate credentials instead of choosing first/last.
        if len(authorization_headers) != 1:
            return None
        header = authorization_headers[0]
        if len(header) > 1024:
            return None
        match = BEARER.fullmatch(header)
        if match is None or not valid_token(match[1]):
            return None
        digest = hashlib.sha256(match[1].encode("ascii")).digest()
        caller = None
        for caller_id, expected in self._callers:
            if hmac.compare_digest(digest, expected):
                caller = caller_id
        return caller
