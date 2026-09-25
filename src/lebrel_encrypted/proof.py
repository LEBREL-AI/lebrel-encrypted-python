"""Proof of Edition, client side: the signed serving manifest and the receipt of every answer.

The same checks as https://lebrel.ai/verify, in your process. Both documents are JSON objects signed
with Ed25519 over their canonical form (keys sorted by code point, no whitespace, raw UTF-8, integers
only). A receipt names the manifest by the digest of its content without the validity window, so it
matches any republication of the same serving configuration.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

MANIFEST_VERSION = 1
RECEIPT_VERSION = 1
MAX_DOCUMENT_BYTES = 65_536
MANIFEST_REQUIRED = frozenset({
    "version", "edition", "weights", "quantization", "engine", "tokenizer_sha256",
    "chat_template_sha256", "runtime", "attestation", "issued_at", "expires_at", "signing_key_id",
})
RECEIPT_REQUIRED = frozenset({
    "version", "request_id", "manifest_sha256", "prompt_sha256", "response_sha256",
    "prompt_tokens", "completion_tokens", "issued_at", "instance_id", "signing_key_id",
})
_HEX64 = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class Check:
    """One verification step and its outcome; ``detail`` never contains conversation content."""
    id: str
    label: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class Signed:
    """A signed document exactly as the API serves it."""
    payload: Dict[str, Any]
    signature: str

    def to_json(self) -> str:
        return json.dumps({"payload": self.payload, "signature": self.signature}, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class Manifest:
    """The serving manifest: which weights, precision, engine and tokenizer are serving, signed by Lebrel."""
    document: Signed
    identity: str
    checks: List[Check]
    verified: bool

    @property
    def payload(self) -> Dict[str, Any]:
        return self.document.payload


@dataclass(frozen=True)
class Receipt:
    """The receipt of one answer: the request id, the digests of the request and the answer, the manifest it ran under."""
    document: Signed
    checks: List[Check]
    verified: bool
    manifest: Optional[Manifest]

    @property
    def payload(self) -> Dict[str, Any]:
        return self.document.payload

    @property
    def request_id(self) -> str:
        return str(self.document.payload.get("request_id", ""))


def _integers_only(value: Any) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        if abs(value) > 2 ** 53 - 1:
            raise ValueError("Canonical JSON allows safe integers only")
        return
    if isinstance(value, float):
        raise ValueError("Canonical JSON does not allow floating point numbers")
    if isinstance(value, list):
        for item in value:
            _integers_only(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Canonical JSON keys must be strings")
            _integers_only(item)
        return
    raise ValueError("Unsupported JSON value")


def canonical(payload: Dict[str, Any]) -> bytes:
    """Python ``json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)``: what is signed."""
    _integers_only(payload)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def key_id(public_key: bytes) -> str:
    """The id every manifest and receipt names: the SHA-256 of the raw public key."""
    return sha256_hex(public_key)


def parse_signed(data: bytes) -> Signed:
    """A signed document from its bytes; raises ``ValueError`` for anything but a well-formed one."""
    if not isinstance(data, (bytes, bytearray)) or len(data) > MAX_DOCUMENT_BYTES:
        raise ValueError("Signed document is missing or too large")
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeError, ValueError):
        raise ValueError("Signed document is not valid JSON") from None
    if not isinstance(raw, dict) or set(raw) != {"payload", "signature"}:
        raise ValueError("Signed document must carry a payload and a signature")
    payload, signature = raw["payload"], raw["signature"]
    if not isinstance(payload, dict) or not isinstance(signature, str):
        raise ValueError("Signed document has the wrong shape")
    try:
        if len(base64.b64decode(signature, validate=True)) != 64:
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError("Signature must be base64 of 64 bytes") from None
    _integers_only(payload)
    return Signed(payload=payload, signature=signature)


def verify_signature(document: Signed, public_key: bytes) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(base64.b64decode(document.signature, validate=True), canonical(document.payload))
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def manifest_identity(payload: Dict[str, Any]) -> str:
    """The digest of the manifest content without its validity window; what every receipt references."""
    return sha256_hex(canonical({k: v for k, v in payload.items() if k not in ("issued_at", "expires_at")}))


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64


def check_manifest(document: Signed, public_key: bytes, *, now: float, edition_id: Optional[str] = None) -> List[Check]:
    payload = document.payload
    missing = sorted(MANIFEST_REQUIRED - set(payload))
    expected_key = key_id(public_key)
    issued, expires = payload.get("issued_at"), payload.get("expires_at")
    window_ok = (isinstance(issued, int) and isinstance(expires, int) and not isinstance(issued, bool) and not isinstance(expires, bool)
                 and issued <= now + 30 and expires > now and expires > issued)
    checks = [
        Check("shape", "Manifest fields", not missing and payload.get("version") == MANIFEST_VERSION,
              "Missing: " + ", ".join(missing) if missing else "Version " + str(payload.get("version"))),
        Check("key", "Signing key", payload.get("signing_key_id") == expected_key,
              "Pinned key " + expected_key[:16] + "…" if payload.get("signing_key_id") == expected_key else "Signed by a key that is not the pinned one"),
        Check("signature", "Signature", verify_signature(document, public_key), "Ed25519 over the canonical manifest"),
        Check("window", "Validity", window_ok, "Valid now" if window_ok else "Not valid at this time"),
    ]
    if edition_id is not None:
        edition = payload.get("edition")
        served = edition.get("id") if isinstance(edition, dict) else None
        checks.append(Check("edition", "Edition", served == edition_id, "Serves " + str(served) if served == edition_id else "The manifest names another edition"))
    return checks


def check_receipt(document: Signed, public_key: bytes, *, request_id: Optional[str] = None, manifest_identity_hex: Optional[str] = None,
                  request_sha256: Optional[str] = None, response_sha256: Optional[str] = None) -> List[Check]:
    """Every check a receipt allows with what the caller knows: the pinned key and signature always; the request id, the
    digests of the request as sent and of the answer as received, and the manifest identity when given."""
    payload = document.payload
    missing = sorted(RECEIPT_REQUIRED - set(payload))
    expected_key = key_id(public_key)
    checks = [
        Check("shape", "Receipt fields", not missing and payload.get("version") == RECEIPT_VERSION,
              "Missing: " + ", ".join(missing) if missing else "Version " + str(payload.get("version"))),
        Check("key", "Signing key", payload.get("signing_key_id") == expected_key,
              "Pinned key " + expected_key[:16] + "…" if payload.get("signing_key_id") == expected_key else "Signed by a key that is not the pinned one"),
        Check("signature", "Signature", verify_signature(document, public_key), "Ed25519 over the canonical receipt"),
    ]
    if request_id is not None:
        ok = payload.get("request_id") == request_id
        checks.append(Check("request_id", "Request id", ok, "The receipt is for this request" if ok else "The receipt names another request"))
    if manifest_identity_hex is not None:
        ok = payload.get("manifest_sha256") == manifest_identity_hex
        checks.append(Check("manifest", "Serving manifest", ok, "Matches manifest " + manifest_identity_hex[:16] + "…" if ok else "The receipt references a different manifest"))
    if request_sha256 is not None:
        ok = _is_hex64(request_sha256) and payload.get("prompt_sha256") == request_sha256
        checks.append(Check("request", "Your request", ok, "SHA-256 of the request as sent matches" if ok else "The request digest does not match what was sent"))
    if response_sha256 is not None:
        ok = _is_hex64(response_sha256) and payload.get("response_sha256") == response_sha256
        checks.append(Check("response", "The answer", ok, "SHA-256 of the answer as received matches" if ok else "The answer digest does not match what was received"))
    return checks


def all_passed(checks: List[Check]) -> bool:
    return bool(checks) and all(check.ok for check in checks)
