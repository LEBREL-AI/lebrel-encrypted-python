"""Signed configuration verification and strict application framing over EHBP.

Cryptography is supplied by the pinned, unmodified upstream EHBP modules. This
wrapper adds Lebrel identity, endpoint, size and application completion checks.
"""

from __future__ import annotations

import base64
import codecs
import hashlib
import ipaddress
import json
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
from urllib.parse import urlsplit

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from . import proof
from ._vendor.ehbp.errors import EHBPError
from ._vendor.ehbp.identity import EncryptedRequest, ServerIdentity

MODEL_ID = "lebrel/deepseek-v4-flash-uncensored"
PRODUCTION_SIGNING_KEY = "beFZtSwt6FnlhIYbX636n7w3/gpaASIkRnIMx52XJwk="
PRODUCTION_SIGNING_KEY_ID = "8f72beb9680a0f1911dce59d1fc103a0a89039998003715d35422d3020e5292d"
MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_EVENT_BYTES = 4 * 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024
RECEIPT_HEADER = "Proof-Of-Edition-Receipt"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_SAFE_CODE = re.compile(r"^[a-zA-Z0-9_.-]{1,64}$")


class LebrelError(Exception):
    """Base exception. Error messages intentionally omit conversation content."""


class EncryptionError(LebrelError):
    """Signature, freshness, cryptographic integrity or plaintext fallback failed."""


class TransportError(LebrelError):
    """Transport failed. No inference request is automatically retried."""


class StreamError(LebrelError):
    """Stream did not authenticate and complete normally."""


class APIError(LebrelError):
    def __init__(self, status_code: int, code: str = "api_error") -> None:
        self.status_code = status_code
        self.code = code if _SAFE_CODE.fullmatch(code) else "api_error"
        super().__init__(f"Lebrel request failed (HTTP {status_code}, {self.code})")


def _reject_constant(_: str) -> None:
    raise ValueError("Non-finite JSON number")


def _object(pairs: List[Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _json(data: Any) -> Any:
    return json.loads(data, parse_constant=_reject_constant, object_pairs_hook=_object)


def _base64(value: Any) -> bytes:
    if not isinstance(value, str):
        raise ValueError("Expected base64 string")
    return base64.b64decode(value, validate=True)


@dataclass(frozen=True)
class VerifiedConfig:
    identity: ServerIdentity
    key_id: str
    expires_at: int


def verify_metadata(envelope: bytes, signing_public_key: bytes, now: float) -> VerifiedConfig:
    """Verify signed bytes before trusting or parsing the runtime configuration."""
    try:
        if len(envelope) > MAX_METADATA_BYTES or len(signing_public_key) != 32:
            raise ValueError("Invalid configuration size")
        outer = _json(envelope)
        if not isinstance(outer, dict) or type(outer.get("version")) is not int or outer["version"] != 1:
            raise ValueError("Unsupported envelope")
        payload = _base64(outer.get("payload"))
        signature = _base64(outer.get("signature"))
        if len(payload) > MAX_METADATA_BYTES or len(signature) != 64:
            raise ValueError("Invalid signed payload")
        Ed25519PublicKey.from_public_bytes(signing_public_key).verify(signature, payload)
        fields = _json(payload)
        required = {"expiresAt", "hpkeConfig", "issuedAt", "keyId", "modelId", "serverInstanceId", "signingKeyId", "version"}
        if not isinstance(fields, dict) or set(fields) != required:
            raise ValueError("Invalid configuration fields")
        if type(fields["version"]) is not int or fields["version"] != 1 or fields["modelId"] != MODEL_ID:
            raise ValueError("Wrong model or version")
        issued, expires = fields["issuedAt"], fields["expiresAt"]
        if type(issued) is not int or type(expires) is not int or not 0 < expires - issued <= 600:
            raise ValueError("Invalid validity interval")
        if issued > now + 30 or expires <= now:
            raise ValueError("Expired or future configuration")
        key_id = fields["keyId"]
        if not isinstance(key_id, str) or not _HEX64.fullmatch(key_id):
            raise ValueError("Invalid key identifier")
        if fields["signingKeyId"] != hashlib.sha256(signing_public_key).hexdigest():
            raise ValueError("Wrong signing key identifier")
        if not isinstance(fields["serverInstanceId"], str) or not _HEX32.fullmatch(fields["serverInstanceId"]):
            raise ValueError("Invalid server instance identifier")
        config = _base64(fields["hpkeConfig"])
        if hashlib.sha256(config).hexdigest() != key_id:
            raise ValueError("Configuration digest mismatch")
        identity = ServerIdentity.unmarshal_public_config(config)
        # The upstream parser accepts extra config bytes; this production profile
        # has exactly one supported suite and no trailing/ambiguous data.
        if identity.marshal_public_config() != config:
            raise ValueError("Non-canonical HPKE configuration")
        return VerifiedConfig(identity, key_id, expires)
    except (ValueError, TypeError, KeyError, InvalidSignature, EHBPError):
        raise EncryptionError("Runtime encryption configuration could not be verified") from None


def _bounded_raw(response: httpx.Response, limit: int) -> bytes:
    chunks, size = [], 0
    for chunk in response.iter_raw():
        size += len(chunk)
        if size > limit:
            raise EncryptionError("Response exceeded its size limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _plaintext(response: httpx.Response, encrypted: EncryptedRequest, key_id: str) -> Iterator[bytes]:
    try:
        values = response.headers.get_list("Ehbp-Response-Nonce")
        if len(values) != 1 or not re.fullmatch(r"[0-9a-fA-F]{64}", values[0]):
            raise EncryptionError("Encrypted response nonce is missing or invalid")
        if response.headers.get_list("X-Lebrel-Encryption-Key-Id") != [key_id]:
            raise EncryptionError("Response encryption key does not match the verified runtime")
        if response.headers.get("content-encoding", "identity").lower() != "identity":
            raise EncryptionError("Unexpected encoding on encrypted response")
        decoder = encrypted.token.create_response_decryptor(bytes.fromhex(values[0]), max_chunk_length=MAX_RESPONSE_BYTES + 16)
        wire_total, plain_total = 0, 0
        for chunk in response.iter_raw():
            wire_total += len(chunk)
            if wire_total > MAX_RESPONSE_BYTES * 2:
                raise EncryptionError("Encrypted response exceeded its size limit")
            for plaintext in decoder.push(chunk):
                plain_total += len(plaintext)
                if plain_total > MAX_RESPONSE_BYTES:
                    raise EncryptionError("Response exceeded its size limit")
                yield plaintext
        decoder.finish()
    except EHBPError:
        raise EncryptionError("Encrypted response failed authentication or was truncated") from None
    except httpx.HTTPError:
        raise TransportError("Encrypted response transport failed; request was not retried") from None
    finally:
        response.close()


def _response_object(data: bytes) -> Dict[str, Any]:
    try:
        result = _json(data)
        if not isinstance(result, dict):
            raise ValueError("Expected object")
        return result
    except (ValueError, TypeError, UnicodeError):
        raise EncryptionError("Decrypted response is not valid JSON") from None


def _api_error(status: int, result: Dict[str, Any]) -> APIError:
    error = result.get("error")
    code = error.get("code", "api_error") if isinstance(error, dict) else "api_error"
    return APIError(status, code if isinstance(code, str) else "api_error")


class Completion(dict):
    """The completion as a dictionary (OpenAI's schema) plus what its receipt binds: the request id, the
    SHA-256 of the request as it was sent and of the answer as it was received. ``Lebrel.receipt`` checks
    a receipt against them."""

    request_id: str = ""
    request_sha256: str = ""
    response_sha256: Optional[str] = None
    receipt_header: Optional[str] = None


def _answer_digest(result: Dict[str, Any]) -> str:
    """What the runtime hashes as the answer: the content of the first choice, empty when there is none."""
    choices = result.get("choices")
    content = choices[0].get("message", {}).get("content") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    return hashlib.sha256((content if isinstance(content, str) else "").encode("utf-8")).hexdigest()


class CompletionStream(Iterator[Dict[str, Any]]):
    """Closeable iterator. Fully consume it to establish authenticated completion."""

    def __init__(self, response: httpx.Response, encrypted: EncryptedRequest, key_id: str, *, request_id: str = "", request_sha256: str = "") -> None:
        self._response = response
        self._source = _plaintext(response, encrypted, key_id)
        self._iterator = self._events()
        self.completed = False
        self.closed = False
        self.request_id = request_id
        self.request_sha256 = request_sha256
        self._text: List[str] = []

    @property
    def response_sha256(self) -> Optional[str]:
        """SHA-256 of the answer as the runtime hashes it (every content delta, in order); None until completed."""
        if not self.completed:
            return None
        return hashlib.sha256("".join(self._text).encode("utf-8")).hexdigest()

    def __iter__(self) -> CompletionStream:
        return self

    def __next__(self) -> Dict[str, Any]:
        try:
            return next(self._iterator)
        except BaseException:
            self.close()
            raise

    def _events(self) -> Iterator[Dict[str, Any]]:
        utf8 = codecs.getincrementaldecoder("utf-8")("strict")
        pending, lines, event_size = "", [], 0
        seen, finished = set(), set()
        done = False

        def event(block: List[str]) -> Optional[Dict[str, Any]]:
            nonlocal done
            data = []
            for line in block:
                if line.startswith("data:"):
                    value = line[5:]
                    data.append(value[1:] if value.startswith(" ") else value)
                if line in ("event: error", "event:error"):
                    raise StreamError("Inference returned an encrypted stream error")
            if not data:
                return None
            if done:
                raise StreamError("Stream contained data after its completion marker")
            value = "\n".join(data)
            if value == "[DONE]":
                if not seen or finished != seen:
                    raise StreamError("Stream completion marker arrived before all choices finished")
                done = True
                return None
            try:
                chunk = _json(value)
                if not isinstance(chunk, dict) or chunk.get("error") is not None:
                    raise ValueError("Invalid event")
                choices = chunk.get("choices", [])
                if not isinstance(choices, list):
                    raise ValueError("Invalid choices")
                for choice in choices:
                    if not isinstance(choice, dict):
                        raise ValueError("Invalid choice")
                    index = choice.get("index", 0)
                    if type(index) is not int or index < 0:
                        raise ValueError("Invalid choice index")
                    seen.add(index)
                    delta = choice.get("delta")
                    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                        self._text.append(delta["content"])
                    reason = choice.get("finish_reason")
                    if reason is not None:
                        if not isinstance(reason, str) or not reason:
                            raise ValueError("Invalid finish reason")
                        finished.add(index)
                return chunk
            except (ValueError, TypeError, UnicodeError):
                raise StreamError("Inference returned an invalid or failed stream event") from None

        try:
            for raw in self._source:
                pending += utf8.decode(raw)
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    line = line.removesuffix("\r")
                    event_size += len(line.encode("utf-8")) + 1
                    if event_size > MAX_EVENT_BYTES:
                        raise StreamError("Stream event exceeded its size limit")
                    if line:
                        lines.append(line)
                    else:
                        result = event(lines)
                        lines, event_size = [], 0
                        if result is not None:
                            yield result
                if len(pending.encode("utf-8")) + event_size > MAX_EVENT_BYTES:
                    raise StreamError("Stream event exceeded its size limit")
            pending += utf8.decode(b"", final=True)
            if pending.strip() or lines or not done:
                raise StreamError("Stream ended before authenticated completion")
            self.completed = True
        except UnicodeError:
            raise StreamError("Stream contains invalid UTF-8") from None

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            # A loopback client's disconnect can close us from a monitor thread
            # while the generator is blocked in HTTP. Close transport first;
            # an executing generator then unwinds through its own finally.
            if not self._response.is_closed:
                network = self._response.extensions.get("network_stream")
                try:
                    active_socket = network.get_extra_info("socket") if network is not None else None
                    if active_socket is not None:
                        # close() alone does not reliably interrupt another
                        # thread's recv; shutdown wakes it immediately. This is
                        # an active HTTP/1.1 response, never a pooled idle socket.
                        active_socket.shutdown(socket.SHUT_RDWR)
                except (AttributeError, OSError):
                    pass
            self._response.close()
            try:
                self._source.close()
            except ValueError:
                pass

    def __enter__(self) -> CompletionStream:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class _Completions:
    def __init__(self, client: Lebrel) -> None:
        self._client = client

    def create(self, *, messages: List[Dict[str, Any]], model: str = MODEL_ID, stream: bool = False, **parameters: Any) -> Any:
        return self._client.create(messages=messages, model=model, stream=stream, **parameters)


class _Chat:
    def __init__(self, client: Lebrel) -> None:
        self.completions = _Completions(client)


class Lebrel:
    """Synchronous encrypted API client; no automatic inference retries.

    ``signing_public_key`` is an explicit trust-anchor override for isolated
    testing or managed key rotation. Do not fetch it from the same metadata URL.
    """

    def __init__(self, api_key: Optional[str] = None, *, base_url: str = "https://api.lebrel.ai", timeout: float = 900.0, signing_public_key: str = PRODUCTION_SIGNING_KEY, _transport: Optional[httpx.BaseTransport] = None, _clock: Callable[[], float] = time.time) -> None:
        key = api_key if api_key is not None else os.getenv("LEBREL_API_KEY", "")
        if not isinstance(key, str) or not key.strip() or any(char in key for char in "\r\n"):
            raise ValueError("A Lebrel API key is required (argument or LEBREL_API_KEY)")
        parsed = urlsplit(base_url)
        local = False
        try:
            local = ipaddress.ip_address(parsed.hostname or "").is_loopback
        except ValueError:
            pass
        if parsed.scheme != "https" and not (parsed.scheme == "http" and local and signing_public_key != PRODUCTION_SIGNING_KEY):
            raise ValueError("HTTPS is required; isolated loopback tests require a separate signing key")
        if not parsed.hostname or parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError("base_url must be a clean API origin")
        try:
            public_key = _base64(signing_public_key)
            if len(public_key) != 32:
                raise ValueError("Invalid signing key")
        except (ValueError, TypeError):
            raise ValueError("signing_public_key must be base64 of a raw 32-byte Ed25519 public key") from None
        self._key, self._origin, self._signer, self._clock = key, base_url.rstrip("/"), public_key, _clock
        self._http = httpx.Client(timeout=httpx.Timeout(timeout, connect=15), follow_redirects=False, trust_env=False, transport=_transport)
        self.chat = _Chat(self)

    def _metadata(self) -> VerifiedConfig:
        try:
            with self._http.stream("GET", self._origin + "/.well-known/lebrel-encryption", headers={"Accept": "application/json", "Accept-Encoding": "identity"}) as response:
                if response.status_code != 200:
                    raise EncryptionError("Verified encryption configuration is unavailable")
                raw = _bounded_raw(response, MAX_METADATA_BYTES)
            return verify_metadata(raw, self._signer, self._clock())
        except httpx.HTTPError:
            raise TransportError("Could not retrieve the runtime encryption configuration") from None

    def create(self, *, messages: List[Dict[str, Any]], model: str = MODEL_ID, stream: bool = False, **parameters: Any) -> Any:
        if model != MODEL_ID:
            raise ValueError("This encrypted endpoint serves only DeepSeek V4 Flash Uncensored")
        if not isinstance(messages, list) or not messages or type(stream) is not bool:
            raise ValueError("messages must be a nonempty list and stream must be boolean")
        try:
            plaintext = json.dumps(dict(parameters, model=model, messages=messages, stream=stream), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        except (ValueError, TypeError, UnicodeError):
            raise ValueError("Completion parameters must be valid finite JSON") from None
        if len(plaintext) + 20 > MAX_REQUEST_BYTES:
            raise ValueError("Completion request exceeds the encrypted request size limit")
        verified = self._metadata()
        try:
            encrypted = verified.identity.encrypt_request_body(plaintext)
        except EHBPError:
            raise EncryptionError("Request encryption failed") from None
        if encrypted is None:
            raise EncryptionError("An encrypted request body is required")
        request_id = str(uuid.uuid4())
        request_sha256 = hashlib.sha256(plaintext).hexdigest()
        request = self._http.build_request("POST", self._origin + "/v1/chat/completions", headers={
            "Authorization": "Bearer " + self._key,
            "Content-Type": "application/octet-stream",
            "Accept": "text/event-stream" if stream else "application/json",
            "Accept-Encoding": "identity",
            "Ehbp-Encapsulated-Key": encrypted.encapsulated_key.hex(),
            "X-Lebrel-Encryption-Key-Id": verified.key_id,
            "X-Lebrel-Request-Id": request_id,
        }, content=encrypted.body)
        try:
            response = self._http.send(request, stream=True)
        except httpx.HTTPError:
            raise TransportError("Inference request transport failed; request was not retried") from None
        try:
            if not response.headers.get("Ehbp-Response-Nonce"):
                # Pre-decryption edge/config failures may be plaintext. Never
                # trust or print their bodies, and never retry their inference.
                if not 200 <= response.status_code < 300:
                    raise APIError(response.status_code, "unencrypted_transport_error")
                raise EncryptionError("Plaintext inference response was rejected")
            if 200 <= response.status_code < 300 and stream:
                if response.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "text/event-stream":
                    raise StreamError("Encrypted response was not an event stream")
                # Validate required headers now, even if the caller never iterates.
                if response.headers.get_list("X-Lebrel-Encryption-Key-Id") != [verified.key_id]:
                    raise EncryptionError("Response encryption key does not match the verified runtime")
                return CompletionStream(response, encrypted, verified.key_id, request_id=request_id, request_sha256=request_sha256)
            result = _response_object(b"".join(_plaintext(response, encrypted, verified.key_id)))
            if not 200 <= response.status_code < 300 or result.get("error") is not None:
                raise _api_error(response.status_code, result)
            choices = result.get("choices")
            if not isinstance(choices, list) or not choices or any(not isinstance(choice, dict) or not isinstance(choice.get("finish_reason"), str) or not choice["finish_reason"] for choice in choices):
                raise EncryptionError("Completion response is missing its final state")
            completion = Completion(result)
            completion.request_id = request_id
            completion.request_sha256 = request_sha256
            completion.response_sha256 = _answer_digest(result)
            header = response.headers.get(RECEIPT_HEADER)
            completion.receipt_header = header if isinstance(header, str) and header else None
            return completion
        except BaseException:
            response.close()
            raise

    def _signed_document(self, path: str, what: str) -> proof.Signed:
        """A public signed document (manifest or receipt), fetched without the API key and parsed strictly."""
        try:
            with self._http.stream("GET", self._origin + path, headers={"Accept": "application/json", "Accept-Encoding": "identity"}) as response:
                if response.status_code == 404:
                    raise APIError(404, what + "_not_found")
                if not 200 <= response.status_code < 300:
                    raise APIError(response.status_code, what + "_unavailable")
                raw = _bounded_raw(response, proof.MAX_DOCUMENT_BYTES)
        except httpx.HTTPError:
            raise TransportError("Could not retrieve the " + what) from None
        try:
            return proof.parse_signed(raw)
        except ValueError:
            raise EncryptionError("The " + what + " is not a valid signed document") from None

    def manifest(self) -> proof.Manifest:
        """The serving manifest, fetched and checked against the pinned key: what is serving right now."""
        document = self._signed_document("/.well-known/proof-of-edition", "manifest")
        checks = proof.check_manifest(document, self._signer, now=self._clock(), edition_id=MODEL_ID)
        return proof.Manifest(document=document, identity=proof.manifest_identity(document.payload), checks=checks, verified=proof.all_passed(checks))

    def receipt(self, subject: Union[str, Completion, CompletionStream], *, manifest: bool = True) -> proof.Receipt:
        """The signed receipt of an answer, fetched and checked: the pinned key and signature, the request id, the
        digests of the request as sent and of the answer as received (when ``subject`` is the completion or the
        fully consumed stream), and the serving manifest the receipt names (``manifest=False`` skips that fetch)."""
        request_id = subject if isinstance(subject, str) else getattr(subject, "request_id", "")
        if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
            raise ValueError("A receipt is looked up by its request id")
        document: Optional[proof.Signed] = None
        header = None if isinstance(subject, str) else getattr(subject, "receipt_header", None)
        if isinstance(header, str) and header:
            try:
                document = proof.parse_signed(base64.b64decode(header, validate=True))
            except (ValueError, TypeError):
                document = None  # a damaged header is not evidence; the API keeps the receipt
        if document is None:
            document = self._signed_document("/v1/receipts/" + request_id, "receipt")
        served = self.manifest() if manifest else None
        known: Dict[str, Optional[str]] = {}
        if not isinstance(subject, str):
            known["request_sha256"] = getattr(subject, "request_sha256", None) or None
            known["response_sha256"] = getattr(subject, "response_sha256", None) or None
        checks = proof.check_receipt(document, self._signer, request_id=request_id, manifest_identity_hex=served.identity if served else None, **known)
        if served is not None:
            checks = list(checks) + [check for check in served.checks if not check.ok]
        return proof.Receipt(document=document, checks=checks, verified=proof.all_passed(checks), manifest=served)

    def list_models(self) -> Dict[str, Any]:
        """Fetch authenticated public metadata, filtered to the exact Flash model."""
        try:
            with self._http.stream("GET", self._origin + "/v1/models", headers={"Authorization": "Bearer " + self._key, "Accept": "application/json", "Accept-Encoding": "identity"}) as response:
                if not 200 <= response.status_code < 300:
                    raise APIError(response.status_code, "model_metadata_unavailable")
                result = _response_object(_bounded_raw(response, 1024 * 1024))
            entries = result.get("data")
            if not isinstance(entries, list):
                raise EncryptionError("Model metadata is invalid")
            matching = [entry for entry in entries if isinstance(entry, dict) and entry.get("id") == MODEL_ID]
            if len(matching) != 1:
                raise EncryptionError("The expected Flash model is unavailable")
            return {"object": "list", "data": matching}
        except httpx.HTTPError:
            raise TransportError("Could not retrieve authenticated model metadata") from None

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Lebrel:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
