"""Loopback adapter for standard desktop and terminal clients.

Three request formats arrive as plaintext HTTP on 127.0.0.1 and leave as one encrypted chat
completion for api.lebrel.ai: OpenAI chat completions (``/v1/chat/completions``), the OpenAI
Responses API (``/v1/responses``, Codex) and the Anthropic Messages API (``/v1/messages``, Claude
Code). There is deliberately no host/network-bind option and no payload logging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import select
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import httpx

from . import __version__, anthropic_messages, openai_responses
from .client import APIError, Lebrel, LebrelError, MAX_REQUEST_BYTES, MODEL_ID, PRODUCTION_SIGNING_KEY_ID, RECEIPT_HEADER, _json

REQUEST_ID_HEADER = "X-Lebrel-Request-Id"
RECEIPT_CHECK_HEADER = "X-Lebrel-Receipt-Check"
PUBLIC_CATALOG_URL = "https://lebrel.ai/api/models"

# What a tool shows its user for the platform's answers, by code (the client keeps the code, never the
# server's own text): in line and the wait ran out, the line full, busy, the platform unreachable.
ERROR_MESSAGES = {
    "queue_wait_timeout": "You were in line for the model and the wait ran out (two minutes). Your client can retry; the line moves as answers finish.",
    "queue_full": "The line for the model is full right now. Try again in a minute.",
    "queue_lapsed": "Your place in line lapsed. Send the request again.",
    "model_paused": "The model is paused until the next drop.",
    "model_not_released": "The model is not available right now.",
    "rate_limit_exceeded": "Too many requests at once for this account. Try again in a few seconds.",
    "insufficient_quota": "Not enough balance for this request. Add funds and try again.",
    "invalid_api_key": "The API key was refused.",
    "database_unavailable": "The platform's database could not be reached just now. Try again in a moment.",
    "authentication_unavailable": "The platform could not check the key just now. Try again in a moment.",
    "billing_unavailable": "The platform could not open the hold for this request just now. Try again in a moment.",
    "encrypted_inference_unavailable": "The model's node could not be reached just now. Try again in a moment.",
    "inference_unavailable": "The model could not be reached just now. Try again in a moment.",
    "queue_unavailable": "The line for the model could not be reached just now. Try again in a moment.",
    "priority_unavailable": "Priority is not available for this key.",
    "context_length_exceeded": "The prompt and the output limit exceed the model's context window.",
    "max_output_tokens_exceeded": "The output limit asked for is above the model's.",
    "model_not_found": "The model is not available for this key.",
    "request_too_large": "The request is larger than the API accepts.",
}
_RECEIPT_PATH = re.compile(r"^/v1/receipts/([A-Za-z0-9_-]{8,128})$")
# The request format each path speaks; every one ends as the same encrypted chat completion.
_FORMATS = {"/v1/chat/completions": "openai", "/v1/responses": "responses", "/v1/messages": "anthropic", "/v1/messages/count_tokens": "anthropic"}

# Capture the loaded module's source identity once. An on-disk pip upgrade must
# not make a still-running old process advertise the new source's identity.
PROXY_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _key_file(path: Path) -> str:
    with path.open("r", encoding="utf-8") as file:
        key = file.read(4097).strip()
    if not key or len(key) > 4096 or "\n" in key or "\r" in key:
        raise ValueError("API key file must contain one API key")
    return key


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port: int = 11437, *, api_key_file: Optional[Path] = None, client_factory: Callable[..., Any] = Lebrel,
                 max_output_tokens: Optional[int] = None, catalog_url: Optional[str] = PUBLIC_CATALOG_URL) -> None:
        if not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("Port must be between 0 and 65535")
        if max_output_tokens is not None and (not isinstance(max_output_tokens, int) or max_output_tokens < 1):
            raise ValueError("max_output_tokens must be a positive integer")
        self.api_key_file = api_key_file
        self.client_factory = client_factory
        self.max_output_tokens = max_output_tokens
        self.catalog_url = catalog_url
        self._limit_lock = threading.Lock()
        self._limit_known: Optional[int] = None
        self._limit_checked: Optional[float] = None  # monotonic time of the last catalog read; None before the first
        if api_key_file is not None:
            _key_file(api_key_file)  # Validate without retaining the key.
        super().__init__(("127.0.0.1", port), ProxyHandler)

    def output_limit(self) -> Optional[int]:
        """The edition's published output budget, so clients that ask for more are capped instead of refused:
        the value given at start, otherwise the public catalog's, read at most once an hour. None when unknown."""
        if self.max_output_tokens is not None:
            return self.max_output_tokens
        if not self.catalog_url:
            return None
        with self._limit_lock:
            now = time.monotonic()
            if self._limit_checked is not None:
                if self._limit_known is not None and now - self._limit_checked < 3600:
                    return self._limit_known
                if self._limit_known is None and now - self._limit_checked < 60:
                    return None
            self._limit_checked = now
            try:
                with httpx.Client(timeout=5.0) as http:
                    catalog = http.get(self.catalog_url).json()
                for edition in catalog.get("editions", []):
                    if isinstance(edition, dict) and edition.get("modelId") == MODEL_ID:
                        value = (edition.get("context") or {}).get("servedOutputTokens")
                        if isinstance(value, int) and value > 0:
                            self._limit_known = value
            except (httpx.HTTPError, ValueError, AttributeError, TypeError):
                pass
            return self._limit_known


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "LebrelEncryptedProxy"
    sys_version = ""

    def log_message(self, *_: Any) -> None:
        pass

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(30)

    def _route(self) -> str:
        return self.path.split("?", 1)[0]

    def _error(self, status: int, code: str, fmt: str = "openai") -> None:
        if fmt == "anthropic":
            body = anthropic_messages.error_body(status, code)
        elif fmt == "responses":
            body = openai_responses.error_body(status, code)
        else:
            body = {"error": {"type": "proxy_error", "code": code, "message": ERROR_MESSAGES.get(code, "Encrypted inference could not be completed")}}
        data = json.dumps(body, separators=(",", ":")).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
        except (OSError, ValueError):
            pass
        self.close_connection = True

    def _authorization(self, fmt: str = "openai") -> Optional[str]:
        """The API key: a Bearer header, Anthropic's x-api-key header, or the key file given at start."""
        if self.headers.get("Origin"):
            self._error(403, "browser_origin_not_allowed", fmt)
            return None
        values = self.headers.get_all("Authorization", [])
        api_keys = self.headers.get_all("x-api-key", [])
        if len(values) > 1 or len(api_keys) > 1:
            self._error(401, "invalid_authorization", fmt)
            return None
        value = values[0] if values else ""
        if value:
            if not value.startswith("Bearer ") or not value[7:].strip() or len(value) > 4103:
                self._error(401, "invalid_authorization", fmt)
                return None
            return value[7:]
        if api_keys:
            key = api_keys[0].strip()
            if not key or len(key) > 4096:
                self._error(401, "invalid_authorization", fmt)
                return None
            return key
        if self.server.api_key_file is not None:
            try:
                return _key_file(self.server.api_key_file)
            except (OSError, UnicodeError, ValueError):
                self._error(503, "key_file_unavailable", fmt)
                return None
        self._error(401, "authorization_required", fmt)
        return None

    def do_OPTIONS(self) -> None:
        self._error(403, "browser_origin_not_allowed")

    def do_GET(self) -> None:
        path = self._route()
        if path == "/healthz":
            if self.headers.get("Origin"):
                self._error(403, "browser_origin_not_allowed")
                return
            self._json({"service": "lebrel-encrypted-proxy", "version": __version__, "upstream": "https://api.lebrel.ai", "signingKeyId": PRODUCTION_SIGNING_KEY_ID, "proxySourceSha256": PROXY_SOURCE_SHA256,
                        "formats": ["openai-chat-completions", "openai-responses", "anthropic-messages"]})
            return
        key = self._authorization()
        if key is None:
            return
        receipt = _RECEIPT_PATH.match(path)
        if receipt:
            self._receipt(key, receipt.group(1))
            return
        if path != "/v1/models":
            self._error(404, "not_found")
            return
        try:
            with self.server.client_factory(api_key=key) as client:
                result = client.list_models()
            self._json(result)
        except APIError as error:
            self._error(error.status_code if 400 <= error.status_code <= 599 else 502, error.code)
        except (LebrelError, ValueError, OSError):
            self._error(502, "metadata_unavailable")

    def _receipt(self, key: str, request_id: str) -> None:
        """The signed receipt of an answer, as the API serves it, with the proxy's verdict in a header: the signature by the
        pinned key and the serving manifest it names are checked here; the digests are yours to check with your text."""
        try:
            with self.server.client_factory(api_key=key) as client:
                receipt = client.receipt(request_id)
            failed = [check.id for check in receipt.checks if not check.ok]
            self._json({"payload": receipt.payload, "signature": receipt.document.signature},
                       {RECEIPT_CHECK_HEADER: "verified" if receipt.verified else "failed:" + ",".join(failed), REQUEST_ID_HEADER: request_id})
        except APIError as error:
            self._error(error.status_code if 400 <= error.status_code <= 599 else 502, error.code)
        except (LebrelError, ValueError, OSError):
            self._error(502, "receipt_unavailable")

    def _json(self, result: Any, headers: Optional[dict] = None) -> None:
        data = json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        for name, value in (headers or {}).items():
            if isinstance(value, str) and value:
                self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True

    def _read_json(self, fmt: str) -> Optional[Dict[str, Any]]:
        """The request body as a JSON object, or None once an error has been answered."""
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self._error(415, "json_required", fmt)
            return None
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) != 1 or not lengths[0].isdigit():
            self._error(411, "content_length_required", fmt)
            return None
        try:
            length = int(lengths[0])
            if not 0 < length <= MAX_REQUEST_BYTES - 20:
                self._error(413, "request_too_large", fmt)
                return None
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete body")
            request = _json(raw)
            if not isinstance(request, dict):
                raise ValueError("Invalid request")
        except (ValueError, UnicodeError, OSError):
            self._error(400, "invalid_request", fmt)
            return None
        return request

    def do_POST(self) -> None:
        path = self._route()
        fmt = _FORMATS.get(path)
        if fmt is None:
            key = self._authorization()
            if key is not None:
                self._error(404, "not_found")
            return
        key = self._authorization(fmt)
        if key is None:
            return
        request = self._read_json(fmt)
        if request is None:
            return
        if path == "/v1/messages/count_tokens":
            try:
                self._json({"input_tokens": anthropic_messages.count_tokens_estimate(request)})
            except (anthropic_messages.MessagesRequestError, ValueError):
                self._error(400, "invalid_request", fmt)
            return
        model = request.get("model") if isinstance(request.get("model"), str) and request.get("model") else MODEL_ID
        try:
            if fmt == "openai":
                if request.get("model", MODEL_ID) != MODEL_ID or not isinstance(request.get("messages"), list) or not request["messages"] or type(request.get("stream", False)) is not bool:
                    raise ValueError("Invalid request")
                chat = request
            elif fmt == "responses":
                chat = openai_responses.to_chat(request, max_output_tokens=self.server.output_limit())
            else:
                chat = anthropic_messages.to_chat(request, max_output_tokens=self.server.output_limit())
        except ValueError:
            self._error(400, "invalid_request", fmt)
            return
        self._complete(key, chat, fmt, model, request)

    def _complete(self, key: str, chat: Dict[str, Any], fmt: str, model: str, original: Dict[str, Any]) -> None:
        """One encrypted completion, answered in the caller's format; a stream is relayed as it arrives."""
        try:
            client = self.server.client_factory(api_key=key)
        except ValueError:
            self._error(401, "invalid_authorization", fmt)
            return
        stream_ref = []
        stop = threading.Event()

        def disconnect_monitor() -> None:
            while not stop.wait(0.2):
                try:
                    readable, _, _ = select.select([self.connection], [], [], 0)
                    if not readable or self.connection.recv(1, socket.MSG_PEEK):
                        continue
                except (OSError, ValueError):
                    pass
                if stream_ref:
                    stream_ref[0].close()
                client.close()
                return

        monitor = threading.Thread(target=disconnect_monitor, daemon=True)
        monitor.start()
        headers_sent = False
        try:
            result = client.create(**chat)
            request_id = getattr(result, "request_id", None)
            if chat.get("stream", False):
                stream_ref.append(result)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                if isinstance(request_id, str) and request_id:
                    # the receipt of a stream exists once it completes: fetch it by this id at /v1/receipts/{id}
                    self.send_header(REQUEST_ID_HEADER, request_id)
                self.end_headers()
                headers_sent = True
                self.close_connection = True
                with result:
                    for name, data in self._events(result, fmt, model, request_id, original):
                        self._write_event(name, data)
                    if not result.completed:
                        raise LebrelError("Stream did not complete")
                    if fmt == "openai":
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
            else:
                if fmt == "anthropic":
                    body: Any = anthropic_messages.from_chat(result, model=model, request_id=request_id)
                elif fmt == "responses":
                    body = openai_responses.from_chat(result, request=original, model=model, request_id=request_id)
                else:
                    body = result
                # the answer's signed receipt travels with it, as the API sends it; the id finds it again later
                self._json(body, {REQUEST_ID_HEADER: request_id, RECEIPT_HEADER: getattr(result, "receipt_header", None)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (LebrelError, ValueError, TypeError, OSError) as error:
            if headers_sent:
                try:
                    self._write_event("error", self._stream_error(fmt))
                except (OSError, ValueError):
                    pass
            elif isinstance(error, APIError):
                self._error(error.status_code if 400 <= error.status_code <= 599 else 502, error.code, fmt)
            else:
                self._error(502, "encrypted_inference_failed", fmt)
        finally:
            stop.set()
            if stream_ref:
                stream_ref[0].close()
            client.close()
            self.close_connection = True

    def _events(self, result: Iterator[Dict[str, Any]], fmt: str, model: str, request_id: Any, original: Dict[str, Any]) -> Iterator[Tuple[Optional[str], Dict[str, Any]]]:
        if fmt == "anthropic":
            return anthropic_messages.stream_events(result, model=model, request_id=request_id)
        if fmt == "responses":
            return openai_responses.stream_events(result, request=original, model=model, request_id=request_id)
        return ((None, event) for event in result)

    def _write_event(self, name: Optional[str], data: Dict[str, Any]) -> None:
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        if name:
            self.wfile.write(b"event: " + name.encode() + b"\n")
        self.wfile.write(b"data: " + payload + b"\n\n")
        self.wfile.flush()

    @staticmethod
    def _stream_error(fmt: str) -> Dict[str, Any]:
        if fmt == "anthropic":
            return {"type": "error", "error": {"type": "api_error", "message": "Encrypted inference did not complete"}}
        if fmt == "responses":
            return {"type": "error", "code": "incomplete_stream", "message": "Encrypted inference did not complete", "param": None, "sequence_number": 0}
        return {"error": {"code": "incomplete_stream", "message": "Encrypted inference did not complete"}}


def main(argv: Optional[Any] = None) -> int:
    parser = argparse.ArgumentParser(description="Local adapter for OpenAI chat completions, OpenAI Responses and Anthropic Messages clients; upstream requests are encrypted")
    parser.add_argument("--port", type=int, default=11437)
    parser.add_argument("--api-key-file", type=Path, help="Optional file containing the API key; incoming Bearer or x-api-key credentials take precedence")
    parser.add_argument("--max-output-tokens", type=int, help="The edition's output budget, to cap what clients ask for; read from the public catalog when omitted")
    args = parser.parse_args(argv)
    try:
        server = ProxyServer(args.port, api_key_file=args.api_key_file, max_output_tokens=args.max_output_tokens)
    except (OSError, ValueError):
        parser.exit(2, "Could not start the loopback proxy or read its key file\n")
    print(f"Lebrel encrypted proxy ready at http://127.0.0.1:{server.server_port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
