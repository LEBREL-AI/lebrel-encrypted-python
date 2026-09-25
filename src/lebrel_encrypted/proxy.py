"""Loopback adapter for standard OpenAI-compatible desktop/terminal clients.

Local HTTP is plaintext on 127.0.0.1; the upstream SDK encrypts for api.lebrel.ai.
There is deliberately no host/network-bind option and no payload logging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import select
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional

from . import __version__
from .client import APIError, Lebrel, LebrelError, MAX_REQUEST_BYTES, MODEL_ID, PRODUCTION_SIGNING_KEY_ID, RECEIPT_HEADER, _json

REQUEST_ID_HEADER = "X-Lebrel-Request-Id"
RECEIPT_CHECK_HEADER = "X-Lebrel-Receipt-Check"
_RECEIPT_PATH = re.compile(r"^/v1/receipts/([A-Za-z0-9_-]{8,128})$")

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

    def __init__(self, port: int = 11437, *, api_key_file: Optional[Path] = None, client_factory: Callable[..., Any] = Lebrel) -> None:
        if not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("Port must be between 0 and 65535")
        self.api_key_file = api_key_file
        self.client_factory = client_factory
        if api_key_file is not None:
            _key_file(api_key_file)  # Validate without retaining the key.
        super().__init__(("127.0.0.1", port), ProxyHandler)


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "LebrelEncryptedProxy"
    sys_version = ""

    def log_message(self, *_: Any) -> None:
        pass

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(30)

    def _error(self, status: int, code: str) -> None:
        data = json.dumps({"error": {"type": "proxy_error", "code": code, "message": "Encrypted inference could not be completed"}}, separators=(",", ":")).encode()
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

    def _authorization(self) -> Optional[str]:
        if self.headers.get("Origin"):
            self._error(403, "browser_origin_not_allowed")
            return None
        values = self.headers.get_all("Authorization", [])
        if len(values) > 1:
            self._error(401, "invalid_authorization")
            return None
        value = values[0] if values else ""
        if value:
            if not value.startswith("Bearer ") or not value[7:].strip() or len(value) > 4103:
                self._error(401, "invalid_authorization")
                return None
            return value[7:]
        if self.server.api_key_file is not None:
            try:
                return _key_file(self.server.api_key_file)
            except (OSError, UnicodeError, ValueError):
                self._error(503, "key_file_unavailable")
                return None
        self._error(401, "authorization_required")
        return None

    def do_OPTIONS(self) -> None:
        self._error(403, "browser_origin_not_allowed")

    def do_GET(self) -> None:
        if self.path == "/healthz":
            if self.headers.get("Origin"):
                self._error(403,"browser_origin_not_allowed")
                return
            self._json({"service":"lebrel-encrypted-proxy","version":__version__,"upstream":"https://api.lebrel.ai","signingKeyId":PRODUCTION_SIGNING_KEY_ID,"proxySourceSha256":PROXY_SOURCE_SHA256})
            return
        key = self._authorization()
        if key is None:
            return
        receipt = _RECEIPT_PATH.match(self.path)
        if receipt:
            self._receipt(key, receipt.group(1))
            return
        if self.path != "/v1/models":
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

    def do_POST(self) -> None:
        key = self._authorization()
        if key is None:
            return
        if self.path != "/v1/chat/completions":
            self._error(404, "not_found")
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self._error(415, "json_required")
            return
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) != 1 or not lengths[0].isdigit():
            self._error(411, "content_length_required")
            return
        try:
            length = int(lengths[0])
            if not 0 < length <= MAX_REQUEST_BYTES - 20:
                self._error(413, "request_too_large")
                return
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete body")
            request = _json(raw)
            if not isinstance(request, dict) or request.get("model", MODEL_ID) != MODEL_ID or not isinstance(request.get("messages"),list) or not request["messages"] or type(request.get("stream",False)) is not bool:
                raise ValueError("Invalid request")
        except (ValueError, UnicodeError, OSError):
            self._error(400, "invalid_request")
            return
        try:
            client = self.server.client_factory(api_key=key)
        except ValueError:
            self._error(401,"invalid_authorization")
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
            result = client.create(**request)
            request_id = getattr(result, "request_id", None)
            if request.get("stream", False):
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
                    for event in result:
                        data = json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
                        self.wfile.write(b"data: " + data + b"\n\n")
                        self.wfile.flush()
                    if not result.completed:
                        raise LebrelError("Stream did not complete")
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
            else:
                # the answer's signed receipt travels with it, as the API sends it; the id finds it again later
                self._json(result, {REQUEST_ID_HEADER: request_id, RECEIPT_HEADER: getattr(result, "receipt_header", None)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (LebrelError, ValueError, TypeError, OSError) as error:
            if headers_sent:
                try:
                    self.wfile.write(b'event: error\ndata: {"error":{"code":"incomplete_stream","message":"Encrypted inference did not complete"}}\n\n')
                    self.wfile.flush()
                except (OSError, ValueError):
                    pass
            elif isinstance(error, APIError):
                self._error(error.status_code if 400 <= error.status_code <= 599 else 502, error.code)
            else:
                self._error(502, "encrypted_inference_failed")
        finally:
            stop.set()
            if stream_ref:
                stream_ref[0].close()
            client.close()
            self.close_connection = True


def main(argv: Optional[Any] = None) -> int:
    parser = argparse.ArgumentParser(description="Local OpenAI-compatible adapter; upstream requests are encrypted")
    parser.add_argument("--port", type=int, default=11437)
    parser.add_argument("--api-key-file", type=Path, help="Optional file containing the API key; incoming Bearer credentials take precedence")
    args = parser.parse_args(argv)
    try:
        server = ProxyServer(args.port, api_key_file=args.api_key_file)
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
