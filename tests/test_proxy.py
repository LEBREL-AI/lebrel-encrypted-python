import json
import hashlib
import socket
import threading
import time

import httpx
import pytest

from lebrel_encrypted import APIError, Completion, MODEL_ID, StreamError, proof
from lebrel_encrypted.proxy import ProxyServer
from lebrel_encrypted import proxy as proxy_module
from pathlib import Path


class MockStream:
    def __init__(self, mode):
        self.mode = mode
        self.completed = False
        self.closed = threading.Event()
        self.request_id = "req-stream-000001"
    def __enter__(self):
        return self
    def __exit__(self, *_):
        self.close()
    def __iter__(self):
        yield {"choices": [{"index": 0, "delta": {"content": "private-answer"}, "finish_reason": None}]}
        if self.mode == "blocking":
            self.closed.wait(5)
            return
        if self.mode == "fail":
            raise StreamError("private-message-must-not-be-logged")
        yield {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        self.completed = True
    def close(self):
        self.closed.set()


class MockClient:
    instances = []
    def __init__(self, api_key):
        self.api_key = api_key
        self.request = None
        self.closed = False
        self.stream = None
        self.instances.append(self)
    def __enter__(self):
        return self
    def __exit__(self, *_):
        self.close()
    def create(self, **request):
        self.request = request
        if request.get("stream"):
            self.stream = MockStream(request["messages"][0]["content"])
            return self.stream
        completion = Completion({"choices": [{"message": {"content": "private-answer"}, "finish_reason": "stop"}]})
        completion.request_id = "req-plain-0000001"
        completion.receipt_header = "eyJwYXlsb2FkIjp7fSwic2lnbmF0dXJlIjoiQUE9PSJ9"
        return completion
    def receipt(self, request_id, manifest=True):
        if request_id != "req-plain-0000001":
            raise APIError(404, "receipt_not_found")
        return proof.Receipt(document=proof.Signed(payload={"request_id": request_id, "prompt_tokens": 3}, signature="AA=="),
                             checks=[proof.Check("signature", "Signature", True, "ok"), proof.Check("manifest", "Serving manifest", manifest, "checked" if manifest else "skipped")],
                             verified=manifest, manifest=None)
    def list_models(self):
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}
    def close(self):
        self.closed = True


@pytest.fixture
def proxy():
    MockClient.instances = []
    server = ProxyServer(0, client_factory=MockClient)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server, "http://127.0.0.1:" + str(server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def body(mode="normal", stream=False):
    return {"model": MODEL_ID, "messages": [{"role": "user", "content": mode}], "stream": stream}


def test_proxy_health_and_required_authorization(proxy):
    server, url = proxy
    assert server.server_address[0] == "127.0.0.1"
    with httpx.Client(trust_env=False) as client:
        result = client.get(url + "/healthz")
        assert result.status_code == 200
        assert result.json()["service"] == "lebrel-encrypted-proxy"
        assert result.json()["upstream"] == "https://api.lebrel.ai"
        assert result.json()["proxySourceSha256"] == hashlib.sha256(Path(proxy_module.__file__).read_bytes()).hexdigest()
        assert client.get(url + "/v1/models").status_code == 401
        assert client.post(url + "/v1/chat/completions", json=body()).status_code == 401
        assert client.post(url + "/v1/chat/completions", json=body(), headers={"Authorization": "Bearer local-key", "Origin": "https://untrusted.example"}).status_code == 403
    assert not MockClient.instances


def test_proxy_json_models_and_no_payload_logs(proxy, capsys):
    _, url = proxy
    with httpx.Client(trust_env=False, headers={"Authorization": "Bearer local-key"}) as client:
        response = client.post(url + "/v1/chat/completions", json=body("private-question"))
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "private-answer"
        models = client.get(url + "/v1/models").json()
        assert models["data"][0]["id"] == MODEL_ID
    assert MockClient.instances[0].api_key == "local-key"
    assert MockClient.instances[0].request["messages"][0]["content"] == "private-question"
    assert all(instance.closed for instance in MockClient.instances)
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("mode,complete", [("normal", True), ("fail", False)])
def test_proxy_streaming_and_failure_has_no_done(proxy, mode, complete):
    _, url = proxy
    with httpx.Client(trust_env=False, headers={"Authorization": "Bearer local-key"}) as client:
        response = client.post(url + "/v1/chat/completions", json=body(mode, True))
    assert response.status_code == 200
    assert ("data: [DONE]" in response.text) == complete
    assert ("incomplete_stream" in response.text) != complete
    assert "private-message-must-not-be-logged" not in response.text
    assert MockClient.instances[-1].stream.closed.is_set()


def test_proxy_disconnect_closes_remote_stream(proxy):
    server, _ = proxy
    payload = json.dumps(body("blocking", True)).encode()
    connection = socket.create_connection(server.server_address, timeout=3)
    connection.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer local-key\r\nContent-Type: application/json\r\nContent-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload)
    data = b""
    while b"private-answer" not in data:
        data += connection.recv(4096)
    connection.close()
    assert MockClient.instances[-1].stream.closed.wait(2)


def test_key_file_fallback_and_bad_model_rejected(tmp_path):
    key = tmp_path / "key"
    key.write_text("file-key\n")
    server = ProxyServer(0, api_key_file=key, client_factory=MockClient)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with httpx.Client(trust_env=False) as client:
            url = "http://127.0.0.1:" + str(server.server_port)
            assert client.post(url + "/v1/chat/completions", json=body()).status_code == 200
            assert MockClient.instances[-1].api_key == "file-key"
            request = body()
            request["model"] = "wrong"
            assert client.post(url + "/v1/chat/completions", json=request).status_code == 400
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_proxy_relays_request_ids_and_receipts(proxy):
    _, base = proxy
    plain = httpx.post(base + "/v1/chat/completions", json=body(), headers={"Authorization": "Bearer leb_live_test"})
    assert plain.status_code == 200 and plain.json()["choices"][0]["message"]["content"] == "private-answer"
    assert plain.headers["x-lebrel-request-id"] == "req-plain-0000001", "the id that finds the receipt later"
    assert plain.headers["proof-of-edition-receipt"] == "eyJwYXlsb2FkIjp7fSwic2lnbmF0dXJlIjoiQUE9PSJ9", "the signed receipt, as the API sent it"
    with httpx.stream("POST", base + "/v1/chat/completions", json=body(stream=True), headers={"Authorization": "Bearer leb_live_test"}) as streamed:
        assert streamed.status_code == 200 and streamed.headers["x-lebrel-request-id"] == "req-stream-000001"
        assert "proof-of-edition-receipt" not in streamed.headers, "a stream's receipt exists only once it completes"
        streamed.read()
    fetched = httpx.get(base + "/v1/receipts/req-plain-0000001", headers={"Authorization": "Bearer leb_live_test"})
    assert fetched.status_code == 200 and fetched.json() == {"payload": {"request_id": "req-plain-0000001", "prompt_tokens": 3}, "signature": "AA=="}
    assert fetched.headers["x-lebrel-receipt-check"] == "verified" and fetched.headers["x-lebrel-request-id"] == "req-plain-0000001"
    missing = httpx.get(base + "/v1/receipts/req-unknown-00001", headers={"Authorization": "Bearer leb_live_test"})
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "receipt_not_found"
    assert httpx.get(base + "/v1/receipts/bad", headers={"Authorization": "Bearer leb_live_test"}).status_code == 404
    assert httpx.get(base + "/v1/receipts/req-plain-0000001").status_code == 401, "receipts through the proxy need the caller's key like everything else"
