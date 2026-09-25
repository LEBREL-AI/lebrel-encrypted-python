import base64
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from lebrel_encrypted import APIError, Completion, EncryptionError, Lebrel, MODEL_ID, StreamError, proof
from lebrel_encrypted.client import PRODUCTION_SIGNING_KEY, PRODUCTION_SIGNING_KEY_ID, verify_metadata
from lebrel_encrypted._vendor.ehbp.identity import ServerIdentity


SIGNER = Ed25519PrivateKey.from_private_bytes(bytes([7]) * 32)
PUBLIC_KEY = SIGNER.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


def metadata(**changes):
    hpke_config = ServerIdentity.from_public_key_bytes(bytes([9]) * 32).marshal_public_config()
    payload = {
        "version": 1, "issuedAt": 1000, "expiresAt": 1300,
        "modelId": MODEL_ID, "keyId": hashlib.sha256(hpke_config).hexdigest(),
        "hpkeConfig": base64.b64encode(hpke_config).decode(),
        "serverInstanceId": "a" * 32,
        "signingKeyId": hashlib.sha256(PUBLIC_KEY).hexdigest(),
    }
    payload.update(changes)
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return json.dumps({"version": 1, "payload": base64.b64encode(encoded).decode(), "signature": base64.b64encode(SIGNER.sign(encoded)).decode()}).encode()


def test_production_pin_and_signed_config():
    assert hashlib.sha256(base64.b64decode(PRODUCTION_SIGNING_KEY)).hexdigest() == PRODUCTION_SIGNING_KEY_ID
    config = verify_metadata(metadata(), PUBLIC_KEY, 1010)
    assert config.expires_at == 1300


@pytest.mark.parametrize("changes", [
    {"modelId": "wrong"}, {"expiresAt": 999}, {"issuedAt": 1100},
    {"expiresAt": 1700}, {"issuedAt": True}, {"version": True},
    {"keyId": "f" * 64}, {"signingKeyId": "f" * 64},
    {"serverInstanceId": "bad"}, {"hpkeConfig": "?"},
])
def test_invalid_signed_metadata_rejected(changes):
    with pytest.raises(EncryptionError):
        verify_metadata(metadata(**changes), PUBLIC_KEY, 1010)


def test_signature_must_match_pinned_key():
    with pytest.raises(EncryptionError):
        verify_metadata(metadata(), bytes([1]) * 32, 1010)


def test_invalid_metadata_never_sends_conversation():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, stream=BytesStream(metadata(modelId="wrong")))
    with Lebrel("leb_live_test", _transport=httpx.MockTransport(handler), _clock=lambda: 1010, signing_public_key=base64.b64encode(PUBLIC_KEY).decode()) as client:
        with pytest.raises(EncryptionError):
            client.create(messages=[{"role": "user", "content": "secret-user-text"}])
    assert len(calls) == 1 and calls[0].method == "GET"
    assert "authorization" not in calls[0].headers
    assert b"secret-user-text" not in calls[0].content


class BytesStream(httpx.SyncByteStream):
    def __init__(self, data):
        self.data = data
    def __iter__(self):
        for offset in range(0, len(self.data), 7):
            yield self.data[offset:offset + 7]


def test_request_headers_encryption_and_no_retry():
    calls = []
    def handler(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(200, stream=BytesStream(metadata()))
        assert b"secret-user-text" not in request.content
        assert b"messages" not in request.content
        assert request.headers["authorization"] == "Bearer leb_live_test"
        assert len(request.headers["ehbp-encapsulated-key"]) == 64
        assert len(request.headers["x-lebrel-request-id"]) == 36
        assert len(request.headers["x-lebrel-encryption-key-id"]) == 64
        return httpx.Response(422, stream=BytesStream(b"secret-user-text"))
    with Lebrel("leb_live_test", _transport=httpx.MockTransport(handler), _clock=lambda: 1010, signing_public_key=base64.b64encode(PUBLIC_KEY).decode()) as client:
        with pytest.raises(APIError) as raised:
            client.chat.completions.create(messages=[{"role": "user", "content": "secret-user-text"}])
    assert "secret-user-text" not in str(raised.value)
    assert len(calls) == 2


def test_wrong_model_and_plain_http_never_sent():
    with pytest.raises(ValueError):
        Lebrel("test", base_url="http://api.lebrel.ai")
    with pytest.raises(ValueError):
        Lebrel("test", base_url="http://127.0.0.1:8000")
    with Lebrel("test") as client:
        with pytest.raises(ValueError):
            client.create(messages=[{"role": "user", "content": "test"}], model="wrong")


@pytest.fixture(scope="session")
def go_server(tmp_path_factory):
    folder = Path(__file__).parent / "interop"
    binary = tmp_path_factory.mktemp("go") / "canary"
    env = dict(os.environ, GOTOOLCHAIN="go1.26.0")
    subprocess.run(["go", "build", "-o", str(binary), "."], cwd=folder, env=env, check=True, capture_output=True, timeout=180)
    process = subprocess.Popen([str(binary)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        line = process.stdout.readline().strip()
        assert line, "Go interoperability server did not start"
        yield line.split(" ")
    finally:
        process.terminate()
        process.communicate(timeout=10)


def test_actual_go_request_response_interoperability(go_server):
    endpoint, public_key = go_server
    with Lebrel("leb_live_test", base_url=endpoint, signing_public_key=public_key) as client:
        result = client.chat.completions.create(messages=[{"role": "user", "content": "secret-user-text"}])
    assert result["model"] == MODEL_ID
    assert result["choices"][0]["message"]["content"] == "verified-python-go"
    # what a receipt binds travels with the completion: the request id sent, the digests of the request and the answer
    assert isinstance(result, Completion) and len(result.request_id) == 36
    assert result.request_sha256 == hashlib.sha256(json.dumps({"model": MODEL_ID, "messages": [{"role": "user", "content": "secret-user-text"}], "stream": False}, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    assert result.response_sha256 == hashlib.sha256(b"verified-python-go").hexdigest()
    assert json.loads(json.dumps(result)) == dict(result), "still a plain dictionary for every JSON consumer"


def test_actual_go_streaming_interoperability(go_server):
    endpoint, public_key = go_server
    with Lebrel("leb_live_test", base_url=endpoint, signing_public_key=public_key) as client:
        with client.chat.completions.create(messages=[{"role": "user", "content": "secret-user-text"}], stream=True) as stream:
            assert stream.response_sha256 is None, "no digest before the stream completes"
            chunks = list(stream)
            assert stream.completed
        assert stream.closed
    assert chunks[0]["choices"][0]["delta"]["content"] == "verified-python-go 🐺"
    assert len(stream.request_id) == 36 and len(stream.request_sha256) == 64
    assert stream.response_sha256 == hashlib.sha256("verified-python-go 🐺".encode("utf-8")).hexdigest(), "every content delta, in order, as the runtime hashes it"


@pytest.mark.parametrize("mode", ["incomplete", "done-no-finish", "after-done"])
def test_actual_go_incomplete_stream_rejected(go_server, mode):
    endpoint, public_key = go_server
    with Lebrel("leb_live_test", base_url=endpoint, signing_public_key=public_key) as client:
        with client.create(messages=[{"role": "user", "content": mode}], stream=True) as stream:
            with pytest.raises(StreamError):
                list(stream)
            assert not stream.completed


def test_actual_go_encrypted_error_and_plaintext_fallback_rejected(go_server):
    endpoint, public_key = go_server
    with Lebrel("leb_live_test", base_url=endpoint, signing_public_key=public_key) as client:
        with pytest.raises(APIError) as raised:
            client.create(messages=[{"role": "user", "content": "encrypted-error"}])
        assert raised.value.status_code == 402 and raised.value.code == "insufficient_credits"
        assert "secret-user-text" not in str(raised.value)
        with pytest.raises(EncryptionError):
            client.create(messages=[{"role": "user", "content": "plain-success"}])


@pytest.mark.parametrize("mode", ["tamper", "truncated-frame"])
def test_actual_go_tampered_and_truncated_ciphertext_rejected(go_server, mode):
    endpoint, public_key = go_server
    with Lebrel("leb_live_test", base_url=endpoint, signing_public_key=public_key) as client:
        with pytest.raises(EncryptionError):
            client.create(messages=[{"role": "user", "content": mode}])


def test_closing_stream_early_is_not_completed(go_server):
    endpoint, public_key = go_server
    with Lebrel("leb_live_test", base_url=endpoint, signing_public_key=public_key) as client:
        with client.create(messages=[{"role": "user", "content": "secret-user-text"}], stream=True) as stream:
            next(stream)
        assert stream.closed and not stream.completed


def test_close_interrupts_actual_network_read(go_server):
    endpoint, public_key = go_server
    finished = threading.Event()
    with Lebrel("leb_live_test", base_url=endpoint, signing_public_key=public_key, timeout=5) as client:
        stream = client.create(messages=[{"role": "user", "content": "blocking"}], stream=True)
        next(stream)
        def consume():
            try:
                next(stream)
            except Exception:
                pass
            finally:
                finished.set()
        reader = threading.Thread(target=consume, daemon=True)
        reader.start()
        time.sleep(0.05)
        stream.close()
        assert finished.wait(2), "Closing a stream must interrupt its outstanding network read"


def signed(payload, signer=SIGNER):
    return proof.Signed(payload=payload, signature=base64.b64encode(signer.sign(proof.canonical(payload))).decode()).to_json().encode()


def served_manifest(**changes):
    payload = {
        "version": 1, "edition": {"id": MODEL_ID, "name": "x", "base_model": "y", "fingerprint_id": None},
        "weights": {"repository": "r", "revision": "a" * 64, "files": {}, "total_bytes": 0}, "quantization": {"method": "nvfp4"}, "engine": {"name": "vllm"},
        "tokenizer_sha256": "b" * 64, "chat_template_sha256": "c" * 64, "runtime": {"provider": "modal"}, "attestation": None,
        "issued_at": 1000, "expires_at": 4600, "signing_key_id": hashlib.sha256(PUBLIC_KEY).hexdigest(),
    }
    payload.update(changes)
    return payload


def served_receipt(**changes):
    payload = {
        "version": 1, "request_id": "req-0123456789", "manifest_sha256": proof.manifest_identity(served_manifest()),
        "prompt_sha256": hashlib.sha256(b"request").hexdigest(), "response_sha256": hashlib.sha256(b"answer").hexdigest(),
        "prompt_tokens": 3, "completion_tokens": 2, "issued_at": 1500, "instance_id": "i" * 32, "signing_key_id": hashlib.sha256(PUBLIC_KEY).hexdigest(),
    }
    payload.update(changes)
    return payload


def proof_client(handler):
    return Lebrel("leb_live_test", _transport=httpx.MockTransport(handler), _clock=lambda: 2000, signing_public_key=base64.b64encode(PUBLIC_KEY).decode())


def test_receipt_is_fetched_without_the_key_and_checked_against_manifest_and_completion():
    calls = []

    def handler(request):
        calls.append(request)
        assert "authorization" not in request.headers, "public documents never carry the API key"
        if request.url.path == "/.well-known/proof-of-edition":
            return httpx.Response(200, stream=BytesStream(signed(served_manifest())))
        if request.url.path == "/v1/receipts/req-0123456789":
            return httpx.Response(200, stream=BytesStream(signed(served_receipt())))
        return httpx.Response(404, stream=BytesStream(b'{"error":{"code":"receipt_not_found"}}'))

    with proof_client(handler) as client:
        checked = client.receipt("req-0123456789")
        assert checked.verified and checked.request_id == "req-0123456789" and checked.manifest is not None and checked.manifest.verified
        assert [c.id for c in checked.checks] == ["shape", "key", "signature", "request_id", "manifest"]
        completion = Completion({"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]})
        completion.request_id, completion.request_sha256, completion.response_sha256 = "req-0123456789", hashlib.sha256(b"request").hexdigest(), hashlib.sha256(b"answer").hexdigest()
        assert client.receipt(completion).verified
        completion.response_sha256 = hashlib.sha256(b"tampered").hexdigest()
        failed = client.receipt(completion)
        assert not failed.verified and [c.id for c in failed.checks if not c.ok] == ["response"]
        # a receipt that came in the answer's header is used as it is, with no fetch
        completion.response_sha256 = hashlib.sha256(b"answer").hexdigest()
        completion.receipt_header = base64.b64encode(signed(served_receipt())).decode()
        before = len(calls)
        assert client.receipt(completion, manifest=False).verified and len(calls) == before
        with pytest.raises(APIError) as raised:
            client.receipt("req-unknown-0000")
        assert raised.value.status_code == 404 and raised.value.code == "receipt_not_found"
        with pytest.raises(ValueError):
            client.receipt("bad id")
        assert all(b"request" not in c.content and b"answer" not in c.content for c in calls)


def test_receipt_signed_by_another_key_or_naming_an_old_manifest_is_not_verified():
    other = Ed25519PrivateKey.from_private_bytes(bytes([8]) * 32)
    state = {"manifest": True}

    def handler(request):
        if request.url.path == "/.well-known/proof-of-edition":
            if not state["manifest"]:
                return httpx.Response(503, stream=BytesStream(b""))
            return httpx.Response(200, stream=BytesStream(signed(served_manifest(engine={"name": "sglang"}))))  # the runtime moved on
        if request.url.path == "/v1/receipts/req-0123456789":
            return httpx.Response(200, stream=BytesStream(signed(served_receipt())))
        if request.url.path == "/v1/receipts/req-forged-00000":
            return httpx.Response(200, stream=BytesStream(signed(served_receipt(request_id="req-forged-00000"), other)))
        if request.url.path == "/v1/receipts/req-broken-00000":
            return httpx.Response(200, stream=BytesStream(b"{}"))
        return httpx.Response(503, stream=BytesStream(b""))

    with proof_client(handler) as client:
        stale = client.receipt("req-0123456789")
        assert not stale.verified and [c.id for c in stale.checks if not c.ok] == ["manifest"], "the receipt is genuine but names a manifest that is not the one serving now"
        assert client.receipt("req-0123456789", manifest=False).verified
        forged = client.receipt("req-forged-00000", manifest=False)
        assert not forged.verified and [c.id for c in forged.checks if not c.ok] == ["signature"]
        with pytest.raises(EncryptionError):
            client.receipt("req-broken-00000", manifest=False)
        state["manifest"] = False  # the model on standby: no manifest to check against
        with pytest.raises(APIError) as raised:
            client.manifest()
        assert raised.value.status_code == 503 and raised.value.code == "manifest_unavailable"
        with pytest.raises(APIError):
            client.receipt("req-0123456789")
        assert client.receipt("req-0123456789", manifest=False).verified, "the receipt alone still checks"
