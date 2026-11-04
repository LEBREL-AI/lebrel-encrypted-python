"""Routed models: the client attests the router, verifies the route's key document with the attested
receipt key, and sends the ciphertext with the model named in a header."""

import base64
import hashlib
import json

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lebrel_encrypted import attestation
from lebrel_encrypted.client import APIError, EncryptionError, Lebrel, ROUTED_MODEL_HEADER, _salted_body, _salted_request_digest, routed_model_id
from lebrel_encrypted._vendor.ehbp.identity import ServerIdentity


class BytesStream(httpx.SyncByteStream):
    def __init__(self, data: bytes) -> None:
        self._data = data

    def __iter__(self):
        yield self._data


def streamed(status, content):
    return httpx.Response(status, stream=BytesStream(content if isinstance(content, bytes) else json.dumps(content).encode()))

ROUTED = "deepseek/deepseek-v4.1-flash"
SLUG = "deepseek--deepseek-v4.1-flash"
RECEIPT_KEY = Ed25519PrivateKey.from_private_bytes(bytes([3]) * 32)
RECEIPT_PUBLIC = RECEIPT_KEY.public_key().public_bytes_raw()
ROUTER = ServerIdentity.from_public_key_bytes(bytes([5]) * 32)
HPKE_CONFIG = ROUTER.marshal_public_config()
KEY_ID = hashlib.sha256(HPKE_CONFIG).hexdigest()


def attested(**changes):
    fields = dict(receipt_public_key=RECEIPT_PUBLIC, receipt_key_id=hashlib.sha256(RECEIPT_PUBLIC).hexdigest(), encryption_key_id=KEY_ID,
                  encryption_public_key=bytes([5]) * 32, app_id=attestation.PINNED_APP_ID, compose_hash="ab" * 32, instance_id="cd" * 20,
                  chain_checked=True, checks=())
    fields.update(changes)
    return attestation.AttestedRouter(**fields)


def route_metadata(model=ROUTED, key=RECEIPT_KEY, **changes):
    payload = {"expiresAt": 1300, "hpkeConfig": base64.b64encode(HPKE_CONFIG).decode(), "issuedAt": 1000, "keyId": KEY_ID, "modelId": model,
               "serverInstanceId": "0" * 32, "signingKeyId": hashlib.sha256(key.public_key().public_bytes_raw()).hexdigest(), "version": 1}
    payload.update(changes)
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    return json.dumps({"version": 1, "payload": base64.b64encode(encoded).decode(), "signature": base64.b64encode(key.sign(encoded)).decode()}).encode()


def client(handler, verify, **kwargs):
    return Lebrel("leb_live_test", _transport=httpx.MockTransport(handler), _clock=lambda: 1010, _verify_attestation=verify, **kwargs)


def test_routed_model_ids():
    assert routed_model_id(ROUTED)
    assert not routed_model_id("lebrel/deepseek-v4.1-flash-uncensored")
    assert not routed_model_id("wrong")
    assert not routed_model_id("a/b/c")


def test_a_routed_request_attests_the_router_then_sends_ciphertext_with_the_model_header():
    seen = []
    verified = []

    def verify(document, nonce, *, allow_unverified_chain, allowed_compose_hashes):
        verified.append((document["kind"], nonce, allow_unverified_chain, allowed_compose_hashes))
        return attested()

    def handler(request):
        seen.append(request)
        path = request.url.path
        if path == f"/m/{SLUG}/.well-known/attestation":
            return streamed(200, {"kind": "lebrel-router-attestation", "nonce": request.url.params["nonce"]})
        if path == f"/m/{SLUG}/.well-known/lebrel-encryption":
            return streamed(200, route_metadata())
        if path == "/v1/chat/completions":
            assert request.headers[ROUTED_MODEL_HEADER] == ROUTED
            assert request.headers["X-Lebrel-Encryption-Key-Id"] == KEY_ID
            assert len(request.headers["Ehbp-Encapsulated-Key"]) == 64
            assert request.headers["Content-Type"] == "application/octet-stream"
            assert b"hello" not in request.content, "the prompt never travels in clear"
            return streamed(503, {"error": {"code": "upstream_unavailable"}})
        raise AssertionError(path)

    with client(handler, verify) as c:
        with pytest.raises(APIError) as error:
            c.chat.completions.create(model=ROUTED, messages=[{"role": "user", "content": "hello"}])
        assert error.value.status_code == 503
        assert [r.url.path for r in seen] == [f"/m/{SLUG}/.well-known/attestation", f"/m/{SLUG}/.well-known/lebrel-encryption", "/v1/chat/completions"]
        assert verified[0][0] == "lebrel-router-attestation" and len(verified[0][1]) == 32 and verified[0][2] is False
        assert verified[0][3] == attestation.PINNED_COMPOSE_HASHES, "the client pins this release's composes by default"
        assert seen[0].url.params["nonce"] == verified[0][1].hex()
        # A second request within five minutes reuses the attestation and fetches only the key document.
        with pytest.raises(APIError):
            c.chat.completions.create(model=ROUTED, messages=[{"role": "user", "content": "again"}])
        assert len(verified) == 1
        assert [r.url.path for r in seen][3:] == [f"/m/{SLUG}/.well-known/lebrel-encryption", "/v1/chat/completions"]


def test_a_key_document_not_signed_by_the_attested_key_or_naming_another_key_is_refused():
    other = Ed25519PrivateKey.from_private_bytes(bytes([7]) * 32)
    cases = {
        "wrong signer": route_metadata(key=other),
        "wrong model": route_metadata(model="other/model"),
    }
    for name, envelope in cases.items():
        sent = []

        def handler(request, envelope=envelope):
            if request.url.path.endswith("/attestation"):
                return streamed(200, {"kind": "lebrel-router-attestation"})
            if request.url.path.endswith("/lebrel-encryption"):
                return streamed(200, envelope)
            sent.append(request)
            return streamed(200, b"{}")

        with client(handler, lambda *a, **k: attested()) as c:
            with pytest.raises(EncryptionError):
                c.chat.completions.create(model=ROUTED, messages=[{"role": "user", "content": "secret"}])
        assert sent == [], name

    def mismatch(request):
        if request.url.path.endswith("/attestation"):
            return streamed(200, {"kind": "lebrel-router-attestation"})
        if request.url.path.endswith("/lebrel-encryption"):
            return streamed(200, route_metadata())
        raise AssertionError("sent")

    with client(mismatch, lambda *a, **k: attested(encryption_key_id="ee" * 32)) as c:
        with pytest.raises(EncryptionError, match="not the one its attestation binds"):
            c.chat.completions.create(model=ROUTED, messages=[{"role": "user", "content": "secret"}])
    # The comparison is of key bytes: a document whose id matches but whose key is another is refused too.
    with client(mismatch, lambda *a, **k: attested(encryption_public_key=bytes([6]) * 32)) as c:
        with pytest.raises(EncryptionError, match="not the one its attestation binds"):
            c.chat.completions.create(model=ROUTED, messages=[{"role": "user", "content": "secret"}])


def test_a_failed_attestation_never_sends_anything():
    sent = []

    def handler(request):
        if request.url.path.endswith("/attestation"):
            return streamed(200, {"kind": "lebrel-router-attestation"})
        sent.append(request)
        return streamed(200, b"{}")

    def refuse(document, nonce, **_):
        raise attestation.AttestationError("report_data: binds nothing")

    with client(handler, refuse) as c:
        with pytest.raises(EncryptionError, match="did not prove"):
            c.chat.completions.create(model=ROUTED, messages=[{"role": "user", "content": "secret"}])
    assert sent == []


def test_an_unknown_model_id_is_refused_before_any_request():
    with client(lambda request: httpx.Response(500), lambda *a, **k: attested()) as c:
        with pytest.raises(ValueError):
            c.chat.completions.create(model="wrong", messages=[{"role": "user", "content": "x"}])


def test_a_routed_request_carries_a_salt_first_inside_the_encrypted_body(monkeypatch):
    plaintexts = []
    original = ServerIdentity.encrypt_request_body

    def recording(self, body):
        plaintexts.append(bytes(body))
        return original(self, body)

    monkeypatch.setattr(ServerIdentity, "encrypt_request_body", recording)

    def handler(request):
        if request.url.path.endswith("/attestation"):
            return streamed(200, {"kind": "lebrel-router-attestation"})
        if request.url.path.endswith("/lebrel-encryption"):
            return streamed(200, route_metadata())
        return streamed(503, {"error": {"code": "upstream_unavailable"}})

    with client(handler, lambda *a, **k: attested()) as c:
        with pytest.raises(APIError):
            c.chat.completions.create(model=ROUTED, messages=[{"role": "user", "content": "hello"}])
    assert len(plaintexts) == 1
    prefix = b'{"lebrel_receipt_salt":"'
    assert plaintexts[0].startswith(prefix)
    salt_hex, rest = plaintexts[0][len(prefix):].split(b'"', 1)
    assert len(bytes.fromhex(salt_hex.decode())) == 16 and rest.startswith(b",")
    inner = b"{" + rest[1:]
    assert json.loads(inner) == {"model": ROUTED, "messages": [{"role": "user", "content": "hello"}], "stream": False}


def test_a_busy_machine_is_asked_again_after_its_retry_after_with_a_new_nonce():
    nonces = []
    waits = []

    def handler(request):
        if request.url.path.endswith("/attestation"):
            nonces.append(request.url.params["nonce"])
            if len(nonces) < 3:
                return httpx.Response(429, headers={"Retry-After": "2"})
            return streamed(200, {"kind": "lebrel-router-attestation"})
        if request.url.path.endswith("/lebrel-encryption"):
            return streamed(200, route_metadata())
        return streamed(503, {"error": {"code": "upstream_unavailable"}})

    with client(handler, lambda *a, **k: attested(), _sleep=waits.append) as c:
        with pytest.raises(APIError):
            c.chat.completions.create(model=ROUTED, messages=[{"role": "user", "content": "x"}])
    assert len(nonces) == 3 and len(set(nonces)) == 3 and waits == [2.0, 2.0]

    # Busy three times, the shared document (no nonce) is taken, and the attestation says it was not fresh.
    taken = []
    verified = []

    def busy_then_shared(request):
        if request.url.path.endswith("/attestation"):
            taken.append(request.url.params.get("nonce"))
            if "nonce" in request.url.params:
                return httpx.Response(429, headers={"Retry-After": "1"})
            return streamed(200, {"kind": "lebrel-router-attestation"})
        if request.url.path.endswith("/lebrel-encryption"):
            return streamed(200, route_metadata())
        return streamed(503, {"error": {"code": "upstream_unavailable"}})

    def verify(document, nonce, **_):
        verified.append(nonce)
        return attested()

    with client(busy_then_shared, verify, _sleep=waits.append) as c:
        with pytest.raises(APIError):
            c.chat.completions.create(model=ROUTED, messages=[{"role": "user", "content": "x"}])
        assert taken[-1] is None and len(taken) == 4 and verified == [b""]
        not_fresh = [check for check in c._attested[SLUG][0].checks if check.id == "nonce"]
        assert not_fresh and not not_fresh[0].ok

    always_busy = lambda request: httpx.Response(429, headers={"Retry-After": "1"})  # noqa: E731
    with client(always_busy, lambda *a, **k: attested(), _sleep=waits.append) as c:
        with pytest.raises(EncryptionError, match="quoting for others"):
            c.chat.completions.create(model=ROUTED, messages=[{"role": "user", "content": "x"}])


def test_the_caller_can_pass_its_own_composes_or_none():
    seen = []

    def verify(document, nonce, **kwargs):
        seen.append(kwargs["allowed_compose_hashes"])
        return attested()

    handler = lambda request: streamed(200, {"kind": "lebrel-router-attestation"}) if request.url.path.endswith("/attestation") else streamed(503, {"error": {"code": "upstream_unavailable"}})  # noqa: E731
    for given, expected in ((("ab" * 32,), ("ab" * 32,)), (None, None)):
        with client(handler, verify, allowed_compose_hashes=given) as c:
            with pytest.raises((APIError, EncryptionError)):
                c.chat.completions.create(model=ROUTED, messages=[{"role": "user", "content": "x"}])
        assert seen[-1] == expected


def test_the_salted_request_digest_is_the_routers():
    inner = b'{"model":"deepseek/deepseek-v4.1-flash","messages":[{"role":"user","content":"a,b"}],"stream":false}'
    salt, body = _salted_body(inner)
    assert len(salt) == 16 and body == b'{"lebrel_receipt_salt":"' + salt.hex().encode() + b'",' + inner[1:]
    assert _salted_request_digest(salt, body) == hashlib.sha256(salt + inner).hexdigest()


def test_a_rotated_key_makes_the_client_attest_again_once():
    attestations = []
    envelopes = [route_metadata(), route_metadata()]
    other = ServerIdentity.from_public_key_bytes(bytes([6]) * 32)
    other_config = other.marshal_public_config()
    rotated = route_metadata(hpkeConfig=base64.b64encode(other_config).decode(), keyId=hashlib.sha256(other_config).hexdigest())

    def handler(request):
        if request.url.path.endswith("/attestation"):
            attestations.append(request.url.params["nonce"])
            return streamed(200, {"kind": "lebrel-router-attestation"})
        if request.url.path.endswith("/lebrel-encryption"):
            return streamed(200, envelopes.pop(0) if envelopes else rotated)
        return streamed(503, {"error": {"code": "upstream_unavailable"}})

    # The first attestation serves two requests (cached); the re-attestation after the rotation binds the new key.
    bound = [attested(), attested(encryption_key_id=hashlib.sha256(other_config).hexdigest(), encryption_public_key=bytes([6]) * 32)]
    with client(handler, lambda *a, **k: bound.pop(0)) as c:
        for _ in range(2):
            with pytest.raises(APIError):
                c.chat.completions.create(model=ROUTED, messages=[{"role": "user", "content": "x"}])
        assert len(attestations) == 1, "within five minutes the attestation is reused"
        # The key document now names a rotated key: the cached attestation is dropped and the machine attested again.
        with pytest.raises(APIError):
            c.chat.completions.create(model=ROUTED, messages=[{"role": "user", "content": "x"}])
        assert len(attestations) == 2
        assert bound == []


def test_a_routed_receipt_is_refused_explicitly_by_this_release():
    with client(lambda request: httpx.Response(500), lambda *a, **k: attested()) as c:
        with pytest.raises(ValueError, match="attested router"):
            c.receipt("req_0123456789abcdef0123456789abcdef")
