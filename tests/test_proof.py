import base64
import hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from lebrel_encrypted import MODEL_ID, proof

SIGNER = Ed25519PrivateKey.from_private_bytes(bytes([7]) * 32)
PUBLIC = SIGNER.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
KEY_ID = hashlib.sha256(PUBLIC).hexdigest()
OTHER = Ed25519PrivateKey.from_private_bytes(bytes([8]) * 32)


def sign(payload, signer=SIGNER):
    return proof.Signed(payload=payload, signature=base64.b64encode(signer.sign(proof.canonical(payload))).decode())


def manifest(**changes):
    payload = {
        "version": 1, "edition": {"id": MODEL_ID, "name": "Lebrel DeepSeek V4 Flash Uncensored", "base_model": "deepseek-ai/DeepSeek-V4-Flash", "fingerprint_id": None},
        "weights": {"repository": "lebrel/deepseek-v4-flash-uncensored", "revision": "a" * 64, "files": {"model.safetensors": {"sha256": "1" * 64, "size": 7}}, "total_bytes": 7},
        "quantization": {"method": "nvfp4", "weights_dtype": "fp8", "kv_cache_dtype": "fp8"}, "engine": {"name": "vllm", "version": "0.30.0", "image_digest": None},
        "tokenizer_sha256": "b" * 64, "chat_template_sha256": "c" * 64, "runtime": {"provider": "modal", "instance_id": "i" * 32, "sidecar_sha256": None},
        "attestation": None, "issued_at": 1000, "expires_at": 4600, "signing_key_id": KEY_ID,
    }
    payload.update(changes)
    return payload


def receipt(**changes):
    payload = {
        "version": 1, "request_id": "req-0123456789", "manifest_sha256": proof.manifest_identity(manifest()),
        "prompt_sha256": hashlib.sha256(b"request").hexdigest(), "response_sha256": hashlib.sha256("answer ñ".encode()).hexdigest(),
        "prompt_tokens": 3, "completion_tokens": 2, "issued_at": 1500, "instance_id": "i" * 32, "signing_key_id": KEY_ID,
    }
    payload.update(changes)
    return payload


def test_canonical_form_is_the_reference_encoding():
    assert proof.canonical({"b": 1, "a": "ñ", "c": [True, None, {"z": 0, "y": ""}]}) == '{"a":"ñ","b":1,"c":[true,null,{"y":"","z":0}]}'.encode("utf-8")
    with pytest.raises(ValueError):
        proof.canonical({"a": 1.5})
    with pytest.raises(ValueError):
        proof.canonical({"a": 2 ** 60})
    assert proof.key_id(PUBLIC) == KEY_ID


def test_manifest_identity_ignores_the_validity_window():
    assert proof.manifest_identity(manifest(issued_at=1, expires_at=2)) == proof.manifest_identity(manifest(issued_at=3, expires_at=4))
    assert proof.manifest_identity(manifest(tokenizer_sha256="d" * 64)) != proof.manifest_identity(manifest())


def test_manifest_checks_pass_and_each_failure_is_named():
    good = proof.check_manifest(sign(manifest()), PUBLIC, now=2000, edition_id=MODEL_ID)
    assert proof.all_passed(good) and [c.id for c in good] == ["shape", "key", "signature", "window", "edition"]
    failed = {
        "key": proof.check_manifest(sign(manifest(signing_key_id="f" * 64)), PUBLIC, now=2000),
        "signature": proof.check_manifest(proof.Signed(sign(manifest()).payload | {"engine": {"name": "other"}}, sign(manifest()).signature), PUBLIC, now=2000),
        "window": proof.check_manifest(sign(manifest()), PUBLIC, now=5000),
        "shape": proof.check_manifest(sign({k: v for k, v in manifest().items() if k != "attestation"}), PUBLIC, now=2000),
        "edition": proof.check_manifest(sign(manifest()), PUBLIC, now=2000, edition_id="lebrel/other"),
    }
    for name, checks in failed.items():
        assert [c.id for c in checks if not c.ok] == [name], name
    with_other_key = proof.check_manifest(sign(manifest(), OTHER), PUBLIC, now=2000)
    assert [c.id for c in with_other_key if not c.ok] == ["signature"], "the pinned key id may be claimed; the signature decides"


def test_receipt_checks_bind_request_answer_and_manifest():
    signed = sign(receipt())
    identity = proof.manifest_identity(manifest())
    known = dict(request_id="req-0123456789", manifest_identity_hex=identity, request_sha256=hashlib.sha256(b"request").hexdigest(),
                 response_sha256=hashlib.sha256("answer ñ".encode()).hexdigest())
    good = proof.check_receipt(signed, PUBLIC, **known)
    assert proof.all_passed(good) and [c.id for c in good] == ["shape", "key", "signature", "request_id", "manifest", "request", "response"]
    assert [c.id for c in proof.check_receipt(signed, PUBLIC) if not c.ok] == [] and len(proof.check_receipt(signed, PUBLIC)) == 3
    for field, wrong in (("request_id", "req-other-000000"), ("manifest_identity_hex", "0" * 64), ("request_sha256", "1" * 64), ("response_sha256", "2" * 64)):
        checks = proof.check_receipt(signed, PUBLIC, **(known | {field: wrong}))
        assert [c.id for c in checks if not c.ok] == [{"manifest_identity_hex": "manifest", "request_sha256": "request", "response_sha256": "response"}.get(field, field)], field
    assert [c.id for c in proof.check_receipt(sign(receipt(), OTHER), PUBLIC) if not c.ok] == ["signature"]
    assert [c.id for c in proof.check_receipt(sign(receipt(signing_key_id="f" * 64)), PUBLIC) if not c.ok] == ["key"]
    assert not any("request" in c.detail and "ñ" in c.detail for c in good), "details never carry conversation content"


@pytest.mark.parametrize("data", [
    b"not json", b"[]", b'{"payload":{},"signature":"AA==","extra":1}', b'{"payload":[],"signature":"AA=="}',
    b'{"payload":{},"signature":"?"}', b'{"payload":{"a":1.5},"signature":"' + b"A" * 88 + b'"}', b"{" + b" " * proof.MAX_DOCUMENT_BYTES + b"}",
])
def test_parse_signed_rejects_malformed_documents(data):
    with pytest.raises(ValueError):
        proof.parse_signed(data)


def test_parse_signed_round_trips_the_api_form():
    document = sign(receipt())
    parsed = proof.parse_signed(document.to_json().encode("utf-8"))
    assert parsed == document and proof.verify_signature(parsed, PUBLIC)
    assert not proof.verify_signature(proof.Signed(parsed.payload | {"prompt_tokens": 4}, parsed.signature), PUBLIC)
