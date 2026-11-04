"""The router's attestation, verified in-process on a document captured from Lebrel's machine."""

import copy
import json
import pathlib

import pytest

from lebrel_encrypted import attestation

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "router-attestation-a.json"


def document():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def nonce_of(doc):
    return bytes.fromhex(doc["nonce"])


def test_a_real_document_verifies_without_the_chain_when_allowed():
    doc = document()
    attested = attestation.verify_attestation(doc, nonce_of(doc), allow_unverified_chain=True)
    assert attested.app_id == attestation.PINNED_APP_ID
    assert attested.compose_hash == doc["info"]["compose_hash"]
    assert attested.encryption_key_id == doc["encryption_key_id"]
    assert attested.receipt_public_key.hex() == doc["public_key"]
    ids = [c.id for c in attested.checks]
    for required in ("report_data", "key_ids", "rtmr_replay", "event_digests", "app_id", "compose_hash", "compose_pinned", "kms", "mr_td", "rtmr0", "rtmr1", "rtmr2", "mr_config_id"):
        assert required in ids
    assert all(c.ok for c in attested.checks if c.id != "intel_chain")


def test_the_compose_is_pinned_to_the_releases_this_client_knows():
    doc = document()
    assert doc["info"]["compose_hash"] in attestation.PINNED_COMPOSE_HASHES
    with pytest.raises(attestation.AttestationError, match="not a release this client knows"):
        attestation.verify_attestation(doc, nonce_of(doc), allow_unverified_chain=True, allowed_compose_hashes=("00" * 32,))
    own = attestation.verify_attestation(doc, nonce_of(doc), allow_unverified_chain=True, allowed_compose_hashes=(doc["info"]["compose_hash"].upper(),))
    assert [c for c in own.checks if c.id == "compose_pinned"][0].ok
    unpinned = attestation.verify_attestation(doc, nonce_of(doc), allow_unverified_chain=True, allowed_compose_hashes=None)
    pinned = [c for c in unpinned.checks if c.id == "compose_pinned"][0]
    assert not pinned.ok and "not checked" in pinned.detail


def test_report_data_is_length_prefixed():
    key, other = bytes([1]) * 32, bytes([2]) * 32
    assert attestation.report_data(key, other, b"") != attestation.report_data(key + other, b"", b"")
    assert attestation.report_data(key, b"", other) != attestation.report_data(key, other, b"")


def test_the_chain_is_checked_with_dcap_qvl_when_installed():
    pytest.importorskip("dcap_qvl")
    doc = document()
    try:
        attested = attestation.verify_attestation(doc, nonce_of(doc))
    except attestation.AttestationError as error:
        if "quote verification failed" in str(error):
            pytest.skip(f"collateral not reachable: {error}")
        raise
    assert attested.chain_checked
    assert [c for c in attested.checks if c.id == "intel_chain"][0].ok


def test_without_the_chain_the_default_refuses(monkeypatch):
    doc = document()
    monkeypatch.setattr(attestation, "_chain", lambda quote: (None, "dcap-qvl is not installed"))
    with pytest.raises(attestation.AttestationError, match="chain was not checked"):
        attestation.verify_attestation(doc, nonce_of(doc))
    attested = attestation.verify_attestation(doc, nonce_of(doc), allow_unverified_chain=True)
    assert not attested.chain_checked


@pytest.mark.parametrize("tamper", [
    "nonce", "compose_payload", "compose_stated", "app_id_pin", "encryption_key", "encryption_key_id", "v2_scheme", "rtmr", "kms",
])
def test_a_tampered_document_or_the_wrong_nonce_is_refused(tamper):
    doc = document()
    nonce = nonce_of(doc)
    kwargs = {"allow_unverified_chain": True}
    if tamper == "nonce":
        nonce = bytes(32)
    elif tamper == "compose_payload":
        for event in doc["event_log"]:
            if event.get("event") == "compose-hash":
                event["event_payload"] = "ab" * 32
    elif tamper == "compose_stated":
        doc["info"]["compose_hash"] = "ab" * 32
    elif tamper == "app_id_pin":
        kwargs["pinned_app_id"] = "00" * 20
    elif tamper == "encryption_key":
        doc["encryption_public_key"] = "ab" * 32
    elif tamper == "encryption_key_id":
        doc["encryption_key_id"] = "ab" * 32
    elif tamper == "v2_scheme":
        doc["scheme"] = 'report_data = sha512("lebrel-router-attestation-v2" || public_key || encryption_public_key || nonce)'
    elif tamper == "rtmr":
        doc["event_log"].append(copy.deepcopy(doc["event_log"][-1]))
    elif tamper == "kms":
        kwargs["pinned_kms_id"] = "00" * 10
    with pytest.raises(attestation.AttestationError):
        attestation.verify_attestation(doc, nonce, **kwargs)


def test_quote_parsing_rejects_other_tees_and_truncation():
    doc = document()
    quote = bytes.fromhex(doc["quote"])
    with pytest.raises(attestation.AttestationError):
        attestation.td_report(quote[:100])
    sgx = bytearray(quote)
    sgx[4:8] = (0).to_bytes(4, "little")
    with pytest.raises(attestation.AttestationError):
        attestation.td_report(bytes(sgx))
