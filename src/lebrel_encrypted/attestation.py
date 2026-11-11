"""Hardware attestation of Lebrel's router, verified in your process before a routed model is used.

A routed model (DeepSeek, GLM, Kimi, Qwen through Lebrel) is served by Lebrel's router inside an
Intel TDX confidential VM. Before the client encrypts a request to that router it asks the machine
for a quote with a fresh nonce and checks, here, with the same steps as
https://github.com/LEBREL-AI/router/blob/main/scripts/verify-attestation.py:

1. Intel's signature chain over the quote and the platform's TCB status (``dcap-qvl``, installed
   with ``lebrel-encrypted[attestation]``); without it the chain is not checked and the client
   refuses unless told to proceed.
2. The quote's report data commits to the receipt-signing key, the encryption key and our nonce,
   each behind its one-byte length (``SHA-512("lebrel-router-attestation-v3" || len || public_key
   || len || encryption_public_key || len || nonce)``), and each key id is the digest of its key.
3. Every runtime event of the log recomputes from its payload and the log replays into the RTMRs
   the quote carries; only then are the measured app id, compose hash, instance id and key provider
   believed.
4. The measured app id and key provider equal the pinned ones; the OS measurements equal the
   pinned reference (values observed on Lebrel's machines, see the router's RELEASES.md).
5. The measured compose hash is one of the releases this client knows (``PINNED_COMPOSE_HASHES``,
   the rows of the router's RELEASES.md at the time of this release): the app owner can register
   other composes under the same app id, and a quote for one of those is refused. Pass your own
   ``allowed_compose_hashes`` after checking RELEASES.md yourself, or ``None`` to leave the compose
   unpinned (then ``compose_pinned`` is reported as not checked).
6. ``mr_config_id`` recomputes from the measured app id, compose hash, instance id and KMS.

What this proves: the key the client encrypts to was born at random inside a measured, published
build of the router on genuine Intel hardware, and exists nowhere else (never in a key service: the
router's KMS derives keys per app, so a key taken from it could be derived again by any compose the
app's owner registers). The receipt-signing key is such an app key, derived inside the machine for
Lebrel's router app: a receipt proves it was signed by that app, and the attestation names the
measured compose that served the request. What it does not prove: what the lab at the other end
does with the request.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ._vendor.ehbp.errors import EHBPError
from ._vendor.ehbp.identity import ServerIdentity

DOMAIN_V3 = b"lebrel-router-attestation-v3"
QUOTE_HEADER = 48
TD_REPORT_OFFSETS = {
    "mr_td": (136, 48), "mr_config_id": (184, 48), "rtmr0": (328, 48), "rtmr1": (376, 48),
    "rtmr2": (424, 48), "rtmr3": (472, 48), "report_data": (520, 64),
}
TD_REPORT10_SIZE = 584
TEE_TYPE_TDX = 0x81
MAX_DOCUMENT_BYTES = 512 * 1024

# Lebrel's router app on Phala Cloud (dstack) since 6 November 2026: the app created under Phala's
# replacement KMS ("Phala KMS 2026-11") after the platform incident of that day, the KMS that derives
# its keys, and the OS measurements as observed on its quotes. The earlier app (4d72eae6…, rows 1.1.0
# to 1.3.7 of the router's RELEASES.md) is retired: its keys were derived under the KMS whose root
# key may have leaked, and this release no longer accepts it.
PINNED_APP_ID = "40b35a054d2eff752b1b5c3e4dee7253ab941f68"
PINNED_KMS_ID = ("3059301306072a8648ce3d020106082a8648ce3d03010703420004e8e325a477b0bf22800dc927a46c68638938d283fe73d1"
                 "f9028eeac7028193f3470122193875f598b7addc503e534712ab437b4120776bbac32ddf8897f844ee")
PINNED_OS = {
    "os_image_hash": "ec43cb8c5e04b6bb6c53b33b16076a01d83d01f7f4b703f2f78a0d947d6247e6",
    "mr_td": "2a674327c50218dba880066b349b8d559d749ed68dce33fd651c184a877d084b07a9e583767a7ad5da13ed91deec2b70",
    "rtmr0": "fb14dc139f33d6fcf474bc8332cac001259fb31cfbcb6b34d4ceeb552a2c4466884a0cbde45ad98a05c5c060c23ad65a",
    "rtmr1": "89ef86c5125b986c325f0a94acfa87718a4ef3a4d4f237dbb72dda3cf777fd2c054555fb4407eb53338d7d67329a7333",
    "rtmr2": "e9000e338f33233e8f51b7fd031e1c158bdcf8572eab041c0bc8eb7fef48b23259a6a556e6c8e2af99999fc39c793179",
}


# The router composes this release knows: the rows of https://github.com/LEBREL-AI/router/blob/main/RELEASES.md
# deployed under the pinned app when it was published (instance A2, the attested endpoint attested.lebrel.ai).
PINNED_COMPOSE_HASHES: Tuple[str, ...] = (
    "b7a8b290d061e36d6f92ee461acf094d87bafdcc90e65abd6e3f4d5d72012ded",  # 1.3.8, instance A2 (attested.lebrel.ai; serving since 11 November 2026)
    "2bcd6faeb21e6c6edb805f923107bca8f3a210a04049361ecce355d2f9316465",  # 1.3.7, instance A2 (attested.lebrel.ai, TLS in the router; Phala KMS 2026-11)
)


def report_data(public_key: bytes, encryption_key: bytes, nonce: bytes) -> bytes:
    """What the quote's report data must equal (v3): the domain, then each field behind its length."""
    digest = hashlib.sha512(DOMAIN_V3)
    for field in (public_key, encryption_key, nonce):
        digest.update(bytes([len(field)]) + field)
    return digest.digest()


class AttestationError(Exception):
    """The machine did not prove what it claims; the request was not sent."""


@dataclass(frozen=True)
class Check:
    id: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class AttestedRouter:
    """What a verified attestation says about the router: the keys the quote binds and the measured identity."""
    receipt_public_key: bytes
    receipt_key_id: str
    encryption_key_id: str
    encryption_public_key: bytes
    app_id: str
    compose_hash: str
    instance_id: str
    chain_checked: bool
    checks: Tuple[Check, ...]


def td_report(quote: bytes) -> Dict[str, bytes]:
    if len(quote) < QUOTE_HEADER + 6:
        raise AttestationError("quote too short")
    version = int.from_bytes(quote[0:2], "little")
    if int.from_bytes(quote[4:8], "little") != TEE_TYPE_TDX:
        raise AttestationError("not a TDX quote")
    if version == 4:
        start = QUOTE_HEADER
    elif version == 5:
        if int.from_bytes(quote[QUOTE_HEADER:QUOTE_HEADER + 2], "little") not in (2, 3):
            raise AttestationError("quote v5 body is not a TD report")
        start = QUOTE_HEADER + 6
    else:
        raise AttestationError("unsupported quote version")
    body = quote[start:start + TD_REPORT10_SIZE]
    if len(body) < TD_REPORT10_SIZE:
        raise AttestationError("quote truncated")
    return {name: body[offset:offset + size] for name, (offset, size) in TD_REPORT_OFFSETS.items()}


def event_digest(event: Dict[str, Any]) -> bytes:
    payload = str(event.get("event_payload", ""))
    try:
        payload_bytes = bytes.fromhex(payload)
    except ValueError:
        payload_bytes = payload.encode()
    return hashlib.sha384(struct.pack("<I", int(event.get("event_type", 0))) + b":" + str(event.get("event", "")).encode() + b":" + payload_bytes).digest()


def replay(event_log: List[Dict[str, Any]]) -> Tuple[Dict[int, bytes], List[str]]:
    rtmrs = {i: bytes(48) for i in range(4)}
    unverified: List[str] = []
    for event in event_log:
        imr = int(event.get("imr", -1))
        digest = bytes.fromhex(str(event.get("digest", "")))
        if imr not in rtmrs or len(digest) != 48:
            raise AttestationError("event log malformed")
        if imr == 3 and event_digest(event) != digest:
            unverified.append(str(event.get("event", "?")))
        rtmrs[imr] = hashlib.sha384(rtmrs[imr] + digest).digest()
    return rtmrs, unverified


def verified_payload(event_log: List[Dict[str, Any]], name: str) -> Optional[str]:
    for event in event_log:
        if int(event.get("imr", -1)) == 3 and event.get("event") == name:
            return str(event.get("event_payload", "")) if event_digest(event) == bytes.fromhex(str(event.get("digest", ""))) else None
    return None


def _chain(quote: bytes) -> Tuple[Optional[bool], str]:
    """Intel's chain and the TCB status through dcap-qvl: (True, detail) verified, (False, detail)
    failed, (None, detail) not checkable here."""
    try:
        import dcap_qvl  # type: ignore
    except ImportError:
        return None, "dcap-qvl is not installed: pip install 'lebrel-encrypted[attestation]'"
    try:
        report = _run(dcap_qvl.get_collateral_and_verify(quote, None))
    except Exception as error:  # noqa: BLE001
        return False, f"quote verification failed: {type(error).__name__}"
    status = str(getattr(report, "status", "") or "")
    if status != "UpToDate":
        return False, f"TCB status {status or 'unknown'}"
    return True, "signature chain ok, TCB UpToDate"


def _run(awaitable: Any) -> Any:
    """Runs dcap-qvl's coroutine to completion, from inside or outside an event loop."""
    import asyncio
    import concurrent.futures
    if not asyncio.iscoroutine(awaitable):
        return awaitable
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, awaitable).result()


def verify_attestation(document: Dict[str, Any], nonce: bytes, *, allow_unverified_chain: bool = False, pinned_app_id: str = PINNED_APP_ID,
                       pinned_kms_id: str = PINNED_KMS_ID, pinned_os: Optional[Dict[str, str]] = PINNED_OS,
                       allowed_compose_hashes: Optional[Iterable[str]] = PINNED_COMPOSE_HASHES) -> AttestedRouter:
    """Verifies the router's attestation document for our nonce; raises AttestationError otherwise."""
    checks: List[Check] = []

    def check(id_: str, ok: bool, detail: str) -> None:
        checks.append(Check(id_, ok, detail))
        if not ok:
            raise AttestationError(f"{id_}: {detail}")

    try:
        scheme = str(document.get("scheme", ""))
        if "attestation-v3" not in scheme:
            raise AttestationError("the document is not a v3 attestation (an older router: its fields are bound without their lengths)")
        public_key = bytes.fromhex(str(document.get("public_key", "")))
        encryption_key = bytes.fromhex(str(document.get("encryption_public_key", "")))
        encryption_key_id = str(document.get("encryption_key_id", ""))
        receipt_key_id = str(document.get("signing_key_id", ""))
        quote = bytes.fromhex(str(document.get("quote", "")))
        if len(public_key) != 32 or len(encryption_key) != 32 or len(encryption_key_id) != 64 or len(receipt_key_id) != 64:
            raise AttestationError("the document names no usable keys")
        report = td_report(quote)
        expected = report_data(public_key, encryption_key, nonce)
        check("report_data", report["report_data"] == expected and document.get("report_data") == expected.hex(), "binds the receipt key, the encryption key and our nonce, each behind its length")
        bound_config = ServerIdentity.from_public_key_bytes(encryption_key).marshal_public_config()
        check("key_ids", receipt_key_id == hashlib.sha256(public_key).hexdigest() and encryption_key_id == hashlib.sha256(bound_config).hexdigest(),
              "each key id is the digest of the key the quote binds")
        event_log = document.get("event_log")
        if isinstance(event_log, str):
            event_log = json.loads(event_log)
        if not isinstance(event_log, list):
            raise AttestationError("no event log")
        rtmrs, unverified = replay(event_log)
        check("rtmr_replay", all(rtmrs[i] == report[f"rtmr{i}"] for i in range(4)), "event log replays into the quote's RTMRs")
        check("event_digests", not unverified, "every runtime event recomputes from its payload" if not unverified else "unverified: " + ", ".join(unverified))
        app_id = verified_payload(event_log, "app-id")
        compose = verified_payload(event_log, "compose-hash")
        instance_id = verified_payload(event_log, "instance-id")
        os_image = verified_payload(event_log, "os-image-hash")
        key_provider = verified_payload(event_log, "key-provider")
        info = document.get("info") or {}
        check("app_id", app_id is not None and app_id == pinned_app_id and info.get("app_id") == app_id, f"measured app id {app_id}")
        check("compose_hash", compose is not None and len(compose) == 64 and info.get("compose_hash") == compose, f"measured compose {compose}")
        if allowed_compose_hashes is None:
            checks.append(Check("compose_pinned", False, "not checked: no list of composes was given"))
        else:
            allowed = {str(h).lower() for h in allowed_compose_hashes}
            check("compose_pinned", str(compose).lower() in allowed,
                  f"measured compose {compose} is a release this client knows" if str(compose).lower() in allowed
                  else f"measured compose {compose} is not a release this client knows: upgrade lebrel-encrypted, or check the router's RELEASES.md and pass allowed_compose_hashes")
        check("instance_id", instance_id is not None and len(instance_id) == 40, f"measured instance {instance_id}")
        provider = json.loads(bytes.fromhex(key_provider).decode()) if key_provider else {}
        check("kms", provider.get("name") == "kms" and str(provider.get("id", "")).lower() == pinned_kms_id.lower(), "key provider is the pinned KMS")
        if pinned_os:
            check("os_image_hash", os_image == pinned_os.get("os_image_hash"), f"measured os image {os_image}")
            for field in ("mr_td", "rtmr0", "rtmr1", "rtmr2"):
                check(field, report[field].hex() == pinned_os.get(field, "").lower(), "equals the pinned reference (observed on Lebrel's machines)")
        record = {"app_id": app_id, "compose_hash": compose, "init_script_hashes": [], "instance_id": instance_id,
                  "key_provider": provider.get("name"), "key_provider_id": provider.get("id"), "version": 3}
        jcs = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        expected_config = (b"\x03" + hashlib.sha256(b"dstack-mr-config-v3:" + b"\x00" + jcs).digest()).ljust(48, b"\x00")
        check("mr_config_id", report["mr_config_id"] == expected_config, "recomputes from app id, compose, instance and KMS")
        chain, detail = _chain(quote)
        if chain is None:
            checks.append(Check("intel_chain", False, detail))
            if not allow_unverified_chain:
                raise AttestationError("Intel's chain was not checked: " + detail)
        else:
            check("intel_chain", chain, detail)
        return AttestedRouter(public_key, receipt_key_id, encryption_key_id, encryption_key, str(app_id), str(compose), str(instance_id), chain is True, tuple(checks))
    except AttestationError:
        raise
    except (ValueError, TypeError, KeyError, json.JSONDecodeError, EHBPError):
        raise AttestationError("the attestation document is malformed") from None
