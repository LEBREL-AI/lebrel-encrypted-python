# Lebrel encrypted Python client

Questions are encrypted locally for the verified runtime. Answers are decrypted
locally. This client serves **`lebrel/deepseek-v4-flash-uncensored`** and verifies
Lebrel's pinned Ed25519 signature before encrypting anything for a runtime key.

Requires Python 3.9 or newer.

```sh
python -m pip install https://lebrel.ai/sdk/lebrel-encrypted-python.zip
export LEBREL_API_KEY='your-Lebrel-API-key'
```

The same archive can be reviewed, extracted and installed locally with
`python -m pip install .`. The import is `lebrel_encrypted`; there is no need for
the OpenAI package. The interface accepts OpenAI chat-completion parameters and
returns dictionaries.

```python
from lebrel_encrypted import Lebrel

with Lebrel() as client:  # reads LEBREL_API_KEY
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Write the opening of a story."}],
        max_tokens=256,
    )
    print(response["choices"][0]["message"]["content"])
```

The model defaults to the exact Flash edition above. Another model ID is
rejected locally. You can also pass `Lebrel(api_key="...")` or use the shorter
`client.create(...)` interface.

## OpenCode and other OpenAI-compatible applications

Run the local adapter in a terminal:

```sh
python -m lebrel_encrypted.proxy --port 11437
```

Configure your application with:

- Base URL: `http://127.0.0.1:11437/v1`
- API key: your usual Lebrel API key
- Model: `lebrel/deepseek-v4-flash-uncensored`

The adapter receives normal OpenAI JSON and SSE on loopback, then uses this SDK
to encrypt requests to the public API and decrypt responses. It binds only to
`127.0.0.1`; there is no network-bind option. The local hop is plaintext, so other
processes with sufficient access on your computer remain within your trust
boundary. Requests require `Authorization: Bearer ...` and JSON bodies with
`Content-Length`. Browser-origin requests are rejected.

The adapter does not persist API keys or conversations. By default the key comes
from each application's request. Optional standalone use can read the key from
`--api-key-file /private/path/to/key`; an incoming Bearer key takes precedence.
The file must contain exactly one key. Do not put the key itself on the command
line. The authenticated `/v1/models` route forwards metadata only for the exact
Flash model. Stream disconnects close the encrypted upstream response.

`GET http://127.0.0.1:11437/healthz` requires no key and identifies the local
adapter, SDK version, upstream, signing-key ID and loaded proxy source hash.
Startup emits one status line; request bodies and credentials are never logged.
The proxy runs in the foreground until stopped. Installing this package does
not automatically start it or modify application configuration.

The console entry point `lebrel-encrypted proxy --port 11437` is equivalent.

## Terminal requests

Put a normal OpenAI completion request in a local JSON file, then:

```sh
export LEBREL_API_KEY='your-Lebrel-API-key'
lebrel-encrypted infer < request.json
```

Equivalent: `python -m lebrel_encrypted infer < request.json`. The command reads
JSON from stdin and the key from the environment. It writes a JSON response, or
SSE when the input sets `"stream": true`. It never silently switches to the plain
public API. On failure it exits nonzero and prints a generic error to stderr.

## Streaming and cancellation

```python
from lebrel_encrypted import Lebrel

with Lebrel() as client:
    with client.chat.completions.create(
        messages=[{"role": "user", "content": "Explain how HPKE works."}],
        max_tokens=512,
        stream=True,
    ) as stream:
        for chunk in stream:
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                print(delta.get("content") or "", end="", flush=True)
        assert stream.completed
```

Fully exhaust the stream to verify its completion. It requires a final
`finish_reason` for every observed choice, then an authenticated `[DONE]`, and
rejects subsequent data or truncated encrypted frames. Partial output may have
been displayed before a later error: do not treat it as a completed answer.
`stream.close()` or leaving the context closes the HTTP response and propagates
cancellation. Closing early leaves `stream.completed` false.

## Receipts: proof of what answered

The runtime signs a receipt for every answer: the SHA-256 of your request exactly as
sent, the SHA-256 of the answer, the serving manifest it ran under (which weights,
precision, engine and tokenizer) and the token counts. The client checks it for you,
with the same checks as https://lebrel.ai/verify:

```python
from lebrel_encrypted import Lebrel

with Lebrel() as client:
    answer = client.chat.completions.create(messages=[{"role": "user", "content": "Name one color."}])
    receipt = client.receipt(answer)
    assert receipt.verified
    print(receipt.request_id, receipt.manifest.payload["weights"]["revision"])
```

`receipt.checks` lists every step: the pinned signing key, the signature, the request
id, the digest of your request, the digest of the answer and the serving manifest the
receipt names, fetched and checked at the same time (`manifest=False` skips that
fetch). A stream is checked the same way once fully consumed: `client.receipt(stream)`.
Any receipt can be looked up later by its request id while the API keeps it:
`client.receipt("…")`; `client.manifest()` returns the current serving manifest,
checked. Every completion carries `request_id`, `request_sha256` and
`response_sha256`; the dictionary itself is unchanged.

The local proxy sends `X-Lebrel-Request-Id` on every answer and, on non-streaming
answers, `Proof-Of-Edition-Receipt` (the signed receipt, base64) as the API sent it.
`GET /v1/receipts/{id}` on the proxy returns the receipt with
`X-Lebrel-Receipt-Check: verified` once the proxy has checked its signature and the
serving manifest; the digests are yours to check against your text.

## Errors and retries

```python
from lebrel_encrypted import APIError, EncryptionError, Lebrel, StreamError, TransportError

try:
    with Lebrel() as client:
        response = client.create(messages=[{"role": "user", "content": "Hello"}])
except APIError as error:
    print(error.status_code, error.code)
except (EncryptionError, StreamError, TransportError) as error:
    print(str(error))
```

The client never automatically retries inference. A connection failure may occur
after execution has begun, so a fresh application retry can incur a new charge.
Each explicit request gets a new UUID in `X-Lebrel-Request-Id` and a fresh HPKE
encapsulation. Redirects, plaintext successful responses and fallback to the
plain API are prohibited. Error messages omit conversation content and ignore
untrusted plaintext error bodies.

The default timeout is 900 seconds for cold starts; configure `Lebrel(timeout=...)`
when needed. Environment HTTP proxy settings are disabled. HTTPS certificate
verification remains enabled.

## What is verified

The client retrieves `https://api.lebrel.ai/.well-known/lebrel-encryption` without
sending the API key, verifies the Ed25519 signature on the original decoded
payload bytes and checks the schema, exact model, signing-key digest, HPKE
configuration digest, freshness and validity interval (at most ten minutes).
The request body is encrypted using the verified HPKE configuration. API
credentials and routing metadata are still HTTP headers protected by HTTPS.

Pinned Ed25519 public key (raw bytes, base64):

```text
beFZtSwt6FnlhIYbX636n7w3/gpaASIkRnIMx52XJwk=
```

SHA-256:

```text
8f72beb9680a0f1911dce59d1fc103a0a89039998003715d35422d3020e5292d
```

The runtime signature authenticates Lebrel's encryption key. It is not hardware
attestation or proof that the operator cannot access the model process's memory.
Questions and answers are decrypted inside the inference runtime. The current
standard Modal runtime remains administered by Lebrel.

## Reused implementation and dependencies

The EHBP Python source is vendored **without changes** from the official Tinfoil
repository at commit `6ae53f8b6270834ca1cb8c29f8d2a79e372bc4b0`, under
`src/lebrel_encrypted/_vendor/ehbp`. Relative imports remain unchanged. Its MIT
license accompanies the source. The wrapper supplies application checks; HPKE,
HKDF and response encryption/decryption use upstream EHBP and its dependencies.

Exact direct dependency versions:

- `pyhpke==0.6.3`
- `cryptography==46.0.7` (latest compatible 46.x patch selected for pyhpke's `<47` bound)
- `httpx==0.28.1`

Tests use `pytest==8.4.2`; builds use `hatchling==1.27.0`. The archive includes
`VENDORED_SHA256.json` so reviewers can compare vendored files to the pinned
upstream revision.

## Offline verification

```sh
python -m pip install '.[test]'
python -m pytest
```

The interoperability fixture builds a local Go EHBP server with the same pinned
upstream revision (`GOTOOLCHAIN=go1.26.0`) and verifies real Python-to-Go encrypted
requests, JSON responses, streaming, truncation and failure handling. No GPU,
production credentials or production API calls are used. Dependency and Go
toolchain downloads are needed once unless already cached.
