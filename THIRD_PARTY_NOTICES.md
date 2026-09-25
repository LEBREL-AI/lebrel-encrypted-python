# Third-party notices

The entire `src/lebrel_encrypted/_vendor/ehbp` Python module is copied unchanged
from https://github.com/tinfoilsh/encrypted-http-body-protocol at commit
`6ae53f8b6270834ca1cb8c29f8d2a79e372bc4b0`, directory `python/src/ehbp`.
The sole added file inside that directory is the upstream repository's LICENSE.
Its original relative imports preserve the upstream implementation.

Copyright (c) 2025 Tinfoil, Inc. — MIT License.

The full license appears in `src/lebrel_encrypted/_vendor/ehbp/LICENSE` and must
remain included with redistributions. `VENDORED_SHA256.json` records the exact
source hashes for comparison to that public revision.

The SDK separately depends on pyhpke (MIT), cryptography
(Apache-2.0 OR BSD-3-Clause), and HTTPX (BSD-3-Clause). These distributions are
installed from their pinned packages and retain their respective notices.
