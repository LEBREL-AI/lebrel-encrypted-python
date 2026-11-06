# Changelog

All notable changes to `lebrel-encrypted`. Versions follow [Semantic Versioning](https://semver.org/).

## 0.5.1 — 2026-11-06

- Pinned to router 1.3.6 as well (LEBREL-AI/router, tag `v1.3.6`, image
  `sha256:d3725f05e6017caee09adeb431c2b2e3d70b010c92cfc99f8bb4c55bd26c8b54`): compose hashes
  `aa4f6387441feb40048718db273a58ab4c0457d8832580fbaff0470d9f44d835` (instance A, router-tee.lebrel.ai) and
  `380bf44b9121927380ff25aa63ae0b787dfdce171de69e830ae4ccf4878205c5` (the replica). Router 1.3.6 changes only
  the routes' prices (Lebrel's published rates, contract router-v3); the 1.3.5 composes stay pinned.

## 0.5.0 — 2026-11-05

- Pinned to router 1.3.5 (LEBREL-AI/router, tag `v1.3.5`, image
  `sha256:b31316b3d211ded8cf320a87f8c02334f787cd0b9113560274e98eeb8800345a`): the measured compose
  of the router that serves a route must be one of `675613867a396b1a453c8d6315b48b343fc57aa41ee4693a701075bb8a65b58a`
  (router-tee.lebrel.ai) or `dabc6a3469d4d9c773e273479397906b4bd1737f437b107a90d1a3ef6df85383` (the
  replica), the rows of that version in the router's RELEASES.md. A later router release needs a
  client release with its composes, or `allowed_compose_hashes` passed by hand after checking the
  ledger. Every routed request carries a salt and the moment of sending inside the ciphertext, as
  router 1.3.3 and later require.
- Routed models (DeepSeek, GLM, Kimi and Qwen through Lebrel) are encrypted end to end, like the
  edition: `create(model="deepseek/deepseek-v4.1-flash", ...)` attests the router that serves the
  route, verifies the route's signed key document with the receipt key the hardware quote binds,
  encrypts the request to that key and names the model in `X-Lebrel-Model`; the answer, streamed or
  not, is decrypted in your process. The local proxy accepts any routed model id.
- `lebrel_encrypted.attestation`: the router's Intel TDX quote is verified in your process (report
  data bound to our nonce and both keys, event log recomputed and replayed, app id and KMS pinned,
  OS measurements compared with the pinned reference, `mr_config_id` recomputed); Intel's signature
  chain and the TCB status through `dcap-qvl`, installed with `pip install 'lebrel-encrypted[attestation]'`.
  Without it the client refuses routed models unless `allow_unverified_chain=True` is passed.
- `routed_model_id()` tells a routed model id from the edition's.
- The proxy turns the edition's thinking on (`thinking: true`) when a request says nothing about
  reasoning: coding tools never do, and the edition thinks only when asked. A request that names
  `thinking`, `profile`, `chat_template_kwargs`, `reasoning_effort` or `custom_params` is forwarded as it is.

## 0.4.2 — 2026-10-29

- The local proxy and the command line answer with the platform's own reason when a request cannot be
  served (`error.code` and a plain message): in line and the wait ran out, line full, place in line lapsed,
  model paused, too many requests at once, not enough balance, platform or node unreachable. A generic
  failure message is no longer the only thing an application sees.
- `service_tier="priority"` on `create()` sends `X-Lebrel-Service-Tier: priority`; the proxy forwards the
  OpenAI `service_tier` parameter the same way. Priority is an account setting (Dashboard → Billing); the
  parameter is accepted as well.

## 0.4.1 — 2026-10-26

- README: where hardware attestation stands, and what the runtime signature proves until then.

## 0.4.0 — 2026-10-15

- The local proxy speaks OpenAI Responses (`/v1/responses`) and Anthropic Messages (`/v1/messages`,
  `/v1/messages/count_tokens`) beside OpenAI chat completions, every format ending as the same encrypted
  completion. Function tools reach the model; hosted and namespaced tools are refused.
- `--max-output-tokens` on the proxy caps what clients ask for; read from the public catalog when omitted.

## 0.3.0 — 2026-10-08

- The client serves DeepSeek V4.1 Flash · Lebrel Uncensored (`lebrel/deepseek-v4.1-flash-uncensored`).

## 0.2.0 — 2026-09-26

- First release on PyPI: encrypted chat completions, streaming with authenticated completion, the signed
  receipt of every answer fetched and checked, the local proxy for OpenAI-compatible applications, the
  `lebrel-encrypted infer` command.
