# Changelog

All notable changes to `lebrel-encrypted`. Versions follow [Semantic Versioning](https://semver.org/).

## 0.5.0 — 2026-11-04

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
