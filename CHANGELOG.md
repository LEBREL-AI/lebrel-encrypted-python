# Changelog

All notable changes to `lebrel-encrypted`. Versions follow [Semantic Versioning](https://semver.org/).

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
