# GPT-6 Astra advertised effort values must be wire-verified — 2026-09-13

## Discovery
OpenAI’s GPT-6 Astra model page named `low`, `medium`, `high`, `xhigh`, and `max`, while the package catalog also exposed `none`. Live Chat Completions validation on September 13, 2026 rejected both `none` and `max` and explicitly enumerated only `low`, `medium`, `high`, and `xhigh`; controlled `low` and `xhigh` calls succeeded.

## Rule
When a first-party model page conflicts with the authenticated production API for selectable reasoning values, publish only the wire-verified intersection. Keep omitted `thinking_level` as a true no-op so the provider default remains backward compatible. Tests must reject every documented-but-wire-invalid value before the SDK call.
