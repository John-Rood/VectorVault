# Model catalog is a routing contract, not just a selector list

## Context
The August 2026 provider refresh added current OpenAI, Anthropic, xAI, and Gemini IDs while preserving old saved integrations.

## Lessons
- Keep `*_MODELS` as the broad runtime compatibility catalog and `*_FRONT_MODELS` as the curated stable selector catalog.
- Preserve invalid, retired, or redirected saved IDs only when there is an explicit local alias to a provider-accepted upstream ID. Never send `chatgpt-latest` or undocumented GPT chat aliases upstream.
- Product defaults and provider aliases are distinct: Vector Vault defaults xAI to `grok-4.5`, while xAI's official `grok-latest` alias resolves to `grok-4.3`.
- Context windows should use exact provider values (`1_050_000`, `1_048_576`) rather than rounded marketing values when the API publishes exact limits.
- Model metadata not represented by the token-limit maps (maximum input/output, multimodal support) needs an explicit registry and tests.
- Test routing behavior as well as membership: verify aliases are rewritten before provider calls and current IDs resolve to the intended platform.
