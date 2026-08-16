# Changelog

## 7.4.9.18 - 2026-08-09

- Require Python 3.10 or newer and `google-genai>=1.56.0` so every advertised Gemini thinking level has SDK enum support.
- Serialize the frontend `default` model row with the default model's integer context window as `token_limit`.
- Add release metadata, serializer, and Gemini level regressions for the compatibility floors.

## 7.4.9.17 - 2026-08-09

- Ship the canonical model/thinking compatibility JSON resource.
- Export helpers for compatibility lookup, defaults, validation, aliases, API metadata, and provider translation.
- Propagate validated thinking levels through Vault chat calls into OpenAI, Anthropic, proven xAI, and Gemini 3 SDK requests.
- Preserve backward compatibility when callers omit `thinking_level`, including private and fine-tuned model IDs.
- Keep Gemini 2.5 generic levels unselectable rather than guessing numeric thinking-budget mappings.
