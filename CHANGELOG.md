# Changelog

## 7.4.9.15 - 2026-08-09

- Ship the canonical model/thinking compatibility JSON resource.
- Export helpers for compatibility lookup, defaults, validation, aliases, API metadata, and provider translation.
- Propagate validated thinking levels through Vault chat calls into OpenAI, Anthropic, xAI, and Gemini SDK requests.
- Preserve backward compatibility when callers omit `thinking_level`.
