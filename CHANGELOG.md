# Changelog

## 7.4.9.22 - 2026-09-06

- Add stable OpenAI `gpt-6-astra` with its 1.05M context, 128k output, multimodal/tool metadata, and `none` through `max` reasoning-effort contract; make it the OpenAI default.
- Add public Anthropic `claude-fable-5-1` with 1M context, 128k output, adaptive thinking, hosted browser/computer tools, and `low` through `max` effort.
- Add GA Google `gemini-3.8-flash` with 1,048,576-token context, 65,536-token output, multimodal/tool metadata, and `minimal`/`low`/`medium`/`high` thinking; make it the Google default and `gemini-latest` target.
- Preserve provider-default omission behavior and all prior stable/legacy IDs while refreshing first-party catalog provenance.

## 7.4.9.21 - 2026-08-30

- Correct xAI Grok 4.3 to expose its official `none`/`low`/`medium`/`high` reasoning-effort contract and current text+image, function-calling, and structured-output metadata.
- Correct Gemini 3.7 Flash's documented provider default thinking level from `high` to `medium` while preserving omission as a provider-default no-op.
- Refresh first-party catalog provenance after the August 30 provider audit.

## 7.4.9.20 - 2026-08-23

- Record Anthropic's current browser and computer-use tool compatibility for Claude Fable 5, Opus 5, Sonnet 5, Opus 4.8, and the `claude-latest` alias.
- Preserve invitation-only Mythos 5 exclusion and older Claude legacy-tool behavior.
- Refresh supported-model documentation to the verified public package and live catalog state.

## 7.4.9.19 - 2026-08-16

- Add xAI `grok-4.6` with its 500k context, multimodal/tool metadata, and `low`/`medium`/`high`/`xhigh` reasoning contract.
- Add Google `gemini-3.7-flash` with its 1,048,576-token input limit, 65,536-token output limit, multimodal/tool metadata, and `low`/`medium`/`high` thinking contract.
- Make Grok 4.6 and Gemini 3.7 Flash the provider defaults while preserving older stable and legacy model IDs.
- Integrate the package-owned model/thinking catalog with the latest historical vector-index compatibility fix on `main`.

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
