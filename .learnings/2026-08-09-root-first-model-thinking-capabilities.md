# Canonical model-compatible thinking levels

## Invariant

`vectorvault/model_catalog.json` is the only model and thinking-capability source of truth shipped by the Python package. Provider model updates must be verified against first-party documentation and applied to that catalog. `vectorvault.ai` derives its legacy model registries from the catalog so existing imports remain compatible. API and UI repositories must consume the public catalog helpers rather than maintain independent compatibility lists.

The release order is **root first**: publish and verify the base `vector-vault` package and its catalog before any API or frontend release consumes the new schema/helpers. Downstream code must pin or otherwise prove it is running the verified root package artifact. Never deploy a downstream selectable level before the root package can validate and translate it.

## First-party capability sources (verified 2026-08-09)

- OpenAI reasoning effort and model-dependent levels/defaults: https://developers.openai.com/api/docs/guides/reasoning
- OpenAI GPT-5.6 model/alias and supported effort values: https://developers.openai.com/api/docs/guides/latest-model
- OpenAI Chat Completions `reasoning_effort` request field: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
- Anthropic effort compatibility, levels, and defaults: https://platform.claude.com/docs/en/build-with-claude/effort
- Anthropic adaptive-thinking request shape: https://platform.claude.com/docs/en/build-with-claude/thinking
- Gemini thinking levels and per-model defaults: https://ai.google.dev/gemini-api/docs/thinking
- xAI Grok 4.5 reasoning effort, levels, and default: https://docs.x.ai/developers/model-capabilities/text/reasoning

Gemini 2.5 exposes numeric thinking budgets, but the first-party documentation does not define a canonical mapping from generic `low` / `medium` / `high` labels to exact budget values. Those models therefore remain cataloged and runtime-supported but expose no selectable generic `thinking_level`; guessing budget numbers is forbidden.

## Compatibility rules

- Omitted `thinking_level` is a true no-op. No provider thinking field is emitted, preserving historical provider defaults and private/fine-tuned model support.
- Explicit levels are normalized and validated before an SDK call. Unknown models and unsupported model/level pairs raise `ModelCapabilityError`.
- Model aliases resolve to one concrete capability row before translation.
- OpenAI and proven xAI paths use `reasoning_effort`; Anthropic uses `output_config.effort` and adaptive thinking where supported; Gemini 3 uses `thinking_config.thinking_level`.
- Sampling parameters are omitted when an explicit capability marks them incompatible.
- Both streaming and non-streaming call paths must use the same translator.
- If first-party semantics for a provider/model cannot be proven, keep the model in the catalog with an explicit unsupported contract rather than inventing selectable levels or provider kwargs.

## Release checklist

1. Update and validate `model_catalog.json` from the first-party sources above.
2. Run catalog/resource/registry-coverage, provider-translation, Vault propagation, and legacy-omission tests.
3. Build both wheel and source distribution and inspect every member for `vectorvault/model_catalog.json`, junk, and accidental secrets.
4. Run metadata checks and clean-install the wheel in a new isolated environment; smoke-test public imports and packaged-resource reads.
5. Publish only the exact approved root artifact, verify PyPI metadata/artifacts and clean installation from PyPI, then authorize downstream API/frontend consumption.
