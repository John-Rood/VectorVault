# Canonical model-compatible thinking levels

## Invariant

`vectorvault/model_catalog.json` is the only model and thinking-capability source of truth shipped by the Python package. Provider model updates must be verified against first-party documentation and applied to that catalog. `vectorvault.ai` derives its legacy model registries from the catalog so existing imports remain compatible. API and UI repositories must consume the public catalog helpers rather than maintain independent compatibility lists.

## Compatibility rules

- Omitted `thinking_level` is a true no-op. No provider thinking field is emitted, preserving historical provider defaults and private/fine-tuned model support.
- Explicit levels are normalized and validated before an SDK call. Unknown models and unsupported model/level pairs raise `ModelCapabilityError`.
- Model aliases resolve to one concrete capability row before translation.
- OpenAI and xAI use `reasoning_effort`; Anthropic uses `output_config.effort` and adaptive thinking where supported; Gemini uses `thinking_config.thinking_level`.
- Sampling parameters are omitted when an explicit capability marks them incompatible.
- Both streaming and non-streaming call paths must use the same translator.

## Release checklist

1. Update and validate `model_catalog.json` from first-party provider documentation.
2. Run the catalog, provider-translation, Vault propagation, and package-content tests.
3. Build both wheel and source distribution and inspect them for `vectorvault/model_catalog.json` and accidental secrets.
4. Clean-install the wheel in a new environment and smoke-test public imports and packaged-resource reads.
5. Publish the exact approved version, verify PyPI metadata/artifacts, and clean-install from PyPI.
