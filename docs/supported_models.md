# Supported AI Models

Vector Vault routes model IDs to OpenAI, Anthropic, xAI, or Google by catalog membership. The aliases below are the recommended VectorVault Cloud/runtime defaults as of **August 2, 2026**; explicit stable IDs are also supported. The SDK source on `main` mirrors this catalog, while the current PyPI package remains `vector-vault==7.4.9.11` until its documented release pipeline is restored; portable SDK examples therefore use `model="default"`.

| Provider | Recommended default | Current stable IDs | Context window | Max output |
| --- | --- | --- | ---: | ---: |
| OpenAI | `gpt-5.6` | `gpt-5.6`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5-mini`, `gpt-5-nano`, `o3`, `o3-pro`, `o4-mini`, `gpt-4o`, `chat-latest` | 400,000–1,050,000 | up to 128,000 |
| Anthropic | `claude-opus-5` (`claude-latest`) | `claude-fable-5`, `claude-opus-5`, `claude-sonnet-5`, supported Claude 4.x stable IDs | up to 1,000,000 | up to 128,000 |
| xAI | `grok-4.5` (product default) | `grok-4.5` (500k), `grok-4.3`, `grok-4.20`, `grok-4.20-0309-reasoning`, `grok-4.20-0309-non-reasoning`, `grok-build-0.1` (256k), `grok-latest` | up to 1,000,000 | model-dependent |
| Google | `gemini-3.6-flash` (`gemini-latest`) | `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite` | up to 1,048,576 | up to 65,536 |

OpenAI GPT-5.6 accepts text and image input and returns text through Chat Completions or Responses. Its 1,050,000-token context allows up to 922,000 input tokens and 128,000 output tokens. The `gpt-5.6` alias routes to GPT-5.6 Sol. The official `chat-latest` model has a 400,000-token context and 128,000-token output limit; the legacy `chatgpt-latest` spelling is translated locally and is never sent upstream.

Anthropic's `claude-opus-5` is the default because Anthropic recommends Opus 5 for complex work; Fable 5 remains available as the highest-capability GA model, and Sonnet 5 is the speed/intelligence option. Invitation-only `claude-mythos-5` is not listed.

Gemini image generation uses the stable backend-only `gemini-3-pro-image` model. It is not shown in text-chat selectors; the retired `gemini-3-pro-image-preview` ID is translated locally for saved integrations.

## Compatibility IDs

The SDK's backend catalog retains older provider-accepted or redirected IDs so existing integrations and saved VectorFlow graphs continue to load. Retired Claude Sonnet 4 IDs (`claude-sonnet-4-0`, `claude-sonnet-4-20250514`), redirected Grok 3/4 aliases, and legacy Gemini shorthand `gemini-3.1-pro` are compatibility-only and are intentionally absent from current model selectors. The Gemini shorthand resolves locally to the accurately labeled preview ID `gemini-3.1-pro-preview`; that preview remains backend-only. The documented deprecated `gpt-5.3-chat-latest` snapshot remains a backend-only pass-through ID at 128k context/16,384 output. The invalid `gpt-5.3` and undocumented GPT-5.4/5.5 chat aliases are translated locally for compatibility but hidden from selectors. Shut-down Gemini 3 preview and Gemini 2.0 IDs are also excluded from the current stable list and migrated locally where a safe successor exists.

xAI's official `grok-latest` alias resolves to `grok-4.3`; Vector Vault deliberately defaults new code/chat work to `grok-4.5` because xAI recommends it for those workloads.

Use an explicit compatibility ID only while migrating an existing workload. New code should use the recommended default alias or a current stable ID from the table above.
