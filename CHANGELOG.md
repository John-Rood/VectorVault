# Changelog

## 7.4.9.33 - 2026-11-01

- Add OpenAI `gpt-6.1-terra` (released October 26, 2026; stable; 1,050,000 context, 922,000 max input, 128,000 output; text+image input). The active Chat Completions route accepts `none|low|medium|high|xhigh` (provider default `medium`) and rejects `minimal` and the Responses-only `max` (wire-verified November 1). Non-default `temperature` is omitted unconditionally and function tools on Chat Completions require `none`, matching the GPT-6 family.
- OpenAI's `GET /v1/models` now reports `shutdown_date: 2026-12-11` for the undated `o3`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5` and `o3-pro` IDs themselves (deprecations page, June 11 notice). Hide `o3`, `gpt-5-mini` and `gpt-5-nano` from current selectors and record deprecation/replacement metadata (`gpt-5.6-sol` / `gpt-5.6-terra` / `gpt-5.6-luna`). They still call themselves upstream until shutdown; saved-flow thinking levels are unchanged. Record `gpt-5.1` deprecation (shutdown 2027-04-01, replacement `gpt-6-sol`).
- Route retired Anthropic IDs that now return HTTP 404 locally to Anthropic's documented replacements so saved flows keep running: `claude-opus-4-1` -> `claude-opus-4-8`; `claude-sonnet-4-0`, `claude-sonnet-4-20250514`, `claude-3-7-sonnet-latest`, `claude-3-5-sonnet-latest` -> `claude-sonnet-5-5`; `claude-3-5-haiku-latest` -> `claude-haiku-4-5`. Retired IDs are never sent upstream and inherit the replacement's limits, thinking contract and sampling rules.
- Gemini Computer Use reached GA on October 30: record `computer_use` as a supported (not preview) Gemini 4 Argon tool.
- Defaults, aliases and every other current thinking contract are unchanged (re-verified on the wire November 1).

## 7.4.9.32 - 2026-10-25

- OpenAI shut down `o4-mini`, `o1`, `o3-mini`, `gpt-4` and `gpt-3.5-turbo` on October 23, 2026 (authenticated Chat Completions now returns HTTP 404 `model_not_found`). Remove `o4-mini` from current selectors and resolve every retired ID locally to OpenAI's documented substitute (`gpt-5.6-terra` for o4-mini/gpt-3.5-turbo, `gpt-5.6-sol` for o1/o3-mini/gpt-4) so saved flows keep running; retired IDs are never sent upstream.
- Apply the same local substitution to previously shut-down `o1-mini`, `chatgpt-4o-latest` and `gpt-5`/`5.1`/`5.2`/`5.3-chat-latest`, which also return 404. `gpt-4o-audio-preview` is recorded as retired without an alias because its successor `gpt-audio-1.5` requires audio input or output.
- Retired IDs inherit the substitute's context window and thinking contract; omitted `thinking_level` remains a provider-default no-op, and saved o4-mini levels (`low|medium|high|xhigh`) stay valid on GPT-5.6 Terra.
- Anthropic SDK 2.x (October 21) defaults to API version `2026-10-21`, which rejects `temperature`/`top_p`/`top_k`. Fresh installs of 7.4.9.31 therefore failed every call to sampling-capable legacy Claude models (Opus 4.5, Sonnet 4.6, Haiku 4.5, Sonnet 4.5). When the canonical catalog allows sampling, the SDK bridge now pins `anthropic-version: 2023-06-01` per Anthropic's migration guide; no-sampling Claude models keep the SDK default. SDK 0.125.0 behavior is unchanged.
- Correct deprecated `claude-sonnet-4-5` context to 200,000 tokens (Models API and context-window docs).
- Defaults, aliases and every current thinking contract are unchanged (re-verified on the wire October 25).

## 7.4.9.31 - 2026-10-18

- Add GA `gemini-4-argon` (released October 13), unified 1,048,576 input/thinking/output tokens, long decode up to the remaining context, `low|medium|high` thinking with `medium` provider default; `minimal` is unsupported.
- Record route-specific output limits (32,768 direct REST non-streaming / 65,536 streaming; SDK automatic continuation), implicit-only caching, modalities and tools. Omit deprecated sampling/budget fields and unsupported explicit caching for Argon only.
- Require `google-genai>=2.30.1` for Argon's automatic long-decode continuation. Existing Gemini 3 sampling and all provider defaults/aliases remain unchanged; missing thinking stays omitted.
- Hide newly deprecated `gpt-4o` from current selectors. Preserve GPT-4o and already backend-only GPT-4o Mini for saved flows until their January 14, 2027 shutdown, with October 14 deprecation metadata.

## 7.4.9.30 - 2026-10-11

- Add Anthropic `claude-haiku-5-5` (released 2026-10-07; stable, 1,000,000 context, 128,000 output, text+image input). It uses adaptive `low|medium|high|xhigh|max` effort with Anthropic's documented `medium` default, rejects `none`, `temperature` other than 1 and all `top_p`/`top_k`, and is omitted from sampling like the other no-temperature Claude 5 models. Wire-verified against the Messages API on 2026-10-11 along with current hosted browser, computer-use, web search/fetch and code-execution tool declarations.
- Correct Claude Opus 5.5 / Sonnet 5.5 computer-use metadata: the accepted Claude API toolset is `computer_toolset_20260801` (the previously recorded `computer_20260922` is rejected by the provider as an unknown tool type).
- Correct xAI Grok 4.3 reasoning contract to `none|low|medium|high|xhigh` with provider default `low` (official model page and authenticated Chat Completions; `max` is rejected). Newer Grok defaults are unchanged.
- Hide newly deprecated `gemini-3.7-flash` (Google deprecated 2026-10-08; auto-routed to `gemini-3.8-flash`), `gemini-3.5-flash` (deprecated; routed to `gemini-3.6-flash`) and `gpt-5.4-nano` (OpenAI deprecated 2026-10-01, shutdown 2027-04-01, replacement `gpt-6-luna`) from current selectors. IDs, aliases, limits and thinking maps remain for saved flows/backend callers.
- Record OpenAI's 2026-12-11 removal of dated GPT-5 Mini/Nano, o3 and o3-pro default snapshots and Gemini 3.1 Flash-Lite's earliest 2027-05-07 shutdown as metadata only; undated IDs are unchanged.
- Defaults, aliases and the wire-verified GPT-6 Astra/GPT-6.1 Sol/Gemini 3.8 Flash contracts are unchanged.

## 7.4.9.29 - 2026-10-04

- Fix unconditional temperature omission for o4-mini, GPT-5 Mini/Nano, GPT-5.5, the GPT-5.6 family, GPT-6 Astra/Sol/Luna and chat-latest: non-default sampling must be omitted even when `thinking_level` is absent, preserving saved-flow/provider-default compatibility.
- Remove GPT-5.6 family `max` from the active Chat Completions effort set: the full live SDK matrix now rejects it on every variant. Keep Responses documentation distinct from the actual adapter contract.
- Bridge Anthropic SDK1.x's removed temperature keyword through documented `extra_body` only for legacy models where direct authenticated API tests confirm sampling support; keep API SDK0.125.0 behavior unchanged and declare that minimum SDK floor.
- Filter xAI streaming role/finish/usage chunks without text so valid thinking responses never yield `None`.
- Hide Gemini 2.5 Flash-Lite from current selectors: the finished live matrix received HTTP 404 "no longer available to new users" on the production key while 2.5 Pro and 2.5 Flash stayed available. Keep the ID for saved-flow/back-compat and record the vendor-recommended successor gemini-3.5-flash-lite.
- Final artifact retains all additions and wire-verified current-family metadata from 7.4.9.27/28, with unchanged model defaults and aliases.

## 7.4.9.28 - 2026-10-04

- Complete the existing stable-family reasoning audit: wire-verified named efforts for o3/o4-mini, GPT-5 Mini/Nano, and GPT-5.4/Mini/Nano were previously absent from the canonical catalog. Preserve true omission behavior and expose only route-accepted efforts.
- Restore Gemini 3.1 Flash-Lite `minimal|low|medium|high` thinking (documented `minimal` default), verified on Generate Content.
- Add first-party input/output, modality, and endpoint metadata for those families.
- Hide Responses-only o3-pro from current Chat Completions selectors while retaining its explicit backend ID and endpoint limitation metadata.
- Preserve the new GPT-6.1 Sol/Sonnet 5.5 additions, Sonnet 4.5 deprecation compatibility, defaults, and aliases from 7.4.9.27.

## 7.4.9.27 - 2026-10-04

- Add stable OpenAI `gpt-6.1-sol` with 1,050,000 context/922,000 input/128,000 output limits and wire-verified Chat Completions `low|medium|high|xhigh` reasoning (`medium` default). Reject unsupported `none`, `minimal`, and `max`; function calling requires Responses.
- Add stable Anthropic `claude-sonnet-5-5` with 1,000,000 context/128,000 output limits and adaptive `low|medium|high|xhigh|max` effort (`high` default). Omit unsupported temperature and preserve provider behavior when thinking is absent.
- Hide newly deprecated Claude Sonnet 4.5 from current selectors (retirement November 30, 2026) while preserving backend/saved-flow compatibility.
- Preserve all existing defaults, aliases, stable models, and compatibility IDs.

## 7.4.9.25 - 2026-09-20

- Restore `temperature`, `top_p`, and `top_k` for Gemini 3.8 Flash after authenticated GenerateContent tests confirmed that the current GA route accepts each sampling control alongside explicit thinking levels. `candidate_count` remains omitted, and unsupported `minimal` thinking remains rejected.

## 7.4.9.24 - 2026-09-13

- Correct GPT-6 Astra's Chat Completions reasoning contract to the wire-verified `low`, `medium`, `high`, and `xhigh` set. Provider-default behavior remains unchanged when reasoning is omitted; `none` and `max` are excluded because the active Chat Completions route rejects them.

## 7.4.9.23 - 2026-09-06

- Add stable OpenAI `gpt-6-astra` with its 1.05M context, 128k output, multimodal/tool metadata, and `none` through `max` reasoning-effort contract; make it the OpenAI default.
- Add public Anthropic `claude-fable-5-1` with 1M context, 128k output, adaptive thinking, hosted browser/computer tools, and `low` through `max` effort.
- Add GA Google `gemini-3.8-flash` with 1,048,576-token context, 65,536-token output, multimodal/tool metadata, and `low`/`medium`/`high` thinking; make it the Google default and `gemini-latest` target.
- Omit sampling parameters rejected by Gemini 3.8 Flash and explicitly reject its unsupported `minimal` thinking level.
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
