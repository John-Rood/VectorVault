
### 2026-10-04 - GPT-6.1 Sol and Sonnet 5.5 contracts are route-specific

- First-party GPT-6.1 Sol docs advertise `max` for Responses, but authenticated Chat Completions accepts only `low|medium|high|xhigh`, rejects `none|minimal|max`, and does not support function calling. Keep generic thinking selectors on the active package route's verified set.
- Claude Sonnet 5.5 accepts adaptive `low|medium|high|xhigh|max`, defaults to `high`, and rejects non-default sampling parameters. `between_tools` is a separate thinking mode, not an effort level.
- Anthropic deprecated Sonnet 4.5 on September 30 (retirement November 30): hide it from current selectors but retain backend/saved-flow compatibility.

### 2026-10-04 - Audit existing stable families, not only new releases

The full-family audit found old GPT-5.4/Mini/Nano, GPT-5 Mini/Nano, o3 and o4-mini named-effort metadata missing. Authenticated Chat Completions verified exact effort sets; notably o3/o4-mini now accept xhigh, whereas GPT-5 Mini/Nano accept minimal but reject none/xhigh. o3-pro is Responses-only and returns a Chat Completions 404; keep its backend identity but never offer it as a current chat selector.
- Gemini 3.1 Flash-Lite's native MINIMAL/LOW/MEDIUM/HIGH values also work on Generate Content; Google's Gemini 3 table explicitly documents minimal as its default. Preserve budget-only Gemini 2.5 and preview-only Pro aliases without inventing named effort mappings or promoting preview IDs.

### 2026-10-04 - Complete wire matrices before the final artifact

The 424-call live SDK matrix found default/omitted-thinking temperature failures across the GPT-5.5/5.6/6 family, Mini/Nano, and chat-latest; all affected models require unconditional omission of non-default sampling. GPT-5.6 `max` is now rejected on the active Chat Completions path on every variant, despite broader Responses documentation and earlier acceptance. Remove that value only from the active route's canonical effort sets; saved-flow clamps preserve provider-default behavior.

Anthropic SDK1.11 removed the temperature keyword, but direct authenticated Messages calls still accept sampling for legacy Opus4.5/Sonnet4.6/Haiku4.5. Use the SDK's documented extra_body only for this already-provider-validated field and only when the callable signature removed the keyword; never resurrect temperature for canonical no-temperature models. SDK0.125.0 remains the pinned API runtime and the minimum declared package floor. xAI streaming role/finish/usage chunks can have null text: skip them rather than returning None or leaking non-text reasoning.

### 2026-10-11 - A new Anthropic tier can appear days after the prior audit; verify tool IDs on the wire

- Claude Haiku 5.5 launched October 7 and was only visible by comparing the first-party models overview and the authenticated Models API against the catalog. Always diff the provider lineup (all tiers: Fable/Opus/Sonnet/Haiku), not just the models already carried.
- The catalog had recorded `computer_20260922` for Opus/Sonnet 5.5; the live Messages API rejects it as an unknown tool type while `computer_toolset_20260801` is accepted by every model Anthropic's docs list. Declared tool IDs in metadata must be probed against the wire, not inferred.
- Provider deprecation pages move without a new model ID: Gemini 3.7/3.5 Flash (Oct 8, auto-routed) and GPT-5.4 Nano (Oct 1) were deprecated. Hide from selectors, keep for saved flows. xAI's Grok 4.3 page gained `xhigh` and a `low` default; authenticated calls confirmed.

### 2026-10-18 - Argon's output limit is not a second input window

Gemini 4 Argon shares 1,048,576 tokens across input, thinking and output. Direct REST non-streaming caps each hop at 32,768, streaming at 65,536; long decode requires google-genai>=2.30.1 and an explicit max_output_tokens above 65,536. Keep those limits distinct in package metadata; do not advertise 1M output in addition to 1M input. Argon omits ignored sampling and deprecated thinking_budget, and cannot use explicit cached_content. Gemini 3.8 Flash still accepts its own sampling parameters, so never generalize Argon restrictions to the whole provider. Keep existing defaults/aliases unchanged for omitted-model and saved-flow compatibility.

### 2026-10-25 - Shutdown dates take selector models offline between audits; SDK majors can change the API version

OpenAI's October 23 shutdown removed `o4-mini` (still a current selector), `o1`, `o3-mini`, `gpt-4` and `gpt-3.5-turbo`; the Models list still returned some IDs with a past `shutdown_date`, so list membership alone is not proof of callability. Check `shutdown_date` on `GET /v1/models` and send a one-token wire call for every catalog ID each audit. Retired IDs stay loadable as local aliases to the provider-documented substitute (never sent upstream) unless the substitute changes modality (`gpt-4o-audio-preview` -> audio-only `gpt-audio-1.5` is left unaliased).

Anthropic Python SDK 2.0 (2026-10-21) sends `anthropic-version: 2026-10-21`, which rejects `temperature`/`top_p`/`top_k` even at default values. The unbounded `anthropic>=0.125.0` floor let fresh 7.4.9.31 installs resolve 2.3.0 and break every sampling-capable legacy Claude call with "temperature: Extra inputs are not permitted". Anthropic's migration guide says such requests may stay on `2023-06-01`; the bridge now sets that header only when the catalog allows sampling. Always clean-install the published package with the newest provider SDKs, not only the API's pinned versions.

### 2026-11-01 - Undated IDs inherit snapshot shutdowns; retired Claude IDs need local routing too

OpenAI's June 11 notice listed `o3`/`gpt-5-mini`/`gpt-5-nano` together with their dated snapshots, and by November `GET /v1/models/<undated id>` itself reports `shutdown_date: 2026-12-11`. An earlier audit treated this as snapshot-only metadata; always query the retrieve endpoint for each selector ID rather than inferring from the table. Hide such IDs before the shutdown and alias them only once they actually 404.

The Anthropic catalog still carried six IDs retired between October 2025 and August 2026 (`claude-opus-4-1`, `claude-sonnet-4-0`, `claude-3-x-*-latest`); they 404 on the Messages API. Every audit must send a one-token call for every non-alias catalog ID of every provider, not only OpenAI. Alias retired IDs to the deprecations page's recommended replacement so saved flows run. GPT-6.1 Terra again documents `max` on the model page while Chat Completions rejects it: catalog only the active route's enum.
