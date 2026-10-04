
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
