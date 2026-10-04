
### 2026-10-04 - GPT-6.1 Sol and Sonnet 5.5 contracts are route-specific

- First-party GPT-6.1 Sol docs advertise `max` for Responses, but authenticated Chat Completions accepts only `low|medium|high|xhigh`, rejects `none|minimal|max`, and does not support function calling. Keep generic thinking selectors on the active package route's verified set.
- Claude Sonnet 5.5 accepts adaptive `low|medium|high|xhigh|max`, defaults to `high`, and rejects non-default sampling parameters. `between_tools` is a separate thinking mode, not an effort level.
- Anthropic deprecated Sonnet 4.5 on September 30 (retirement November 30): hide it from current selectors but retain backend/saved-flow compatibility.

### 2026-10-04 - Audit existing stable families, not only new releases

The full-family audit found old GPT-5.4/Mini/Nano, GPT-5 Mini/Nano, o3 and o4-mini named-effort metadata missing. Authenticated Chat Completions verified exact effort sets; notably o3/o4-mini now accept xhigh, whereas GPT-5 Mini/Nano accept minimal but reject none/xhigh. o3-pro is Responses-only and returns a Chat Completions 404; keep its backend identity but never offer it as a current chat selector.
- Gemini 3.1 Flash-Lite's native MINIMAL/LOW/MEDIUM/HIGH values also work on Generate Content; Google's Gemini 3 table explicitly documents minimal as its default. Preserve budget-only Gemini 2.5 and preview-only Pro aliases without inventing named effort mappings or promoting preview IDs.
