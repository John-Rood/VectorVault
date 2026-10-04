
### 2026-10-04 - GPT-6.1 Sol and Sonnet 5.5 contracts are route-specific

- First-party GPT-6.1 Sol docs advertise `max` for Responses, but authenticated Chat Completions accepts only `low|medium|high|xhigh`, rejects `none|minimal|max`, and does not support function calling. Keep generic thinking selectors on the active package route's verified set.
- Claude Sonnet 5.5 accepts adaptive `low|medium|high|xhigh|max`, defaults to `high`, and rejects non-default sampling parameters. `between_tools` is a separate thinking mode, not an effort level.
- Anthropic deprecated Sonnet 4.5 on September 30 (retirement November 30): hide it from current selectors but retain backend/saved-flow compatibility.
