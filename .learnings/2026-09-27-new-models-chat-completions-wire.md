# September 27 model release and Chat Completions wire contract

## What Happened
OpenAI published GPT-6 Sol/Luna, Anthropic recommended Claude Opus 5.5 as the new default, and xAI recommended Grok 4.7. The OpenAI pages advertise `max` reasoning, but authenticated requests against VectorVault's active Chat Completions route reject `max` for both GPT-6 Sol and Luna; the provider error enumerates `none`, `low`, `medium`, `high`, `xhigh`. Anthropic Opus 5.5 accepted `low` and `max` with adaptive thinking/output_config; xAI Grok 4.7 accepted `low` and `xhigh`.

## Root Cause
General model pages span multiple endpoints; their reasoning/tool claims are broader than the Chat Completions endpoint actually used by vectorvault.ai. Promoting all documented choices would create invalid customer requests. The previous product defaults also lagged provider recommendations.

## The Fix
Catalog only the wire-verified OpenAI Chat Completions levels while keeping omitted thinking as a true no-op. Add the newly stable IDs in the package-owned JSON, repoint Anthropic's local `claude-latest` alias to Opus 5.5, choose the current recommended Anthropic/xAI defaults, preserve older provider-accepted IDs and xAI's distinct legacy `grok-latest` alias, and test SDK parameter interception and rejection before provider calls. Update tests that asserted obsolete release-specific defaults and pin the uv.lock version alongside pyproject.toml.

## Prevention
For every model release, compare official model docs and authenticated wire behavior on the *actual* provider endpoint; expose the verified intersection. Include exact defaults, package-resource tests, API pin, UI-derived metadata, and clean-install verification in the root-first release gates. Don't treat the old package source tree or an existing serving revision as immutable deployment provenance without comparing Cloud Build source and image digest.

Sources verified 2026-09-27: https://developers.openai.com/api/docs/models/gpt-6-sol ; https://developers.openai.com/api/docs/models/gpt-6-luna ; https://platform.claude.com/docs/en/models/overview ; https://platform.claude.com/docs/en/models/opus-5-5/overview ; https://docs.x.ai/developers/models/grok-4.7 ; https://ai.google.dev/gemini-api/docs/models .
