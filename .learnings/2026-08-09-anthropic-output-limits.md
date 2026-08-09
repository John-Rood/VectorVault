# Anthropic output limits must come from the Models API — 2026-08-09

## What changed
Anthropic's first-party Models API now reports `max_tokens: 128000` for Claude Sonnet 5 and Claude Sonnet 4.6, matching the current models overview. The SDK catalog still recorded Sonnet 5 at 64,000 output tokens and did not publish capability metadata for several supported current Claude IDs.

## Root cause
The August catalog refresh copied the launch-era Sonnet 5 output limit into a one-off metadata entry. That made the metadata registry incomplete and allowed a provider-side capability increase to drift from source/docs.

## Fix and prevention
Keep one explicit capability table for all selector-visible Anthropic IDs and verify it against `GET /v1/models/{model_id}` during weekly audits. Context dictionaries remain routing/truncation limits; `MODEL_METADATA` is the authoritative capability/output registry. Compatibility-only retired IDs may remain outside the current capability table when first-party metadata is unavailable.

The same sweep found active SDK/doc/demo examples still hard-coding GPT-4 or GPT-3.5. Portable examples should use `default`; examples that intentionally demonstrate an explicit current OpenAI model should use `gpt-5.6`. Keep those old IDs only in backend compatibility registries and their tests.
