# November 15, 2026: public model integration and SDK enum floor

Official provider model APIs/docs and authenticated Generate Content/Messages calls confirm Claude Fable 5.2 (November 9), Gemini 4 Carbon (November 10) and Gemini 4 Hydrogen (November 12). Fable uses adaptive output_config.effort low/medium/high/xhigh/max, default high; its retained thinking is account/conversation-prefix bound and it requires30-day data retention unless Anthropic expressly authorizes ZDR. Mythos 5.2 remains restricted; don't expose it merely because a credential can see it.

Carbon retains Argon's unified 1,048,576-token window and low/medium/high, medium default, REST nonstream hop32,768. Hydrogen has low/medium/high/xhigh, high default, REST nonstream hop65,536. Both streaming hops65,536; long-decode output occupies the remaining shared context, not a second 1M budget. Both ignore sampling, reject minimal/off, and use implicit caching only. Keep omission as a true no-op and retain all current aliases/defaults.

Add every new Claude to ANTHROPIC_NO_TEMPERATURE_LIST as well as thinking.omit_parameters: without the former, omitted thinking leaves a default temperature in the SDK payload. Add Gemini IDs to the package compatibility multimodal/thinking lists as well as catalog models.

Inspected public google-genai wheels2.30.1,3.0.0,3.1.0,3.2.0,3.3.0:3.3.0 is first to define ThinkingLevel.XHIGH. It requires Python>=3.11. Therefore7.4.9.35 intentionally requires Python>=3.11 and google-genai>=3.3.0; existing production already uses Python3.11. Do not publish a floor that silently resolves to an older SDK and loses a selectable level. uv lock's cross-Python resolution caught the incompatible initial3.10 declaration before publication.

First-party OpenAI model pages still describe max for GPT6 families, but active Chat Completions rejects it; retain wire-verified intersection. Never change to Responses-only semantics through generic metadata claims.
