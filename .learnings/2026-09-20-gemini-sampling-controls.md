# Gemini 3.8 Flash sampling controls changed after launch

## Problem

The initial Gemini 3.8 Flash rollout correctly omitted `temperature`, `top_p`, and `top_k` after the production GenerateContent route rejected those fields. On 2026-09-20, authenticated wire tests against the same GA route accepted each sampling control alongside explicit `HIGH` thinking, while `MINIMAL` thinking remained rejected.

## Durable rule

Provider launch constraints are not permanent capability contracts. Re-test model-specific parameter omissions during weekly audits and remove an omission only when the active production endpoint accepts the exact provider payload shape used by VectorVault. Keep unrelated restrictions (here, `candidate_count`) until independently proven.

## Evidence

- Official model page: https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash
- Official thinking guide: https://ai.google.dev/gemini-api/docs/thinking
- 2026-09-20 authenticated GenerateContent tests: `temperature`, `topP`, and `topK` each returned HTTP 200 with Gemini 3.8 Flash and explicit thinking; `MINIMAL` returned HTTP 400.
