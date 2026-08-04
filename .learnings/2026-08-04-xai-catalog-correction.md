# xAI catalog correction — 2026-08-04

## What changed
xAI's current first-party model index and individual model pages now agree on the limits that conflicted during the August 2 audit:

- `grok-4.5`: 500,000-token context.
- `grok-4.20-0309-reasoning` and `grok-4.20-0309-non-reasoning`: 1,000,000-token context.
- `grok-4.20` is an alias of the dated reasoning model and therefore also uses 1,000,000.

This supersedes the August 2 conservative 256k decision for Grok 4.5 and the prior 2M Grok 4.20 interpretation.

## Alias contract
Current provider aliases belong in the broad backend catalog but do not need duplicate selector entries. Normalize them locally before provider calls:

- `grok-4.5-latest` and `grok-build-latest` → `grok-4.5`.
- `grok-4.3-latest` and `grok-latest` → `grok-4.3`.
- Stable Grok 4.20 reasoning/non-reasoning aliases → their dated `0309` canonical IDs.
- `grok-code-fast`, `grok-code-fast-1`, and `grok-code-fast-1-0825` → `grok-build-0.1`.

The last rule matters: xAI documents `grok-code-fast*` as Grok Build 0.1 aliases, so routing `grok-code-fast` to Grok 4.3 changes the requested model family and is incorrect.
