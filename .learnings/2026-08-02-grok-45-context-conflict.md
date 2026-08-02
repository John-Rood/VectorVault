# Grok 4.5 Context-Window Conflict — 2026-08-02

## What happened
xAI's public model index listed a 500,000-token context for `grok-4.5`, while the individual first-party Grok 4.5 model page listed 256,000.

## Decision
Use 256,000 in VectorVault's runtime catalog and public docs until an authenticated xAI Models API response resolves the discrepancy.

## Why
The runtime token limit is an enforcement boundary. Overstating it can send requests that the provider rejects; understating it safely switches or rejects sooner. When first-party sources conflict, the conservative limit is the release-safe choice.
