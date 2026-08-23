# Model catalog learnings

## Code issues

### Provider capability releases can require catalog work without a new model ID

The 2026-08-23 audit found no new generally available text-model ID, but Anthropic released new hosted browser and computer-use tools for specific existing models after the prior audit. A weekly model audit must compare per-model modalities and tool compatibility, not only model IDs and context limits.

## Expected behavior

### Tool metadata must identify the exact compatible model set

Anthropic's current browser/computer toolsets support `claude-fable-5`, `claude-opus-5`, `claude-sonnet-5`, and `claude-opus-4-8`; invitation-only `claude-mythos-5` remains excluded. Older Claude models use a different legacy computer-use tool version and must not be presented as supporting the current hosted toolset.

## Solutions

### Test capability metadata at the package root and let downstream APIs serialize it

Record first-party tool compatibility in the package-owned `model_catalog.json`, assert the exact model/tool combinations in package tests, and let the API continue serializing package metadata dynamically. Do not duplicate the table downstream.
