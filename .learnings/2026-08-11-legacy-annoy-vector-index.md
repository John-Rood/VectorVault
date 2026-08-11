# Historical Annoy vector indexes — Learnings from 2026-08-11

## What Happened
Cloud reads began failing after the SDK's FAISS migration when existing vector objects were downloaded to extensionless temporary files. NumPy rejected the historical Annoy bytes, then the loader hid that error by appending `.npz` to the temporary path and raising `FileNotFoundError`.

## Root Cause
The January Annoy-to-FAISS change updated serialization but did not implement the promised read compatibility for existing cloud objects. The temporary filename cannot identify the object format; GCS had already downloaded the complete object to a unique path.

## The Fix
Inspect the downloaded bytes. Load current ZIP/NPZ archives with `allow_pickle=False`; reject standalone NPY/pickle payloads; load historical Annoy indexes with the known dimensions and metric, extract their vectors, and rebuild the current FAISS index in memory. Unknown or corrupt formats fail with one bounded `ValueError` instead of a fake sibling-path retry.

## Prevention
Every persisted-format migration needs fixture tests for both the prior and current bytes, extensionless cloud downloads, corrupt/pickle rejection, repeated loads, and concurrency. Never use a bare `except` to switch formats and never enable pickle loading for storage compatibility.
