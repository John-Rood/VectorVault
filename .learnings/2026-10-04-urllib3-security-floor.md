# urllib3 security floor — 2026-10-04

## What happened
Dependabot reported three urllib3 advisories against `uv.lock`: CVE-2026-97687 / GHSA-8988-9cw3-xx77 (high, HTTPS proxy TLS policy), CVE-2026-97689 / GHSA-vxq7-64xx-v4gw (high, unbounded streaming chunk-size line), and CVE-2026-97688 / GHSA-gh4c-6fx4-qh6g (medium, Deflate streaming infinite loop). All three are fixed in urllib3 2.8.0.

## Root cause
The lockfile retained urllib3 2.7.0 through Requests. Requests permits older urllib3 releases, and VectorVault did not declare its own security floor. Updating the model catalog/package version does not refresh unrelated locked dependencies. A fresh unlocked installation can choose a safe release while a locked installation or an existing pip environment retains a vulnerable one.

## Fix
Declare `urllib3>=2.8.0` in the canonical `pyproject.toml`, regenerate only urllib3's lock entry with `uv lock --upgrade-package 'urllib3==2.8.0'`, and guard the manifest, lockfile, and installed version in release metadata tests. No other locked package version changes. The Python >=3.10 floor is unchanged and compatible with urllib3 2.8.0.

## Prevention and release boundary
Check both the lockfile and built wheel requirements when addressing transitive dependency alerts. Verify that the wheel rejects an explicitly vulnerable urllib3 pin; a lockfile alone does not protect pip consumers. GitHub alerts can clear after the patch reaches the default branch and its dependency graph updates. Existing installed environments still need an upgrade, and new package metadata needs a separately authorized version bump/release. This security PR does not publish a package or deploy a runtime.
