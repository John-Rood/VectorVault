# AGENT INSTRUCTIONS - READ FIRST

## Workspace Guidelines

This document contains critical workflow instructions. Read and follow these before making any changes.

---

## 1. Documentation Updates

**When you make changes to the code, update existing documentation.**

- If you modify a function, endpoint, or feature, update the relevant docs
- Don't let documentation drift from implementation
- If no relevant docs exist, note what was changed for future documentation

---

## 2. Push Protocol

**Follow the push protocol.**

After completing work:

1. **Audit** - Verify all changes are correct and complete
2. **Bug check** - Scan for syntax errors, missing imports, logic issues
3. **Commit** - Use descriptive commit messages
4. **Push** - Push to remote
5. **Deploy** - If a deploy script exists, run it after code changes

Reference: See the full push workflow in the [Cline Error Orchestrator](../Cline-Error-Orchestrator/workflows/push.md)

---

## 3. Keep Codebase Clean

**Don't leave extra files around.**

- Remove temporary files, debug logs, and test artifacts
- Don't commit files that don't belong (`.DS_Store`, `*.pyc`, `__pycache__/`, etc.)
- Clean up any scaffolding or placeholder code before pushing
- If you create a file for testing, delete it when done

---

## 4. Sensitive Data Protection

**NEVER commit credentials, API keys, or sensitive data.**

- Before committing, verify no `.env`, `.env.local`, or credential files are staged
- Add all sensitive files to `.gitignore` BEFORE creating them
- Never hardcode API keys, tokens, passwords, or secrets in source code
- Use environment variables for all sensitive configuration
- If you accidentally commit sensitive data, alert immediately - git history must be rewritten

**Always check before committing:**
```bash
git status  # Review staged files - no .env, credentials, or secrets
```

---

## Quick Reference

```bash
# After making changes:
git status  # VERIFY no sensitive files staged
git add -A
git commit -m "Title: Brief summary" -m "Body: Detailed changes"
git push origin main
./deploy.sh  # Only if deploy script exists and code changed
```
