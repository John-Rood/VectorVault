# AGENT INSTRUCTIONS - READ FIRST

## Workspace Guidelines

This document contains critical workflow instructions. Read and follow these before making any changes.

---

## 1. Push Protocol

**ALWAYS follow this protocol after making changes:**

### 1. Review ALL Changes
Before committing, review EVERYTHING that changed since the last commit:
```bash
git status
git diff
git diff --cached
```
Make sure you understand and can document ALL changes, not just your current task.

### 2. Update Documentation
If you modified any function, endpoint, feature, or behavior:
- Update relevant docs (README.md, Architecture.md, inline comments)
- Update any user-facing command docs
- Don't let documentation drift from implementation

### 3. Audit & Bug Check
- Verify all changes are correct and complete
- Scan for syntax errors, missing imports, logic issues
- Run any available tests

### 4. Commit with Full Description
```bash
git add -A
git commit -m "type: brief summary" -m "- Change 1
- Change 2
- Change 3"
```
The commit message MUST cover ALL changes since last commit.

### 5. Push & Deploy
```bash
git push origin main
./deploy.sh  # If code changes were made and deploy script exists
```

---

## 2. Documentation Updates

**When you make code changes, you MUST update existing documentation.**

- If you modify a function, endpoint, or feature → update the relevant docs
- If you add new functionality → document it
- If you change behavior → update any affected docs
- Check: README.md, Architecture.md, any *_COMMANDS.txt files, inline comments

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
