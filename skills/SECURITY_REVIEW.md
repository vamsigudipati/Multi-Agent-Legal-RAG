# Skill Security Review Checklist

Skills that are appended to agent prompts must be reviewed for prompt-injection risks.

Checklist:

- [ ] Does the skill include instructions that override system-level constraints?
- [ ] Are any external URLs or code execution hints present?
- [ ] Are there escape sequences or role-switching text fragments?
- [ ] Is the skill signed or reviewed by a trusted reviewer?

Required metadata on skill files:

- `Reviewed: YES/NO`
- `Reviewer: <username>`
- `ReviewedAt: <ISO timestamp>`

If a skill fails review, do not add it to the runtime prompt; instead, open an issue and quarantine it under `skills/quarantine/`.
