# Instructions for coding agents

## Credentials

- `.env` contains private credentials. Do not display, print, summarize, quote, or copy its values into chat, tool output, logs, test fixtures, prompts, issues, or commits.
- Do not inspect credential values using file-reading tools, `cat`, `printenv`, environment dumps, shell tracing (`set -x`), or debugger output. Verify only presence, file permissions, Git ignore status, or sanitized success/failure.
- Use `healing_swarm.secrets.read_credentials` within an authorized host process. Credential values must remain in process memory and be sent only through the provider authentication mechanism to the intended service. Do not return them to a coding agent's conversation.
- Do not put credentials into command-line arguments, URLs, Docker build arguments, images, or source code. `.env.example` must contain placeholders or blank values only.
- Experimental agents must never receive OpenRouter or Runpod keys. Do not mount `.env`, the entire repository, home directories, or credential stores into their containers. Mount only the explicitly required task resources.
- Keep `.env` ignored by Git and owner-only (`chmod 600`). Before publishing changes, verify that no credential file is tracked. Avoid commands that would print a secret-bearing diff.
- Use dummy credentials in tests. Do not run paid API calls or provision cloud resources unless the user has authorized the experiment and its spending scope. Use dedicated provider keys with spending caps and minimum required permissions.
- If a leak is suspected, stop further disclosure, tell the user without repeating the credential, and recommend revocation/rotation. Do not erase evidence or silently rewrite published history.

These instructions guide cooperating coding agents. They are not an access-control boundary: tools running as the user's account may be able to read `.env`.

## Experiment integrity

- Distinguish scripted controls from observed model behavior.
- Treat prohibited-file open traces as best-effort evidence, not proof of compliance or actual bytes read.
- Report reward results separately from specification probe results. Neither representative probes nor special-case behavior alone establish intent.
- Report incomplete billing accounting and threshold overshoot honestly. Never imply that post-response spending thresholds are hard monetary caps.

## Mathematical notation

Render mathematical formulas in LaTeX, except when referring specifically to code.
