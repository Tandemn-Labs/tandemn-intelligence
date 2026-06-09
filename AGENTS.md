# AGENTS.md

This file is for coding agents. It is laid out as organization wide rules followed by repo-specific information.

Current repo: tandemn-labs/cloud-setup

## Organization guide

### Overall coding style
- Avoid clever one-liners that hurt readability.
- Use comments only for non-obvious operational logic, failure modes, or cross-service contracts. Do not comment what the code already says.
- Keep things simple and functions direct. Do not add unnecessary complexity in order to attain goals like scalability and security.
- Follow the existing local patterns before inventing a new one.

### Python rules
- Use PEP 8 as code style guide and PEP 257 as docstrings style guide.
- Ensure `pyproject.toml` exists with `ruff`, `mypy` rules
- Use the `./src/` layout for code
- Use `uv` for virtual environment
- Use the python stdlib `logging` library instead of `print()` in the codebase

### YAML rules
- Use `.yaml` for new files.

### Other rules
- Ensure `.pre-commit-config.yaml` exists


## Repo-specific guide

This repository implements koi. TODO - add more stuff