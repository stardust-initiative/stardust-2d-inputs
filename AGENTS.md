# AGENTS — Contribution Conventions

This repository follows the Stardust public-repo conventions.

## Workflow

- **All work via feature branches.** Direct pushes to `main` are blocked.
- Branch naming: `<type>/<kebab-description>` where `<type>` is from the
  Conventional Commits set (`feat/`, `fix/`, `chore/`, `docs/`, …).
- **Conventional Commits** for commit messages: `<type>(<scope>): <subject>`.
- PRs are required for merging into `main`. Allowed merge methods: rebase
  or squash (linear history).
- CI must pass before merge.

## Security

Never commit credentials, private paths, internal URLs, or unpublished
data. The pre-commit secret-scanner is the first line of defense; CI is
the backstop. For vulnerabilities, see [`SECURITY.md`](SECURITY.md).
