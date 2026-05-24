# Contributing

Contributors are physicists, not necessarily software engineers; the
workflow is designed to be low-friction. All enforcement is at the PR
into `main`.

## Flow

1. Fork the repo (or, if you have write access, create a feature branch).
2. Branch naming: `<type>/<kebab-description>` (`feat/`, `fix/`,
   `chore/`, `docs/`, …).
3. Commit using Conventional Commits.
4. Open a PR against `main`.
5. After CI green, the PR is rebase- or squash-merged into `main`.

See [`AGENTS.md`](AGENTS.md) for the full conventions.

## Pre-commit

A pre-commit configuration is provided. To install it:

```
pip install pre-commit
pre-commit install
```

The hooks run a secret scan and basic hygiene checks on each commit.
