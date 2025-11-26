# Repository Guidelines

## Project Structure & Module Organization
- Place application code in `src/` grouped by feature; avoid grab-bag utilities.
- Mirror `src/` paths in `tests/` (e.g., `tests/api/user.test.*` for `src/api/user.*`).
- Use `scripts/` for helper tooling, `config/` for environment-aware settings, `docs/` for documentation, and `assets/` for static files.
- Keep modules small with a clear entry point (e.g., `src/index.*` or `cmd/<tool>/main.*`) and exports that match folder names.

## Build, Test, and Development Commands
- Add a `Makefile` (or package scripts) as the single entry for tasks:
  - `make install` – install dependencies.
  - `make lint` – run all formatters/linters.
  - `make test` – execute the full test suite.
  - `make check` – run lint + test for CI parity.
- If using Node.js, mirror with `npm install`, `npm run lint`, and `npm test`. Keep commands idempotent so they run from a clean checkout.

## Coding Style & Naming Conventions
- Rely on automated formatters (e.g., Prettier for JS/TS, Black for Python, gofmt for Go); commit configs.
- Default to 2-space indentation for web stacks; 100–120 char lines.
- Name files/directories `kebab-case` for web assets and `snake_case` for scripts; use `PascalCase` for types/classes and `camelCase` for functions/variables.
- Keep public APIs small; prefer explicit exports and typed interfaces.

## Testing Guidelines
- Place tests under `tests/` with the same basename as the subject; use `*.test.*` or `*_test.*` per language norms.
- Favor fast, deterministic tests; mock external services and avoid real network calls.
- Prioritize coverage on critical paths (auth, persistence, feature flags). Add regression tests with each bug fix.

## Commit & Pull Request Guidelines
- Prefer Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`, `test:`, `refactor:`). Write imperatively and keep the subject under ~72 chars.
- Include in each PR: short summary, linked issue (if any), test evidence (`make check` output or equivalent), and screenshots for UI changes.
- Keep changes focused and reviewable; split refactors into preparatory commits before behavior changes.

## Security & Configuration Tips
- Never commit secrets; use `.env.example` to document required variables and load via environment.
- Check dependencies for vulnerabilities as part of `make check`; keep lockfiles up to date and reviewed in PRs.
- Avoid logging sensitive identifiers; scrub PII in fixtures and recorded test data.
