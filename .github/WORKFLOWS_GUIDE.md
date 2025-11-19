# MLPotion CI/CD Workflows - Quick Reference

## Available Workflows

### 1. UTESTS.yml - Unit Tests
**Triggers:**
- Push to `dev` or `main` branches (when Python files or tests change)
- Pull requests to `dev` or `main`
- Manual dispatch with options

**What it does:**
- **Smoke test**: Quick unit tests on PRs for fast feedback (3 min timeout)
- **Test matrix**: Runs tests across Python 3.10 and 3.11
  - Unit tests
  - Integration tests
  - TensorFlow-specific tests (Python 3.11 only)
  - PyTorch-specific tests (Python 3.11 only)
  - Keras-specific tests (Python 3.11 only)
- **Coverage**: Generates coverage report and uploads to Codecov

**Manual run:**
```bash
# Via GitHub UI: Actions → Unit-tests → Run workflow
# Choose Python version and test type
```

### 2. PRECOMMITS.yml - Pre-commit Checks
**Triggers:**
- Pull requests to `dev` or `main`
- Manual dispatch

**What it does:**
- Validates PR title format (conventional commits)
- Runs pre-commit hooks on changed files
- Checks code formatting, linting, etc.

**Expected PR title format:**
```
(feat|fix|docs|style|refactor|perf|test|build|ci|chore|ops|hotfix)(scope): description
```

Examples:
- `feat(tensorflow): add new layer type`
- `fix(core): resolve memory leak`
- `docs(readme): update installation instructions`

### 3. RELEASE.yml - Semantic Release
**Triggers:**
- Manual dispatch only

**What it does:**
1. **Semantic Release**: Analyzes commits and determines version bump
2. **PyPI Publish**: Builds and publishes package to PyPI
3. **Update Docs**: Deploys versioned documentation to GitHub Pages

**Manual run:**
```bash
# Via GitHub UI: Actions → Semantic Release → Run workflow
# Options:
# - DRY_RUN: Test without actually releasing
# - SKIP_RELEASE: Use existing version (for recovery)
# - RELEASE_VERSION: Manually specify version
```

**Commit format for versioning:**
- `feat(scope): description` → Minor version bump (0.1.0 → 0.2.0)
- `fix(scope): description` → Patch version bump (0.1.0 → 0.1.1)
- `perf(scope): description` → Patch version bump
- `BREAKING CHANGE:` in commit body → Major version bump (0.1.0 → 1.0.0)

### 4. MANUAL_PYPI_PUBLISH.yml - Manual PyPI Publish
**Triggers:**
- Manual dispatch only

**What it does:**
- Publishes a specific version to PyPI
- Useful for recovery or re-publishing

**Manual run:**
```bash
# Via GitHub UI: Actions → Manual PyPI Publish → Run workflow
# Specify: RELEASE_VERSION (e.g., "0.1.0")
```

### 5. MANUAL_DOCS_PUBLISH.yml - Manual Documentation Publish
**Triggers:**
- Manual dispatch only

**What it does:**
- Publishes documentation for a specific version
- Optionally updates the 'latest' alias

**Manual run:**
```bash
# Via GitHub UI: Actions → Manual Documentation Publish → Run workflow
# Specify:
# - RELEASE_VERSION (e.g., "0.1.0")
# - UPDATE_LATEST (true/false)
```

### 6. PR_PREVIEW.yml - PR Documentation Preview
**Triggers:**
- Push to `dev` or `main` (when docs change)
- Pull requests to `dev` or `main` (when docs change)

**What it does:**
- Builds documentation preview
- Deploys to PR preview environment
- Adds comment to PR with preview link

## Common Commands

### Local Testing

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests
make test-integration

# Run framework-specific tests
make test-tensorflow
make test-pytorch
make test-keras

# Run tests with coverage
make coverage

# Run pre-commit checks
make pre-commit

# Build documentation locally
make docs-serve
```

### Installation for CI/CD Testing

```bash
# Development setup
poetry install --with dev

# Testing setup (all frameworks)
poetry install --with dev --all-extras

# Documentation setup
poetry install --with docs

# Specific framework
poetry install -E tensorflow
poetry install -E pytorch
poetry install -E keras
```

## Troubleshooting

### Tests failing in CI but passing locally?
- Ensure you're testing with the same Python version (3.10 or 3.11)
- Install all extras: `poetry install --all-extras --with dev`
- Check if you have uncommitted changes

### Pre-commit checks failing?
- Run locally: `pre-commit run --all-files`
- Check PR title format
- Ensure code is formatted: `make format`

### Documentation build failing?
- Test locally: `make docs-build`
- Check for broken links or missing files
- Ensure all dependencies installed: `poetry install --with docs`

### Release workflow failing?
- Check commit messages follow conventional format
- Ensure PYPI_TOKEN secret is set
- Verify version doesn't already exist on PyPI

## Secrets Required

The following GitHub secrets must be configured:

1. **PYPI_TOKEN**: PyPI API token for publishing packages
   - Get from: https://pypi.org/manage/account/token/
   - Scope: Project-specific or account-wide

2. **GITHUB_TOKEN**: Automatically provided by GitHub Actions
   - Used for: Creating releases, updating docs

## Best Practices

1. **Commit Messages**: Always use conventional commit format
2. **Testing**: Run tests locally before pushing
3. **Documentation**: Update docs when adding features
4. **Versioning**: Let semantic release handle version bumps
5. **Pre-commit**: Install hooks: `make setup`

## Workflow Dependencies

```
RELEASE.yml
├── SEMANTIC_RELEASE (determines version)
├── PYPI_PUBLISH (publishes to PyPI)
└── UPDATE_DOCS (updates documentation)

UTESTS.yml
├── smoke-test (fast feedback)
├── test-matrix (comprehensive testing)
└── coverage (coverage report)

PRECOMMITS.yml
└── pre-commit-checks (code quality)

PR_PREVIEW.yml
└── deploy-pr-preview (documentation preview)
```

## Support

For issues with CI/CD:
1. Check workflow logs in GitHub Actions
2. Review this guide and CI_CD_FIXES.md
3. Test locally using make commands
4. Contact maintainers if issues persist
