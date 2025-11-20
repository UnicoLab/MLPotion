# ------------------------------------
# MLPotion - Makefile
# Modular ML Pipeline Components
# ------------------------------------

# ------------------------------------
# Variables
# ------------------------------------

HAS_POETRY := $(shell command -v poetry 2> /dev/null)
POETRY_VERSION := $(shell poetry version $(shell git describe --tags --abbrev=0 2>/dev/null || echo "0.1.0"))
PYTHON := poetry run python
PYTEST := poetry run pytest

# ------------------------------------
# Help
# ------------------------------------

.DEFAULT_GOAL := help

.PHONY: help
## Show this help message
help:
	@echo "$$(tput bold)MLPotion - Available Commands:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=25 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

# ------------------------------------
# Installation & Setup
# ------------------------------------

.PHONY: install
## Install package with core dependencies only
install:
	poetry install

.PHONY: install-dev
## Install package with development dependencies
install-dev:
	poetry install --with dev

.PHONY: install-docs
## Install package with documentation dependencies
install-docs:
	poetry install --with docs

.PHONY: install-tensorflow
## Install package with TensorFlow support
install-tensorflow:
	poetry install -E tensorflow

.PHONY: install-pytorch
## Install package with PyTorch support
install-pytorch:
	poetry install -E pytorch

.PHONY: install-keras
## Install package with Keras support
install-keras:
	poetry install -E keras

.PHONY: install-zenml
## Install package with ZenML integration
install-zenml:
	poetry install -E zenml

.PHONY: install-all
## Install package with all extras (TensorFlow, PyTorch, Keras, ZenML)
install-all:
	poetry install -E all

.PHONY: setup
## Complete development setup (install all + pre-commit hooks)
setup: install-all install-dev install-docs
	poetry run pre-commit install
	@echo "✅ Development environment ready!"

# ------------------------------------
# Testing
# ------------------------------------

.PHONY: test
## Run all tests with coverage
test:
	$(PYTEST) --cov=mlpotion --cov-report=term-missing --cov-report=html --cov-report=xml

.PHONY: test-fast
## Run fast tests only (unit tests, no integration)
test-fast:
	$(PYTEST) -m "unit or fast" --maxfail=5

.PHONY: test-unit
## Run unit tests only
test-unit:
	$(PYTEST) -m "unit" --maxfail=10

.PHONY: test-integration
## Run integration tests only
test-integration:
	$(PYTEST) -m "integration" --maxfail=3

.PHONY: test-tensorflow
## Run TensorFlow-specific tests
test-tensorflow:
	$(PYTEST) -m "tensorflow" --maxfail=5

.PHONY: test-pytorch
## Run PyTorch-specific tests
test-pytorch:
	$(PYTEST) -m "pytorch" --maxfail=5

.PHONY: test-keras
## Run Keras-specific tests
test-keras:
	$(PYTEST) -m "keras" --maxfail=5

.PHONY: test-zenml
## Run ZenML integration tests
test-zenml:
	$(PYTEST) -m "zenml" --maxfail=3

.PHONY: test-verbose
## Run tests with verbose output
test-verbose:
	$(PYTEST) -v --tb=long

.PHONY: test-parallel
## Run tests in parallel (faster)
test-parallel:
	$(PYTEST) -n auto --maxfail=3

.PHONY: test-smoke
## Run quick smoke test
test-smoke:
	$(PYTEST) -m "fast" --maxfail=1 --tb=no -q

.PHONY: test-watch
## Run tests in watch mode (re-run on changes)
test-watch:
	$(PYTEST) -f

.PHONY: coverage
## Generate detailed coverage report
coverage:
	$(PYTEST) --cov=mlpotion --cov-report=html --cov-report=term-missing
	@echo "✅ Coverage report generated in htmlcov/index.html"

# ------------------------------------
# Code Quality & Formatting
# ------------------------------------

.PHONY: format
## Format code with black and isort
format:
	poetry run black mlpotion tests docs/examples
	poetry run isort mlpotion tests docs/examples
	@echo "✅ Code formatted!"

.PHONY: lint
## Run linting with ruff
lint:
	pre-commit run --all-files

.PHONY: lint-fix
## Run linting and auto-fix issues
lint-fix:
	pre-commit run --all-files --fix

.PHONY: typecheck
## Run type checking with mypy
typecheck:
	poetry run mypy mlpotion

.PHONY: check
## Run all quality checks (format, lint, typecheck)
check: format lint typecheck
	@echo "✅ All quality checks passed!"

.PHONY: pre-commit
## Run pre-commit hooks on all files
pre-commit:
	poetry run pre-commit run --all-files

# ------------------------------------
# Documentation
# ------------------------------------

.PHONY: docs-serve
## Serve documentation locally
docs-serve:
	poetry run mkdocs serve

.PHONY: docs-build
## Build documentation
docs-build:
	poetry run mkdocs build

.PHONY: docs-deploy
## Deploy documentation to GitHub Pages
docs-deploy:
	poetry run mkdocs gh-deploy --force

.PHONY: docs-deploy-version
## Deploy versioned documentation with mike
docs-deploy-version:
ifdef HAS_POETRY
	@$(POETRY_VERSION)
	poetry version -s | xargs -I {} sh -c 'echo Deploying version {} && mike deploy --push --update-aliases {} latest'
else
	@echo "❌ Poetry is required to deploy versioned docs"
	exit 1
endif

.PHONY: docs-version-list
## List available documentation versions
docs-version-list:
	mike list

.PHONY: docs-version-serve
## Serve versioned documentation
docs-version-serve:
	mike serve

# ------------------------------------
# Build & Release
# ------------------------------------

.PHONY: build
## Build package distribution
build: clean-build
	@echo "🔨 Building package..."
ifdef HAS_POETRY
	@$(POETRY_VERSION)
	poetry build
	@echo "✅ Package built successfully!"
else
	@echo "❌ Poetry is required to build the package"
	exit 1
endif

.PHONY: publish-test
## Publish package to TestPyPI
publish-test: build
	poetry publish -r testpypi

.PHONY: publish
## Publish package to PyPI
publish: build
	poetry publish

.PHONY: version-patch
## Bump patch version (0.1.0 -> 0.1.1)
version-patch:
	poetry version patch
	@echo "✅ Version bumped to $$(poetry version -s)"

.PHONY: version-minor
## Bump minor version (0.1.0 -> 0.2.0)
version-minor:
	poetry version minor
	@echo "✅ Version bumped to $$(poetry version -s)"

.PHONY: version-major
## Bump major version (0.1.0 -> 1.0.0)
version-major:
	poetry version major
	@echo "✅ Version bumped to $$(poetry version -s)"

# ------------------------------------
# Examples
# ------------------------------------

.PHONY: run-example-tensorflow
## Run TensorFlow example
run-example-tensorflow:
	$(PYTHON) docs/examples/tensorflow/basic_usage.py

.PHONY: run-example-pytorch
## Run PyTorch example
run-example-pytorch:
	$(PYTHON) docs/examples/pytorch/basic_usage.py

.PHONY: run-example-keras
## Run Keras example
run-example-keras:
	$(PYTHON) docs/examples/keras/basic_usage.py

.PHONY: run-zenml-tensorflow
## Run TensorFlow ZenML pipeline example
run-zenml-tensorflow:
	$(PYTHON) docs/examples/tensorflow/zenml_pipeline.py

.PHONY: run-zenml-pytorch
## Run PyTorch ZenML pipeline example
run-zenml-pytorch:
	$(PYTHON) docs/examples/pytorch/zenml_pipeline.py

# ------------------------------------
# Cleaning
# ------------------------------------

.PHONY: clean-tests
## Remove test artifacts and cache
clean-tests:
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*junit_report.xml' -exec rm -f {} +
	find . -type f -name '*.pyc' -exec rm -f {} +
	rm -rf htmlcov/
	rm -f .coverage
	rm -f coverage.xml
	rm -f pytest.xml
	@echo "✅ Test artifacts cleaned!"

.PHONY: clean-build
## Remove build artifacts
clean-build:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	find . -type f -name '*.py[co]' -delete
	find . -type d -name __pycache__ -delete
	@echo "✅ Build artifacts cleaned!"

.PHONY: clean-docs
## Remove documentation build artifacts
clean-docs:
	rm -rf site/
	@echo "✅ Documentation artifacts cleaned!"

.PHONY: clean-cache
## Remove all Python cache files
clean-cache:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	@echo "✅ Cache cleaned!"

.PHONY: clean
## Remove all generated files and caches
clean: clean-tests clean-build clean-docs clean-cache
	@echo "✅ All clean!"

# ------------------------------------
# Development Workflow
# ------------------------------------

.PHONY: dev
## Run complete development workflow (format, lint, test)
dev: format lint test
	@echo "✅ Development workflow complete!"

.PHONY: ci
## Run CI workflow (lint, typecheck, test with coverage)
ci: lint typecheck test
	@echo "✅ CI workflow complete!"

.PHONY: quick-check
## Quick sanity check (format, fast tests)
quick-check: format test-smoke
	@echo "✅ Quick check passed!"

# ------------------------------------
# Utilities
# ------------------------------------

.PHONY: info
## Show project information
info:
	@echo "📦 MLPotion - Machine Learning Potion"
	@echo "Version: $$(poetry version -s)"
	@echo "Python: $$(poetry run python --version)"
	@echo ""
	@echo "Available Frameworks:"
	@$(PYTHON) -c "from mlpotion.utils import get_available_frameworks; print('  -', '\n  - '.join(get_available_frameworks()))" || echo "  (none installed)"

.PHONY: shell
## Open Python shell with mlpotion imported
shell:
	poetry run python -i -c "import mlpotion; print('MLPotion loaded. Available in namespace.')"

.PHONY: jupyter
## Start Jupyter notebook server
jupyter:
	poetry run jupyter notebook

.PHONY: update-deps
## Update all dependencies
update-deps:
	poetry update
	@echo "✅ Dependencies updated!"

.PHONY: lock
## Lock dependencies without updating
lock:
	poetry lock --no-update

.PHONY: show-deps
## Show dependency tree
show-deps:
	poetry show --tree

# ------------------------------------
# Git Helpers
# ------------------------------------

.PHONY: git-clean
## Clean untracked files and directories (WARNING: destructive)
git-clean:
	@echo "⚠️  This will remove all untracked files and directories!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		git clean -fdx; \
		echo "✅ Git repository cleaned!"; \
	else \
		echo "❌ Cancelled"; \
	fi

.PHONY: git-status
## Show detailed git status
git-status:
	@echo "📊 Git Status:"
	@git status
	@echo ""
	@echo "📝 Recent Commits:"
	@git log --oneline -5

# ------------------------------------
# Docker (Future)
# ------------------------------------

.PHONY: docker-build
## Build Docker image (TODO)
docker-build:
	@echo "🐳 Docker support coming soon!"

.PHONY: docker-run
## Run Docker container (TODO)
docker-run:
	@echo "🐳 Docker support coming soon!"

# ------------------------------------
# Performance
# ------------------------------------

.PHONY: profile
## Profile code performance (TODO)
profile:
	@echo "⚡ Profiling support coming soon!"

.PHONY: benchmark
## Run performance benchmarks (TODO)
benchmark:
	@echo "⚡ Benchmark support coming soon!"
