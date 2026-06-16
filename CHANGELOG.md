# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 0: Project foundation
  - Created `__init__.py` for all 11 packages
  - Implemented `src/config/settings.py` with dataclass-based configuration
  - Implemented `src/config/paths.py` with PathConfig for all project directories
  - Implemented `src/config/constants.py` with project constants
  - Implemented `src/utils/logger.py` with structured logging
  - Implemented `src/utils/seed.py` for reproducible experiments
  - Implemented `src/utils/helpers.py` with utility functions
  - Implemented `src/utils/file_manager.py` for structured file I/O
  - Created 5 YAML configuration files with hyperparameters
  - Created `requirements.txt` with all dependencies
  - Created `pyproject.toml` with project metadata
  - Created `.gitignore` for dataset/model exclusion
  - Created `.env.example` for environment configuration
  - Created `README.md` with project documentation

## [0.1.0] - 2026-06-15

### Added
- Initial project structure
- Governance documents (CLAUDE.md, PROJECT_CONTEXT.md, REQUIREMENTS.md, etc.)
- Empty module stubs for all packages
