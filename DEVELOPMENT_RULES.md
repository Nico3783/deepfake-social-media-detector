# DEVELOPMENT_RULES.md

## General Principles

The repository shall be developed according to software engineering best practices.

Code quality is equally important as model performance.

---

# Repository Standards

Every component must:

* Be modular
* Be reusable
* Be testable
* Be documented

No file should exceed reasonable complexity.

---

# Python Standards

Use:

* Python 3.11+
* Type hints
* Dataclasses where appropriate
* Pathlib instead of os paths

Avoid:

* Global variables
* Hardcoded paths
* Hardcoded credentials

---

# Naming Conventions

Files:

snake_case.py

Variables:

snake_case

Classes:

PascalCase

Constants:

UPPER_CASE

---

# Documentation Rules

Every file must include:

1. Purpose
2. Responsibilities
3. Dependencies

Every function must include:

* Description
* Parameters
* Returns
* Exceptions

---

# Logging Standards

Use structured logging.

Log:

* Dataset operations
* Training operations
* Evaluation operations
* Inference operations

Never use excessive print statements.

---

# Error Handling

All critical operations must:

* Validate inputs
* Catch expected exceptions
* Log failures
* Fail gracefully

---

# Testing Rules

Every major module requires tests.

Minimum areas:

* Dataset loading
* Face extraction
* Preprocessing
* Model loading
* Training pipeline
* Evaluation pipeline
* Inference pipeline

---

# Configuration Rules

Never hardcode:

* Paths
* Hyperparameters
* Dataset locations

Use:

configs/*.yaml

---

# Experiment Rules

Each experiment must have:

* Configuration
* Metrics
* Logs
* Checkpoints
* Notes

Stored under:

experiments/

---

# Git Rules

Commit messages should follow:

feat:
fix:
docs:
test:
refactor:

Examples:

feat: add xception training pipeline

fix: correct frame extraction bug

docs: update architecture documentation

---

# Production Readiness

Code should be suitable for:

* Academic review
* Supervisor review
* Demonstration
* Future deployment

Every implementation should assume it may later be used in a real-world cybersecurity environment.
