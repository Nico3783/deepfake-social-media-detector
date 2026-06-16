#!/usr/bin/env python3
"""
Repository Cleanup Script

Purpose: Clean up repository for supervisor review and final submission.
Responsibilities: Remove empty files, organize structure, verify completeness.
Dependencies: pathlib

Research Traceability:
    Research Objective: Final repository readiness
    Methodology: Systematic cleanup and verification
    Implementation: scripts/cleanup_repository.py

Usage:
    python scripts/cleanup_repository.py
"""

from __future__ import annotations

import os
from pathlib import Path


# Files to keep (even if empty - they're required __init__.py files)
KEEP_EMPTY = {
    "__init__.py",
}

# Directories that should exist
REQUIRED_DIRS = [
    "src/config",
    "src/utils",
    "src/data",
    "src/preprocessing",
    "src/models",
    "src/training",
    "src/evaluation",
    "src/inference",
    "src/visualization",
    "src/api",
    "tests",
    "configs",
    "docs",
    "scripts",
    "outputs/experiments",
    "outputs/reports",
    "outputs/models",
    "outputs/plots",
    "thesis/assets",
]


def find_empty_files(root: Path) -> list[Path]:
    """Find empty files that are not __init__.py."""
    empty_files = []
    for file in root.rglob("*"):
        if file.is_file() and file.stat().st_size == 0:
            if file.name not in KEEP_EMPTY:
                empty_files.append(file)
    return empty_files


def check_required_directories(root: Path) -> list[str]:
    """Check for missing required directories."""
    missing = []
    for dir_path in REQUIRED_DIRS:
        full_path = root / dir_path
        if not full_path.exists():
            missing.append(dir_path)
    return missing


def check_required_files(root: Path) -> dict[str, bool]:
    """Check for required files."""
    required_files = {
        "README.md": False,
        "CHANGELOG.md": False,
        "TODO.md": False,
        "requirements.txt": False,
        "pyproject.toml": False,
        ".gitignore": False,
        "CLAUDE.md": False,
        "PROJECT_DOCUMENT.md": False,
        "IMPLEMENTATION_ROADMAP.md": False,
        "ARCHITECTURE.md": False,
        "REQUIREMENTS.md": False,
        "RESEARCH_GUIDELINES.md": False,
        "DEVELOPMENT_RULES.md": False,
    }

    for file_name in required_files:
        if (root / file_name).exists():
            required_files[file_name] = True

    return required_files


def check_src_init_files(root: Path) -> dict[str, bool]:
    """Check for __init__.py files in all src subdirectories."""
    src_packages = [
        "src",
        "src/config",
        "src/utils",
        "src/data",
        "src/preprocessing",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/inference",
        "src/visualization",
        "src/api",
    ]

    init_files = {}
    for pkg in src_packages:
        init_path = root / pkg / "__init__.py"
        init_files[pkg] = init_path.exists()

    return init_files


def main() -> None:
    """Main function to clean up repository."""
    root = Path(__file__).parent.parent

    print("=" * 60)
    print("Repository Cleanup Script")
    print("=" * 60)

    # Find empty files
    print("\n1. Checking for empty files...")
    empty_files = find_empty_files(root)
    if empty_files:
        print(f"   Found {len(empty_files)} empty file(s):")
        for f in empty_files:
            print(f"   - {f.relative_to(root)}")
    else:
        print("   No empty files found.")

    # Check required directories
    print("\n2. Checking required directories...")
    missing_dirs = check_required_directories(root)
    if missing_dirs:
        print(f"   Missing {len(missing_dirs)} directory(ies):")
        for d in missing_dirs:
            print(f"   - {d}")
    else:
        print("   All required directories exist.")

    # Check required files
    print("\n3. Checking required files...")
    file_status = check_required_files(root)
    missing_files = [f for f, exists in file_status.items() if not exists]
    if missing_files:
        print(f"   Missing {len(missing_files)} file(s):")
        for f in missing_files:
            print(f"   - {f}")
    else:
        print("   All required files exist.")

    # Check __init__.py files
    print("\n4. Checking __init__.py files...")
    init_status = check_src_init_files(root)
    missing_init = [pkg for pkg, exists in init_status.items() if not exists]
    if missing_init:
        print(f"   Missing {len(missing_init)} __init__.py file(s):")
        for pkg in missing_init:
            print(f"   - {pkg}/__init__.py")
    else:
        print("   All __init__.py files exist.")

    # Summary
    print("\n" + "=" * 60)
    print("Cleanup Summary")
    print("=" * 60)
    print(f"Empty files: {len(empty_files)}")
    print(f"Missing directories: {len(missing_dirs)}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Missing __init__.py: {len(missing_init)}")

    if not empty_files and not missing_dirs and not missing_files and not missing_init:
        print("\nRepository is clean and ready for supervisor review!")
    else:
        print("\nSome issues found. Please address them before submission.")


if __name__ == "__main__":
    main()