
"""Shared fixtures for the recovery and validation tests."""

import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def manifest(project_root: Path) -> dict:
    """Load the project's actual manifest."""
    manifest_path = project_root / "data" / "manifest.json"

    with manifest_path.open("r", encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture
def valid_records() -> pd.DataFrame:
    """Create a small dataset containing valid canonical records."""
    return pd.DataFrame([
        {
            "country": "Afghanistan",
            "continent": "Asia",
            "year": 1952,
            "lifeExp": 28.801,
            "pop": 8425333,
            "gdpPercap": 779.445314,
            "source_file": "test.csv",
            "source_record": 1,
        },
        {
            "country": "Albania",
            "continent": "Europe",
            "year": 1952,
            "lifeExp": 55.230,
            "pop": 1282697,
            "gdpPercap": 1601.056136,
            "source_file": "test.csv",
            "source_record": 2,
        },
    ])


@pytest.fixture
def canonical_columns(manifest: dict) -> list[str]:
    """Return the canonical column order from the manifest."""
    return manifest["canonical_order"]


@pytest.fixture
def fixtures_dir() -> Path:
    """Create the test fixtures directory outside the data directory."""
    directory = Path(__file__).parent / "fixtures"
    directory.mkdir(parents=True, exist_ok=True)
    return directory
