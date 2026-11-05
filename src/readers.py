from pathlib import Path
import json

import pandas as pd


CANONICAL_COLUMNS = [
    "country",
    "continent",
    "year",
    "lifeExp",
    "pop",
    "gdpPercap",
]


def extract_nested_value(record: dict, path: str):
    """Retrieve a value using a dot-separated dictionary path."""
    value = record

    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Missing JSON field: {path}")

        value = value[key]

    return value


def normalize_columns(
    df: pd.DataFrame,
    mapping: dict,
    source_file: str,
) -> pd.DataFrame:
    """Rename source columns and preserve original record positions."""

    required_columns = set(mapping.keys())
    available_columns = set(df.columns)

    missing = required_columns - available_columns

    if missing:
        raise ValueError(
            f"{source_file}: Missing required columns: {sorted(missing)}"
        )

    renamed = df.rename(columns=mapping)

    result = renamed[CANONICAL_COLUMNS].copy()

    result.insert(0, "source_record", range(1, len(result) + 1))
    result.insert(0, "source_file", source_file)

    return result


def read_csv(path: Path, settings: dict) -> pd.DataFrame:
    """Read and normalize the supplied CSV file."""

    df = pd.read_csv(
        path,
        encoding=settings["encoding"],
        sep=settings["separator"],
        comment=settings["comment_prefix"],
        dtype=str,
        keep_default_na=False,
    )

    mapping = settings["source_to_canonical"]

    return normalize_columns(df, mapping, path.name)


def read_excel(path: Path, settings: dict) -> pd.DataFrame:
    """Read the observations sheet from the supplied Excel file."""

    df = pd.read_excel(
        path,
        sheet_name=settings["sheet_name"],
        engine=settings["engine"],
        dtype=str,
        keep_default_na=False,
    )

    mapping = settings["source_to_canonical"]

    return normalize_columns(df, mapping, path.name)


def read_json(path: Path, settings: dict) -> pd.DataFrame:
    """Read nested JSON records and flatten them into canonical columns."""

    with open(path, "r", encoding=settings["encoding"]) as file:
        data = json.load(file)

    records = data

    for key in settings["record_path"].split("."):
        if not isinstance(records, dict) or key not in records:
            raise ValueError(f"Missing JSON record path: {settings['record_path']}")

        records = records[key]

    if not isinstance(records, list):
        raise ValueError("JSON records must be stored in a list.")

    mapping = settings["source_to_canonical"]

    flattened_records = []

    for record in records:
        flattened = {}

        for source_field, canonical_field in mapping.items():
            flattened[canonical_field] = extract_nested_value(
                record,
                source_field,
            )

        flattened_records.append(flattened)

    df = pd.DataFrame(
        flattened_records,
        columns=CANONICAL_COLUMNS,
    )

    df.insert(0, "source_record", range(1, len(df) + 1))
    df.insert(0, "source_file", path.name)

    return df


def read_parquet(path: Path, settings: dict) -> pd.DataFrame:
    """Read Parquet records while preserving their original values."""

    df = pd.read_parquet(
        path,
        engine=settings["engine"],
    )

    mapping = settings["source_to_canonical"]

    return normalize_columns(df, mapping, path.name)


def read_input(path: Path, settings: dict) -> pd.DataFrame:
    """Select the appropriate reader based on the configured format."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    file_format = settings["format"]

    readers = {
        "csv": read_csv,
        "xlsx": read_excel,
        "json": read_json,
        "parquet": read_parquet,
    }

    if file_format not in readers:
        raise ValueError(f"Unsupported file format: {file_format}")

    return readers[file_format](path, settings)
