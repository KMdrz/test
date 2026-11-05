
from __future__ import annotations

from pathlib import Path

import pandas as pd


CANONICAL_COLUMNS = [
    "country",
    "continent",
    "year",
    "lifeExp",
    "pop",
    "gdpPercap",
]


def load_exported_file(path: Path) -> pd.DataFrame:
    """
    Reload an exported file according to its file extension.
    """
    extension = path.suffix.lower()

    if extension == ".csv":
        df = pd.read_csv(path)

    elif extension == ".xlsx":
        df = pd.read_excel(
            path,
            sheet_name="observations",
            engine="openpyxl",
        )

    elif extension == ".json":
        df = pd.read_json(
            path,
            orient="records",
        )

    elif extension == ".parquet":
        df = pd.read_parquet(
            path,
            engine="pyarrow",
        )

    else:
        raise ValueError(f"Unsupported export format: {extension}")

    return df


def verify_schema(df: pd.DataFrame, filename: str) -> None:
    """
    Verify that an exported dataset has the canonical columns and types.
    """
    if list(df.columns) != CANONICAL_COLUMNS:
        raise ValueError(
            f"{filename}: Incorrect column names or column order."
        )

    if df["country"].isna().any():
        raise ValueError(f"{filename}: Missing country values.")

    if df["continent"].isna().any():
        raise ValueError(f"{filename}: Missing continent values.")

    if not pd.api.types.is_integer_dtype(df["year"]):
        raise ValueError(f"{filename}: Year must be an integer.")

    if not pd.api.types.is_integer_dtype(df["pop"]):
        raise ValueError(f"{filename}: Population must be an integer.")

    if not pd.api.types.is_numeric_dtype(df["lifeExp"]):
        raise ValueError(f"{filename}: Life expectancy must be numeric.")

    if not pd.api.types.is_numeric_dtype(df["gdpPercap"]):
        raise ValueError(f"{filename}: GDP per capita must be numeric.")


def compare_export_to_canonical(
    original: pd.DataFrame,
    exported: pd.DataFrame,
    filename: str,
    absolute_tolerance: float = 1e-8,
) -> dict:
    """
    Compare a reloaded export against the original canonical dataset.
    """
    if len(original) != len(exported):
        raise ValueError(
            f"{filename}: Row count mismatch. "
            f"Expected {len(original)}, got {len(exported)}."
        )

    original_sorted = original.sort_values(
        ["country", "year"]
    ).reset_index(drop=True)

    exported_sorted = exported.sort_values(
        ["country", "year"]
    ).reset_index(drop=True)

    for column in ["country", "continent", "year", "pop"]:
        if not original_sorted[column].equals(exported_sorted[column]):
            raise ValueError(
                f"{filename}: Values differ in column '{column}'."
            )

    for column in ["lifeExp", "gdpPercap"]:
        differences = (
            original_sorted[column].astype(float)
            - exported_sorted[column].astype(float)
        ).abs()

        maximum_difference = (
            float(differences.max()) if len(differences) else 0.0
        )

        if not (differences <= absolute_tolerance).all():
            raise ValueError(
                f"{filename}: Floating-point mismatch in '{column}'. "
                f"Maximum difference: {maximum_difference}."
            )

    return {
        "file": filename,
        "expected_rows": len(original),
        "actual_rows": len(exported),
        "result": "PASSED",
    }


def verify_all_exports(
    canonical: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Reload and verify all four canonical exports.

    Raises ValueError if any file fails schema or value verification.
    """
    export_names = [
        "canonical.csv",
        "canonical.xlsx",
        "canonical.json",
        "canonical.parquet",
    ]

    results = []

    for filename in export_names:
        path = output_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Required export is missing: {path}"
            )

        exported = load_exported_file(path)

        verify_schema(exported, filename)

        result = compare_export_to_canonical(
            canonical,
            exported,
            filename,
        )

        results.append(result)

    return pd.DataFrame(results)
