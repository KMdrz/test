
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def export_canonical_formats(
    canonical: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Export the canonical dataset into CSV, Excel, JSON, and Parquet.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV export
    canonical.to_csv(
        output_dir / "canonical.csv",
        index=False,
        encoding="utf-8",
    )

    # Excel export
    canonical.to_excel(
        output_dir / "canonical.xlsx",
        sheet_name="observations",
        index=False,
        engine="openpyxl",
    )

    # JSON export
    canonical.to_json(
        output_dir / "canonical.json",
        orient="records",
        indent=4,
    )

    # Parquet export
    canonical.to_parquet(
        output_dir / "canonical.parquet",
        index=False,
        engine="pyarrow",
    )


def export_analysis_outputs(
    asia: pd.DataFrame,
    philippines: pd.DataFrame,
    population_index: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Export the three analytical datasets as CSV files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    asia.to_csv(
        output_dir / "asia_selected_year.csv",
        index=False,
        encoding="utf-8",
    )

    philippines.to_csv(
        output_dir / "country_history.csv",
        index=False,
        encoding="utf-8",
    )

    population_index.to_csv(
        output_dir / "population_index.csv",
        index=False,
        encoding="utf-8",
    )


def export_validation_outputs(
    quarantine: pd.DataFrame,
    duplicates: pd.DataFrame,
    comparison_report: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Export quarantine, duplicate, and comparison reports.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    quarantine.to_csv(
        output_dir / "quarantine.csv",
        index=False,
        encoding="utf-8",
    )

    duplicates.to_csv(
        output_dir / "duplicate_log.csv",
        index=False,
        encoding="utf-8",
    )

    comparison_report.to_csv(
        output_dir / "comparison.csv",
        index=False,
        encoding="utf-8",
    )


def export_audit_log(
    audit_records: list[dict],
    output_dir: Path,
) -> None:
    """
    Export processing audit records.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_df = pd.DataFrame(audit_records)

    audit_df.to_csv(
        output_dir / "audit_log.csv",
        index=False,
        encoding="utf-8",
    )


def export_validation_report(
    checks: list[dict],
    output_dir: Path,
) -> None:
    """
    Export the validation report containing expected and actual values.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_df = pd.DataFrame(checks)

    validation_df.to_csv(
        output_dir / "validation.csv",
        index=False,
        encoding="utf-8",
    )
