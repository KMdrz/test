
from __future__ import annotations

import pandas as pd


KEY_COLUMNS = ["country", "year"]

TEXT_COLUMNS = ["country", "continent"]

INTEGER_COLUMNS = ["year", "pop"]

FLOAT_COLUMNS = ["lifeExp", "gdpPercap"]

CANONICAL_COLUMNS = [
    "country",
    "continent",
    "year",
    "lifeExp",
    "pop",
    "gdpPercap",
]


def compare_datasets(
    datasets: dict[str, pd.DataFrame],
    absolute_tolerance: float = 1e-8,
) -> pd.DataFrame:
    """
    Compare recovered datasets against the first dataset.

    Checks country-year keys, text fields, integer fields, and
    floating-point fields. Returns a report of comparison results.
    Raises ValueError if any comparison fails.
    """
    if len(datasets) < 2:
        raise ValueError("At least two datasets are required for comparison.")

    dataset_names = list(datasets.keys())
    reference_name = dataset_names[0]
    reference = datasets[reference_name].copy()

    report = []

    # Ensure the reference dataset has the expected columns.
    if list(reference.columns) != CANONICAL_COLUMNS:
        raise ValueError(
            f"{reference_name} does not have the canonical column order."
        )

    # Sort by the country-year key so row order does not affect comparison.
    reference = reference.sort_values(KEY_COLUMNS).reset_index(drop=True)

    for dataset_name, dataset in datasets.items():
        candidate = dataset.copy()

        if list(candidate.columns) != CANONICAL_COLUMNS:
            raise ValueError(
                f"{dataset_name} does not have the canonical column order."
            )

        candidate = candidate.sort_values(KEY_COLUMNS).reset_index(drop=True)

        # 1. Compare the number of records.
        row_count_pass = len(reference) == len(candidate)
        report.append({
            "comparison": f"{dataset_name}: row count",
            "expected": len(reference),
            "actual": len(candidate),
            "passed": row_count_pass,
        })

        if not row_count_pass:
            continue

        # 2. Compare country-year keys.
        keys_pass = reference[KEY_COLUMNS].equals(candidate[KEY_COLUMNS])
        report.append({
            "comparison": f"{dataset_name}: country-year keys",
            "expected": "Same keys as reference",
            "actual": "Match" if keys_pass else "Different keys",
            "passed": keys_pass,
        })

        if not keys_pass:
            continue

        # 3. Compare text and integer fields exactly.
        for column in TEXT_COLUMNS + INTEGER_COLUMNS:
            passed = reference[column].equals(candidate[column])

            report.append({
                "comparison": f"{dataset_name}: {column}",
                "expected": "Exact match",
                "actual": "Match" if passed else "Different values",
                "passed": passed,
            })

        # 4. Compare floating-point fields using absolute tolerance.
        for column in FLOAT_COLUMNS:
            differences = (
                reference[column].astype(float)
                - candidate[column].astype(float)
            ).abs()

            maximum_difference = (
                float(differences.max()) if len(differences) else 0.0
            )

            passed = bool((differences <= absolute_tolerance).all())

            report.append({
                "comparison": f"{dataset_name}: {column}",
                "expected": f"Difference <= {absolute_tolerance}",
                "actual": f"Maximum difference = {maximum_difference}",
                "passed": passed,
            })

    report_df = pd.DataFrame(report)

    # Stop the pipeline if even one comparison fails.
    if not report_df["passed"].all():
        failed = report_df.loc[~report_df["passed"], "comparison"].tolist()
        raise ValueError(
            "Dataset comparison failed:\n" + "\n".join(failed)
        )

    return report_df
