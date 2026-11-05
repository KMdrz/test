
from dataclasses import dataclass
import math

import pandas as pd


CANONICAL_COLUMNS = [
    "country",
    "continent",
    "year",
    "lifeExp",
    "pop",
    "gdpPercap",
]

QUARANTINE_COLUMNS = [
    "source_file",
    "source_record",
    "reason",
    *CANONICAL_COLUMNS,
]


@dataclass
class ValidationResult:
    """Store the results of validating one input dataset."""

    valid_records: pd.DataFrame
    quarantine: pd.DataFrame
    duplicates: pd.DataFrame


def is_missing(value) -> bool:
    """Check whether a value is missing or empty."""

    if value is None or pd.isna(value):
        return True

    return isinstance(value, str) and value.strip() == ""


def parse_number(value, column: str) -> float:
    """Convert a value into a finite number."""

    if is_missing(value):
        raise ValueError(f"{column} is missing")

    try:
        number = float(str(value).strip())
    except (ValueError, TypeError):
        raise ValueError(f"{column} is not a valid number")

    if not math.isfinite(number):
        raise ValueError(f"{column} must be finite")

    return number


def validate_record(
    record: pd.Series,
    manifest: dict,
) -> tuple[dict, list[str]]:
    """Validate one record and return normalized values and errors."""

    normalized = {}
    reasons = []

    # Validate country
    country = str(record["country"]).strip() if not is_missing(
        record["country"]
    ) else ""

    if not country:
        reasons.append("country is empty")

    normalized["country"] = country

    # Validate continent
    continent = str(record["continent"]).strip() if not is_missing(
        record["continent"]
    ) else ""

    if not continent:
        reasons.append("continent is empty")
    elif continent not in manifest["allowed_continents"]:
        reasons.append("continent is outside allowed values")

    normalized["continent"] = continent

    # Validate year
    try:
        year = parse_number(record["year"], "year")

        if not year.is_integer():
            reasons.append("year must be integral")
        elif int(year) not in manifest["allowed_years"]:
            reasons.append("year is outside allowed values")
        else:
            normalized["year"] = int(year)

    except ValueError as error:
        reasons.append(str(error))

    # Validate life expectancy
    try:
        life_exp = parse_number(record["lifeExp"], "lifeExp")

        if not 0 < life_exp <= 120:
            reasons.append("lifeExp must be greater than 0 and at most 120")
        else:
            normalized["lifeExp"] = life_exp

    except ValueError as error:
        reasons.append(str(error))

    # Validate population
    try:
        population = parse_number(record["pop"], "pop")

        if not population.is_integer():
            reasons.append("pop must be integral")
        elif population <= 0:
            reasons.append("pop must be greater than 0")
        else:
            normalized["pop"] = int(population)

    except ValueError as error:
        reasons.append(str(error))

    # Validate GDP per capita
    try:
        gdp = parse_number(record["gdpPercap"], "gdpPercap")

        if gdp <= 0:
            reasons.append("gdpPercap must be greater than 0")
        else:
            normalized["gdpPercap"] = gdp

    except ValueError as error:
        reasons.append(str(error))

    return normalized, reasons


def detect_conflicting_keys(records: pd.DataFrame) -> None:
    """Stop processing when a country-year key has different values."""

    if records.empty:
        return

    key_columns = ["country", "year"]

    grouped = records.groupby(key_columns, dropna=False)

    for key, group in grouped:
        distinct_records = group[CANONICAL_COLUMNS].drop_duplicates()

        if len(distinct_records) > 1:
            raise ValueError(
                f"Conflicting key detected: country={key[0]}, year={key[1]}"
            )


def validate_dataset(
    df: pd.DataFrame,
    manifest: dict,
) -> ValidationResult:
    """Validate records, quarantine invalid rows and remove duplicates."""

    valid_rows = []
    quarantined_rows = []

    for _, record in df.iterrows():

        normalized, reasons = validate_record(record, manifest)

        if reasons:
            quarantine_record = {
                "source_file": record["source_file"],
                "source_record": record["source_record"],
                "reason": "; ".join(reasons),
            }

            # Preserve the original six field values
            for column in CANONICAL_COLUMNS:
                quarantine_record[column] = record[column]

            quarantined_rows.append(quarantine_record)

        else:
            valid_record = {
                "source_file": record["source_file"],
                "source_record": record["source_record"],
                **normalized,
            }

            valid_rows.append(valid_record)

    valid_df = pd.DataFrame(valid_rows)
    quarantine_df = pd.DataFrame(
        quarantined_rows,
        columns=QUARANTINE_COLUMNS,
    )

    # Check conflicts before removing exact duplicates
    detect_conflicting_keys(valid_df)

    # Remove exact duplicate canonical records
    unique_rows = []
    duplicate_rows = []
    seen_records = {}

    for _, record in valid_df.iterrows():

        record_key = tuple(
            record[column] for column in CANONICAL_COLUMNS
        )

        if record_key in seen_records:

            duplicate_rows.append({
                "source_file": record["source_file"],
                "source_record": record["source_record"],
                "duplicate_of_record": seen_records[record_key],
                "country": record["country"],
                "year": record["year"],
            })

        else:
            seen_records[record_key] = record["source_record"]
            unique_rows.append(record.to_dict())

    clean_df = pd.DataFrame(unique_rows)

    duplicate_df = pd.DataFrame(
        duplicate_rows,
        columns=[
            "source_file",
            "source_record",
            "duplicate_of_record",
            "country",
            "year",
        ],
    )

    return ValidationResult(
        valid_records=clean_df,
        quarantine=quarantine_df,
        duplicates=duplicate_df,
    )
