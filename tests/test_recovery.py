
"""Automated tests for the four-format recovery project."""

import pandas as pd
import pytest

from src.readers import read_input
from src.validator import validate_dataset
from src.comparator import compare_datasets
from src.export_validator import compare_export_to_canonical


def test_01_normal_recovery(project_root, manifest):
    """All four original files should recover identical valid records."""
    recovered = {}

    for settings in manifest["files"]:
        path = project_root / "data" / settings["file"]

        raw_data = read_input(path, settings)
        result = validate_dataset(raw_data, manifest)

        recovered[settings["file"]] = result.valid_records[
            manifest["canonical_order"]
        ].copy()

        assert len(raw_data) == 1712
        assert len(result.quarantine) == 6
        assert len(result.duplicates) == 2
        assert len(result.valid_records) == 1704

    comparison = compare_datasets(recovered)

    assert comparison["passed"].all()


def test_02_missing_file(tmp_path, manifest):
    """The reader should report an input file that does not exist."""
    missing_path = tmp_path / "missing.csv"

    settings = {
        "format": "csv",
        "source_to_canonical": {
            "CountryName": "country",
        },
        "encoding": "utf-8",
        "separator": ",",
    }

    with pytest.raises(FileNotFoundError):
        read_input(missing_path, settings)


def test_03_unsupported_format(tmp_path):
    """The reader should reject unsupported file formats."""
    unsupported_path = tmp_path / "records.txt"
    unsupported_path.write_text("test data", encoding="utf-8")

    settings = {
        "format": "txt",
        "source_to_canonical": {},
    }

    with pytest.raises((ValueError, NotImplementedError)):
        read_input(unsupported_path, settings)


def test_04_missing_column(tmp_path):
    """The reader should reject files missing required source columns."""
    csv_path = tmp_path / "missing_column.csv"

    pd.DataFrame({
        "CountryName": ["Afghanistan"],
        "ContinentName": ["Asia"],
    }).to_csv(csv_path, index=False)

    settings = {
        "format": "csv",
        "source_to_canonical": {
            "CountryName": "country",
            "ContinentName": "continent",
            "ObservationYear": "year",
            "LifeYears": "lifeExp",
            "PopulationPersons": "pop",
            "GDPPerPerson": "gdpPercap",
        },
        "encoding": "utf-8",
        "separator": ",",
    }

    with pytest.raises((ValueError, KeyError)):
        read_input(csv_path, settings)


def test_05_missing_key(manifest, valid_records):
    """A record with an empty country key should be quarantined."""
    invalid_data = valid_records.copy()
    invalid_data.loc[0, "country"] = ""

    result = validate_dataset(invalid_data, manifest)

    assert len(result.quarantine) == 1
    assert len(result.valid_records) == 1


def test_06_unparseable_number(manifest, valid_records):
    invalid_data = valid_records.copy()
    invalid_data["lifeExp"] = invalid_data["lifeExp"].astype(object)
    invalid_data.loc[0, "lifeExp"] = "not-a-number"

    result = validate_dataset(invalid_data, manifest)

    assert len(result.quarantine) == 1
    assert len(result.valid_records) == 1


@pytest.mark.parametrize(
    "column, invalid_value",
    [
        ("year", 1952.5),
        ("pop", 1234.5),
    ],
)
def test_07_fractional_values(manifest, valid_records, column, invalid_value):
    invalid_data = valid_records.copy()
    invalid_data[column] = invalid_data[column].astype(float)
    invalid_data.loc[0, column] = invalid_value

    result = validate_dataset(invalid_data, manifest)

    assert len(result.quarantine) == 1
    assert len(result.valid_records) == 1


@pytest.mark.parametrize(
    "column, invalid_value",
    [
        ("lifeExp", float("inf")),
        ("lifeExp", 121),
        ("pop", float("nan")),
        ("gdpPercap", -1),
    ],
)
def test_08_non_finite_or_out_of_range(manifest, valid_records, column, invalid_value):
    invalid_data = valid_records.copy()
    invalid_data.loc[0, column] = invalid_value

    result = validate_dataset(invalid_data, manifest)

    assert len(result.quarantine) == 1
    assert len(result.valid_records) == 1


def test_09_exact_duplicates(manifest, valid_records):
    duplicated_data = pd.concat(
        [valid_records, valid_records.iloc[[0]]],
        ignore_index=True,
    )

    result = validate_dataset(duplicated_data, manifest)

    assert len(result.valid_records) == 2
    assert len(result.duplicates) == 1
    assert len(result.quarantine) == 0


def test_10_conflicting_keys(manifest, valid_records):
    conflicting_data = valid_records.copy()

    conflicting_row = valid_records.iloc[[0]].copy()
    conflicting_row.loc[:, "pop"] = 99999999

    conflicting_data = pd.concat(
        [conflicting_data, conflicting_row],
        ignore_index=True,
    )

    with pytest.raises(ValueError):
        validate_dataset(conflicting_data, manifest)


def test_11_reordered_records(manifest, valid_records):
    canonical_data = valid_records[manifest["canonical_order"]].copy()

    reordered_data = canonical_data.iloc[::-1].reset_index(drop=True)

    comparison = compare_datasets({
        "original": canonical_data,
        "reordered": reordered_data,
    })

    assert comparison["passed"].all()


def test_12_corrupted_export(manifest, valid_records):
    canonical_data = valid_records[manifest["canonical_order"]].copy()

    corrupted_data = canonical_data.copy()
    corrupted_data.loc[0, "pop"] += 1000

    with pytest.raises(ValueError):
        compare_export_to_canonical(
            canonical_data,
            corrupted_data,
            "corrupted.csv",
        )
