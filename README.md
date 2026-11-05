# Four-Format Data Recovery and Validation

## Project Overview

This project recovers, validates, compares, and analyzes the same Gapminder dataset stored in four different formats: CSV, Excel, JSON, and Parquet.

It identifies invalid records, quarantines rejected rows, removes exact duplicates, detects conflicting country-year records, and verifies that all four recovered datasets agree before generating analytical outputs.

## Features

- Reads CSV, Excel, JSON, and Parquet files.
- Validates records using a shared manifest and canonical schema.
- Preserves rejected records in a quarantine report.
- Detects exact duplicates and conflicting keys.
- Compares recovered datasets across all four formats.
- Performs pandas and NumPy analysis.
- Exports canonical and analytical datasets.
- Reloads and verifies exported files.
- Includes automated tests for 12 required test categories.

## Project Structure

```text
Four-Format-Data-Recovery-and-Validation/
├── data/                  # Input datasets and manifest
├── outputs/               # Generated outputs and reports
├── src/                   # Processing and validation modules
├── tests/                 # Automated tests
├── main.py                # Command-line entry point
├── config.json            # Pipeline configuration
├── report.ipynb           # Jupyter Notebook analysis
├── report.html            # HTML version of the notebook
├── requirements.txt       # Python dependencies
└── test_results.txt       # Automated test results
```

## Installation

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Pipeline

From the project root directory, execute:

```bash
python main.py --config config.json
```

The reporting year can be changed in `config.json`. The default reporting year is 2007.

## Running Automated Tests

Execute:

```bash
python -m pytest tests/test_recovery.py -v
```

## Main Outputs

- `canonical.csv`, `canonical.xlsx`, `canonical.json`, and `canonical.parquet`
- `asia_selected_year.csv`
- `country_history.csv`
- `population_index.csv`
- `quarantine.csv`
- `duplicate_log.csv`
- `comparison.csv`
- `validation.csv`
- `audit_log.csv`

## Dataset Note

The project uses a historical Gapminder teaching dataset. Its records are not intended to represent current population, life expectancy, or GDP observations.

## Team Contributions

See `contributions.csv` for individual responsibilities, contributions, peer reviews, and feedback records.