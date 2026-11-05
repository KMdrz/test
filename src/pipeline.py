
"""Main processing pipeline for the four-format data recovery project."""

from __future__ import annotations

import hashlib
import json
import shutil
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from src.readers import read_input
from src.validator import validate_dataset
from src.comparator import compare_datasets
from src.processor import (
    create_canonical_dataset,
    create_asia_analysis,
    create_country_history,
    create_population_index,
    calculate_asian_population,
)
from src.outputs import (
    export_canonical_formats,
    export_analysis_outputs,
    export_validation_outputs,
    export_audit_log,
    export_validation_report,
)
from src.export_validator import verify_all_exports


class ProcessingRun:
    """Manage one complete data recovery and validation run."""

    def __init__(self, config_path: Path) -> None:
        """Load configuration and initialize the processing run."""
        self.config_path = config_path
        self.project_dir = config_path.parent

        with config_path.open("r", encoding="utf-8") as file:
            self.config: dict[str, Any] = json.load(file)

        manifest_path = self.project_dir / self.config["manifest_path"]

        with manifest_path.open("r", encoding="utf-8") as file:
            self.manifest: dict[str, Any] = json.load(file)

        self.data_dir = self.project_dir / self.config["data_dir"]
        self.output_dir = self.project_dir / self.config["output_dir"]
        self.staging_dir = self.project_dir / self.config["staging_dir"]

        self.reporting_year = int(self.config.get("reporting_year", 2007))
        self.audit_records: list[dict[str, Any]] = []
        self.checks: list[dict[str, Any]] = []

    def add_audit(
        self,
        action: str,
        source: str,
        details: str,
        status: str = "SUCCESS",
    ) -> None:
        """Append an audit record."""
        self.audit_records.append({
            "action": action,
            "source": source,
            "details": details,
            "status": status,
        })

    def add_check(
        self,
        check: str,
        expected: Any,
        actual: Any,
        tolerance: str = "Exact",
    ) -> None:
        """Record a validation check."""
        self.checks.append({
            "check": check,
            "expected": expected,
            "actual": actual,
            "tolerance": tolerance,
            "pass": expected == actual,
        })

    def verify_input_file(self, settings: dict[str, Any]) -> Path:
        """Verify that an input file exists and matches its manifest hash."""
        path = self.data_dir / settings["file"]

        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

        actual_size = path.stat().st_size
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()

        if actual_size != settings["bytes"]:
            raise ValueError(
                f"File size mismatch for {path.name}: "
                f"expected {settings['bytes']}, got {actual_size}"
            )

        if actual_hash != settings["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for {path.name}. "
                "The input file may have been modified or corrupted."
            )

        self.add_audit(
            "INPUT_HASH_CHECK",
            path.name,
            f"SHA-256 and file size verified ({actual_size} bytes).",
        )
        return path

    def prepare_staging(self) -> None:
        """Create a clean staging directory for this run."""
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)

        self.staging_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> bool:
        """Execute recovery, validation, analysis, export, and verification."""
        try:
            print("\n========== DATA RECOVERY PIPELINE ==========")
            print(f"Reporting year: {self.reporting_year}")

            # 1. Read and validate all four input files.
            cleaned_datasets: dict[str, pd.DataFrame] = {}
            quarantine_frames: list[pd.DataFrame] = []
            duplicate_frames: list[pd.DataFrame] = []

            expected = self.manifest["expected_per_file"]

            for settings in self.manifest["files"]:
                path = self.verify_input_file(settings)

                print(f"\nReading: {path.name}")
                raw_data = read_input(path, settings)

                result = validate_dataset(raw_data, self.manifest)

                canonical_columns = self.manifest["canonical_order"]

                cleaned_datasets[path.name] = (
                result.valid_records[canonical_columns].copy()
                )
                quarantine_frames.append(result.quarantine)
                duplicate_frames.append(result.duplicates)

                self.add_check(
                    f"{path.name} raw rows",
                    expected["raw_rows"],
                    len(raw_data),
                )
                self.add_check(
                    f"{path.name} rejected rows",
                    expected["rejected_rows"],
                    len(result.quarantine),
                )
                self.add_check(
                    f"{path.name} valid before deduplication",
                    expected["valid_before_deduplication"],
                    len(result.valid_records) + len(result.duplicates),
                )
                self.add_check(
                    f"{path.name} exact duplicates removed",
                    expected["exact_duplicates_removed"],
                    len(result.duplicates),
                )
                self.add_check(
                    f"{path.name} canonical rows",
                    expected["canonical_rows"],
                    len(result.valid_records),
                )

                self.add_audit(
                    "INPUT_VALIDATION",
                    path.name,
                    (
                        f"Raw: {len(raw_data)}, "
                        f"quarantined: {len(result.quarantine)}, "
                        f"duplicates: {len(result.duplicates)}, "
                        f"valid: {len(result.valid_records)}"
                    ),
                )

            quarantine = pd.concat(
                quarantine_frames, ignore_index=True
            )
            duplicates = pd.concat(
                duplicate_frames, ignore_index=True
            )

            print("\nAll four files have been read and validated.")

            # 2. Compare the four cleaned datasets.
            print("\nComparing the four recovered datasets...")

            comparison_report = compare_datasets(cleaned_datasets)

            if not comparison_report["passed"].all():
                failed_comparisons = comparison_report.loc[
                    ~comparison_report["passed"]
                ]

                raise ValueError(
                     "Cross-format comparison failed:\n"
                        + failed_comparisons.to_string(index=False)
                )

            self.add_audit(
                "CROSS_FORMAT_COMPARISON",
                "All four input files",
                "Cleaned datasets compared using country/year keys.",
            )

            # 3. Create the canonical dataset only after comparison.
            canonical = create_canonical_dataset(
                cleaned_datasets[next(iter(cleaned_datasets))]
            )

            # 4. Run the required analysis.
            asia = create_asia_analysis(canonical, self.reporting_year)
            philippines = create_country_history(canonical)
            population_index = create_population_index(canonical)

            asian_population = calculate_asian_population(
                canonical, self.reporting_year
            )

            controls = self.manifest["controls"]

            self.add_check(
                "Number of countries",
                controls["countries"],
                canonical["country"].nunique(),
            )
            self.add_check(
                "Philippines historical rows",
                controls["philippines_rows"],
                len(philippines),
            )
            self.add_check(
                f"Asia rows in {self.reporting_year}",
                controls["asia_2007_rows"]
                if self.reporting_year == 2007
                else len(asia),
                len(asia),
            )

            if self.reporting_year == 2007:
                self.add_check(
                    "Asia rows in 2007",
                    controls["asia_2007_rows"],
                    len(asia),
                )

                philippines_2007 = canonical.loc[
                    (canonical["country"] == "Philippines")
                    & (canonical["year"] == 2007),
                    "pop",
                ].iloc[0]

                self.add_check(
                    "Philippines population in 2007",
                    controls["philippines_2007_population"],
                    int(philippines_2007),
                )

            print("\nAnalysis completed.")
            print(
                f"Asian population in {self.reporting_year}: "
                f"{asian_population:,}"
            )

            # 5. Stop before exporting if any data checks failed.
            failed_checks = [
                check for check in self.checks if not check["pass"]
            ]

            if failed_checks:
                raise ValueError(
                    f"{len(failed_checks)} validation check(s) failed."
                )

            # 6. Export everything into staging first.
            self.prepare_staging()

            export_canonical_formats(canonical, self.staging_dir)

            export_analysis_outputs(
                asia,
                philippines,
                population_index,
                self.staging_dir,
            )

            export_validation_outputs(
                quarantine,
                duplicates,
                comparison_report,
                self.staging_dir,
            )

            export_audit_log(self.audit_records, self.staging_dir)

            # 7. Reload and verify the four canonical exports.
            print("\nVerifying exported files...")

            round_trip_report = verify_all_exports(
                canonical, self.staging_dir
            )

            self.add_check(
                "Canonical export round-trip verification",
                True,
                True,
                "All four exports must match the canonical dataset.",
            )

            self.add_audit(
                "ROUND_TRIP_VERIFICATION",
                "Four canonical exports",
                "All exported formats successfully reloaded and verified.",
            )

            # Include the final audit record and validation report.
            export_audit_log(self.audit_records, self.staging_dir)
            export_validation_report(self.checks, self.staging_dir)

            # 8. Publish staged files only after all checks pass.
            self.output_dir.mkdir(parents=True, exist_ok=True)

            for staged_file in self.staging_dir.iterdir():
                if staged_file.is_file():
                    shutil.copy2(
                        staged_file,
                        self.output_dir / staged_file.name,
                    )

            print("\n========== PIPELINE SUCCESS ==========")
            print(f"Canonical records: {len(canonical)}")
            print(f"Quarantined records: {len(quarantine)}")
            print(f"Duplicate records removed: {len(duplicates)}")
            print(f"Reporting year: {self.reporting_year}")
            print(f"Final outputs: {self.output_dir}")

            return True

        except Exception as error:
            print("\n========== PIPELINE FAILED ==========")
            print(f"Reason: {error}")
            traceback.print_exc()

            # Preserve a failure report without publishing staged outputs.
            if self.checks:
                pd.DataFrame(self.checks).to_csv(
                    self.output_dir / "validation_failed.csv",
                    index=False,
                )

            pd.DataFrame(self.audit_records).to_csv(
                self.output_dir / "audit_log_failed.csv",
                index=False,
            )

            return False