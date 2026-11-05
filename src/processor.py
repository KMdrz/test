
from __future__ import annotations

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = [
    "country",
    "continent",
    "year",
    "lifeExp",
    "pop",
    "gdpPercap",
]


def create_canonical_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Create the shared canonical dataset from one verified input.

    The four recovered datasets must already have passed comparison.
    Only one copy is used to avoid multiplying the same observations.
    """
    canonical = dataset[CANONICAL_COLUMNS].copy()

    canonical = canonical.sort_values(
        ["country", "year"]
    ).reset_index(drop=True)

    return canonical


def create_asia_analysis(
    canonical: pd.DataFrame,
    reporting_year: int,
) -> pd.DataFrame:
    """
    Select Asian observations for the configured year.

    Adds population in millions and ranks life expectancy using
    the minimum ranking method.
    """
    selected = canonical.loc[
        (canonical["continent"] == "Asia")
        & (canonical["year"] == reporting_year)
    ].copy()

    selected["pop_millions"] = selected["pop"] / 1_000_000

    selected["lifeExp_rank"] = selected["lifeExp"].rank(
        method="min",
        ascending=False,
    )

    selected = selected.sort_values(
        ["lifeExp_rank", "country"]
    ).reset_index(drop=True)

    return selected


def create_country_history(canonical: pd.DataFrame) -> pd.DataFrame:
    """
    Create the Philippines' historical observations.

    Calculates population in millions and life expectancy change
    relative to the first recorded year.
    """
    history = canonical.loc[
        canonical["country"] == "Philippines"
    ].copy()

    history = history.sort_values("year").reset_index(drop=True)

    history["pop_millions"] = history["pop"] / 1_000_000

    if history.empty:
        raise ValueError("Philippines records are missing.")

    first_life_expectancy = history.loc[0, "lifeExp"]

    history["lifeExp_change_from_first"] = (
        history["lifeExp"] - first_life_expectancy
    )

    return history


def create_population_index(canonical: pd.DataFrame) -> pd.DataFrame:
    """
    Create a population index for three Asian countries.

    Population in 1982 is the baseline (100).
    Broadcasting is used to calculate all index values.
    """
    countries = ["Philippines", "Indonesia", "Thailand"]
    years = [1982, 1987, 1992, 1997, 2002, 2007]

    selected = canonical.loc[
        canonical["country"].isin(countries)
        & canonical["year"].isin(years),
        ["country", "year", "pop"],
    ]

    population_table = selected.pivot(
        index="country",
        columns="year",
        values="pop",
    )

    population_table = population_table.reindex(
        index=countries,
        columns=years,
    )

    if population_table.isna().any().any():
        raise ValueError(
            "Missing population observations for the population index."
        )

    population_array = population_table.to_numpy(dtype=float)

    # Keep the original shape for reporting.
    original_shape = population_array.shape

    # Select the first year's population for each country.
    baseline = population_array[:, 0:1]

    # Broadcasting: divide every year by the 1982 baseline.
    indexed_array = (population_array / baseline) * 100

    # Independently verify the broadcasting result with loops.
    loop_result = np.zeros_like(indexed_array)

    for row in range(population_array.shape[0]):
        for column in range(population_array.shape[1]):
            loop_result[row, column] = (
                population_array[row, column]
                / population_array[row, 0]
            ) * 100

    if not np.allclose(indexed_array, loop_result):
        raise ValueError(
            "Broadcasting result does not match the loop calculation."
        )

    result = pd.DataFrame(
        indexed_array,
        index=countries,
        columns=years,
    )

    result.index.name = "country"
    result = result.reset_index()

    print("Population array shape before broadcasting:", original_shape)
    print("Baseline array shape:", baseline.shape)
    print("Population array shape after broadcasting:", indexed_array.shape)
    print("Independent loop verification: PASSED")

    return result


def calculate_asian_population(
    canonical: pd.DataFrame,
    reporting_year: int,
) -> int:
    """
    Calculate the total population of Asia for a given year.

    NumPy Boolean selection is used to select the matching records.
    """
    continents = canonical["continent"].to_numpy()
    years = canonical["year"].to_numpy()
    populations = canonical["pop"].to_numpy()

    selection = (
        (continents == "Asia")
        & (years == reporting_year)
    )

    selected_populations = populations[selection]

    return int(np.sum(selected_populations))
