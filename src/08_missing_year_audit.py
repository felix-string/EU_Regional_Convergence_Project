from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
processed = root / "data" / "processed"
outputs = root / "outputs"

master = pd.read_csv(
    processed / "eu_nuts2_master_panel.csv"
)

excluded = pd.read_csv(
    outputs / "excluded_regions.csv"
)

excluded_codes = excluded["region_code"].tolist()

audit = master[
    master["region_code"].isin(excluded_codes)
].copy()

audit["employment_missing"] = (
    audit["employment_rate"].isna()
)

audit["unemployment_missing"] = (
    audit["unemployment_rate"].isna()
)

missing_rows = audit[
    audit["employment_missing"]
    | audit["unemployment_missing"]
].copy()

missing_rows = missing_rows[
    [
        "region_code",
        "region_name",
        "country_code",
        "year",
        "employment_rate",
        "unemployment_rate",
        "employment_rate_flag",
        "unemployment_rate_flag"
    ]
]

missing_rows = missing_rows.sort_values(
    ["country_code", "region_code", "year"]
)

print("Missing region-year observations:")
print(
    missing_rows.to_string(
        index=False
    )
)

missing_rows.to_csv(
    outputs / "missing_year_audit.csv",
    index=False
)

print("\nSaved to:")
print(outputs / "missing_year_audit.csv")

decision_table = (
    missing_rows.groupby(
        [
            "region_code",
            "region_name",
            "country_code"
        ]
    )
    .agg(
        first_missing_year=("year", "min"),
        last_missing_year=("year", "max"),
        number_missing_years=("year", "nunique"),
        employment_missing_years=(
            "employment_rate",
            lambda x: x.isna().sum()
        ),
        unemployment_missing_years=(
            "unemployment_rate",
            lambda x: x.isna().sum()
        )
    )
    .reset_index()
)

decision_table["suspected_reason"] = (
    "Requires review"
)

decision_table["provisional_treatment"] = (
    "Keep in master panel; exclude from balanced panel"
)

decision_table.to_csv(
    outputs / "nuts_boundary_decisions.csv",
    index=False
)

print("\nNUTS decision table saved to:")
print(outputs / "nuts_boundary_decisions.csv")