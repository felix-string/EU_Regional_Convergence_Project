from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
processed = root / "data" / "processed"
outputs = root / "outputs"

gdp = pd.read_csv(processed / "gdp_clean.csv")
employment = pd.read_csv(
    processed / "employment_clean.csv"
)
unemployment = pd.read_csv(
    processed / "unemployment_clean.csv"
)

# Keep only the columns required from the labour datasets
employment = employment[
    [
        "region_code",
        "year",
        "employment_rate",
        "employment_rate_flag"
    ]
]

unemployment = unemployment[
    [
        "region_code",
        "year",
        "unemployment_rate",
        "unemployment_rate_flag"
    ]
]

# GDP is the regional base because all GDP codes
# also occur in the two labour-market datasets
master = gdp.merge(
    employment,
    on=["region_code", "year"],
    how="left",
    validate="one_to_one"
)

master = master.merge(
    unemployment,
    on=["region_code", "year"],
    how="left",
    validate="one_to_one"
)

# Arrange the columns clearly
column_order = [
    "region_code",
    "region_name",
    "country_code",
    "year",
    "gdp_pps_index",
    "employment_rate",
    "unemployment_rate",
    "gdp_pps_index_flag",
    "employment_rate_flag",
    "unemployment_rate_flag"
]

master = master[column_order]

# Sort consistently
master = master.sort_values(
    ["region_code", "year"]
).reset_index(drop=True)

# Save the full panel
master.to_csv(
    processed / "eu_nuts2_master_panel.csv",
    index=False
)

print("Merge completed.")
print("Master rows:", len(master))
print(
    "Unique regions:",
    master["region_code"].nunique()
)
print(
    "Duplicate region-year rows:",
    master.duplicated(
        ["region_code", "year"]
    ).sum()
)

print("\nMissing values:")
print(
    master[
        [
            "gdp_pps_index",
            "employment_rate",
            "unemployment_rate"
        ]
    ].isna().sum()
)

analysis_columns = [
    "gdp_pps_index",
    "employment_rate",
    "unemployment_rate"
]

master["complete_row"] = (
    master[analysis_columns]
    .notna()
    .all(axis=1)
)

region_audit = (
    master.groupby(
        ["region_code", "region_name", "country_code"]
    )
    .agg(
        years_present=("year", "nunique"),
        complete_years=("complete_row", "sum"),
        gdp_missing=(
            "gdp_pps_index",
            lambda x: x.isna().sum()
        ),
        employment_missing=(
            "employment_rate",
            lambda x: x.isna().sum()
        ),
        unemployment_missing=(
            "unemployment_rate",
            lambda x: x.isna().sum()
        )
    )
    .reset_index()
)

region_audit["complete_all_years"] = (
    region_audit["complete_years"] == 9
)

region_audit.to_csv(
    outputs / "region_completeness_audit.csv",
    index=False
)

balanced_codes = region_audit.loc[
    region_audit["complete_all_years"],
    "region_code"
]

balanced = master[
    master["region_code"].isin(balanced_codes)
].copy()

balanced = balanced.drop(columns="complete_row")

balanced.to_csv(
    processed / "eu_nuts2_balanced_panel.csv",
    index=False
)

print("\nBalanced panel created.")
print(
    "Regions complete in all 9 years:",
    balanced["region_code"].nunique()
)
print("Balanced-panel rows:", len(balanced))
print(
    "Regions excluded:",
    244 - balanced["region_code"].nunique()
)