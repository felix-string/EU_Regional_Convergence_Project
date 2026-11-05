from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
processed = root / "data" / "processed"
outputs = root / "outputs"

master = pd.read_csv(
    processed / "eu_nuts2_master_panel.csv"
)

balanced = pd.read_csv(
    processed / "eu_nuts2_balanced_panel.csv"
)

decisions = pd.read_csv(
    outputs / "nuts_boundary_decisions_reviewed.csv"
)

summary = pd.DataFrame({
    "measure": [
        "Master panel rows",
        "Master panel regions",
        "Master panel start year",
        "Master panel end year",
        "Master GDP missing",
        "Master employment missing",
        "Master unemployment missing",
        "Balanced panel rows",
        "Balanced panel regions",
        "Regions excluded from balanced panel",
        "NUTS transition regions",
        "Unemployment quality regions",
        "Other incomplete coverage regions"
    ],
    "value": [
        len(master),
        master["region_code"].nunique(),
        master["year"].min(),
        master["year"].max(),
        master["gdp_pps_index"].isna().sum(),
        master["employment_rate"].isna().sum(),
        master["unemployment_rate"].isna().sum(),
        len(balanced),
        balanced["region_code"].nunique(),
        (
            master["region_code"].nunique()
            - balanced["region_code"].nunique()
        ),
        (
            decisions["suspected_reason"]
            == "NUTS classification transition"
        ).sum(),
        (
            decisions["suspected_reason"]
            == "Unemployment unavailable or low reliability"
        ).sum(),
        (
            decisions["suspected_reason"]
            == "Both labour indicators unavailable in 2021-2023"
        ).sum()
    ]
})

summary.to_csv(
    outputs / "final_dataset_summary.csv",
    index=False
)

print(summary.to_string(index=False))

print("\nSaved to:")
print(outputs / "final_dataset_summary.csv")