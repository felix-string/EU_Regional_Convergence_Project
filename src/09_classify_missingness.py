from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
outputs = root / "outputs"

decisions = pd.read_csv(
    outputs / "nuts_boundary_decisions.csv"
)

boundary_codes = {
    "NL35", "NL36",
    "PT19", "PT1A", "PT1B", "PT1C", "PT1D"
}

unemployment_quality_codes = {
    "DE22", "DE23", "DE24", "DE26", "DE50",
    "DE73", "DEB1", "DEB2", "DEC0", "DED4",
    "FI20", "PL43"
}

decisions["suspected_reason"] = "Other incomplete coverage"

decisions.loc[
    decisions["region_code"].isin(boundary_codes),
    "suspected_reason"
] = "NUTS classification transition"

decisions.loc[
    decisions["region_code"].isin(
        unemployment_quality_codes
    ),
    "suspected_reason"
] = "Unemployment unavailable or low reliability"

decisions.loc[
    decisions["region_code"] == "FRY5",
    "suspected_reason"
] = "Both labour indicators unavailable in 2021-2023"

decisions["provisional_treatment"] = (
    "Keep in GDP master panel; exclude from "
    "three-indicator balanced panel"
)

decisions.loc[
    decisions["region_code"].isin(boundary_codes),
    "provisional_treatment"
] = (
    "Do not backfill; use from 2019 onward or "
    "exclude from 2016-2024 balanced analysis"
)

decisions.loc[
    decisions["region_code"] == "FI20",
    "provisional_treatment"
] = (
    "Retain for GDP and employment; exclude from "
    "analyses requiring unemployment"
)

decisions.to_csv(
    outputs / "nuts_boundary_decisions_reviewed.csv",
    index=False
)

print(
    decisions[
        [
            "region_code",
            "region_name",
            "suspected_reason",
            "provisional_treatment"
        ]
    ].to_string(index=False)
)

print("\nSaved to:")
print(
    outputs /
    "nuts_boundary_decisions_reviewed.csv"
)