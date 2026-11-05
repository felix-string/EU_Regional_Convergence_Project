from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
outputs = root / "outputs"

audit = pd.read_csv(
    outputs / "region_completeness_audit.csv"
)

excluded = audit[
    audit["complete_all_years"] == False
].copy()

excluded = excluded.sort_values(
    [
        "unemployment_missing",
        "employment_missing",
        "region_code"
    ],
    ascending=[False, False, True]
)

print("Excluded regions:", len(excluded))

print("\nExcluded regions by country:")
print(
    excluded.groupby("country_code")
    .size()
    .sort_values(ascending=False)
)

print("\nExcluded-region details:")
print(
    excluded[
        [
            "region_code",
            "region_name",
            "country_code",
            "employment_missing",
            "unemployment_missing"
        ]
    ].to_string(index=False)
)

excluded.to_csv(
    outputs / "excluded_regions.csv",
    index=False
)

print("\nSaved to:")
print(outputs / "excluded_regions.csv")
