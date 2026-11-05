from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
processed = root / "data" / "processed"
outputs = root / "outputs"

files = {
    "GDP": processed / "gdp_clean.csv",
    "Employment": processed / "employment_clean.csv",
    "Unemployment": processed / "unemployment_clean.csv"
}

value_columns = {
    "GDP": "gdp_pps_index",
    "Employment": "employment_rate",
    "Unemployment": "unemployment_rate"
}

results = []

for dataset, file_path in files.items():
    df = pd.read_csv(file_path)
    value_column = value_columns[dataset]

    print(f"\n{'=' * 50}")
    print(dataset)
    print("=" * 50)

    print("Rows:", len(df))
    print("Unique regions:", df["region_code"].nunique())
    print("Years:", sorted(df["year"].unique()))
    print("Duplicate region-year rows:",
          df.duplicated(["region_code", "year"]).sum())
    print("Missing values:", df[value_column].isna().sum())
    print("Minimum value:", df[value_column].min())
    print("Maximum value:", df[value_column].max())

    yearly = (
        df.groupby("year")
        .agg(
            total_regions=("region_code", "nunique"),
            available_values=(value_column, "count"),
            missing_values=(value_column, lambda x: x.isna().sum())
        )
        .reset_index()
    )

    yearly["dataset"] = dataset
    results.append(yearly)

    print("\nRegions and available values by year:")
    print(yearly.to_string(index=False))

quality_report = pd.concat(results, ignore_index=True)

quality_report.to_csv(
    outputs / "region_counts_by_year.csv",
    index=False
)

print("\nQuality checking completed.")
print("Report saved to:")
print(outputs / "region_counts_by_year.csv")