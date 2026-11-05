from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
processed = root / "data" / "processed"
outputs = root / "outputs"

gdp = pd.read_csv(processed / "gdp_clean.csv")
employment = pd.read_csv(processed / "employment_clean.csv")
unemployment = pd.read_csv(processed / "unemployment_clean.csv")

gdp_regions = set(gdp["region_code"].dropna().unique())
employment_regions = set(
    employment["region_code"].dropna().unique()
)
unemployment_regions = set(
    unemployment["region_code"].dropna().unique()
)

common_regions = (
    gdp_regions
    & employment_regions
    & unemployment_regions
)

employment_not_gdp = sorted(
    employment_regions - gdp_regions
)

unemployment_not_gdp = sorted(
    unemployment_regions - gdp_regions
)

gdp_not_employment = sorted(
    gdp_regions - employment_regions
)

gdp_not_unemployment = sorted(
    gdp_regions - unemployment_regions
)

print("GDP regions:", len(gdp_regions))
print("Employment regions:", len(employment_regions))
print("Unemployment regions:", len(unemployment_regions))
print("Common regions:", len(common_regions))

print("\nEmployment regions not in GDP:")
print(employment_not_gdp)

print("\nUnemployment regions not in GDP:")
print(unemployment_not_gdp)

print("\nGDP regions not in employment:")
print(gdp_not_employment)

print("\nGDP regions not in unemployment:")
print(gdp_not_unemployment)

comparison = pd.DataFrame({
    "region_code": sorted(
        gdp_regions
        | employment_regions
        | unemployment_regions
    )
})

comparison["in_gdp"] = comparison["region_code"].isin(
    gdp_regions
)

comparison["in_employment"] = comparison["region_code"].isin(
    employment_regions
)

comparison["in_unemployment"] = comparison["region_code"].isin(
    unemployment_regions
)

comparison["in_all_three"] = comparison["region_code"].isin(
    common_regions
)

comparison.to_csv(
    outputs / "region_code_comparison.csv",
    index=False
)

print("\nComparison saved to:")
print(outputs / "region_code_comparison.csv")