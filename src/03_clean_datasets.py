from pathlib import Path
import re
import warnings

import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="Workbook contains no default style"
)

root = Path(__file__).resolve().parents[1]
raw = root / "data" / "raw"
processed = root / "data" / "processed"
outputs = root / "outputs"

processed.mkdir(parents=True, exist_ok=True)
outputs.mkdir(parents=True, exist_ok=True)

EU27 = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI",
    "FR", "DE", "EL", "HU", "IE", "IT", "LV", "LT", "LU",
    "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE"
}

YEARS = list(range(2016, 2025))


def is_eu_nuts2(code):
    code = str(code).strip()

    return (
        bool(re.fullmatch(r"[A-Z]{2}[A-Z0-9]{2}", code))
        and code[:2] in EU27
        and not code.endswith("ZZ")
    )


def clean_wide_excel(
    file_path,
    sheet_name,
    year_row,
    data_start_row,
    value_name
):
    raw_table = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=None
    )

    year_values = raw_table.iloc[year_row].tolist()

    records = []

    for row_number in range(data_start_row, len(raw_table)):
        region_code = raw_table.iloc[row_number, 0]
        region_name = raw_table.iloc[row_number, 1]

        if not is_eu_nuts2(region_code):
            continue

        for column_number, year_value in enumerate(year_values):
            if column_number < 2:
                continue

            try:
                year = int(float(year_value))
            except (TypeError, ValueError):
                continue

            if year not in YEARS:
                continue

            value = raw_table.iloc[row_number, column_number]

            if column_number + 1 < raw_table.shape[1]:
                quality_flag = raw_table.iloc[
                    row_number,
                    column_number + 1
                ]
            else:
                quality_flag = None

            numeric_value = pd.to_numeric(value, errors="coerce")

            records.append({
                "region_code": str(region_code).strip(),
                "region_name": str(region_name).strip(),
                "country_code": str(region_code).strip()[:2],
                "year": year,
                value_name: numeric_value,
                f"{value_name}_flag": (
                    "" if pd.isna(quality_flag)
                    else str(quality_flag).strip()
                )
            })

    cleaned = pd.DataFrame(records)

    cleaned = cleaned.drop_duplicates(
        subset=["region_code", "year"]
    )

    return cleaned


gdp = clean_wide_excel(
    file_path=raw / "gdp_raw.xlsx",
    sheet_name="Sheet 1",
    year_row=7,
    data_start_row=9,
    value_name="gdp_pps_index"
)

employment = clean_wide_excel(
    file_path=raw / "employment_raw.xlsx",
    sheet_name="Sheet 5",
    year_row=9,
    data_start_row=11,
    value_name="employment_rate"
)

unemployment = clean_wide_excel(
    file_path=raw / "unemployment_raw.xlsx",
    sheet_name="Sheet 1",
    year_row=10,
    data_start_row=12,
    value_name="unemployment_rate"
)

gdp.to_csv(
    processed / "gdp_clean.csv",
    index=False
)

employment.to_csv(
    processed / "employment_clean.csv",
    index=False
)

unemployment.to_csv(
    processed / "unemployment_clean.csv",
    index=False
)

print("Cleaning completed.")
print(f"GDP rows: {len(gdp)}")
print(f"Employment rows: {len(employment)}")
print(f"Unemployment rows: {len(unemployment)}")