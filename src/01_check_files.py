from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
raw = root / "data" / "raw"

files = {
    "GDP": raw / "gdp_raw.xlsx",
    "Employment": raw / "employment_raw.xlsx",
    "Unemployment": raw / "unemployment_raw.xlsx"
}

for dataset, path in files.items():
    print(f"\n--- {dataset} ---")
    print("File:", path.name)
    print("Found:", path.exists())

    if path.exists():
        excel_file = pd.ExcelFile(path)
        print("Sheets:", excel_file.sheet_names)