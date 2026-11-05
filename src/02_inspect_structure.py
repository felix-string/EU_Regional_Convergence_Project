from pathlib import Path
from contextlib import redirect_stdout

import pandas as pd
import warnings


warnings.filterwarnings(
    "ignore",
    message="Workbook contains no default style"
)

root = Path(__file__).resolve().parents[1]
raw = root / "data" / "raw"
output_file = root / "outputs" / "structure_inspection.txt"

files = {
    "GDP": raw / "gdp_raw.xlsx",
    "Employment": raw / "employment_raw.xlsx",
    "Unemployment": raw / "unemployment_raw.xlsx"
}

output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "w", encoding="utf-8") as report:
    with redirect_stdout(report):

        for dataset, path in files.items():
            print(f"\n{'=' * 50}")
            print(dataset)
            print("=" * 50)
            print(f"File: {path.name}")

            workbook = pd.ExcelFile(path)

            for sheet in workbook.sheet_names:
                print(f"\nSheet: {sheet}")

                preview = pd.read_excel(
                    path,
                    sheet_name=sheet,
                    header=None,
                    nrows=12
                )

                print(preview.to_string(index=True, header=False))

print("Inspection saved to:")
print(output_file)