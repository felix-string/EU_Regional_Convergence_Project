"""Sector shares per region, built from the Eurostat sector employment download (lfst_r_lfe2en2).

Reads  data/raw/sector_employment_raw.xlsx
Writes data/processed/sector_shares.csv and outputs/sector_data_audit.csv
"""
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")  # openpyxl complains about the Eurostat file styles, harmless

root = Path(__file__).resolve().parents[1]
RAW = root / "data" / "raw" / "sector_employment_raw.xlsx"
PANEL = root / "data" / "processed" / "eu_nuts2_balanced_panel.csv"
OUT_CSV = root / "data" / "processed" / "sector_shares.csv"
OUT_AUDIT = root / "outputs" / "sector_data_audit.csv"
OUT_AUDIT.parent.mkdir(exist_ok=True)

# the workbook has one sheet per NACE sector, so read them all into a dict
book = pd.ExcelFile(RAW)
sectors = {}
for name in book.sheet_names:
    if not name.startswith("Sheet"):
        continue  # first sheet is only a summary
    t = pd.read_excel(book, name, header=None)
    nace = t.iloc[5, 2].split("[")[-1].strip("]")  # sector code, e.g. B-E or TOTAL
    header = t.iloc[10]
    # years sit in every second column, the columns in between hold the flags
    year_cols = {int(header[c]): c for c in range(2, t.shape[1]) if pd.notna(header[c])}
    body = t.iloc[12:]
    body = body[body[0].astype(str).str.len() == 4]  # NUTS2 codes have 4 characters
    sectors[nace] = pd.DataFrame(
        {y: pd.to_numeric(body[c], errors="coerce") for y, c in year_cols.items()}
    ).set_axis(body[0].values)  # ':' in the file turns into NaN

# only keep the regions that are in our balanced panel
regions = sorted(pd.read_csv(PANEL)["region_code"].unique())
get = lambda k: sectors[k].reindex(regions)
total, agri, ind, cons = get("TOTAL"), get("A"), get("B-E"), get("F")

# agriculture is missing for some regions (mostly cities where farming is tiny).
# missing years get the region's own average, regions with no value at all get 0
agri_missing = agri.isna()
agri_filled = agri.apply(lambda row: row.fillna(row.mean()), axis=1).fillna(0)

industry = ind + cons                       # industry incl. construction
services = total - agri_filled - ind - cons  # everything that is left
share = lambda x: x / total * 100

yearly = {
    "share_agri": share(agri_filled),
    "share_industry": share(industry),
    "share_services": share(services),
    "share_trade": share(get("G-I")),
    "share_business": share(get("M_N")),
    "share_public": share(get("O-Q")),
}

# average over the years
S = pd.DataFrame({k: v.mean(axis=1) for k, v in yearly.items()})
S.index.name = "region_code"
S["years_used"] = industry.notna().sum(axis=1)
S = S.round(2)
S.to_csv(OUT_CSV)

audit = pd.DataFrame({
    "years_agri_missing": agri_missing.sum(axis=1),
    "years_industry_missing": industry.isna().sum(axis=1),
    "agri_never_reported_set_to_zero": agri_missing.all(axis=1),
})
audit.index.name = "region_code"
audit.to_csv(OUT_AUDIT)

# quick check
cols = ["share_agri", "share_industry", "share_services"]
print("regions:", len(S), "| with industry share:", S["share_industry"].notna().sum())
print("agriculture missing in at least one year:", int((audit.years_agri_missing > 0).sum()))
print("agriculture never reported (set to 0):", int(audit.agri_never_reported_set_to_zero.sum()))
print(S[cols].describe().round(1))
print("rows adding up to 100:", ((S[cols].sum(axis=1) - 100).abs() < 0.5).sum())
print("saved", OUT_CSV)
