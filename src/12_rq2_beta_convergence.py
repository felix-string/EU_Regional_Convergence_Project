"""RQ2: Do poorer regions grow faster? Absolute beta convergence.

Question: using 2016 GDP and annualised growth to 2024, report the
slope, the fit, and commuter-region sensitivity.

Input : data/processed/gdp_clean.csv
        (region_code, region_name, country_code, year,
         gdp_pps_index, gdp_pps_index_flag)
Output: outputs/rq2_results.csv            all model results
        outputs/rq2_region_level.csv       growth and residual per region
        outputs/rq2_plot1_scatter.png      main convergence figure

Run from the project root, from src, or from notebooks:
    python src/12_rq2_beta_convergence.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path

# ===============================================================
# 1. SETTINGS
# ===============================================================
START = 2016
END = 2024
ALT_END = 2023  # earlier end year: less provisional data

# Commuter definition: capital-city region = core, the region
# directly around it = commuter ring. Simple rule, state it in
# the report (seven capital areas only).
CORE = ["BE10", "DE30", "AT13", "CZ01", "HU11", "DK01", "IE06"]
RING = ["BE24", "BE31", "DE40", "AT12", "CZ02", "HU12", "DK02"]

# Team outlier list (from RQ4): high-GDP outliers, not commuter regions
OUTLIERS = ["LU00", "IE05", "IE06", "BE10"]

# Regions to inspect for data artefacts (flags, jumps)
INSPECT = ["IE05", "IE06", "EL53", "EL41", "EL54", "FRY3", "FRY1"]

# ===============================================================
# 2. LOCATE DATA AND OUTPUT FOLDER
# ===============================================================
candidates = [Path("."), Path("..")]
try:
    candidates.append(Path(__file__).resolve().parent.parent)
except NameError:
    pass

ROOT = None
for c in candidates:
    if (c / "data" / "processed" / "gdp_clean.csv").exists():
        ROOT = c
        break
if ROOT is None:
    raise SystemExit("Cannot find data/processed/gdp_clean.csv")

OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

g = pd.read_csv(ROOT / "data" / "processed" / "gdp_clean.csv")


# ===============================================================
# 3. HELPER FUNCTIONS
# ===============================================================
def build(df, start, end):
    """One row per region: start level, end level, annualised growth."""
    sub = df[df["year"].isin([start, end])]
    wide = sub.pivot(
        index="region_code",
        columns="year",
        values="gdp_pps_index",
    )
    wide = wide.dropna()
    wide = wide.rename(columns={start: "y0", end: "y1"})

    meta = df.drop_duplicates("region_code")
    meta = meta.set_index("region_code")
    meta = meta[["region_name", "country_code"]]

    out = meta.join(wide, how="inner")
    out["ln_y0"] = np.log(out["y0"])
    diff = np.log(out["y1"]) - np.log(out["y0"])
    out["growth"] = diff / (end - start)
    return out


def fit(d, label, T):
    """OLS of growth on ln(start level), country-clustered errors."""
    X = sm.add_constant(d["ln_y0"])
    groups = pd.factorize(d["country_code"])[0]
    model = sm.OLS(d["growth"], X)
    m = model.fit(cov_type="cluster", cov_kwds={"groups": groups})

    b = m.params["ln_y0"]
    inner = 1 + b * T
    if inner > 0:
        speed = -np.log(inner) / T
    else:
        speed = np.nan

    if speed == speed and speed > 0:
        half_life = np.log(2) / speed
    else:
        half_life = np.nan

    return {
        "model": label,
        "n_regions": int(m.nobs),
        "slope": b,
        "se_cluster": m.bse["ln_y0"],
        "p_value": m.pvalues["ln_y0"],
        "r_squared": m.rsquared,
        "speed": speed,
        "half_life_yrs": half_life,
    }


def present(codes, index, name):
    """Keep only codes found in the data; warn about the rest."""
    found = [c for c in codes if c in index]
    lost = [c for c in codes if c not in index]
    if lost:
        print("WARNING: " + name + " codes not in data:", lost)
    return found


# ===============================================================
# 4. DATA CHECKS
# ===============================================================
print("Regions:", g["region_code"].nunique())
print("Years:", g["year"].min(), "to", g["year"].max())
print("Duplicate region-year rows:",
      g.duplicated(["region_code", "year"]).sum())

full = build(g, START, END)
T = END - START
core = present(CORE, full.index, "core")
ring = present(RING, full.index, "ring")
outl = present(OUTLIERS, full.index, "outlier")

# ===============================================================
# 5. MAIN MODEL AND SENSITIVITY MODELS
# ===============================================================
results = []

# A. Full sample (the answer to RQ2)
results.append(fit(full, "A. Full sample", T))

# B. Commuter-region sensitivity
results.append(fit(full.drop(index=ring),
                   "B. Excl. commuter ring", T))
results.append(fit(full.drop(index=core + ring),
                   "B2. Excl. core + ring", T))

# C. Regions with a break-in-series flag
is_break = g["gdp_pps_index_flag"] == "b"
break_codes = g.loc[is_break, "region_code"].unique()
drop_c = [r for r in break_codes if r in full.index]
results.append(fit(full.drop(index=drop_c),
                   "C. Excl. break-flag regions", T))

# D. Influential regions (Cook's distance > 4/n)
X_full = sm.add_constant(full["ln_y0"])
ols = sm.OLS(full["growth"], X_full).fit()
cooks = ols.get_influence().cooks_distance[0]
infl = list(full.index[cooks > 4 / len(full)])
results.append(fit(full.drop(index=infl),
                   "D. Excl. influential regions", T))

# E. Earlier end year
r_alt = build(g, START, ALT_END)
results.append(fit(r_alt, "E. End year " + str(ALT_END),
                   ALT_END - START))

# F, G, H. Ireland and French overseas regions
no_ie = full[full["country_code"] != "IE"]
no_fry = full[~full.index.str.startswith("FRY")]
no_both = no_ie[~no_ie.index.str.startswith("FRY")]
results.append(fit(no_ie, "F. Excl. Ireland", T))
results.append(fit(no_fry, "G. Excl. French overseas", T))
results.append(fit(no_both, "H. Excl. Ireland + overseas", T))

# I. Team outlier list
results.append(fit(full.drop(index=outl),
                   "I. Excl. team outlier list", T))

res = pd.DataFrame(results)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
print()
print("RESULTS (slope < 0 means poorer regions grew faster)")
print(res.round(4).to_string(index=False))
res.to_csv(OUT_DIR / "rq2_results.csv", index=False)

print()
print("Influential regions dropped in model D:", infl)
print("Break-flag regions dropped in model C:", drop_c)

# ===============================================================
# 6. REGIONS WHERE THE ANSWER DOES NOT HOLD
# ===============================================================
full["fitted"] = ols.fittedvalues
full["resid"] = ols.resid
full["resid_z"] = full["resid"] / full["resid"].std()

show = ["region_name", "y0", "y1", "growth", "resid_z"]
fastest = full.sort_values("resid_z", ascending=False)[show].head(10)
slowest = full.sort_values("resid_z")[show].head(10)

print()
print("Growing fastest relative to the fitted line:")
print(fastest.round(4).to_string())
print()
print("Growing slowest relative to the fitted line:")
print(slowest.round(4).to_string())
full.to_csv(OUT_DIR / "rq2_region_level.csv")

# ===============================================================
# 7. DATA-ARTEFACT CHECK ON SUSPICIOUS REGIONS
# ===============================================================
chk = present(INSPECT, set(g["region_code"]), "inspect")
sub = g[g["region_code"].isin(chk)]
vals = sub.pivot(index="region_code", columns="year",
                 values="gdp_pps_index")
flags = sub.pivot(index="region_code", columns="year",
                  values="gdp_pps_index_flag").fillna("-")
print()
print("GDP index by year (inspected regions):")
print(vals)
print()
print("Flags (p provisional, e estimated, b break):")
print(flags)

# ===============================================================
# 8. MAIN FIGURE: BETA CONVERGENCE SCATTER
# ===============================================================
fig, ax = plt.subplots(figsize=(10, 7))

is_core = full.index.isin(core)
is_ring = full.index.isin(ring)
other = ~(is_core | is_ring)

ax.scatter(full.loc[other, "ln_y0"], full.loc[other, "growth"],
           s=25, alpha=0.6, color="grey", label="Other regions")
ax.scatter(full.loc[is_ring, "ln_y0"], full.loc[is_ring, "growth"],
           s=45, color="orange", label="Commuter ring")
ax.scatter(full.loc[is_core, "ln_y0"], full.loc[is_core, "growth"],
           s=45, color="red", label="Capital core")

xs = np.linspace(full["ln_y0"].min(), full["ln_y0"].max(), 100)
ys = ols.params["const"] + ols.params["ln_y0"] * xs
ax.plot(xs, ys, color="black", linewidth=2, label="Fitted line")

far = full["resid_z"].abs().sort_values(ascending=False).head(10).index
for code in far:
    row = full.loc[code]
    name = str(row["region_name"])[:18]
    ax.annotate(name, (row["ln_y0"], row["growth"]),
                fontsize=8, xytext=(4, 4),
                textcoords="offset points")

ax.axhline(0, color="lightgrey", linewidth=1)
ax.set_xlabel("Log of GDP per head in " + str(START) +
              " (PPS index, EU = 100)")
ax.set_ylabel("Annualised growth of the index, " +
              str(START) + "-" + str(END))
ax.set_title("Beta convergence across " + str(len(full)) +
             " EU NUTS 2 regions")
b = ols.params["ln_y0"]
ax.text(0.02, 0.04,
        "Slope = " + str(round(b, 4)) +
        ", R2 = " + str(round(ols.rsquared, 2)),
        transform=ax.transAxes, fontsize=10)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(OUT_DIR / "rq2_plot1_scatter.png", dpi=200)
plt.show()

print()
print("Saved to", OUT_DIR.resolve())
