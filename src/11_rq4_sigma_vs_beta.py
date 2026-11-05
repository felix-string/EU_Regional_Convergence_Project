"""
RQ4: Where do the two convergence tests disagree, and which conclusion is better supported?

Compares sigma convergence (is the spread shrinking?) with beta convergence
(do poorer regions grow faster?) for EU NUTS 2 regions, 2016-2024.

Data: data/processed/eu_nuts2_master_panel.csv (244 regions, GDP complete).
GDP is GDP per inhabitant in PPS as an index (EU27 = 100). Because every
region is measured against the same EU average, the index growth of a region
equals its real relative growth versus the EU, so beta slopes and log
dispersion are unaffected by using the index instead of levels.

Steps
  1. Sigma: dispersion each year with three measures (SD of log GDP, CV,
     P90/P10), for all regions and without the outlier regions.
  2. Beta: OLS of annualised log growth 2016-2024 on log GDP 2016, with
     robust (HC1) standard errors, speed of convergence and half-life.
  3. Year-by-year comparison: annual beta vs annual change in sigma.
  4. Variance decomposition linking the two tests:
        Var(y_T) = (1 + B)^2 * Var(y_0) + Var(e)
     catch-up effect = [(1 + B)^2 - 1] * Var(y_0)   (from beta, always <= 0 if B < 0)
     shock effect    = Var(e)                        (new region-specific shocks, >= 0)
     Sigma falls only if the catch-up effect is larger than the shock effect.
  5. Between- vs within-country split of the variance of log GDP.
  6. Exceptions: regions and countries that go against the EU-wide result.
  7. Robustness: without outliers, balanced panel, within-country (demeaned) beta.

Outputs go to outputs/rq4/.
Requires matplotlib in addition to requirements.txt (pip install matplotlib).
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:  # charts are skipped, tables still produced
    HAVE_MPL = False

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "data" / "processed" / "eu_nuts2_master_panel.csv"
BALANCED = ROOT / "data" / "processed" / "eu_nuts2_balanced_panel.csv"
OUT = ROOT / "outputs" / "rq4"
OUT.mkdir(parents=True, exist_ok=True)

START, END = 2016, 2024
T = END - START

# Regions whose GDP per inhabitant is distorted: commuters work (and create GDP)
# there but live elsewhere, or multinational profit booking inflates GDP.
# Agree this list with the RQ2 owner so both questions use the same sensitivity set.
OUTLIERS = {
    "LU00": "Luxembourg - cross-border commuters",
    "IE05": "Southern Ireland - multinational profit booking",
    "IE06": "Eastern and Midland Ireland - multinational profit booking",
    "BE10": "Brussels - commuter region",
}

MIN_REGIONS_PER_COUNTRY = 4  # country-level sigma/beta needs enough regions


# ---------------------------------------------------------------- helpers
def ols(x, y):
    """Simple OLS y = a + b x with HC1 robust standard errors (numpy only)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ X.T @ y
    resid = y - X @ coef
    meat = X.T @ (X * resid[:, None] ** 2)
    cov = XtX_inv @ meat @ XtX_inv * n / (n - 2)
    se = np.sqrt(np.diag(cov))
    r2 = 1 - resid.var() / y.var()
    return {
        "n": n,
        "alpha": coef[0],
        "beta": coef[1],
        "se_beta": se[1],
        "t_beta": coef[1] / se[1],
        "r2": r2,
        "resid": resid,
    }


def speed_and_half_life(beta_annual, years):
    """Implied convergence speed lambda and half-life from b = -(1 - e^{-lambda T}) / T."""
    inside = 1 + beta_annual * years
    if beta_annual >= 0 or inside <= 0:
        return np.nan, np.nan
    lam = -np.log(inside) / years
    return lam, np.log(2) / lam


def sigma_measures(levels):
    logs = np.log(levels)
    return {
        "sd_log": logs.std(ddof=1),
        "cv": levels.std(ddof=1) / levels.mean(),
        "p90_p10": levels.quantile(0.9) / levels.quantile(0.1),
    }


# ---------------------------------------------------------------- data
panel = pd.read_csv(MASTER)
wide = panel.pivot(index="region_code", columns="year", values="gdp_pps_index")
wide = wide.loc[:, START:END].dropna()
names = panel.drop_duplicates("region_code").set_index("region_code")["region_name"]
country = panel.drop_duplicates("region_code").set_index("region_code")["country_code"]
logs = np.log(wide)
years = list(wide.columns)
no_out = [r for r in wide.index if r not in OUTLIERS]

print(f"Regions: {len(wide)}  years: {years[0]}-{years[-1]}")

# ---------------------------------------------------------------- 1. sigma
rows = []
for t in years:
    all_ = sigma_measures(wide[t])
    excl = sigma_measures(wide.loc[no_out, t])
    rows.append({
        "year": t,
        "sd_log": all_["sd_log"],
        "cv": all_["cv"],
        "p90_p10": all_["p90_p10"],
        "sd_log_excl_outliers": excl["sd_log"],
        "cv_excl_outliers": excl["cv"],
    })
sigma = pd.DataFrame(rows).set_index("year")
# trend of each measure: slope per year of an OLS on year, and % change over the period
sigma_trend = pd.DataFrame({
    "start": sigma.loc[START],
    "end": sigma.loc[END],
    "pct_change": (sigma.loc[END] / sigma.loc[START] - 1) * 100,
    "trend_per_year": [np.polyfit(years, sigma[c], 1)[0] for c in sigma.columns],
    "years_rising": [(sigma[c].diff() > 0).sum() for c in sigma.columns],
})
sigma.round(4).to_csv(OUT / "rq4_sigma_by_year.csv")
sigma_trend.round(4).to_csv(OUT / "rq4_sigma_trend_summary.csv")

# ---------------------------------------------------------------- 2. beta (full period)
growth = (logs[END] - logs[START]) / T  # annualised log growth relative to EU
beta_rows = []


def add_beta(label, idx, x=None, y=None):
    xx = logs.loc[idx, START] if x is None else x
    yy = growth.loc[idx] if y is None else y
    r = ols(xx, yy)
    lam, hl = speed_and_half_life(r["beta"], T)
    beta_rows.append({
        "specification": label, "n": r["n"], "beta": r["beta"], "se_hc1": r["se_beta"],
        "t_stat": r["t_beta"], "r2": r["r2"], "speed_lambda": lam, "half_life_years": hl,
    })
    return r


main = add_beta("All regions (master panel)", wide.index)
add_beta("Without outlier regions", no_out)

bal = pd.read_csv(BALANCED)
bal_regions = [r for r in bal["region_code"].unique() if r in wide.index]
add_beta("Balanced panel regions", bal_regions)

# within-country beta: subtract country means -> do poorer regions grow faster
# than richer regions of the SAME country?
x_dm = logs[START] - logs[START].groupby(country).transform("mean")
g_dm = growth - growth.groupby(country).transform("mean")
add_beta("Within countries (country-demeaned)", wide.index, x_dm, g_dm)

# between countries: one observation per country (mean log GDP of its regions)
cx = logs[START].groupby(country).mean()
cg = growth.groupby(country).mean()
add_beta("Between countries (country means)", cx.index, cx, cg)

beta_table = pd.DataFrame(beta_rows)
beta_table.round(5).to_csv(OUT / "rq4_beta_results.csv", index=False)

# ---------------------------------------------------------------- 3. year by year
annual = []
for t0, t1 in zip(years[:-1], years[1:]):
    r = ols(logs[t0], logs[t1] - logs[t0])
    d_sd = sigma.loc[t1, "sd_log"] - sigma.loc[t0, "sd_log"]
    d_cv = sigma.loc[t1, "cv"] - sigma.loc[t0, "cv"]
    beta_says = "converging" if r["beta"] < 0 else "diverging"
    annual.append({
        "period": f"{t0}-{t1}",
        "beta": r["beta"],
        "beta_significant_5pct": abs(r["t_beta"]) > 1.96,
        "change_sd_log": d_sd,
        "change_cv": d_cv,
        "beta_says": beta_says,
        "sd_log_says": "converging" if d_sd < 0 else "diverging",
        "cv_says": "converging" if d_cv < 0 else "diverging",
    })
annual = pd.DataFrame(annual)
annual["agree_sd_log"] = annual["beta_says"] == annual["sd_log_says"]
annual["agree_cv"] = annual["beta_says"] == annual["cv_says"]
annual.round(5).to_csv(OUT / "rq4_annual_agreement.csv", index=False)

# ---------------------------------------------------------------- 4. variance decomposition
def decompose(t0, t1, idx):
    y0, y1 = logs.loc[idx, t0], logs.loc[idx, t1]
    r = ols(y0, y1)  # y1 = a + (1+B) y0 + e  -> slope = 1 + B
    B = r["beta"] - 1
    v0, v1 = y0.var(ddof=0), y1.var(ddof=0)
    catch_up = ((1 + B) ** 2 - 1) * v0
    shocks = r["resid"].var(ddof=0)
    return {
        "period": f"{t0}-{t1}", "sample": "all" if len(idx) == len(wide) else "excl. outliers",
        "var_start": v0, "var_end": v1, "change": v1 - v0,
        "catch_up_effect": catch_up, "shock_effect": shocks,
        "check_sum": catch_up + shocks,  # equals 'change' exactly
    }


dec = [decompose(START, END, wide.index), decompose(START, END, no_out)]
dec += [decompose(a, b, wide.index) for a, b in zip(years[:-1], years[1:])]
decomp = pd.DataFrame(dec)
decomp.round(6).to_csv(OUT / "rq4_variance_decomposition.csv", index=False)

# ---------------------------------------------------------------- 5. between / within countries
bw = []
for t in years:
    s = logs[t]
    cm = s.groupby(country).transform("mean")
    total = s.var(ddof=0)
    between = cm.var(ddof=0)
    within = (s - cm).var(ddof=0)
    bw.append({"year": t, "total": total, "between_countries": between,
               "within_countries": within, "share_within": within / total})
between_within = pd.DataFrame(bw).set_index("year")
between_within.round(5).to_csv(OUT / "rq4_between_within.csv")

# ---------------------------------------------------------------- 6. exceptions
# 6a. regions: residual from the beta line + simple quadrant label
reg = pd.DataFrame({
    "region_name": names.reindex(wide.index),
    "country": country.reindex(wide.index),
    "index_2016": wide[START],
    "index_2024": wide[END],
    "change_points": wide[END] - wide[START],
    "growth_rel_eu_pct_pa": growth * 100,
    "predicted_growth_pct_pa": (main["alpha"] + main["beta"] * logs[START]) * 100,
    "residual_pct_pa": main["resid"] * 100,
})
poor = reg["index_2016"] < 75          # EU cohesion threshold for "less developed"
rich = reg["index_2016"] > 125
losing = reg["change_points"] < 0      # index fell = lost ground versus EU average
reg["pattern"] = np.select(
    [poor & ~losing, poor & losing, rich & ~losing, rich & losing],
    ["poor, catching up (fits beta)", "poor, falling behind (exception)",
     "rich, pulling further ahead (exception)", "rich, losing ground (fits beta)"],
    default="middle group",
)
reg["outlier_flag"] = reg.index.isin(OUTLIERS.keys())
reg.sort_values("residual_pct_pa").round(3).to_csv(OUT / "rq4_region_exceptions.csv")

# 6b. countries: is the gap between a country's own regions shrinking?
crow = []
for c, idx in reg.groupby("country").groups.items():
    if len(idx) < MIN_REGIONS_PER_COUNTRY:
        continue
    sd0, sd1 = logs.loc[idx, START].std(ddof=1), logs.loc[idx, END].std(ddof=1)
    r = ols(logs.loc[idx, START], growth.loc[idx])
    top = reg.loc[idx, "index_2024"].idxmax()
    crow.append({
        "country": c, "n_regions": len(idx),
        "sd_log_2016": sd0, "sd_log_2024": sd1, "sd_change_pct": (sd1 / sd0 - 1) * 100,
        "within_beta": r["beta"],
        "sigma_says": "converging" if sd1 < sd0 else "diverging",
        "beta_says": "converging" if r["beta"] < 0 else "diverging",
        "richest_region_2024": f"{top} {names[top]}",
    })
countries = pd.DataFrame(crow)
countries["tests_agree"] = countries["sigma_says"] == countries["beta_says"]
countries.sort_values("sd_change_pct", ascending=False).round(4).to_csv(
    OUT / "rq4_country_within_convergence.csv", index=False)

# ---------------------------------------------------------------- 7. charts
if HAVE_MPL:
    plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9})
    INK, MUTED, A, B_, C_ = "#222222", "#888888", "#2a6f97", "#c8553d", "#6a994e"

    # Figure 1: sigma measures indexed to 2016 = 100
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, lab, colr, ls in [
        ("sd_log", "SD of log GDP", A, "-"),
        ("cv", "Coefficient of variation", B_, "-"),
        ("cv_excl_outliers", "CV without LU, IE, Brussels", B_, "--"),
        ("p90_p10", "P90 / P10 ratio", C_, "-"),
    ]:
        s = sigma[col] / sigma.loc[START, col] * 100
        ax.plot(s.index, s.values, ls, color=colr, lw=2, label=lab)
        ax.annotate(f"{s.iloc[-1]:.0f}", (END, s.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", color=colr)
    ax.axhline(100, color=MUTED, lw=0.8)
    ax.axvspan(2019, 2022, color="#eeeeee", zorder=0)
    ax.text(2020.5, ax.get_ylim()[1], "CV rises\n2019-2022", ha="center", va="top", color=MUTED)
    ax.set_ylabel("Dispersion, 2016 = 100")
    ax.set_title("Sigma convergence depends on the measure", loc="left", color=INK)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_sigma_measures.png")
    plt.close(fig)

    # Figure 2: beta scatter with exceptions labelled
    fig, ax = plt.subplots(figsize=(7, 4.5))
    col = np.where(reg["pattern"].str.contains("exception"), B_, "#bbbbbb")
    ax.scatter(reg["index_2016"], reg["growth_rel_eu_pct_pa"], s=14, c=col, lw=0)
    xs = np.linspace(logs[START].min(), logs[START].max(), 100)
    ax.plot(np.exp(xs), (main["alpha"] + main["beta"] * xs) * 100, color=A, lw=2)
    ax.set_xscale("log")
    ax.set_xticks([30, 50, 75, 100, 150, 250])
    ax.set_xticklabels(["30", "50", "75", "100", "150", "250"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.axhline(0, color=MUTED, lw=0.8)
    label_ids = list(reg["residual_pct_pa"].abs().nlargest(8).index)
    for r_id in label_ids:
        ax.annotate(r_id, (reg.loc[r_id, "index_2016"], reg.loc[r_id, "growth_rel_eu_pct_pa"]),
                    xytext=(3, 3), textcoords="offset points", fontsize=7, color=INK)
    ax.set_xlabel("GDP per inhabitant 2016 (PPS, EU27 = 100, log scale)")
    ax.set_ylabel("Growth relative to EU, % per year 2016-2024")
    ax.set_title(f"Beta convergence: slope {main['beta']:.4f}, R² {main['r2']:.2f}  "
                 "(red = exceptions)", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_beta_scatter.png")
    plt.close(fig)

    # Figure 3: decomposition per year (catch-up vs shocks)
    d = decomp.iloc[2:]
    fig, ax = plt.subplots(figsize=(7, 4))
    xpos = np.arange(len(d))
    ax.bar(xpos, d["catch_up_effect"], color=A, label="Catch-up effect (beta)")
    ax.bar(xpos, d["shock_effect"], color=B_, label="New shocks")
    ax.plot(xpos, d["change"], "o", color=INK, label="Net change in variance")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(d["period"], rotation=0, fontsize=7)
    ax.set_ylabel("Change in variance of log GDP")
    ax.set_title("Sigma falls only when catch-up outweighs new shocks", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_variance_decomposition.png")
    plt.close(fig)

    # Figure 4: between vs within countries
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.stackplot(between_within.index, between_within["between_countries"],
                 between_within["within_countries"], colors=[A, "#9ecae1"],
                 labels=["Between countries", "Within countries"])
    ax.set_ylabel("Variance of log GDP")
    ax.set_title("Convergence happens between countries, not inside them", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_between_within.png")
    plt.close(fig)

# ---------------------------------------------------------------- 8. text summary
bt = beta_table.set_index("specification")
dfull, dexcl = decomp.iloc[0], decomp.iloc[1]
bw0, bw1 = between_within.loc[START], between_within.loc[END]
disagree_cv = annual.loc[~annual["agree_cv"], "period"].tolist()
disagree_sd = annual.loc[~annual["agree_sd_log"], "period"].tolist()
n_poor_behind = (reg["pattern"] == "poor, falling behind (exception)").sum()
n_rich_ahead = (reg["pattern"] == "rich, pulling further ahead (exception)").sum()
div_countries = countries.loc[countries["sigma_says"] == "diverging", "country"].tolist()

lines = [
    "RQ4 SUMMARY: sigma vs beta convergence, EU NUTS 2, 2016-2024",
    "=" * 62,
    "",
    "BETA (full period)",
    f"  All regions:      beta = {bt.loc['All regions (master panel)', 'beta']:.4f} "
    f"(t = {bt.loc['All regions (master panel)', 't_stat']:.1f}), "
    f"R2 = {bt.loc['All regions (master panel)', 'r2']:.2f}, "
    f"half-life = {bt.loc['All regions (master panel)', 'half_life_years']:.0f} years",
    f"  Without outliers: beta = {bt.loc['Without outlier regions', 'beta']:.4f}, "
    f"R2 = {bt.loc['Without outlier regions', 'r2']:.2f}",
    f"  Within countries: beta = {bt.loc['Within countries (country-demeaned)', 'beta']:.4f} "
    f"(t = {bt.loc['Within countries (country-demeaned)', 't_stat']:.1f})",
    f"  Between countries: beta = {bt.loc['Between countries (country means)', 'beta']:.4f} "
    f"(t = {bt.loc['Between countries (country means)', 't_stat']:.1f})",
    "",
    "SIGMA (% change 2016 -> 2024)",
] + [f"  {k:<22} {v:+.1f}%  (rose in {int(sigma_trend.loc[k, 'years_rising'])} of {T} years)"
     for k, v in sigma_trend["pct_change"].items()] + [
    "",
    "WHERE THEY DISAGREE",
    f"  Years where beta < 0 but CV rose:          {', '.join(disagree_cv) or 'none'}",
    f"  Years where beta < 0 but SD of log rose:   {', '.join(disagree_sd) or 'none'}",
    f"  Poor regions (<75) that fell further behind: {n_poor_behind}",
    f"  Rich regions (>125) that pulled further ahead: {n_rich_ahead}",
    f"  Countries whose own regions diverged (sigma): {', '.join(div_countries) or 'none'}",
    "",
    "LINKING THE TWO TESTS (variance of log GDP, 2016-2024)",
    f"  All regions:    change {dfull['change']:+.4f} = catch-up {dfull['catch_up_effect']:+.4f}"
    f" + shocks {dfull['shock_effect']:+.4f}",
    f"  Excl. outliers: change {dexcl['change']:+.4f} = catch-up {dexcl['catch_up_effect']:+.4f}"
    f" + shocks {dexcl['shock_effect']:+.4f}",
    "",
    "BETWEEN vs WITHIN COUNTRIES (variance of log GDP)",
    f"  Between: {bw0['between_countries']:.4f} -> {bw1['between_countries']:.4f} "
    f"({(bw1['between_countries'] / bw0['between_countries'] - 1) * 100:+.0f}%)",
    f"  Within:  {bw0['within_countries']:.4f} -> {bw1['within_countries']:.4f} "
    f"({(bw1['within_countries'] / bw0['within_countries'] - 1) * 100:+.0f}%)",
    f"  Share of inequality that is within countries: "
    f"{bw0['share_within'] * 100:.0f}% -> {bw1['share_within'] * 100:.0f}%",
]
(OUT / "rq4_summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(f"\nAll outputs written to {OUT}")
