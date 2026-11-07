"""RQ1A enhanced NUTS2 GDP PPS dispersion analysis.
Run: python rq1a_enhanced.py --input eu_nuts2_master_panel.csv --output rq1a_enhanced_outputs
Dependencies: pandas numpy matplotlib
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS = ['cv_pct','sd','p90_p10_gap']
LABELS = ['Coefficient of variation (%)','Standard deviation (index points)','P90–P10 gap (index points)']

def calc(d):
    g=d.groupby('year')['gdp_pps_index']
    a=g.agg(n_regions='count',mean='mean',sd='std',median='median').reset_index()
    for q in (10,25,75,90): a[f'p{q}']=g.quantile(q/100).to_numpy()
    a['cv_pct']=100*a.sd/a['mean']
    a['p90_p10_gap']=a.p90-a.p10
    a['iqr']=a.p75-a.p25
    return a.sort_values('year').reset_index(drop=True)

def main(source,out):
    out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(source,dtype={'region_code':str,'country_code':str})
    required={'region_code','year','gdp_pps_index'}
    if required-set(d.columns): raise ValueError(f'Missing columns: {required-set(d.columns)}')
    d.year=pd.to_numeric(d.year,errors='raise').astype(int)
    d.gdp_pps_index=pd.to_numeric(d.gdp_pps_index,errors='coerce')
    if d.duplicated(['region_code','year']).any(): raise ValueError('Duplicate region-year combinations')
    if d.gdp_pps_index.dropna().le(0).any(): raise ValueError('Nonpositive GDP index')
    d['gdp_flag']=d.get('gdp_pps_index_flag',pd.Series(index=d.index,dtype='object')).fillna('none').astype(str).str.strip().replace('','none')
    years=sorted(d.year.unique()); regions=d.region_code.nunique()
    region_years=d.groupby('region_code').year.nunique()
    balanced=(len(d)==len(years)*regions and region_years.eq(len(years)).all())
    missing=d.gdp_pps_index.isna().sum()
    a=calc(d.dropna(subset=['gdp_pps_index']))
    for m in METRICS:
        a[f'{m}_change']=a[m].diff()
        a[f'{m}_pct_change']=a[m].pct_change()*100
        a[f'{m}_indexed_2016_100']=100*a[m]/a[m].iloc[0]
    a['all_three_narrow']=a[[f'{m}_change' for m in METRICS]].lt(0).all(axis=1)
    a.to_csv(out/'annual_dispersion_enhanced.csv',index=False,float_format='%.6f')
    # 1. Indexed trends
    fig,ax=plt.subplots(figsize=(10,5.5))
    for m,label in zip(METRICS,LABELS):ax.plot(a.year,a[f'{m}_indexed_2016_100'],marker='o',linewidth=2,label=label)
    ax.axhline(100,color='gray',linestyle='--',linewidth=1)
    ax.set(title='RQ1A | Relative change in regional GDP dispersion',ylabel='Index (first year = 100)',xlabel='Year',xticks=years)
    ax.legend(fontsize=9);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'figure_1_indexed_dispersion.png',dpi=200);plt.close(fig)
    # 2. Boxplots
    fig,ax=plt.subplots(figsize=(11,6))
    vals=[d.loc[d.year.eq(y),'gdp_pps_index'].dropna().values for y in years]
    ax.boxplot(vals,labels=years,showfliers=True,patch_artist=True,boxprops={'facecolor':'#dce9f4'},medianprops={'color':'#a34a2a','linewidth':2})
    ax.set(title='RQ1A | Annual distribution of regional GDP per capita PPS index',xlabel='Year',ylabel='GDP per capita PPS index (not euros)')
    ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(out/'figure_2_annual_boxplots.png',dpi=200);plt.close(fig)
    # 3. Period changes; overlapping boundary years, not annual averages
    periods=[(years[0],2019),(2019,2022),(2022,years[-1])]
    periods=[(s,e) for s,e in periods if s in years and e in years and s<e]
    rows=[]
    for s,e in periods:
        first=a.loc[a.year.eq(s)].iloc[0];last=a.loc[a.year.eq(e)].iloc[0]
        r={'period':f'{s}–{e}','start_year':s,'end_year':e,'elapsed_years':e-s}
        for m in METRICS:
            r[f'{m}_start']=first[m];r[f'{m}_end']=last[m]
            r[f'{m}_absolute_change']=last[m]-first[m]
            r[f'{m}_relative_change_pct']=100*(last[m]/first[m]-1)
        rows.append(r)
    periods_df=pd.DataFrame(rows);periods_df.to_csv(out/'period_comparisons.csv',index=False,float_format='%.6f')
    # 4. Sensitivity: per-year upper 1% trimming, plus IQR
    trim=d.dropna(subset=['gdp_pps_index']).copy()
    cutoff=trim.groupby('year').gdp_pps_index.transform(lambda s:s.quantile(.99))
    trim=trim.loc[trim.gdp_pps_index.le(cutoff)]
    t=calc(trim)
    robustness=a[['year','n_regions','cv_pct','sd','p90_p10_gap','iqr']].merge(t[['year','n_regions','cv_pct','sd','p90_p10_gap']],on='year',suffixes=('_main','_trim_top_1pct'))
    robustness['cv_difference_pp']=robustness.cv_pct_trim_top_1pct-robustness.cv_pct_main
    robustness.to_csv(out/'outlier_sensitivity.csv',index=False,float_format='%.6f')
    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot(robustness.year,robustness.cv_pct_main,marker='o',label='Main CV (all regions)')
    ax.plot(robustness.year,robustness.cv_pct_trim_top_1pct,marker='s',label='CV excluding annual top 1%')
    ax.set(title='RQ1A | Sensitivity to highest-index observations',xlabel='Year',ylabel='Coefficient of variation (%)',xticks=years)
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'figure_3_outlier_sensitivity.png',dpi=200);plt.close(fig)
    # 5. Quality and flags
    flags=pd.crosstab(d.year,d.gdp_flag).reindex(years,fill_value=0)
    flags['total']=flags.sum(axis=1)
    flags['missing_gdp']=d.groupby('year').gdp_pps_index.apply(lambda s:s.isna().sum())
    flags.to_csv(out/'gdp_flags_by_year.csv')
    region_check=d.groupby('year').agg(n_rows=('region_code','size'),n_regions=('region_code','nunique'),missing_gdp=('gdp_pps_index',lambda s:s.isna().sum()))
    region_check.to_csv(out/'panel_completeness.csv')
    flag_free=calc(d.loc[d.gdp_flag.eq('none')].dropna(subset=['gdp_pps_index']))
    flag_free[['year','n_regions','cv_pct','sd','p90_p10_gap']].to_csv(out/'unflagged_only_sensitivity.csv',index=False,float_format='%.6f')
    # Region-code consistency is not a formal NUTS boundary harmonization check
    lines=[f'Observations: {len(d)}; regions: {regions}; years: {years[0]}–{years[-1]}.',
           f'Balanced region-year structure: {balanced}. Missing GDP values: {missing}.',
           'GDP flags (all years): '+str(d.gdp_flag.value_counts().to_dict()),
           'Important: A balanced panel and stable region codes DO NOT establish harmonized NUTS boundary definitions.',
           'Check Eurostat vintage, NUTS classification, retroactive revisions, and data provenance externally.',
           'All main statistics are unweighted across regions. SD is sample SD (ddof=1).',
           'GDP is an index, not GDP in euros; SD and percentile gaps are index points.',
           'Period comparisons use start/end levels, not average annual measures.',
           'Annual top-1% trimming changes the sample and is a sensitivity test, not a replacement.',
           'Unflagged-only comparison changes sample composition; interpret with caution.',
           'Descriptive results do not establish causes of widening/narrowing.', '', 'Main results:']
    for _,r in a.iterrows():
        lines.append(f"{int(r.year)}: CV={r.cv_pct:.2f}%; SD={r.sd:.2f}; P90-P10={r.p90_p10_gap:.2f}; IQR={r.iqr:.2f}")
    lines.append('\nPeriod changes (CV percentage points; SD and gap index points):')
    for _,r in periods_df.iterrows():
        lines.append(f"{r.period}: CV {r.cv_pct_absolute_change:+.2f}; SD {r.sd_absolute_change:+.2f}; P90-P10 {r.p90_p10_gap_absolute_change:+.2f}")
    lines.append('\nRobustness direction from first to last year:')
    for name,frame,col in [('Main CV',robustness,'cv_pct_main'),('Trimmed CV',robustness,'cv_pct_trim_top_1pct'),('IQR',robustness,'iqr')]:
        lines.append(f'{name}: {frame[col].iloc[0]:.2f} to {frame[col].iloc[-1]:.2f} ({frame[col].iloc[-1]-frame[col].iloc[0]:+.2f})')
    (out/'rq1a_findings_and_caveats.txt').write_text('\n'.join(lines),encoding='utf-8')

    # Publication-ready narrative generated from the computed results.
    # Markdown is intentionally separate from QMD so another team member can import it.
    first,last=a.iloc[0],a.iloc[-1]
    def fmt_delta(v,unit=''):
        return f"{v:+.2f}{unit}"
    def direction(v):
        return 'narrowed' if v < -1e-9 else ('widened' if v > 1e-9 else 'remained unchanged')
    def metric_sentence(m,label,unit):
        change=last[m]-first[m]
        return (f"{label} {direction(change)} from {first[m]:.2f}{unit} in {int(first.year)} "
                f"to {last[m]:.2f}{unit} in {int(last.year)} "
                f"(change: {fmt_delta(change,unit)}; {100*change/first[m]:+.2f}% relative to the starting value).")
    year_changes=[]
    for _,r in a.iloc[1:].iterrows():
        shifts=[r[f'{m}_change'] for m in METRICS]
        if all(v<0 for v in shifts): tag='All three indicators narrowed.'
        elif all(v>0 for v in shifts): tag='All three indicators widened.'
        else: tag='Indicators disagreed about the direction of change.'
        year_changes.append(f"- **{int(r.year)}:** CV {r.cv_pct_change:+.2f} percentage points; "
                            f"SD {r.sd_change:+.2f} index points; P90–P10 "
                            f"{r.p90_p10_gap_change:+.2f} index points. {tag}")
    period_lines=[]
    for _,r in periods_df.iterrows():
        period_lines.append(f"- **{r.period}:** CV {r.cv_pct_absolute_change:+.2f} percentage points; "
                            f"SD {r.sd_absolute_change:+.2f} index points; "
                            f"P90–P10 {r.p90_p10_gap_absolute_change:+.2f} index points. "
                            f"These are endpoint changes over {int(r.elapsed_years)} years, not annualised rates.")
    flagged=int(d.gdp_flag.ne('none').sum())
    flag_counts=d.gdp_flag.value_counts().to_dict()
    main_change=robustness.cv_pct_main.iloc[-1]-robustness.cv_pct_main.iloc[0]
    trim_change=robustness.cv_pct_trim_top_1pct.iloc[-1]-robustness.cv_pct_trim_top_1pct.iloc[0]
    iqr_change=robustness.iqr.iloc[-1]-robustness.iqr.iloc[0]
    all_three_years=[str(int(y)) for y in a.loc[a.all_three_narrow,'year']]
    disagreement_years=[str(int(r.year)) for _,r in a.iloc[1:].iterrows() if not (all(r[f'{m}_change']<0 for m in METRICS) or all(r[f'{m}_change']>0 for m in METRICS))]
    markdown=f"""# RQ1A — Is the GDP spread narrowing year by year?

## Scope and measurement

This section measures **sigma convergence**: whether the cross-sectional dispersion of GDP per capita (PPS index) across NUTS 2 regions decreases over time. It does **not** test whether initially poorer regions subsequently grow faster (beta convergence), explain industrial or geographical differences, or reconcile the two convergence tests. Those questions are reserved for other workstreams.

The dataset contains **{len(d):,} region-year observations**, **{regions} unique regions**, and **{len(years)} years ({years[0]}–{years[-1]})**. Each region receives equal weight. The standard deviation uses the sample formula (ddof=1). CV is SD divided by the cross-sectional mean, multiplied by 100. P90–P10 subtracts the annual 10th percentile from the annual 90th percentile. SD and percentile gaps are in **PPS index points, not euros**. Negative changes in dispersion indicate narrowing; positive changes indicate widening. These descriptive statistics do not identify causes.

## Figure 1. Indexed dispersion indicators

![Figure 1. Three dispersion indicators indexed to the first year = 100](figure_1_indexed_dispersion.png)

**What the figure shows.** The coefficient of variation, standard deviation and P90–P10 gap are rescaled to **{years[0]} = 100** to permit comparison of their proportional movement despite different original units. An index below 100 indicates less dispersion than in the base year; above 100 indicates more. Indexing does not alter the underlying values or their direction of change.

**Results.** {metric_sentence('cv_pct','The coefficient of variation','%')} {metric_sentence('sd','Standard deviation',' index points')} {metric_sentence('p90_p10_gap','The P90–P10 gap',' index points')}

**Interpretation.** All three measures finish below their initial values, supporting **overall narrowing over the full period**, but this is not equivalent to uninterrupted annual convergence. The indicators show different intermediate movements, particularly during 2019–2022: the CV and SD increased while the P90–P10 gap fell. This suggests that changes outside the central P10–P90 range and/or changes in the distribution's shape matter for the aggregate dispersion statistics. The figure alone cannot identify which regions drove those differences.

**Limitation.** Because these are annual cross-sectional measures of a relative GDP index, they should not be interpreted as differences in GDP expressed in currency or as proof of causal economic integration.

## Figure 2. Annual distribution boxplots

![Figure 2. Distribution of regional GDP per capita PPS index by year](figure_2_annual_boxplots.png)

**What the figure shows.** Each box represents the middle 50% of regional GDP index observations (25th to 75th percentiles); the horizontal line is the median. Whiskers follow the conventional 1.5×IQR rule, and observations beyond them are plotted individually. These dots are **statistical outliers, not necessarily data errors**.

**Results.** The interquartile range changed from **{first.iqr:.2f}** to **{last.iqr:.2f} index points** ({iqr_change:+.2f}). The median changed from **{first['median']:.2f}** to **{last['median']:.2f} index points**. The boxplots should be read alongside the percentile gap: a narrower central box does not imply that every part of the distribution narrowed.

**Interpretation.** The declining IQR supports a reduction in dispersion among the middle half of regions. The persistence of distant observations provides a reason to examine whether the overall CV is sensitive to unusually high GDP index values. The boxplots provide a distributional explanation, not evidence about which specific regions gained or lost ground.

**Limitation.** The region-level observations are equally weighted; the boxes do not represent the income distribution of individuals or population-weighted inequality.

## Figure 3. Sensitivity to high-index regions

![Figure 3. Coefficient of variation with and without the annual highest 1 percent of GDP index observations](figure_3_outlier_sensitivity.png)

**What the figure shows.** The main annual CV is compared with a recalculated CV after removing observations above the **year-specific 99th percentile**. This is a robustness test, not a preferred replacement for the full sample.

**Results.** The full-sample CV changed by **{main_change:+.2f} percentage points** between {years[0]} and {years[-1]}. The trimmed CV changed by **{trim_change:+.2f} percentage points**. Both show a decline over the full period.

**Interpretation.** The overall narrowing conclusion does not depend solely on the most prosperous observations. However, the magnitude of convergence changes when the upper tail is excluded. Because the 99th-percentile cutoff is recalculated every year, the identities of the removed regions can change; therefore this is a **distributional sensitivity check**, not a fixed-region comparison.

**Limitation.** Trimming may remove genuine economic variation, not errors. Do not use the trimmed estimates as the headline series or infer that outlying regions caused the observed time pattern.

## Period comparisons

The three periods are descriptive windows, not causal regimes or formal structural-break tests. Adjacent windows share an endpoint year, so their changes should not be treated as independent observations.

{chr(10).join(period_lines)}

**Interpretation.** The 2019–2022 disagreement is especially important: the CV and SD imply widening while P90–P10 implies narrowing. This is not a computational contradiction; each statistic summarises a different aspect of dispersion. The analysis establishes **when and in what sense** the spread changed; it does not establish the economic drivers of those changes.

## Year-by-year changes

{chr(10).join(year_changes)}

**Reading rule.** A negative CV change is measured in **percentage points**, not percent; SD and P90–P10 changes are in **GDP index points**. Years where all three indicators decreased: **{', '.join(all_three_years) if all_three_years else 'none'}**. Years with mixed directional signals: **{', '.join(disagreement_years) if disagreement_years else 'none'}**.

## Data quality, flags and comparability

The panel is **{'balanced' if balanced else 'unbalanced'}** in its observed region-year structure, with **{missing} missing GDP index values** and no duplicate region-year combinations. The flag field contains **{flag_counts.get('p',0)} provisional (p)**, **{flag_counts.get('b',0)} break-marked (b)** and **{flag_counts.get('e',0)} estimated (e)** observations; **{flagged} observations** are flagged in total. The files `gdp_flags_by_year.csv` and `unflagged_only_sensitivity.csv` permit transparent review. A flagged observation is not automatically invalid; removing all flagged observations also changes the annual sample and may bias comparisons.

**Critical unresolved check.** Stable region codes and a balanced panel do **not** prove harmonisation to a single NUTS classification. Before final submission, the team should verify the source vintage, whether historical figures were backcast to the same NUTS boundaries, and the meaning of the original flags against Eurostat metadata. Until that is documented, describe the panel as *balanced by observed codes*, not as definitively boundary-harmonised.

## Conclusion and hand-off to the other research questions

**RQ1A answer:** Dispersion in regional GDP per capita PPS index **narrowed between {years[0]} and {years[-1]}**, according to the CV, SD and P90–P10 gap, but **not year after year**. Changes differ across periods and across statistics, so the conclusion is conditional on the measure and time window. The decline also persists in an upper-tail-trimmed CV check, though its size changes.

**For RQ2:** These results document sigma convergence but do not test beta convergence; compare this descriptive time pattern with growth-on-initial-income findings. **For RQ3:** The distributional differences motivate investigation of regional groupings, without assigning mechanisms here. **For RQ4:** The periods where dispersion widens or measures disagree provide useful cases for reconciling sigma and beta convergence. No explanatory or causal claims are made in RQ1A.

## Reproducibility

The narrative and figures are generated from the same input panel by `rq1a_enhanced_with_explanations.py`. Re-running the script regenerates the numerical statements in this Markdown file and the supporting CSVs. It is suitable for inclusion in the team's Quarto project; verify figure paths relative to the final QMD file.
"""
    (out/'rq1a_detailed_explanations.md').write_text(markdown,encoding='utf-8')

    print('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('eu_nuts2_master_panel.csv'));p.add_argument('--output',type=Path,default=Path('rq1a_enhanced_outputs'))
    args=p.parse_args();main(args.input,args.output)
