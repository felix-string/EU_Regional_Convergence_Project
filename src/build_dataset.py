from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from config import START_YEAR, END_YEAR, EU27, DATASETS, FILTER_CANDIDATES
from eurostat_io import download_json, jsonstat_to_long

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
OUTPUTS = ROOT / 'outputs'


def is_nuts2(code: str) -> bool:
    # EU NUTS 2 codes contain exactly four characters. Aggregates and NUTS 0, 1, 3 are rejected.
    return bool(re.fullmatch(r'[A-Z]{2}[A-Z0-9]{2}', str(code))) and str(code)[:2] in EU27


def choose_and_filter(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict]:
    selected = {}
    out = df.copy()
    for col, candidates in FILTER_CANDIDATES.get(name, {}).items():
        if col not in out.columns:
            continue
        available = set(out[col].dropna().astype(str))
        choice = next((x for x in candidates if x in available), None)
        if choice is None:
            raise ValueError(f'{name}: none of {candidates} found in {col}. Available sample: {sorted(available)[:30]}')
        out = out[out[col].astype(str).eq(choice)]
        selected[col] = choice
    return out, selected


def prepare_core(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df, selected = choose_and_filter(df, name)
    df = df[df['geo'].map(is_nuts2)].copy()
    df['year'] = pd.to_numeric(df['time'], errors='coerce').astype('Int64')
    df['country'] = df['geo'].str[:2]
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['quality_flag'] = df['obs_status'].fillna('')
    print(f'{name}: filters used {selected}; rows={len(df):,}')
    keep = ['geo', 'country', 'year', 'value', 'quality_flag']
    return df[keep].drop_duplicates(['geo', 'year'])


def prepare_sector(df: pd.DataFrame) -> pd.DataFrame:
    df, selected = choose_and_filter(df, 'sector_employment')
    df = df[df['geo'].map(is_nuts2)].copy()
    df['year'] = pd.to_numeric(df['time'], errors='coerce').astype('Int64')
    df['country'] = df['geo'].str[:2]
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    nace_col = next((c for c in ['nace_r2', 'nace_r2_1'] if c in df.columns), None)
    if nace_col is None:
        raise ValueError('Sector file has no NACE dimension. Inspect the raw CSV columns.')
    # Broad, interpretable groups. Total is used as denominator where available.
    total = df[df[nace_col].eq('TOTAL')][['geo','year','value']].rename(columns={'value':'sector_total'})
    broad = {
        'agriculture_share': ['A'],
        'industry_share': ['B-E', 'B-F'],
        'construction_share': ['F'],
        'market_services_share': ['G-I', 'G-N'],
        'knowledge_services_share': ['J', 'K', 'M_N', 'J-N'],
        'public_services_share': ['O-Q', 'O-U'],
    }
    result = total.copy()
    available = set(df[nace_col].astype(str))
    for output, choices in broad.items():
        code = next((c for c in choices if c in available), None)
        if code:
            x = df[df[nace_col].eq(code)][['geo','year','value']].rename(columns={'value':output})
            result = result.merge(x, on=['geo','year'], how='left')
            result[output] = 100 * result[output] / result['sector_total']
    print(f'sector_employment: filters used {selected}')
    return result.drop(columns='sector_total')


def count_regions(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    return (df.dropna(subset=[variable]).groupby('year')['geo'].nunique()
              .rename(variable).reset_index())


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    prepared = {}
    audit = []

    for name, code in DATASETS.items():
        raw_json = RAW / f'{code}.json'
        if raw_json.exists():
            payload = json.loads(raw_json.read_text(encoding='utf-8'))
        else:
            print(f'Downloading {code} ...')
            payload = download_json(code, raw_json, START_YEAR, END_YEAR)
        long = jsonstat_to_long(payload)
        long.to_csv(RAW / f'{code}_long.csv', index=False)
        audit.append({'dataset': name, 'code': code, 'rows_raw': len(long),
                      'columns': ', '.join(long.columns)})
        if name == 'sector_employment':
            prepared[name] = prepare_sector(long)
        else:
            prepared[name] = prepare_core(long, name).rename(columns={'value': name, 'quality_flag': f'{name}_flag'})

    panel = prepared['gdp']
    for name in ['employment_rate','youth_unemployment','population']:
        add = prepared[name].drop(columns='country')
        panel = panel.merge(add, on=['geo','year'], how='outer')
    panel = panel.merge(prepared['sector_employment'], on=['geo','year'], how='left')
    panel['country'] = panel['geo'].str[:2]

    # Remove impossible values and preserve missingness rather than filling it.
    for c in ['employment_rate','youth_unemployment']:
        if c in panel:
            panel.loc[~panel[c].between(0,100), c] = np.nan
    panel.loc[panel['population'].le(0), 'population'] = np.nan

    # Main analysis panel: codes observed in GDP in every included year.
    years = list(range(START_YEAR, END_YEAR + 1))
    g = panel[panel['year'].isin(years)].dropna(subset=['gdp'])
    stable = g.groupby('geo')['year'].nunique()
    stable_codes = set(stable[stable.eq(len(years))].index)
    panel['stable_code_full_period'] = panel['geo'].isin(stable_codes)

    panel.sort_values(['geo','year']).to_csv(PROCESSED / 'eu_nuts2_master_panel.csv', index=False)
    panel[panel['stable_code_full_period']].sort_values(['geo','year']).to_csv(
        PROCESSED / 'eu_nuts2_balanced_panel.csv', index=False)

    counts = None
    for variable in ['gdp','employment_rate','youth_unemployment','population']:
        c = count_regions(panel, variable)
        counts = c if counts is None else counts.merge(c, on='year', how='outer')
    counts.to_csv(OUTPUTS / 'region_counts_by_year.csv', index=False)
    pd.DataFrame(audit).to_csv(OUTPUTS / 'download_audit.csv', index=False)

    missing = panel.groupby('year').agg(
        rows=('geo','size'),
        gdp_missing=('gdp', lambda s: s.isna().sum()),
        employment_missing=('employment_rate', lambda s: s.isna().sum()),
        youth_unemployment_missing=('youth_unemployment', lambda s: s.isna().sum()),
        population_missing=('population', lambda s: s.isna().sum()),
    ).reset_index()
    missing.to_csv(OUTPUTS / 'missingness_by_year.csv', index=False)

    print('\nCreated:')
    print(PROCESSED / 'eu_nuts2_master_panel.csv')
    print(PROCESSED / 'eu_nuts2_balanced_panel.csv')
    print(OUTPUTS / 'region_counts_by_year.csv')
    print(f'Stable NUTS 2 codes across {START_YEAR}-{END_YEAR}: {len(stable_codes)}')

if __name__ == '__main__':
    main()
