from __future__ import annotations
import itertools
from pathlib import Path
import requests
import pandas as pd

BASE = 'https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data'


def download_json(dataset: str, out_file: Path, start: int, end: int) -> dict:
    params = [('lang', 'en')]
    for year in range(start, end + 1):
        params.append(('time', str(year)))
    url = f'{BASE}/{dataset}'
    response = requests.get(url, params=params, timeout=180)
    response.raise_for_status()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(response.text, encoding='utf-8')
    return response.json()


def ordered_codes(dimension: dict) -> list[str]:
    index = dimension['category']['index']
    if isinstance(index, list):
        return index
    return [code for code, pos in sorted(index.items(), key=lambda x: x[1])]


def jsonstat_to_long(payload: dict) -> pd.DataFrame:
    dims = payload['id']
    codes = [ordered_codes(payload['dimension'][d]) for d in dims]
    values = payload.get('value', {})
    status = payload.get('status', {})
    rows = []
    for flat_index, combo in enumerate(itertools.product(*codes)):
        value = values.get(str(flat_index), values.get(flat_index))
        flag = status.get(str(flat_index), status.get(flat_index))
        if value is None:
            continue
        row = dict(zip(dims, combo))
        row['value'] = value
        row['obs_status'] = flag
        rows.append(row)
    return pd.DataFrame(rows)
