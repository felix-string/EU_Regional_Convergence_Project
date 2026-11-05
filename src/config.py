START_YEAR = 2010
END_YEAR = 2024

EU27 = {
    'AT','BE','BG','HR','CY','CZ','DK','EE','FI','FR','DE','EL','HU','IE',
    'IT','LV','LT','LU','MT','NL','PL','PT','RO','SK','SI','ES','SE'
}

DATASETS = {
    'gdp': 'nama_10r_2gdp',
    'employment_rate': 'lfst_r_lfe2emprt',
    'youth_unemployment': 'lfst_r_lfu3rt',
    'population': 'demo_r_d2jan',
    'sector_employment': 'lfst_r_lfe2en2',
}

# The program chooses the first code that is actually present in the downloaded file.
FILTER_CANDIDATES = {
    'gdp': {
        'unit': ['PPS_HAB_EU27_2020', 'PPS_HAB'],
    },
    'employment_rate': {
        'sex': ['T'],
        'age': ['Y20-64'],
        'unit': ['PC'],
    },
    'youth_unemployment': {
        'sex': ['T'],
        'age': ['Y15-29', 'Y15-24'],
        'unit': ['PC_ACT'],
    },
    'population': {
        'sex': ['T'],
        'age': ['TOTAL'],
        'unit': ['NR'],
    },
    'sector_employment': {
        'sex': ['T'],
        'age': ['Y15-64', 'Y15-74', 'Y_GE15'],
        'unit': ['THS_PER'],
    },
}
