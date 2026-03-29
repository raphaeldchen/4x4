"""
Compute intraday shape: for each (group, day_of_week, interval),
divide interval means by the daily mean for that group/day_of_week.
Daily means and staffing are computed from April-June 2025 data only.
"""

import pandas as pd

GROUPS = ['a', 'b', 'c', 'd']

# Must match SHAPE_MONTHS in agg.py.
SHAPE_MONTHS = [4, 5, 6]

# Must match EXCLUDE_DATES in agg.py.
EXCLUDE_DATES = {
    '2025-04-18',  # Good Friday
    '2025-04-20',  # Easter Sunday
    '2025-05-11',  # Mother's Day
    '2025-05-26',  # Memorial Day
}

METRICS = [
    ('mean_call_volume',    'Call Volume'),
    ('mean_service_level',  'Service Level'),
    ('mean_abandoned_rate', 'Abandon Rate'),
    ('mean_cct',            'CCT'),
]

# --- Step 1: Load daily data, filter Apr-Jun 2025, compute means by (group, dow) ---

daily_frames = []
for group in GROUPS:
    df = pd.read_csv(f'cleaned_data/{group}_daily_cleaned.csv', encoding='utf-8-sig')
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
    df = df[(df['Date'].dt.year == 2025) & df['Date'].dt.month.isin(SHAPE_MONTHS)]
    df = df[~df['Date'].dt.strftime('%Y-%m-%d').isin(EXCLUDE_DATES)]
    df['day_of_week'] = df['Date'].dt.day_name()
    df['group'] = group.upper()
    daily_frames.append(df)

daily = pd.concat(daily_frames, ignore_index=True)

def trimmed_mean(x, trim=1):
    s = sorted(x.dropna())
    if len(s) <= 2 * trim:
        return x.mean()
    return pd.Series(s[trim:-trim]).mean()

daily_means = (
    daily
    .groupby(['group', 'day_of_week'])[['Call Volume', 'CCT', 'Service Level', 'Abandon Rate']]
    .agg(trimmed_mean)
    .rename(columns={
        'Call Volume':   'daily_call_volume',
        'CCT':           'daily_cct',
        'Service Level': 'daily_service_level',
        'Abandon Rate':  'daily_abandoned_rate',
    })
    .reset_index()
)

# --- Step 2: Load staffing, filter Apr-Jun 2025, compute means by dow per group ---

staffing = pd.read_csv('cleaned_data/daily_staffing_cleaned.csv', encoding='utf-8-sig')
staffing['Date'] = pd.to_datetime(staffing['Date'], format='%m/%d/%y')
staffing = staffing[(staffing['Date'].dt.year == 2025) & staffing['Date'].dt.month.isin(SHAPE_MONTHS)]
staffing['day_of_week'] = staffing['Date'].dt.day_name()

# Melt wide (A, B, C, D columns) to long form so we can join on (group, day_of_week)
staffing_long = staffing.melt(
    id_vars=['day_of_week'],
    value_vars=['A', 'B', 'C', 'D'],
    var_name='group',
    value_name='staffing',
)
staffing_means = (
    staffing_long
    .groupby(['group', 'day_of_week'])['staffing']
    .mean()
    .rename('daily_staffing')
    .reset_index()
)

# --- Step 3: Load interval_aggregated.csv, join daily means and staffing ---

shape = pd.read_csv('cleaned_data/interval_aggregated.csv', encoding='utf-8-sig')

shape = shape.merge(daily_means, on=['group', 'day_of_week'], how='left')
shape = shape.merge(staffing_means, on=['group', 'day_of_week'], how='left')

# --- Step 4: Compute shape ratios ---

for interval_col, _ in METRICS:
    short = interval_col.replace('mean_', '')
    daily_col = f'daily_{short}'
    shape[f'shape_{short}'] = shape[interval_col] / shape[daily_col]

# --- Step 5: Select and order output columns ---

output_cols = [
    'group', 'day_of_week', 'interval',
    'interval_call_volume',    'daily_call_volume',    'shape_call_volume',
    'interval_service_level',  'daily_service_level',  'shape_service_level',
    'interval_abandoned_rate', 'daily_abandoned_rate', 'shape_abandoned_rate',
    'interval_cct',            'daily_cct',            'shape_cct',
    'daily_staffing',
]

shape = shape.rename(columns={
    'mean_call_volume':    'interval_call_volume',
    'mean_service_level':  'interval_service_level',
    'mean_abandoned_rate': 'interval_abandoned_rate',
    'mean_cct':            'interval_cct',
})

# If SHAPE_MONTHS excludes some months, certain (group, DOW, interval) combos
# may be missing. Fill gaps using the full Apr-Jun shape as a fallback.
if len(shape) < 1344:
    fallback = pd.read_csv('cleaned_data/intraday_shape_aprijun.csv', encoding='utf-8-sig')
    shape = (
        fallback[['group', 'day_of_week', 'interval']]
        .merge(shape[output_cols], on=['group', 'day_of_week', 'interval'], how='left')
    )
    for col in output_cols:
        if col not in shape.columns:
            shape[col] = fallback[col]
    # Fill any remaining NaN shape columns from fallback
    for col in [c for c in output_cols if c.startswith('shape_') or c.startswith('daily_') or c.startswith('interval_')]:
        shape[col] = shape[col].fillna(fallback[col])
    missing = 1344 - shape[output_cols].dropna().shape[0]
    print(f'Filled {1344 - len(shape[shape[output_cols[3]].notna()])} missing intervals from Apr-Jun fallback')

out_path = 'cleaned_data/intraday_shape.csv'
shape[output_cols].to_csv(out_path, index=False)

print(f'Wrote {len(shape)} rows to {out_path}')
print(f'Expected 1344 rows, got {len(shape)}')
