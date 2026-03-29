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

# =============================================================================
# DOM intramonth correction factors
#
# Goal: isolate the within-month call pattern from DOW effects.
#
# Method:
#   1. For each (group, date, interval) in Apr-Jun, compute the per-date shape:
#        actual_shape = interval_cv / daily_cv_that_day
#   2. Look up the DOW baseline shape for that (group, DOW, interval).
#   3. DOM residual = actual_shape / DOW_baseline  (~1.0 on average)
#   4. Average residuals by (group, day_of_month, interval) across the 3 months.
#
# Usage in forecast: predicted = daily_cv × DOW_shape × DOM_correction
# DOM_correction ≈ 1.0 means no intramonth adjustment; values above/below 1.0
# indicate that interval tends to run higher/lower than the DOW baseline on
# that day of the month (e.g., billing-cycle effects).
#
# Caveats:
#   - Only 3 observations per (group, DOM, interval) cell — high variance.
#   - Day 31 has only 1 observation (May 31); its correction is unreliable.
# =============================================================================

# Build (group, date) daily CV lookup from cleaned daily files
daily_cv_lookup = {}
for grp in GROUPS:
    df = pd.read_csv(f'cleaned_data/{grp}_daily_cleaned.csv', encoding='utf-8-sig')
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
    df = df[(df['Date'].dt.year == 2025) & df['Date'].dt.month.isin(SHAPE_MONTHS)]
    df = df[~df['Date'].dt.strftime('%Y-%m-%d').isin(EXCLUDE_DATES)]
    for _, row in df.iterrows():
        daily_cv_lookup[(grp.upper(), row['Date'].strftime('%Y-%m-%d'))] = row['Call Volume']

# Load per-interval observations from cleaned interval files
intv_rows = []
for grp in GROUPS:
    df = pd.read_csv(f'cleaned_data/{grp}_interval_cleaned.csv', encoding='utf-8-sig')
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
    df = df[df['Date'].dt.month.isin(SHAPE_MONTHS)]
    df = df[~df['Date'].dt.strftime('%Y-%m-%d').isin(EXCLUDE_DATES)]
    df['group'] = grp.upper()
    df['date_str'] = df['Date'].dt.strftime('%Y-%m-%d')
    df['day_of_month'] = df['Date'].dt.day
    df['day_of_week'] = df['Date'].dt.day_name()
    # Normalize interval to zero-padded HH:MM
    df['interval'] = df['Interval'].apply(
        lambda x: f"{int(x.split(':')[0]):02d}:{x.split(':')[1]}" if isinstance(x, str) else None
    )
    intv_rows.append(df[['group', 'date_str', 'day_of_month', 'day_of_week', 'interval', 'Call Volume']])

intv_all = pd.concat(intv_rows, ignore_index=True)
intv_all = intv_all[intv_all['interval'].notna() & intv_all['Call Volume'].notna()]

# Look up daily CV for each row
intv_all['daily_cv'] = intv_all.apply(
    lambda r: daily_cv_lookup.get((r['group'], r['date_str'])), axis=1
)
intv_all = intv_all[intv_all['daily_cv'].notna() & (intv_all['daily_cv'] > 0)]

# Compute per-date actual shape ratio for each (group, date, interval)
intv_all['actual_shape'] = intv_all['Call Volume'] / intv_all['daily_cv']

# Load DOW baseline shape for lookup
dow_shape_lookup = (
    shape[['group', 'day_of_week', 'interval', 'shape_call_volume']]
    .set_index(['group', 'day_of_week', 'interval'])['shape_call_volume']
)

intv_all['dow_shape'] = intv_all.set_index(['group', 'day_of_week', 'interval']).index.map(
    dow_shape_lookup
)
intv_all['dow_shape'] = intv_all.apply(
    lambda r: dow_shape_lookup.get((r['group'], r['day_of_week'], r['interval'])), axis=1
)

# DOM residual = actual_shape / DOW_baseline (exclude rows where DOW shape is 0 or missing)
valid = intv_all['dow_shape'].notna() & (intv_all['dow_shape'] > 0)
intv_all.loc[valid, 'dom_residual'] = (
    intv_all.loc[valid, 'actual_shape'] / intv_all.loc[valid, 'dow_shape']
)

# Average DOM residuals by (group, day_of_month, interval) using trimmed mean
def trimmed_mean_series(s, trim=1):
    vals = s.dropna().tolist()
    if len(vals) == 0:
        return None
    if len(vals) <= 2 * trim:
        return sum(vals) / len(vals)
    s_sorted = sorted(vals)
    trimmed = s_sorted[trim:-trim]
    return sum(trimmed) / len(trimmed)

dom_correction = (
    intv_all[valid]
    .groupby(['group', 'day_of_month', 'interval'])['dom_residual']
    .agg(trimmed_mean_series)
    .rename('dom_correction')
    .reset_index()
)

# Count observations per cell for diagnostics
dom_n_obs = (
    intv_all[valid]
    .groupby(['group', 'day_of_month', 'interval'])['dom_residual']
    .count()
    .rename('n_obs')
    .reset_index()
)
dom_correction = dom_correction.merge(dom_n_obs, on=['group', 'day_of_month', 'interval'])

dom_out_path = 'cleaned_data/intraday_shape_dom.csv'
dom_correction.to_csv(dom_out_path, index=False)

print(f'\nWrote {len(dom_correction)} rows to {dom_out_path}')
print(f'Expected at most: 4 groups x 31 days x 48 intervals = {4*31*48} rows')
print(f'DOM correction factor stats:')
print(dom_correction['dom_correction'].describe().round(4).to_string())
print(f'Cells with n_obs == 1 (day-31 only, unreliable): {(dom_correction["n_obs"] == 1).sum()}')

# ── Bucketed DOM correction (early=1-10, mid=11-20, late=21-31) ──────────────
# ~30 observations per (group, bucket, interval) cell vs ~3 for per-DOM.
# Much lower variance; this is the version used in forecast_cv.py.

def dom_to_bucket(dom):
    if dom <= 10:  return 'early'
    if dom <= 20:  return 'mid'
    return 'late'

intv_all['dom_bucket'] = intv_all['day_of_month'].apply(dom_to_bucket)

dom_bucket_correction = (
    intv_all[valid]
    .groupby(['group', 'dom_bucket', 'interval'])['dom_residual']
    .agg(trimmed_mean_series)
    .rename('dom_correction')
    .reset_index()
)
dom_bucket_n_obs = (
    intv_all[valid]
    .groupby(['group', 'dom_bucket', 'interval'])['dom_residual']
    .count()
    .rename('n_obs')
    .reset_index()
)
dom_bucket_correction = dom_bucket_correction.merge(
    dom_bucket_n_obs, on=['group', 'dom_bucket', 'interval']
)

dom_bucket_out_path = 'cleaned_data/intraday_shape_dom_bucket.csv'
dom_bucket_correction.to_csv(dom_bucket_out_path, index=False)

print(f'\nWrote {len(dom_bucket_correction)} rows to {dom_bucket_out_path}')
print(f'Expected: 4 groups x 3 buckets x 48 intervals = {4*3*48} rows')
print(f'Bucket DOM correction factor stats:')
print(dom_bucket_correction['dom_correction'].describe().round(4).to_string())
print(f'n_obs range: {dom_bucket_correction["n_obs"].min()}–{dom_bucket_correction["n_obs"].max()}')
