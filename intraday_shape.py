"""
Compute intraday shape: for each (group, day_of_week, interval),
divide interval CV by the daily CV for that group/day_of_week.

SHAPE_METHOD options:
  'ratio_of_means' — original: trimmed_mean(interval_cv) / trimmed_mean(daily_cv)
  'ratio_of_sums'  — Σ(interval_cv) / Σ(daily_cv) across all training days for that DOW.
                     Equivalent to a volume-weighted mean of per-day shapes: high-volume
                     (normal) days dominate; low-volume (holiday-adjacent) days are
                     naturally down-weighted without needing explicit exclusion.
                     Only applied to shape_call_volume; CCT/ABD/SL keep ratio_of_means.
"""

import pandas as pd

GROUPS = ['a', 'b', 'c', 'd']

# Must match SHAPE_MONTHS in agg.py.
SHAPE_MONTHS = [4, 5, 6]

SHAPE_METHOD = 'ratio_of_sums'  # 'ratio_of_means' = original trimmed-mean approach

# Must match EXCLUDE_DATES in agg.py.
EXCLUDE_DATES = {
    '2025-04-18',  # Good Friday
    '2025-04-20',  # Easter Sunday
    '2025-05-11',  # Mother's Day
    '2025-05-26',  # Memorial Day
    '2025-06-15',  # Father's Day     — 5-12% below normal Sunday
    '2025-06-19',  # Juneteenth       — 6-9% below normal Thursday
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

# --- Step 1b: Ratio-of-sums CV shape (volume-weighted, uses raw interval data) ---
# Computed here because it reuses the already-filtered `daily` DataFrame.
# Only affects shape_call_volume; other metrics use ratio_of_means below.

if SHAPE_METHOD == 'ratio_of_sums':
    # Load raw interval data with the same SHAPE_MONTHS + EXCLUDE_DATES filter.
    intv_frames = []
    for group in GROUPS:
        df = pd.read_csv(f'cleaned_data/{group}_interval_cleaned.csv', encoding='utf-8-sig')
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
        df = df[(df['Date'].dt.year == 2025) & df['Date'].dt.month.isin(SHAPE_MONTHS)]
        df = df[~df['Date'].dt.strftime('%Y-%m-%d').isin(EXCLUDE_DATES)]
        df['day_of_week'] = df['Date'].dt.day_name()
        df['group'] = group.upper()
        intv_frames.append(df[['group', 'day_of_week', 'Interval', 'Call Volume']])

    intervals_raw = pd.concat(intv_frames, ignore_index=True)
    # Normalize single-digit hours to zero-padded HH:MM (e.g. '9:00' → '09:00')
    intervals_raw['Interval'] = intervals_raw['Interval'].str.replace(
        r'^(\d):(\d{2})$', r'0\1:\2', regex=True
    )

    # Numerator: Σ(interval_cv) per (group, DOW, interval)
    interval_cv_sums = (
        intervals_raw.groupby(['group', 'day_of_week', 'Interval'])['Call Volume']
        .sum()
        .reset_index()
        .rename(columns={'Interval': 'interval', 'Call Volume': 'sum_interval_cv'})
    )

    # Denominator: Σ(interval_cv) across ALL intervals per (group, DOW).
    # Using the interval data as its own denominator guarantees shape sums to exactly 1.0
    # per (group, DOW), regardless of any daily↔interval data discrepancies.
    total_cv_per_dow = (
        intervals_raw.groupby(['group', 'day_of_week'])['Call Volume']
        .sum()
        .rename('total_interval_cv')
        .reset_index()
    )

    ros_shape = interval_cv_sums.merge(total_cv_per_dow, on=['group', 'day_of_week'])
    ros_shape['shape_call_volume_ros'] = ros_shape['sum_interval_cv'] / ros_shape['total_interval_cv']

    # Sanity check: verify shapes sum to 1.0 per (group, DOW)
    shape_check = (
        ros_shape.groupby(['group', 'day_of_week'])['shape_call_volume_ros'].sum().round(4)
    )
    assert (shape_check == 1.0).all(), f"Shape sums not 1.0: {shape_check[shape_check != 1.0]}"
    print(
        f"Ratio-of-sums shape: {len(ros_shape)} cells, "
        f"mean={ros_shape['shape_call_volume_ros'].mean():.5f}, "
        f"max={ros_shape['shape_call_volume_ros'].max():.5f}, "
        f"all DOW sums=1.0 OK"
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

# --- Step 4b: Override shape_call_volume with ratio-of-sums if selected ---

if SHAPE_METHOD == 'ratio_of_sums':
    shape = shape.drop(columns=['shape_call_volume'])
    shape = shape.merge(
        ros_shape[['group', 'day_of_week', 'interval', 'shape_call_volume_ros']]
        .rename(columns={'shape_call_volume_ros': 'shape_call_volume'}),
        on=['group', 'day_of_week', 'interval'],
        how='left',
    )

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
