import pandas as pd

GROUPS = ['a', 'b', 'c', 'd']
SHAPE_MONTHS = [4, 5, 6]

# keep in sync with agg.py
EXCLUDE_DATES = {
    '2025-04-18',  # Good Friday
    '2025-04-20',  # Easter Sunday
    '2025-04-21',  # Easter return day (Mon)
    '2025-05-11',  # Mother's Day
    '2025-05-12',  # Mother's Day return day (Mon)
    '2025-05-26',  # Memorial Day
    '2025-05-27',  # Memorial Day return day (Tue)
    '2025-06-15',  # Father's Day
    '2025-06-16',  # Father's Day return day (Mon)
    '2025-06-19',  # Juneteenth
    '2025-06-20',  # Juneteenth return day (Fri)
}

# metrics to compute ratio-of-means shapes for (CV gets overwritten with ratio-of-sums)
SHAPE_METRICS = ['call_volume', 'service_level', 'abandoned_rate', 'cct']

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
        'Call Volume': 'daily_call_volume',
        'CCT': 'daily_cct',
        'Service Level': 'daily_service_level',
        'Abandon Rate': 'daily_abandoned_rate',
    })
    .reset_index()
)

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
intervals_raw['Interval'] = intervals_raw['Interval'].str.replace(
    r'^(\d):(\d{2})$', r'0\1:\2', regex=True
)

# ratio-of-sums CV shape: volume-weighted, sums to exactly 1.0 per (group, DOW)
iv_sums = (
    intervals_raw.groupby(['group', 'day_of_week', 'Interval'])['Call Volume']
    .sum().reset_index()
    .rename(columns={'Interval': 'interval', 'Call Volume': 'sum_cv'})
)
dow_totals = (
    intervals_raw.groupby(['group', 'day_of_week'])['Call Volume']
    .sum().rename('total_cv').reset_index()
)
ros_shape = iv_sums.merge(dow_totals, on=['group', 'day_of_week'])
ros_shape['shape_call_volume'] = ros_shape['sum_cv'] / ros_shape['total_cv']

shape_check = ros_shape.groupby(['group', 'day_of_week'])['shape_call_volume'].sum().round(4)
assert (shape_check == 1.0).all(), f"Shape sums not 1.0: {shape_check[shape_check != 1.0]}"
print(f"Ratio-of-sums shape: {len(ros_shape)} cells, all DOW sums=1.0 OK")

staffing = pd.read_csv('cleaned_data/daily_staffing_cleaned.csv', encoding='utf-8-sig')
staffing['Date'] = pd.to_datetime(staffing['Date'], format='%m/%d/%y')
staffing = staffing[(staffing['Date'].dt.year == 2025) & staffing['Date'].dt.month.isin(SHAPE_MONTHS)]
staffing['day_of_week'] = staffing['Date'].dt.day_name()

staffing_long = staffing.melt(
    id_vars=['day_of_week'],
    value_vars=['A', 'B', 'C', 'D'],
    var_name='group',
    value_name='staffing',
)
staffing_means = (
    staffing_long
    .groupby(['group', 'day_of_week'])['staffing']
    .mean().rename('daily_staffing').reset_index()
)

shape = pd.read_csv('cleaned_data/interval_aggregated.csv', encoding='utf-8-sig')
shape = shape.merge(daily_means, on=['group', 'day_of_week'], how='left')
shape = shape.merge(staffing_means, on=['group', 'day_of_week'], how='left')

for m in SHAPE_METRICS:
    shape[f'shape_{m}'] = shape[f'mean_{m}'] / shape[f'daily_{m}']

# replace CV shape with ratio-of-sums version
shape = shape.drop(columns=['shape_call_volume'])
shape = shape.merge(
    ros_shape[['group', 'day_of_week', 'interval', 'shape_call_volume']],
    on=['group', 'day_of_week', 'interval'],
    how='left',
)

shape = shape.rename(columns={
    'mean_call_volume': 'interval_call_volume',
    'mean_service_level': 'interval_service_level',
    'mean_abandoned_rate': 'interval_abandoned_rate',
    'mean_cct': 'interval_cct',
})

out_cols = [
    'group', 'day_of_week', 'interval',
    'interval_call_volume', 'daily_call_volume', 'shape_call_volume',
    'interval_service_level', 'daily_service_level', 'shape_service_level',
    'interval_abandoned_rate', 'daily_abandoned_rate', 'shape_abandoned_rate',
    'interval_cct', 'daily_cct', 'shape_cct',
    'daily_staffing',
]

shape[out_cols].to_csv('cleaned_data/intraday_shape.csv', index=False)
print(f'Wrote {len(shape)} rows to cleaned_data/intraday_shape.csv')
print(f'Expected 1344 rows, got {len(shape)}')
