import csv
from collections import defaultdict
import math
import os

GROUPS = ['a', 'b', 'c', 'd']

# Months to include when building the interval shape (1=Jan ... 12=Dec).
# Must match SHAPE_MONTHS in intraday_shape.py.
# Options: [4, 5, 6] = Apr-Jun (full dataset), [6] = June only
SHAPE_MONTHS = [4, 5, 6]

# Weight multiplier for the most recent month vs earlier months.
# Set to 1 for equal weighting (best empirically — June-heavy shapes hurt EV).
RECENCY_WEIGHT = 1

# Dates to exclude from the shape calculation (YYYY-MM-DD).
# These are major holidays with atypical intraday distributions that would
# distort the shape when applied to normal August days.
# Must match EXCLUDE_DATES in intraday_shape.py.
EXCLUDE_DATES = {
    '2025-04-18',  # Good Friday      — 7-13% below normal Friday
    '2025-04-20',  # Easter Sunday     — 40-55% below normal Sunday; morning-shifted shape
    '2025-05-11',  # Mother's Day      — 14-23% below normal Sunday
    '2025-05-26',  # Memorial Day      — 43-51% below normal Monday
    '2025-06-15',  # Father's Day      — 5-12% below normal Sunday (Groups C, D)
    '2025-06-19',  # Juneteenth        — 6-9% below normal Thursday (Groups A, C, D)
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'cleaned_data')
OUTPUT_FILE = os.path.join(BASE_DIR, 'cleaned_data', 'interval_aggregated.csv')

METRICS = ['Call Volume', 'Abandoned Calls', 'Abandoned Rate', 'Service Level', 'CCT']
METRIC_KEYS = ['call_volume', 'abandoned_calls', 'abandoned_rate', 'service_level', 'cct']

DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def parse_datetime(dt_str):
    date_part, time_part = dt_str.strip().split(' ')
    year, month, day = map(int, date_part.split('-'))
    import datetime
    d = datetime.date(year, month, day)
    return d.weekday(), time_part[:5]

def trimmed_mean(values, trim=1):
    """Drop `trim` lowest and `trim` highest values, return mean of remainder.
    Falls back to plain mean if not enough observations to trim."""
    if len(values) <= 2 * trim:
        return sum(values) / len(values)
    s = sorted(values)
    trimmed = s[trim:-trim]
    return sum(trimmed) / len(trimmed)

def std_dev(values, mean):
    if len(values) < 2:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)

data = defaultdict(lambda: defaultdict(list))

for group in GROUPS:
    filepath = os.path.join(INPUT_DIR, f'{group}_interval_cleaned.csv')
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row['DateTime']:
                continue
            date_str = row['DateTime'].strip().split(' ')[0]  # 'YYYY-MM-DD'
            # Filter to SHAPE_MONTHS only
            month = int(date_str.split('-')[1])
            if month not in SHAPE_MONTHS:
                continue
            # Exclude holiday dates with atypical intraday distributions
            if date_str in EXCLUDE_DATES:
                continue
            dow, interval = parse_datetime(row['DateTime'])
            key = (group.upper(), dow, interval)
            # Repeat most recent month's observations by RECENCY_WEIGHT
            most_recent = max(SHAPE_MONTHS)
            repeat = RECENCY_WEIGHT if month == most_recent else 1
            for col, metric_key in zip(METRICS, METRIC_KEYS):
                try:
                    val = float(row[col])
                    data[key][metric_key].extend([val] * repeat)
                except (ValueError, KeyError):
                    pass
                
output_columns = ['group', 'day_of_week', 'interval']
for mk in METRIC_KEYS:
    output_columns += [f'mean_{mk}', f'std_{mk}']
output_columns += ['n_observations']

rows_out = []
for (group, dow, interval), metrics in sorted(data.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
    row = {
        'group': group,
        'day_of_week': DAY_NAMES[dow],
        'interval': interval,
    }
    n = len(metrics.get('call_volume', []))
    row['n_observations'] = n
    for mk in METRIC_KEYS:
        vals = metrics.get(mk, [])
        if vals:
            mean = trimmed_mean(vals)
            sd = std_dev(vals, mean)
        else:
            mean, sd = None, None
        row[f'mean_{mk}'] = round(mean, 4) if mean is not None else ''
        row[f'std_{mk}'] = round(sd, 4) if sd is not None else ''
    rows_out.append(row)

with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=output_columns)
    writer.writeheader()
    writer.writerows(rows_out)

print(f"Written {len(rows_out)} rows to {OUTPUT_FILE}")
print(f"Expected: 4 groups x 7 days x 48 intervals = {4*7*48} rows")

# =============================================================================
# DOM (day-of-month) aggregation
# Aggregates by (group, day_of_month, interval) instead of (group, DOW, interval).
# Each cell has ~3 observations (one per shape month), one from each distinct DOW.
# Used to compute DOW-normalized intramonth correction factors in intraday_shape.py.
# =============================================================================

data_dom = defaultdict(lambda: defaultdict(list))

for group in GROUPS:
    filepath = os.path.join(INPUT_DIR, f'{group}_interval_cleaned.csv')
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row['DateTime']:
                continue
            date_str = row['DateTime'].strip().split(' ')[0]  # 'YYYY-MM-DD'
            month = int(date_str.split('-')[1])
            if month not in SHAPE_MONTHS:
                continue
            if date_str in EXCLUDE_DATES:
                continue
            day_of_month = int(date_str.split('-')[2])
            _, interval = parse_datetime(row['DateTime'])
            key = (group.upper(), day_of_month, interval)
            most_recent = max(SHAPE_MONTHS)
            repeat = RECENCY_WEIGHT if month == most_recent else 1
            for col, metric_key in zip(METRICS, METRIC_KEYS):
                try:
                    val = float(row[col])
                    data_dom[key][metric_key].extend([val] * repeat)
                except (ValueError, KeyError):
                    pass

output_columns_dom = ['group', 'day_of_month', 'interval']
for mk in METRIC_KEYS:
    output_columns_dom += [f'mean_{mk}', f'std_{mk}']
output_columns_dom += ['n_observations']

rows_out_dom = []
for (group, dom, interval), metrics in sorted(
    data_dom.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])
):
    row = {'group': group, 'day_of_month': dom, 'interval': interval}
    n = len(metrics.get('call_volume', []))
    row['n_observations'] = n
    for mk in METRIC_KEYS:
        vals = metrics.get(mk, [])
        if vals:
            mean = trimmed_mean(vals)
            sd   = std_dev(vals, mean)
        else:
            mean, sd = None, None
        row[f'mean_{mk}'] = round(mean, 4) if mean is not None else ''
        row[f'std_{mk}']  = round(sd,   4) if sd   is not None else ''
    rows_out_dom.append(row)

OUTPUT_FILE_DOM = os.path.join(BASE_DIR, 'cleaned_data', 'interval_aggregated_dom.csv')
with open(OUTPUT_FILE_DOM, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=output_columns_dom)
    writer.writeheader()
    writer.writerows(rows_out_dom)

print(f"\nWritten {len(rows_out_dom)} rows to {OUTPUT_FILE_DOM}")
print(f"Expected at most: 4 groups x 31 days x 48 intervals = {4*31*48} rows")