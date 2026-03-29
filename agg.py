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
# e.g. RECENCY_WEIGHT = 1 repeats the most recent month's observations 2×,
# giving it twice the influence on the shape means.
# Set to 1 for equal weighting.
RECENCY_WEIGHT = 1

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
            # Filter to SHAPE_MONTHS only
            month = int(row['DateTime'].strip().split('-')[1])
            if month not in SHAPE_MONTHS:
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
            mean = sum(vals) / len(vals)
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