import pandas as pd

GROUPS = ['a', 'b', 'c', 'd']

# Toggle between 'shaped' and 'flat' to compare submission scores.
# 'shaped' — multiply daily CV by Apr-Jun intraday shape ratio (original approach)
# 'flat'   — divide daily CV evenly across all 48 intervals
SHAPE_MODE = 'shaped'

# Upward bias per group to reduce underprediction penalty (Pt).
# In 'flat' mode set to 1.0 first to isolate the shape effect.
BIAS = {
    'shaped': {'A': 1.03, 'B': 1.03, 'C': 1.03, 'D': 1.03},
    'flat':   {'A': 1.0,  'B': 1.0,  'C': 1.0,  'D': 1.0 },
}

# Zero out interval CV predictions below this threshold.
# Overnight slots often have near-zero actual volume; small positive predictions
# add pure error. Set to 0 to disable, or try 1-3 to suppress low-volume noise.
OVERNIGHT_ZERO_THRESHOLD = 0  # set > 0 to zero out low overnight predictions
RECENCY_WEIGHT = 1            # set > 1 to upweight most recent month in shape (configure in agg.py)

# --- Load shape ---

shape = pd.read_csv('cleaned_data/intraday_shape.csv')[
    ['group', 'day_of_week', 'interval', 'shape_call_volume']
]

# --- Load August 2025 daily CV for each group, combine into one dataframe ---

daily_frames = []
for g in GROUPS:
    df = pd.read_csv(f'cleaned_data/{g}_daily_cleaned.csv', encoding='utf-8-sig')
    df['Date']  = pd.to_datetime(df['Date'], format='%m/%d/%y')
    df['group'] = g.upper()
    df = df[(df['Date'].dt.year == 2025) & (df['Date'].dt.month == 8)]
    daily_frames.append(df[['group', 'Date', 'Call Volume']])

daily = pd.concat(daily_frames, ignore_index=True)
daily['day_of_week'] = daily['Date'].dt.day_name()

# --- Join shape onto daily (each daily row fans out to 48 interval rows) ---

forecast = daily.merge(shape, on=['group', 'day_of_week'], how='left')

# --- Compute interval CV ---

bias = BIAS[SHAPE_MODE]

if SHAPE_MODE == 'shaped':
    forecast['interval_cv'] = forecast['Call Volume'] * forecast['shape_call_volume'] * forecast['group'].map(bias)
elif SHAPE_MODE == 'flat':
    forecast['interval_cv'] = forecast['Call Volume'] / 48 * forecast['group'].map(bias)

forecast['interval_cv'] = forecast['interval_cv'].clip(lower=0).round().astype(int)

# Zero out low overnight predictions
if OVERNIGHT_ZERO_THRESHOLD > 0:
    forecast.loc[forecast['interval_cv'] < OVERNIGHT_ZERO_THRESHOLD, 'interval_cv'] = 0

# --- Validate: interval sums vs daily totals ---

validation = (
    forecast
    .groupby(['group', 'Date'])['interval_cv']
    .sum()
    .reset_index()
    .rename(columns={'interval_cv': 'interval_sum'})
    .merge(daily[['group', 'Date', 'Call Volume']], on=['group', 'Date'])
)
validation['pct_diff'] = ((validation['interval_sum'] - validation['Call Volume']) / validation['Call Volume'] * 100).round(2)
print(f"SHAPE_MODE = '{SHAPE_MODE}'")
print("Validation — interval sum vs daily total (pct diff from daily):")
print(validation.groupby('group')['pct_diff'].describe().round(2).to_string())
print()

# --- Output ---

out = forecast[['group', 'Date', 'day_of_week', 'interval', 'interval_cv']].sort_values(
    ['group', 'Date', 'interval']
).reset_index(drop=True)

out_path = 'forecasts/cv_forecast.csv'
out.to_csv(out_path, index=False)

print(f"Wrote {len(out)} rows to {out_path}")
print(f"Expected {4 * 31 * 48} rows (4 groups × 31 days × 48 intervals)")