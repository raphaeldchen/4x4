import pandas as pd

GROUPS = ['a', 'b', 'c', 'd']

# Upward bias per group to reduce underprediction penalty (Pt).
# CV shape sums are all ~1.0, so 1.03 is a clean ~3% lean for each group.
BIAS = {
    'A': 1.03,
    'B': 1.03,
    'C': 1.03,
    'D': 1.03,
}

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

forecast['interval_cv'] = forecast['Call Volume'] * forecast['shape_call_volume'] * forecast['group'].map(BIAS)
forecast['interval_cv'] = forecast['interval_cv'].clip(lower=0).round().astype(int)

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
print("Validation — interval sum vs daily total (sample):")
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