"""
Forecast interval-level Abandoned Rate for August 2025.
Method: August daily Abandon Rate (known) × intraday shape ratio (from Apr-Jun 2025).
ABD does not appear in the workload penalty (Pt), so no upward bias is needed —
but per-group bias variables are included for manual tuning if desired.
Output is the rate only (clamped to [0, 1]); Abandoned Calls are not computed here.
"""

import pandas as pd

GROUPS = ['a', 'b', 'c', 'd']

# ABD is not in the Pt penalty, so bias defaults to 1.0 for all groups.
# Tune upward if you observe systematic underprediction after review.
BIAS = {
    'A': 1.0,
    'B': 1.0,
    'C': 1.0,
    'D': 1.0,
}

# --- Load shape ---

shape = pd.read_csv('cleaned_data/intraday_shape.csv')[
    ['group', 'day_of_week', 'interval', 'shape_abandoned_rate']
]

# --- Load August 2025 daily Abandon Rate for each group ---

daily_frames = []
for g in GROUPS:
    df = pd.read_csv(f'cleaned_data/{g}_daily_cleaned.csv', encoding='utf-8-sig')
    df['Date']  = pd.to_datetime(df['Date'], format='%m/%d/%y')
    df['group'] = g.upper()
    df = df[(df['Date'].dt.year == 2025) & (df['Date'].dt.month == 8)]
    daily_frames.append(df[['group', 'Date', 'Abandon Rate']])

daily = pd.concat(daily_frames, ignore_index=True)
daily['day_of_week'] = daily['Date'].dt.day_name()

# --- Join shape onto daily (each daily row fans out to 48 interval rows) ---

forecast = daily.merge(shape, on=['group', 'day_of_week'], how='left')

# --- Compute interval Abandon Rate ---

forecast['interval_abd'] = (
    forecast['Abandon Rate'] * forecast['shape_abandoned_rate'] * forecast['group'].map(BIAS)
)
forecast['interval_abd'] = forecast['interval_abd'].clip(0, 1).round(6)

# --- Validate: mean interval ABD vs mean daily ABD per group ---

validation = (
    forecast
    .groupby('group')
    .agg(
        daily_abd_mean=('Abandon Rate', 'mean'),
        interval_abd_mean=('interval_abd', 'mean'),
    )
)
validation['ratio'] = (validation['interval_abd_mean'] / validation['daily_abd_mean']).round(4)
print("Validation — mean interval ABD vs mean daily ABD (ratio should be ~BIAS):")
print(validation.round(6).to_string())
print()

# --- Output ---

out = forecast[['group', 'Date', 'day_of_week', 'interval', 'interval_abd']].sort_values(
    ['group', 'Date', 'interval']
).reset_index(drop=True)

out_path = 'forecasts/abd_forecast.csv'
out.to_csv(out_path, index=False)

print(f"Wrote {len(out)} rows to {out_path}")
print(f"Expected {4 * 31 * 48} rows (4 groups × 31 days × 48 intervals)")
