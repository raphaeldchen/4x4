"""
Forecast interval-level CCT for August 2025.
Method: August daily CCT (known) × intraday shape ratio (from Apr-Jun 2025).
CCT is set to 0 for any interval where forecasted CV = 0 (those intervals
are excluded from scoring entirely, so they should not contribute noise).

SHAPE_MODE options:
  'shaped' — multiply daily CCT by the Apr-Jun intraday shape ratio (original approach)
  'flat'   — use daily CCT uniformly across all intervals (no shape adjustment)
"""

import pandas as pd

GROUPS = ['a', 'b', 'c', 'd']

# Toggle between 'shaped' and 'flat' to compare submission scores
SHAPE_MODE = 'flat'

# Upward bias per group to reduce underprediction penalty (Pt).
# Only applies in 'shaped' mode — in 'flat' mode set all to 1.0 first to isolate the shape effect.
BIAS = {
    'shaped': {'A': 1.05, 'B': 1.06, 'C': 1.10, 'D': 1.15},
    'flat':   {'A': 1.0,  'B': 1.0,  'C': 1.0,  'D': 1.0 },
}

# --- Load shape (only used in 'shaped' mode) ---

shape = pd.read_csv('cleaned_data/intraday_shape.csv')[
    ['group', 'day_of_week', 'interval', 'shape_cct']
]

# --- Load August 2025 daily CCT for each group ---

daily_frames = []
for g in GROUPS:
    df = pd.read_csv(f'cleaned_data/{g}_daily_cleaned.csv', encoding='utf-8-sig')
    df['Date']  = pd.to_datetime(df['Date'], format='%m/%d/%y')
    df['group'] = g.upper()
    df = df[(df['Date'].dt.year == 2025) & (df['Date'].dt.month == 8)]
    daily_frames.append(df[['group', 'Date', 'CCT']])

daily = pd.concat(daily_frames, ignore_index=True)
daily['day_of_week'] = daily['Date'].dt.day_name()

# --- Join shape onto daily (each daily row fans out to 48 interval rows) ---

forecast = daily.merge(shape, on=['group', 'day_of_week'], how='left')

# --- Compute interval CCT ---

bias = BIAS[SHAPE_MODE]

if SHAPE_MODE == 'shaped':
    forecast['interval_cct'] = forecast['CCT'] * forecast['shape_cct'] * forecast['group'].map(bias)
elif SHAPE_MODE == 'flat':
    forecast['interval_cct'] = forecast['CCT'] * forecast['group'].map(bias)

forecast['interval_cct'] = forecast['interval_cct'].clip(lower=0).round(2)

# --- Zero out CCT where CV forecast is zero ---

cv = pd.read_csv('forecasts/cv_forecast.csv')
cv['Date'] = pd.to_datetime(cv['Date'])

forecast = forecast.merge(
    cv[['group', 'Date', 'interval', 'interval_cv']],
    on=['group', 'Date', 'interval'],
    how='left'
)
forecast.loc[forecast['interval_cv'] == 0, 'interval_cct'] = 0

# --- Validate ---

validation = (
    forecast
    .groupby('group')
    .agg(
        daily_cct_mean=('CCT', 'mean'),
        interval_cct_mean=('interval_cct', 'mean'),
    )
)
validation['ratio'] = (validation['interval_cct_mean'] / validation['daily_cct_mean']).round(4)
print(f"SHAPE_MODE = '{SHAPE_MODE}'")
print("Validation — mean interval CCT vs mean daily CCT (ratio should be ~BIAS):")
print(validation.round(2).to_string())
print()

# --- Output ---

out = forecast[['group', 'Date', 'day_of_week', 'interval', 'interval_cct']].sort_values(
    ['group', 'Date', 'interval']
).reset_index(drop=True)

out_path = 'forecasts/cct_forecast.csv'
out.to_csv(out_path, index=False)

print(f"Wrote {len(out)} rows to {out_path}")
print(f"Expected {4 * 31 * 48} rows (4 groups × 31 days × 48 intervals)")
