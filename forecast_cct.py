"""
Forecast interval-level CCT for August 2025.
Method: August daily CCT (known) × blended shape ratio.

SHAPE_MODE options:
  'shaped' — full Apr-Jun intraday shape ratio (scored 20% EC — worse than flat)
  'flat'   — daily CCT uniformly across all intervals (scored 16.77% EC)
  'blend'  — weighted mix: α × shape + (1-α) × 1.0
             α=0 is equivalent to flat, α=1 is equivalent to shaped
             tune ALPHA between 0 and 1 to find the sweet spot

CCT is set to 0 for any interval where forecasted CV = 0 (excluded from scoring).
"""

import pandas as pd

GROUPS = ['a', 'b', 'c', 'd']

SHAPE_MODE = 'blend'

# Blend weight: how much of the Apr-Jun CCT shape to apply.
# 0.0 = fully flat (16.77% EC), 1.0 = fully shaped (20.00% EC)
# Try 0.1, 0.2, 0.3 in sequence to find if any partial shape helps.
ALPHA = 0.7

# Upward bias per group (applies in all modes).
# Bias doesn't affect EC (symmetric metric) but helps Pt.
# Keep at 1.0 until ALPHA is tuned, to isolate the blend effect.
BIAS = {
    'A': 1.0,
    'B': 1.0,
    'C': 1.0,
    'D': 1.0,
}

# --- Load shape ---

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

# --- Join shape onto daily ---

forecast = daily.merge(shape, on=['group', 'day_of_week'], how='left')

# --- Compute interval CCT ---

if SHAPE_MODE == 'shaped':
    blended = forecast['shape_cct']
elif SHAPE_MODE == 'flat':
    blended = 1.0
elif SHAPE_MODE == 'blend':
    blended = ALPHA * forecast['shape_cct'] + (1 - ALPHA) * 1.0

forecast['interval_cct'] = forecast['CCT'] * blended * forecast['group'].map(BIAS)
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
print(f"SHAPE_MODE = '{SHAPE_MODE}' | ALPHA = {ALPHA}")
print("Validation — mean interval CCT vs mean daily CCT:")
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
