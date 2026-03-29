"""
Forecast interval-level CCT for August 2025.
Method: August daily CCT (known) × intraday shape ratio (from Apr-Jun 2025).
CCT is set to 0 for any interval where forecasted CV = 0 (those intervals
are excluded from scoring entirely, so they should not contribute noise).
"""

import pandas as pd

GROUPS = ['a', 'b', 'c', 'd']

# Upward bias per group to reduce underprediction penalty (Pt).
# Groups C and D run low vs daily CCT due to more low-CCT overnight intervals,
# so they need a larger correction than A and B.
BIAS = {
    'A': 1.05,
    'B': 1.06,
    'C': 1.10,
    'D': 1.15,
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

# --- Join shape onto daily (each daily row fans out to 48 interval rows) ---

forecast = daily.merge(shape, on=['group', 'day_of_week'], how='left')

# --- Compute interval CCT ---

forecast['interval_cct'] = forecast['CCT'] * forecast['shape_cct'] * forecast['group'].map(BIAS)
forecast['interval_cct'] = forecast['interval_cct'].clip(lower=0).round(2)

# --- Zero out CCT where CV forecast is zero ---
# Load CV forecast to identify zero-volume intervals

cv = pd.read_csv('forecasts/cv_forecast.csv')
cv['Date'] = pd.to_datetime(cv['Date'])

forecast = forecast.merge(
    cv[['group', 'Date', 'interval', 'interval_cv']],
    on=['group', 'Date', 'interval'],
    how='left'
)
forecast.loc[forecast['interval_cv'] == 0, 'interval_cct'] = 0

# --- Validate: check CCT stays in a reasonable range vs daily CCT ---

validation = (
    forecast
    .groupby('group')
    .agg(
        daily_cct_mean=('CCT', 'mean'),
        interval_cct_mean=('interval_cct', 'mean'),
    )
)
validation['ratio'] = (validation['interval_cct_mean'] / validation['daily_cct_mean']).round(4)
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
