"""
Forecast interval-level CCT for August 2025.

SHAPE_MODE options:
  'shaped'       — daily_cct_aug × shape_ratio  (shape_ratio = apr-jun_interval / apr-jun_daily)
                   scored 20% EC — worse than flat
  'flat'         — daily_cct_aug uniformly across all intervals (scored 16.77% EC)
  'blend'        — daily_cct_aug × (α × shape_ratio + (1-α) × 1.0)
                   α=0 = flat, α=1 = shaped. Monotonically improving up to α=0.7 (15.56% EC).
  'direct_blend' — α × interval_cct_aprjun + (1-α) × daily_cct_aug
                   Analogous to ABD direct_blend: blends observed Apr-Jun interval CCT with
                   August daily CCT directly, avoiding ratio amplification when Aug daily
                   CCT differs from Apr-Jun daily CCT.

CCT is set to 0 for any interval where forecasted CV = 0 (excluded from scoring).
"""

import pandas as pd

GROUPS = ['a', 'b', 'c', 'd']

SHAPE_MODE = 'direct_blend'

# For blend: how much of the Apr-Jun CCT shape ratio to apply.
# For direct_blend: weight on Apr-Jun observed interval CCT vs August daily CCT.
# blend α=0.7 scored 15.56% EC (best so far); try 0.9 and 1.0 next.
ALPHA = 1.0

# Upward bias per group (applies in all modes).
# Bias doesn't affect EC (symmetric metric) but helps Pt.
BIAS = {
    'A': 1.0,
    'B': 1.0,
    'C': 1.0,
    'D': 1.0,
}

# --- Load shape ---

shape = pd.read_csv('cleaned_data/intraday_shape.csv')[
    ['group', 'day_of_week', 'interval', 'shape_cct', 'interval_cct']
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

if SHAPE_MODE in ('shaped', 'flat', 'blend'):
    forecast['interval_cct'] = forecast['CCT'] * blended * forecast['group'].map(BIAS)
elif SHAPE_MODE == 'direct_blend':
    forecast['interval_cct'] = (
        ALPHA * forecast['interval_cct'] +
        (1 - ALPHA) * forecast['CCT']
    ) * forecast['group'].map(BIAS)

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
