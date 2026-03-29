"""
Forecast interval-level CCT for August 2025.

SHAPE_MODE options:
  'shaped'       — daily_cct_aug × shape_ratio  (scored 20% EC — worse than flat)
  'flat'         — daily_cct_aug uniformly (16.77% EC)
  'blend'        — daily_cct_aug × (α × shape_ratio + (1-α) × 1.0)  (best: α=0.7, 15.56%)
  'direct_blend' — α × interval_cct_aprjun + (1-α) × daily_cct_aug
                   α=0.9 → 14.40% EC (best overall)
  'additive'     — interval_cct_aprjun + α × (daily_cct_aug_date - daily_cct_aprjun_dow)
                   Did NOT improve — additive level shift hurts noisy overnight slots.
  'cv_gated'     — hybrid: direct_blend (ALPHA) when interval CV >= CV_THRESHOLD,
                   flat daily CCT when CV < CV_THRESHOLD.
                   Rationale: overnight CCT cv=0.87 (pure noise); business-hours CCT cv=0.10
                   (stable signal). Apr-Jun interval means for low-CV overnight slots are
                   unreliable — flat daily is a better estimate for those.

CCT is set to 0 for intervals where forecasted CV = 0 (excluded from scoring).
"""

import pandas as pd

GROUPS = ['a', 'b', 'c', 'd']

SHAPE_MODE = 'cv_gated'

# For direct_blend and cv_gated: weight on Apr-Jun interval CCT vs August daily CCT.
# Best so far: 0.9 (14.40% EC). Do not increase above 1.0.
ALPHA = 0.9

# cv_gated only: intervals with CV_pred < this threshold use flat daily CCT.
# Hours 2-6 have mean CV 3-14 — high noise zone. Threshold of 10 captures most of that.
# Try 10 first, then 15 or 5 if no improvement.
CV_THRESHOLD = 15

# Upward bias per group — helps Pt but not EC (symmetric).
BIAS = {
    'A': 1.0,
    'B': 1.0,
    'C': 1.0,
    'D': 1.0,
}

# --- Load shape ---

shape = pd.read_csv('cleaned_data/intraday_shape.csv')[
    ['group', 'day_of_week', 'interval', 'shape_cct', 'interval_cct', 'daily_cct']
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

# --- Load CV forecast for cv_gated mode ---

cv = pd.read_csv('forecasts/cv_forecast.csv')
cv['Date'] = pd.to_datetime(cv['Date'])
forecast = forecast.merge(
    cv[['group', 'Date', 'interval', 'interval_cv']],
    on=['group', 'Date', 'interval'],
    how='left'
)

# --- Compute interval CCT ---

if SHAPE_MODE == 'shaped':
    forecast['interval_cct_pred'] = forecast['CCT'] * forecast['shape_cct']

elif SHAPE_MODE == 'flat':
    forecast['interval_cct_pred'] = forecast['CCT']

elif SHAPE_MODE == 'blend':
    blended = ALPHA * forecast['shape_cct'] + (1 - ALPHA) * 1.0
    forecast['interval_cct_pred'] = forecast['CCT'] * blended

elif SHAPE_MODE == 'direct_blend':
    forecast['interval_cct_pred'] = (
        ALPHA * forecast['interval_cct'] + (1 - ALPHA) * forecast['CCT']
    )

elif SHAPE_MODE == 'additive':
    level_correction = forecast['CCT'] - forecast['daily_cct']
    forecast['interval_cct_pred'] = forecast['interval_cct'] + ALPHA * level_correction

elif SHAPE_MODE == 'cv_gated':
    # High-CV intervals: direct_blend (reliable Apr-Jun shape signal)
    # Low-CV overnight intervals: flat daily CCT (Apr-Jun mean is noise-dominated)
    high_cv = forecast['interval_cv'] >= CV_THRESHOLD
    forecast['interval_cct_pred'] = forecast['CCT'].copy()  # default: flat
    forecast.loc[high_cv, 'interval_cct_pred'] = (
        ALPHA * forecast.loc[high_cv, 'interval_cct'] +
        (1 - ALPHA) * forecast.loc[high_cv, 'CCT']
    )

forecast['interval_cct_pred'] = (
    forecast['interval_cct_pred'] * forecast['group'].map(BIAS)
).clip(lower=0).round(2)

# --- Zero out CCT where CV = 0 ---

forecast.loc[forecast['interval_cv'] == 0, 'interval_cct_pred'] = 0

# --- Validate ---

n_gated = (forecast['interval_cv'] < CV_THRESHOLD).sum() if SHAPE_MODE == 'cv_gated' else 0
validation = (
    forecast
    .groupby('group')
    .agg(
        aug_daily_mean=('CCT', 'mean'),
        interval_cct_pred_mean=('interval_cct_pred', 'mean'),
    )
)
validation['ratio'] = (validation['interval_cct_pred_mean'] / validation['aug_daily_mean']).round(4)
print(f"SHAPE_MODE = '{SHAPE_MODE}' | ALPHA = {ALPHA}" +
      (f" | CV_THRESHOLD = {CV_THRESHOLD} ({n_gated} intervals use flat)" if SHAPE_MODE == 'cv_gated' else ""))
print("Validation — mean interval CCT vs Aug daily CCT:")
print(validation.round(2).to_string())
print()

# --- Output ---

out = forecast[['group', 'Date', 'day_of_week', 'interval', 'interval_cct_pred']].rename(
    columns={'interval_cct_pred': 'interval_cct'}
).sort_values(['group', 'Date', 'interval']).reset_index(drop=True)

out_path = 'forecasts/cct_forecast.csv'
out.to_csv(out_path, index=False)

print(f"Wrote {len(out)} rows to {out_path}")
print(f"Expected {4 * 31 * 48} rows (4 groups × 31 days × 48 intervals)")
