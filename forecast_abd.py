"""
Forecast interval-level Abandoned Rate for August 2025.

SHAPE_MODE options:
  'shaped'       — daily_abd_aug × shape_ratio  (original; shape_ratio = apr-jun_interval / apr-jun_daily,
                   produces extreme values at opening/closing slots where ratio can reach 47x)
  'flat'         — daily_abd_aug uniformly across all intervals
  'blend'        — daily_abd_aug × (α × shape_ratio + (1-α) × 1.0)  [same amplification problem]
  'direct_blend' — α × interval_abd_aprjun + (1-α) × daily_abd_aug
                   Blends the observed Apr-Jun interval rate with August's daily rate directly.
                   Avoids ratio amplification: opening-slot 43% doesn't become 43×(aug_daily/apr-jun_daily).
                   α=0 → flat, α=1 → use Apr-Jun rates as-is regardless of August level.

Why 'direct_blend' is better:
  The shaped approach compounds errors: if Aug daily ABD is 3× higher than Apr-Jun daily,
  the 06:30 opening slot prediction (already 43% in Apr-Jun) becomes 130% (capped to 95%).
  direct_blend anchors on the observed Apr-Jun interval rate and adjusts proportionally
  toward August's daily level, so opening-slot predictions stay realistic.
"""

import pandas as pd

GROUPS = ['a', 'b', 'c', 'd']

SHAPE_MODE = 'direct_blend'

# For direct_blend: weight given to Apr-Jun observed interval rate vs August daily rate.
# α=0 → pure flat (August daily rate everywhere)
# α=1 → pure Apr-Jun observed rates (ignores August level shift)
# Try 0.3–0.7; higher α gives more intraday variation but risks outdated Apr-Jun patterns.
ALPHA = 1.0

# ABD is not in the Pt penalty, so bias defaults to 1.0 for all groups.
BIAS = {
    'A': 1.0,
    'B': 1.0,
    'C': 1.0,
    'D': 1.0,
}

# --- Load shape ---

shape = pd.read_csv('cleaned_data/intraday_shape.csv')[
    ['group', 'day_of_week', 'interval', 'shape_abandoned_rate', 'interval_abandoned_rate']
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

if SHAPE_MODE == 'shaped':
    forecast['interval_abd'] = forecast['Abandon Rate'] * forecast['shape_abandoned_rate']

elif SHAPE_MODE == 'flat':
    forecast['interval_abd'] = forecast['Abandon Rate']

elif SHAPE_MODE == 'blend':
    blended_ratio = ALPHA * forecast['shape_abandoned_rate'] + (1 - ALPHA) * 1.0
    forecast['interval_abd'] = forecast['Abandon Rate'] * blended_ratio

elif SHAPE_MODE == 'direct_blend':
    # Blend observed Apr-Jun interval rate with August daily rate directly.
    # For zero-shape intervals (overnight no-call slots), interval_abandoned_rate=0,
    # so the blend pulls toward 0 from the flat anchor.
    forecast['interval_abd'] = (
        ALPHA * forecast['interval_abandoned_rate'] +
        (1 - ALPHA) * forecast['Abandon Rate']
    )

forecast['interval_abd'] = (
    forecast['interval_abd'] * forecast['group'].map(BIAS)
).clip(0, 1).round(6)

# --- Validate ---

validation = (
    forecast
    .groupby('group')
    .agg(
        daily_abd_mean=('Abandon Rate', 'mean'),
        interval_abd_mean=('interval_abd', 'mean'),
        max_interval_abd=('interval_abd', 'max'),
    )
)
validation['ratio'] = (validation['interval_abd_mean'] / validation['daily_abd_mean']).round(4)
print(f"SHAPE_MODE = '{SHAPE_MODE}' | ALPHA = {ALPHA}")
print("Validation — mean interval ABD vs mean daily ABD:")
print(validation.round(6).to_string())
print()
print(f"Intervals with ABD > 10%: {(forecast['interval_abd'] > 0.1).sum()}")
print(f"Intervals with ABD > 30%: {(forecast['interval_abd'] > 0.3).sum()}")
print(f"Intervals with ABD > 50%: {(forecast['interval_abd'] > 0.5).sum()}")
print()

# --- Output ---

out = forecast[['group', 'Date', 'day_of_week', 'interval', 'interval_abd']].sort_values(
    ['group', 'Date', 'interval']
).reset_index(drop=True)

out_path = 'forecasts/abd_forecast.csv'
out.to_csv(out_path, index=False)

print(f"Wrote {len(out)} rows to {out_path}")
print(f"Expected {4 * 31 * 48} rows (4 groups × 31 days × 48 intervals)")
