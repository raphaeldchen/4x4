import pandas as pd

GROUPS = ['a', 'b', 'c', 'd']

# direct_blend alpha=1.0: use Apr-Jun observed interval abandon rates directly.
# Avoids ratio amplification that shaped mode produces at opening/closing slots
# (e.g., a 43% Apr-Jun opening-slot rate multiplied by an August level shift
# would exceed 100% — direct_blend keeps predictions realistic).
# alpha=1.0 means fully Apr-Jun interval rates; alpha=0.0 would be flat daily rate.
ALPHA = 1.0

# --- Load shape ---

shape = pd.read_csv('cleaned_data/intraday_shape.csv')[
    ['group', 'day_of_week', 'interval', 'interval_abandoned_rate']
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
# Blend Apr-Jun observed interval rate with August daily rate.
# At alpha=1.0 this is purely the Apr-Jun interval rate.

forecast['interval_abd'] = (
    ALPHA * forecast['interval_abandoned_rate'] +
    (1 - ALPHA) * forecast['Abandon Rate']
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
print(f"direct_blend | ALPHA={ALPHA}")
print("Validation — mean interval ABD vs mean daily ABD:")
print(validation.round(6).to_string())
print()

# --- Output ---

out = forecast[['group', 'Date', 'day_of_week', 'interval', 'interval_abd']].sort_values(
    ['group', 'Date', 'interval']
).reset_index(drop=True)

out.to_csv('forecasts/abd_forecast.csv', index=False)
print(f"Wrote {len(out)} rows to forecasts/abd_forecast.csv")
print(f"Expected {4 * 31 * 48} rows (4 groups × 31 days × 48 intervals)")
