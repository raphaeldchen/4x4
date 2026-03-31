import pandas as pd
import numpy as np

GROUPS = ['a', 'b', 'c', 'd']

# dont reduce C/D below A/B
BIAS = {'A': 1.044, 'B': 1.044, 'C': 1.044, 'D': 1.044}

# ~12 obs/cell from Apr-Jun so raw shape is noisy, blend with circular-smoothed version.
SHAPE_SMOOTH_ALPHA = 0.5

WEEKDAYS = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'}
WEEKEND_BIAS = 1.044

daily_frames = []
for g in GROUPS:
    df = pd.read_csv(f'cleaned_data/{g}_daily_cleaned.csv', encoding='utf-8-sig')
    df['Date']  = pd.to_datetime(df['Date'], format='%m/%d/%y')
    df['group'] = g.upper()
    df = df[(df['Date'].dt.year == 2025) & (df['Date'].dt.month == 8)]
    daily_frames.append(df[['group', 'Date', 'Call Volume']])

daily = pd.concat(daily_frames, ignore_index=True)
daily['day_of_week'] = daily['Date'].dt.day_name()

shape = pd.read_csv('cleaned_data/intraday_shape.csv')[
    ['group', 'day_of_week', 'interval', 'shape_call_volume']
].copy()

kernel = np.array([0.10, 0.20, 0.40, 0.20, 0.10])
half = len(kernel) // 2

smooth_parts = []
for (g, dow), grp in shape.groupby(['group', 'day_of_week']):
    grp = grp.sort_values('interval').copy()
    vals = grp['shape_call_volume'].values.astype(float)
    n = len(vals)
    smoothed = np.array([
        sum(kernel[k] * vals[(i + k - half) % n] for k in range(len(kernel)))
        for i in range(n)
    ])
    grp['shape_call_volume'] = (1 - SHAPE_SMOOTH_ALPHA) * vals + SHAPE_SMOOTH_ALPHA * smoothed
    smooth_parts.append(grp)
shape = pd.concat(smooth_parts).reset_index(drop=True)

forecast = daily.merge(shape, on=['group', 'day_of_week'], how='left')
forecast['_bias'] = forecast.apply(
    lambda r: BIAS[r['group']] if r['day_of_week'] in WEEKDAYS else WEEKEND_BIAS,
    axis=1
)
forecast['interval_cv'] = (
    forecast['Call Volume'] * forecast['shape_call_volume'] * forecast['_bias']
)
forecast['interval_cv'] = forecast['interval_cv'].clip(lower=0).round().astype(int)

validation = (
    forecast
    .groupby(['group', 'Date'])['interval_cv']
    .sum()
    .reset_index()
    .rename(columns={'interval_cv': 'interval_sum'})
    .merge(daily[['group', 'Date', 'Call Volume']], on=['group', 'Date'])
)
validation['pct_diff'] = (
    (validation['interval_sum'] - validation['Call Volume']) / validation['Call Volume'] * 100
).round(2)
print("Validation — interval sum vs daily total (pct diff from daily):")
print(validation.groupby('group')['pct_diff'].describe().round(2).to_string())
print()

out = forecast[['group', 'Date', 'day_of_week', 'interval', 'interval_cv']].sort_values(
    ['group', 'Date', 'interval']
).reset_index(drop=True)

out.to_csv('forecasts/cv_forecast.csv', index=False)
print(f"Wrote {len(out)} rows to forecasts/cv_forecast.csv")
print(f"Expected {4 * 31 * 48} rows (4 groups × 31 days × 48 intervals)")
