import pandas as pd

GROUPS = ['a', 'b', 'c', 'd']

ALPHA = 1.0

shape = pd.read_csv('cleaned_data/intraday_shape.csv')[
    ['group', 'day_of_week', 'interval', 'interval_abandoned_rate']
]

daily_frames = []
for g in GROUPS:
    df = pd.read_csv(f'cleaned_data/{g}_daily_cleaned.csv', encoding='utf-8-sig')
    df['Date']  = pd.to_datetime(df['Date'], format='%m/%d/%y')
    df['group'] = g.upper()
    df = df[(df['Date'].dt.year == 2025) & (df['Date'].dt.month == 8)]
    daily_frames.append(df[['group', 'Date', 'Abandon Rate']])
daily = pd.concat(daily_frames, ignore_index=True)
daily['day_of_week'] = daily['Date'].dt.day_name()

forecast = daily.merge(shape, on=['group', 'day_of_week'], how='left')
forecast['interval_abd'] = (
    ALPHA * forecast['interval_abandoned_rate'] +
    (1 - ALPHA) * forecast['Abandon Rate']
).clip(0, 1).round(6)

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

out = forecast[['group', 'Date', 'day_of_week', 'interval', 'interval_abd']].sort_values(
    ['group', 'Date', 'interval']
).reset_index(drop=True)
out.to_csv('forecasts/abd_forecast.csv', index=False)