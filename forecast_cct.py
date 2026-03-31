import pandas as pd

GROUPS = ['a', 'b', 'c', 'd']

# high-CV slots: blend Apr-Jun shape with August daily (90/10)
# low-CV slots: flat daily CCT
ALPHA = 0.9
CV_THRESHOLD = 15

shape = pd.read_csv('cleaned_data/intraday_shape.csv')[
    ['group', 'day_of_week', 'interval', 'interval_cct', 'daily_cct']
]

daily_frames = []
for g in GROUPS:
    df = pd.read_csv(f'cleaned_data/{g}_daily_cleaned.csv', encoding='utf-8-sig')
    df['Date']  = pd.to_datetime(df['Date'], format='%m/%d/%y')
    df['group'] = g.upper()
    df = df[(df['Date'].dt.year == 2025) & (df['Date'].dt.month == 8)]
    daily_frames.append(df[['group', 'Date', 'CCT']])
daily = pd.concat(daily_frames, ignore_index=True)
daily['day_of_week'] = daily['Date'].dt.day_name()

forecast = daily.merge(shape, on=['group', 'day_of_week'], how='left')
cv = pd.read_csv('forecasts/cv_forecast.csv')
cv['Date'] = pd.to_datetime(cv['Date'])
forecast = forecast.merge(
    cv[['group', 'Date', 'interval', 'interval_cv']],
    on=['group', 'Date', 'interval'],
    how='left'
)

high_cv = forecast['interval_cv'] >= CV_THRESHOLD
forecast['interval_cct_pred'] = forecast['CCT'].copy()  # defaults flat daily CCT
forecast.loc[high_cv, 'interval_cct_pred'] = (
    ALPHA * forecast.loc[high_cv, 'interval_cct'] +
    (1 - ALPHA) * forecast.loc[high_cv, 'CCT']
)

forecast['interval_cct_pred'] = forecast['interval_cct_pred'].clip(lower=0).round(2)
forecast.loc[forecast['interval_cv'] == 0, 'interval_cct_pred'] = 0
n_gated = (forecast['interval_cv'] < CV_THRESHOLD).sum()
validation = (
    forecast
    .groupby('group')
    .agg(
        aug_daily_mean=('CCT', 'mean'),
        interval_cct_pred_mean=('interval_cct_pred', 'mean'),
    )
)
validation['ratio'] = (validation['interval_cct_pred_mean'] / validation['aug_daily_mean']).round(4)
out = forecast[['group', 'Date', 'day_of_week', 'interval', 'interval_cct_pred']].rename(
    columns={'interval_cct_pred': 'interval_cct'}
).sort_values(['group', 'Date', 'interval']).reset_index(drop=True)
out.to_csv('forecasts/cct_forecast.csv', index=False)
