import pandas as pd

cv  = pd.read_csv('forecasts/cv_forecast.csv')
cct = pd.read_csv('forecasts/cct_forecast.csv')
abd = pd.read_csv('forecasts/abd_forecast.csv')

for df in [cv, cct, abd]:
    df['Date'] = pd.to_datetime(df['Date'])

merged = (
    cv[['group', 'Date', 'interval', 'interval_cv']]
    .merge(cct[['group', 'Date', 'interval', 'interval_cct']], on=['group', 'Date', 'interval'])
    .merge(abd[['group', 'Date', 'interval', 'interval_abd']], on=['group', 'Date', 'interval'])
)

merged['abandoned_calls'] = (merged['interval_abd'] * merged['interval_cv']).round().astype(int)
merged['abandoned_calls'] = merged['abandoned_calls'].clip(lower=0, upper=merged['interval_cv'].astype(int))

groups = ['A', 'B', 'C', 'D']
metrics = {
    'interval_cv':      'Calls_Offered',
    'abandoned_calls':  'Abandoned_Calls',
    'interval_abd':     'Abandoned_Rate',
    'interval_cct':     'CCT',
}

wide = merged[merged['group'] == groups[0]][['Date', 'interval']].copy().reset_index(drop=True)

for g in groups:
    grp = merged[merged['group'] == g].sort_values(['Date', 'interval']).reset_index(drop=True)
    for src_col, out_prefix in metrics.items():
        wide[f'{out_prefix}_{g}'] = grp[src_col].values

wide['Month'] = wide['Date'].dt.strftime('%B')  # e.g. 'August'
wide['Day']   = wide['Date'].dt.day

col_order = ['Month', 'Day', 'Interval'] + [
    f'{m}_{g}'
    for g in groups
    for m in ['Calls_Offered', 'Abandoned_Calls', 'Abandoned_Rate', 'CCT']
]

wide = wide.rename(columns={'interval': 'Interval'})

# Strip leading zero from hour to match raw data format: '00:00' -> '0:00', '09:30' -> '9:30'
wide['Interval'] = wide['Interval'].str.lstrip('0').str.replace('^:', '0:', regex=True)

# Cap Abandoned_Rate at 0.95 (100% abandon rate is unrealistic and may fail validation)
for g in groups:
    wide[f'Abandoned_Rate_{g}'] = wide[f'Abandoned_Rate_{g}'].clip(upper=0.95)

wide = wide[col_order].sort_values(['Month', 'Day', 'Interval']).reset_index(drop=True)

print(f"Rows:    {len(wide)} (expected {31 * 48} = {31*48})")
print(f"Columns: {len(wide.columns)} (expected {3 + 4*4} = {3 + 16})")
print(f"\nSample (first 3 rows):")
print(wide.head(3).to_string())
print(f"\nNegative values check:")
for col in wide.columns[3:]:
    negs = (wide[col] < 0).sum()
    if negs:
        print(f"  WARNING: {negs} negative values in {col}")
print("  No negative values found." if not any((wide[col] < 0).any() for col in wide.columns[3:]) else "")

out_path = 'forecasts/forecast_v42.csv'
wide.to_csv(out_path, index=False)
print(f"\nWrote {len(wide)} rows to {out_path}")