import pandas as pd
import numpy as np

GROUPS = ['a', 'b', 'c', 'd']

# Toggle between 'shaped', 'flat', and 'regression' to compare submission scores.
# 'shaped'     — multiply daily CV by Apr-Jun intraday shape ratio (original approach)
# 'flat'       — divide daily CV evenly across all 48 intervals
# 'regression' — per-cell OLS: interval_cv = a * daily_cv + b, trained on Apr-Jun intervals
#                The intercept captures base load independent of daily volume.
#                In-sample vs shaped: 5.01% EV-like vs 5.26% (~4.7% MAE improvement)
SHAPE_MODE = 'shaped'

# Upward bias per group (multiplicative, applied after prediction).
# Reduces underprediction penalty Pt. Tune after confirming base approach.
BIAS = {
    'shaped':     {'A': 1.03, 'B': 1.03, 'C': 1.03, 'D': 1.03},
    'flat':       {'A': 1.0,  'B': 1.0,  'C': 1.0,  'D': 1.0 },
    'regression': {'A': 1.03, 'B': 1.03, 'C': 1.03, 'D': 1.03},
}

# Zero out interval CV predictions below this threshold.
OVERNIGHT_ZERO_THRESHOLD = 0

# --- Load August 2025 daily CV for each group ---

daily_frames = []
for g in GROUPS:
    df = pd.read_csv(f'cleaned_data/{g}_daily_cleaned.csv', encoding='utf-8-sig')
    df['Date']  = pd.to_datetime(df['Date'], format='%m/%d/%y')
    df['group'] = g.upper()
    df = df[(df['Date'].dt.year == 2025) & (df['Date'].dt.month == 8)]
    daily_frames.append(df[['group', 'Date', 'Call Volume']])

daily = pd.concat(daily_frames, ignore_index=True)
daily['day_of_week'] = daily['Date'].dt.day_name()

# --- Load shape (used by 'shaped' mode and as column scaffold for 'regression') ---

shape = pd.read_csv('cleaned_data/intraday_shape.csv')[
    ['group', 'day_of_week', 'interval', 'shape_call_volume']
]

bias = BIAS[SHAPE_MODE]

# ============================================================
# REGRESSION MODE
# ============================================================

if SHAPE_MODE == 'regression':
    # Build training data: Apr-Jun interval CV joined with same-day daily CV
    intv_frames, train_daily_frames = [], []
    for g in GROUPS:
        intv = pd.read_csv(f'cleaned_data/{g}_interval_cleaned.csv', encoding='utf-8-sig')
        intv['Date'] = pd.to_datetime(intv['Date'], format='%m/%d/%y')
        intv['group'] = g.upper()
        intv_frames.append(intv[['group', 'Date', 'Interval', 'Call Volume']])

        tr_daily = pd.read_csv(f'cleaned_data/{g}_daily_cleaned.csv', encoding='utf-8-sig')
        tr_daily['Date'] = pd.to_datetime(tr_daily['Date'], format='%m/%d/%y')
        tr_daily['group'] = g.upper()
        tr_daily = tr_daily[
            (tr_daily['Date'].dt.year == 2025) & tr_daily['Date'].dt.month.isin([4, 5, 6])
        ]
        train_daily_frames.append(tr_daily[['group', 'Date', 'Call Volume']].rename(
            columns={'Call Volume': 'daily_cv'}
        ))

    intv_all = pd.concat(intv_frames).rename(
        columns={'Call Volume': 'interval_cv', 'Interval': 'interval'}
    )
    # Normalize interval format to zero-padded HH:MM (matches intraday_shape.csv)
    intv_all['interval'] = intv_all['interval'].str.replace(
        r'^(\d):(\d{2})$', r'0\1:\2', regex=True
    )
    train_daily = pd.concat(train_daily_frames)
    train = intv_all.merge(train_daily, on=['group', 'Date'])
    train['day_of_week'] = train['Date'].dt.day_name()

    # Fit OLS per (group, day_of_week, interval): interval_cv = a * daily_cv + b
    # Falls back to shape-ratio (through-origin) if slope is negative or n < 3.
    coeff_records = []
    for (g, dow, interval), cell in train.groupby(['group', 'day_of_week', 'interval']):
        X = cell['daily_cv'].values
        y = cell['interval_cv'].values
        shape_ratio = y.mean() / X.mean() if X.mean() > 0 else 0.0

        if len(X) >= 3:
            a, b = np.polyfit(X, y, 1)
            if a < 0:
                # Negative slope — degenerate fit, fall back to shape ratio (no intercept)
                a, b = shape_ratio, 0.0
        else:
            a, b = shape_ratio, 0.0

        coeff_records.append({'group': g, 'day_of_week': dow, 'interval': interval,
                               'slope': a, 'intercept': b})

    coeffs = pd.DataFrame(coeff_records)

    # Apply to August: fan out daily rows to intervals, then predict
    # Use shape as the interval scaffold (guarantees all 48 intervals per day/group)
    forecast = daily.merge(shape[['group', 'day_of_week', 'interval']], on=['group', 'day_of_week'], how='left')
    forecast = forecast.merge(coeffs, on=['group', 'day_of_week', 'interval'], how='left')

    forecast['interval_cv'] = (
        forecast['slope'] * forecast['Call Volume'] + forecast['intercept']
    ) * forecast['group'].map(bias)
    forecast['interval_cv'] = forecast['interval_cv'].clip(lower=0).round().astype(int)

# ============================================================
# SHAPED / FLAT MODES
# ============================================================

elif SHAPE_MODE == 'shaped':
    forecast = daily.merge(shape, on=['group', 'day_of_week'], how='left')
    forecast['interval_cv'] = (
        forecast['Call Volume'] * forecast['shape_call_volume'] * forecast['group'].map(bias)
    )
    forecast['interval_cv'] = forecast['interval_cv'].clip(lower=0).round().astype(int)

elif SHAPE_MODE == 'flat':
    forecast = daily.merge(shape[['group', 'day_of_week', 'interval']], on=['group', 'day_of_week'], how='left')
    forecast['interval_cv'] = (
        forecast['Call Volume'] / 48 * forecast['group'].map(bias)
    )
    forecast['interval_cv'] = forecast['interval_cv'].clip(lower=0).round().astype(int)

# --- Zero out low overnight predictions ---

if OVERNIGHT_ZERO_THRESHOLD > 0:
    forecast.loc[forecast['interval_cv'] < OVERNIGHT_ZERO_THRESHOLD, 'interval_cv'] = 0

# --- Validate: interval sums vs daily totals ---

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
print(f"SHAPE_MODE = '{SHAPE_MODE}'")
print("Validation — interval sum vs daily total (pct diff from daily):")
print(validation.groupby('group')['pct_diff'].describe().round(2).to_string())
print()

# --- Output ---

out = forecast[['group', 'Date', 'day_of_week', 'interval', 'interval_cv']].sort_values(
    ['group', 'Date', 'interval']
).reset_index(drop=True)

out_path = 'forecasts/cv_forecast.csv'
out.to_csv(out_path, index=False)

print(f"Wrote {len(out)} rows to {out_path}")
print(f"Expected {4 * 31 * 48} rows (4 groups × 31 days × 48 intervals)")
