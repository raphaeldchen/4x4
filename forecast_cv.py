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

# ── Seasonal / DOW adjustment (Fix 1 + Fix 3) ──────────────────────────────
# Apply a per-(group, DOW) scaling factor that corrects for the mismatch
# between the Apr-Jun training shape and August intraday behaviour.
#
# adj[g, dow] = Aug2024_meanCV(g, dow) / AprJun2025_meanCV(g, dow)
#
# This captures:
#   Fix 1 — month-of-year level: how August volume compares to Q2 volume
#   Fix 3 — DOW-specific intensity: how each DOW's relative weight shifts in Aug
#
# Both adjustments use the same formula; they are applied once as a combined
# multiplier so we do not double-count.
APPLY_SEASONAL_ADJ = False

# Holidays to exclude when computing AprJun2025 daily means (matches agg.py)
SEASONAL_ADJ_EXCLUDE = {
    '2025-04-18',  # Good Friday
    '2025-04-20',  # Easter Sunday
    '2025-05-11',  # Mother's Day
    '2025-05-26',  # Memorial Day
}

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

# ── Compute seasonal adjustment factors ────────────────────────────────────
if APPLY_SEASONAL_ADJ:
    adj_frames = []
    for g in GROUPS:
        df_all = pd.read_csv(f'cleaned_data/{g}_daily_cleaned.csv', encoding='utf-8-sig')
        df_all['Date'] = pd.to_datetime(df_all['Date'], format='%m/%d/%y')
        df_all['day_of_week'] = df_all['Date'].dt.day_name()

        # Aug 2024 — no major US holidays in August
        aug2024 = df_all[(df_all['Date'].dt.year == 2024) & (df_all['Date'].dt.month == 8)]
        aug2024_mean = aug2024.groupby('day_of_week')['Call Volume'].mean().rename('aug2024_cv')

        # Apr-Jun 2025 — exclude same holidays used when building the shape
        aprjun = df_all[
            (df_all['Date'].dt.year == 2025) &
            df_all['Date'].dt.month.isin([4, 5, 6])
        ]
        aprjun = aprjun[~aprjun['Date'].dt.strftime('%Y-%m-%d').isin(SEASONAL_ADJ_EXCLUDE)]
        aprjun_mean = aprjun.groupby('day_of_week')['Call Volume'].mean().rename('aprjun_cv')

        adj = pd.concat([aug2024_mean, aprjun_mean], axis=1).reset_index()
        adj.columns = ['day_of_week', 'aug2024_cv', 'aprjun_cv']
        adj['seasonal_adj'] = adj['aug2024_cv'] / adj['aprjun_cv']
        adj['group'] = g.upper()
        adj_frames.append(adj[['group', 'day_of_week', 'seasonal_adj']])

    seasonal_adj = pd.concat(adj_frames, ignore_index=True)

    # Normalize so the weighted-average adj across August 2025 DOWs = 1.0 per group.
    # This redistributes volume across DOWs (some heavier, some lighter in Aug vs Q2)
    # without inflating the total — prevents compounding with BIAS.
    aug_dow_counts = (
        daily[['group', 'day_of_week']]
        .value_counts()
        .reset_index(name='count')
    )
    seasonal_adj = seasonal_adj.merge(aug_dow_counts, on=['group', 'day_of_week'], how='left')
    seasonal_adj['count'] = seasonal_adj['count'].fillna(1)
    seasonal_adj['_weighted'] = seasonal_adj['seasonal_adj'] * seasonal_adj['count']
    grp_avg = (
        seasonal_adj
        .groupby('group')[['_weighted', 'count']]
        .sum()
        .assign(avg_adj=lambda x: x['_weighted'] / x['count'])
        [['avg_adj']]
        .reset_index()
    )
    seasonal_adj = seasonal_adj.drop(columns=['_weighted'])
    seasonal_adj = seasonal_adj.merge(grp_avg, on='group')
    seasonal_adj['seasonal_adj'] = seasonal_adj['seasonal_adj'] / seasonal_adj['avg_adj']
    seasonal_adj = seasonal_adj[['group', 'day_of_week', 'seasonal_adj']]

    print("Raw seasonal adjustment factors (Aug2024 / AprJun2025):")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Rebuild for display: recompute raw before normalization
    raw_display = pd.concat(adj_frames, ignore_index=True)
    pivot_raw = raw_display.pivot_table(values='seasonal_adj', index='group', columns='day_of_week')
    pivot_raw = pivot_raw[[c for c in day_order if c in pivot_raw.columns]]
    print(pivot_raw.round(3).to_string())

    print("\nNormalized seasonal adjustment factors (volume-neutral DOW redistribution):")
    pivot_norm = seasonal_adj.pivot_table(values='seasonal_adj', index='group', columns='day_of_week')
    pivot_norm = pivot_norm[[c for c in day_order if c in pivot_norm.columns]]
    print(pivot_norm.round(3).to_string())
    print()
else:
    seasonal_adj = None

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
    if APPLY_SEASONAL_ADJ and seasonal_adj is not None:
        forecast = forecast.merge(seasonal_adj, on=['group', 'day_of_week'], how='left')
        forecast['interval_cv'] = (
            forecast['Call Volume'] * forecast['shape_call_volume'] *
            forecast['seasonal_adj'] * forecast['group'].map(bias)
        )
    else:
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
