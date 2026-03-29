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

# Per-month blend weights for the intraday shape (Apr, May, Jun).
# Set to None to use the prebuilt intraday_shape.csv (trimmed mean of all 3 months).
# Otherwise, specify weights that sum to 1.0 — June gets the most weight since it's
# closest to August seasonally.
# Examples:
#   (0.2, 0.3, 0.5)  — user-proposed blend
#   (0.0, 0.0, 1.0)  — June only
#   None             — use prebuilt shape (trimmed mean Apr-Jun, current baseline)
SHAPE_MONTH_WEIGHTS = None

# Dates excluded from shape (same as agg.py / intraday_shape.py)
_EXCLUDE_DATES = {'2025-04-18', '2025-04-20', '2025-05-11', '2025-05-26'}

# Upward bias per group (multiplicative, applied after prediction).
# Reduces underprediction penalty Pt. Tune after confirming base approach.
BIAS = {
    'shaped':     {'A': 1.03, 'B': 1.03, 'C': 1.03, 'D': 1.03},
    'flat':       {'A': 1.0,  'B': 1.0,  'C': 1.0,  'D': 1.0 },
    'regression': {'A': 1.03, 'B': 1.03, 'C': 1.03, 'D': 1.03},
}

# Zero out interval CV predictions below this threshold.
OVERNIGHT_ZERO_THRESHOLD = 0

# Per-group, per-DOW-type morning boost for 07:00-07:30.
# Jun/Apr ratios (Easter-week excluded, weekday vs weekend split):
#   A  weekday: 1.011 → 1.01   A  weekend: 0.731 → 0.85 (strong decline → reduce)
#   B  weekday: 1.075 → 1.08   B  weekend: 1.118 → 1.12
#   C  weekday: 1.102 → 1.10   C  weekend: 1.169 → 1.17
#   D  weekday: 1.057 → 1.06   D  weekend: 1.275 → 1.20 (capped; noisy small sample)
# Previous submission used per-group uniform boost → worse. Key miss: A weekend
# morning is declining (0.73) so uniform 1.0 was still too high for weekends.
MORNING_BOOST_WEEKDAY = {'A': 1.0, 'B': 1.0, 'C': 1.0, 'D': 1.0}
MORNING_BOOST_WEEKEND = {'A': 1.0, 'B': 1.0, 'C': 1.0, 'D': 1.0}
MORNING_INTERVALS = {'07:00', '07:30'}
WEEKDAYS = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'}

# Afternoon boost reverted — no leaderboard improvement observed.
AFTERNOON_BOOST_WEEKDAY = {'A': 1.0, 'B': 1.0, 'C': 1.0, 'D': 1.0}
AFTERNOON_INTERVALS = {'15:00', '15:30', '16:00', '16:30'}

# DOM intramonth correction blend weight (only applies in 'shaped' mode).
# 0.0 = pure DOW shape (current baseline).
# DOM_correction is a multiplicative factor on the DOW shape, isolating
# within-month billing-cycle / payday patterns after removing DOW effects.
# Because DOM observations are noisy (only ~3 per cell), start with low weights.
# Day-31 DOM correction is based on 1 observation (May 31 only) — unreliable;
# it is automatically excluded (falls back to DOW-only) via the n_obs < 2 filter.
DOM_CORRECTION_WEIGHT = 0.0   # tune via leaderboard; try 0.1, 0.2, 0.3
# Bucket boundaries: early = days 1-10, mid = 11-20, late = 21-31 (~30 obs/cell).
# DOM_MIN_OBS kept for safety but all bucket cells should comfortably exceed it.
DOM_MIN_OBS = 5               # require at least this many obs; else no DOM correction

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
].copy()

# --- Load DOM correction factors (for 'shaped' mode only) ---

if DOM_CORRECTION_WEIGHT > 0 and SHAPE_MODE == 'shaped':
    _dom_raw = pd.read_csv('cleaned_data/intraday_shape_dom_bucket.csv')
    _dom_raw = _dom_raw[_dom_raw['n_obs'] >= DOM_MIN_OBS][
        ['group', 'dom_bucket', 'interval', 'dom_correction']
    ]
    shape_dom = _dom_raw.copy()
    print(f'DOM bucket correction loaded: {len(shape_dom)} cells with n_obs >= {DOM_MIN_OBS}')
    print(f'DOM correction weight: {DOM_CORRECTION_WEIGHT}')
    print(shape_dom.groupby('dom_bucket')['dom_correction'].describe().round(4).to_string())
else:
    shape_dom = None

# --- Per-month blend: replace shape_call_volume with weighted blend of monthly shapes ---

if SHAPE_MONTH_WEIGHTS is not None:
    w_apr, w_may, w_jun = SHAPE_MONTH_WEIGHTS
    assert abs(w_apr + w_may + w_jun - 1.0) < 1e-6, "SHAPE_MONTH_WEIGHTS must sum to 1.0"

    month_shapes = {}
    for month, weight in zip([4, 5, 6], [w_apr, w_may, w_jun]):
        if weight == 0.0:
            continue
        intv_frames, daily_frames_shape = [], []
        for g in GROUPS:
            intv = pd.read_csv(f'cleaned_data/{g}_interval_cleaned.csv', encoding='utf-8-sig')
            intv['Date'] = pd.to_datetime(intv['Date'], format='%m/%d/%y')
            intv = intv[intv['Date'].dt.month == month]
            intv = intv[~intv['Date'].dt.strftime('%Y-%m-%d').isin(_EXCLUDE_DATES)]
            intv['group'] = g.upper()
            intv['day_of_week'] = intv['Date'].dt.day_name()
            intv['interval'] = intv['Interval'].str.replace(r'^(\d):(\d{2})$', r'0\1:\2', regex=True)
            intv_frames.append(intv[['group', 'day_of_week', 'interval', 'Call Volume']])

            d = pd.read_csv(f'cleaned_data/{g}_daily_cleaned.csv', encoding='utf-8-sig')
            d['Date'] = pd.to_datetime(d['Date'], format='%m/%d/%y')
            d = d[(d['Date'].dt.year == 2025) & (d['Date'].dt.month == month)]
            d = d[~d['Date'].dt.strftime('%Y-%m-%d').isin(_EXCLUDE_DATES)]
            d['group'] = g.upper()
            d['day_of_week'] = d['Date'].dt.day_name()
            daily_frames_shape.append(d[['group', 'day_of_week', 'Call Volume']])

        intv_all = pd.concat(intv_frames)
        daily_all = pd.concat(daily_frames_shape)

        intv_mean = (
            intv_all.groupby(['group', 'day_of_week', 'interval'])['Call Volume']
            .mean().rename('interval_mean').reset_index()
        )
        daily_mean = (
            daily_all.groupby(['group', 'day_of_week'])['Call Volume']
            .mean().rename('daily_mean').reset_index()
        )
        m = intv_mean.merge(daily_mean, on=['group', 'day_of_week'])
        m['shape'] = m['interval_mean'] / m['daily_mean']
        month_shapes[month] = m.set_index(['group', 'day_of_week', 'interval'])['shape']

    # Build blended shape
    blended = None
    for month, weight in zip([4, 5, 6], [w_apr, w_may, w_jun]):
        if weight == 0.0:
            continue
        contrib = month_shapes[month] * weight
        blended = contrib if blended is None else blended.add(contrib, fill_value=0)
    blended = blended.rename('shape_call_volume').reset_index()

    shape = shape[['group', 'day_of_week', 'interval']].merge(
        blended, on=['group', 'day_of_week', 'interval'], how='left'
    )
    # Fall back to prebuilt shape for any missing cells
    prebuilt = pd.read_csv('cleaned_data/intraday_shape.csv')[
        ['group', 'day_of_week', 'interval', 'shape_call_volume']
    ].rename(columns={'shape_call_volume': 'shape_fallback'})
    shape = shape.merge(prebuilt, on=['group', 'day_of_week', 'interval'], how='left')
    shape['shape_call_volume'] = shape['shape_call_volume'].fillna(shape['shape_fallback'])
    shape = shape.drop(columns=['shape_fallback'])

    print(f'Per-month blend applied: Apr={w_apr}, May={w_may}, Jun={w_jun}')

# --- Morning boost: weekday vs weekend differentiated, renormalize to preserve daily total ---
all_boost_vals = list(MORNING_BOOST_WEEKDAY.values()) + list(MORNING_BOOST_WEEKEND.values())
if any(v != 1.0 for v in all_boost_vals):
    orig_sums = (
        shape.groupby(['group', 'day_of_week'])['shape_call_volume']
        .sum().rename('orig_sum').reset_index()
    )
    for grp in ['A', 'B', 'C', 'D']:
        wd_factor = MORNING_BOOST_WEEKDAY[grp]
        we_factor = MORNING_BOOST_WEEKEND[grp]
        wd_mask = (shape['group'].eq(grp) & shape['interval'].isin(MORNING_INTERVALS)
                   & shape['day_of_week'].isin(WEEKDAYS))
        we_mask = (shape['group'].eq(grp) & shape['interval'].isin(MORNING_INTERVALS)
                   & ~shape['day_of_week'].isin(WEEKDAYS))
        if wd_factor != 1.0:
            shape.loc[wd_mask, 'shape_call_volume'] *= wd_factor
        if we_factor != 1.0:
            shape.loc[we_mask, 'shape_call_volume'] *= we_factor
    new_sums = (
        shape.groupby(['group', 'day_of_week'])['shape_call_volume']
        .sum().rename('new_sum').reset_index()
    )
    shape = shape.merge(orig_sums, on=['group', 'day_of_week'])
    shape = shape.merge(new_sums,  on=['group', 'day_of_week'])
    shape['shape_call_volume'] = shape['shape_call_volume'] * shape['orig_sum'] / shape['new_sum']
    shape = shape.drop(columns=['orig_sum', 'new_sum'])
    print(f'Morning boost applied (weekday/weekend) for {sorted(MORNING_INTERVALS)}')
    print(f'  Weekday: {MORNING_BOOST_WEEKDAY}')
    print(f'  Weekend: {MORNING_BOOST_WEEKEND}')

# --- Afternoon boost: back-to-school secondary peak 15:00-16:30, weekdays only ---
if any(v != 1.0 for v in AFTERNOON_BOOST_WEEKDAY.values()):
    orig_sums = (
        shape.groupby(['group', 'day_of_week'])['shape_call_volume']
        .sum().rename('orig_sum').reset_index()
    )
    for grp in ['A', 'B', 'C', 'D']:
        factor = AFTERNOON_BOOST_WEEKDAY[grp]
        if factor != 1.0:
            mask = (shape['group'].eq(grp) & shape['interval'].isin(AFTERNOON_INTERVALS)
                    & shape['day_of_week'].isin(WEEKDAYS))
            shape.loc[mask, 'shape_call_volume'] *= factor
    new_sums = (
        shape.groupby(['group', 'day_of_week'])['shape_call_volume']
        .sum().rename('new_sum').reset_index()
    )
    shape = shape.merge(orig_sums, on=['group', 'day_of_week'])
    shape = shape.merge(new_sums,  on=['group', 'day_of_week'])
    shape['shape_call_volume'] = shape['shape_call_volume'] * shape['orig_sum'] / shape['new_sum']
    shape = shape.drop(columns=['orig_sum', 'new_sum'])
    print(f'Afternoon boost applied (weekday) for {sorted(AFTERNOON_INTERVALS)}: {AFTERNOON_BOOST_WEEKDAY}')

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

    if shape_dom is not None:
        def _dom_bucket(d):
            if d <= 10: return 'early'
            if d <= 20: return 'mid'
            return 'late'
        forecast['dom_bucket'] = forecast['Date'].dt.day.apply(_dom_bucket)
        forecast = forecast.merge(
            shape_dom, on=['group', 'dom_bucket', 'interval'], how='left'
        )
        # DOM correction is multiplicative on top of DOW shape.
        # Missing DOM correction (n_obs < DOM_MIN_OBS or unmapped) → correction = 1.0 (no change).
        forecast['dom_correction'] = forecast['dom_correction'].fillna(1.0)
        forecast['effective_shape'] = (
            forecast['shape_call_volume']
            * (1 + DOM_CORRECTION_WEIGHT * (forecast['dom_correction'] - 1))
        )
    else:
        forecast['effective_shape'] = forecast['shape_call_volume']

    forecast['interval_cv'] = (
        forecast['Call Volume'] * forecast['effective_shape'] * forecast['group'].map(bias)
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
