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

# Dates excluded from shape (must match EXCLUDE_DATES in agg.py / intraday_shape.py)
_EXCLUDE_DATES = {
    '2025-04-18',  # Good Friday
    '2025-04-20',  # Easter Sunday
    '2025-05-11',  # Mother's Day
    '2025-05-26',  # Memorial Day
    '2025-06-15',  # Father's Day
    '2025-06-19',  # Juneteenth
}

# Upward bias per group (multiplicative, applied after prediction).
# Empirically confirmed optimal: uniform ~1.042-1.044 (EV U-shape minimum).
# Decoded weights: w1=0.450, w2=0.202, w3=0.146, w4=0.199.
# NEVER reduce C/D below A/B — C is 46% of EV. Uniform or C/D ≥ A/B only.
BIAS = {
    'shaped':     {'A': 1.044, 'B': 1.044, 'C': 1.044, 'D': 1.044},
    'flat':       {'A': 1.0,   'B': 1.0,   'C': 1.0,   'D': 1.0  },
    'regression': {'A': 1.044, 'B': 1.044, 'C': 1.044, 'D': 1.044},
}

# ── Shape smoothing ────────────────────────────────────────────────────────
# Blend raw shape with a smoothed version to reduce estimation noise.
# Rationale: with ~12 obs/cell, the shape has meaningful noise at some intervals.
# Smoothing reduces noise while preserving broad patterns. Circular (wraps midnight).
#
# SHAPE_SMOOTH_ALPHA: 0.0 = no smoothing, 1.0 = fully smoothed
# SHAPE_SMOOTH_WINDOW: 3 = [0.25, 0.5, 0.25],  5 = [0.1, 0.2, 0.4, 0.2, 0.1]
SHAPE_SMOOTH_ALPHA  = 0.5   # v31 best: alpha=0.5, window=5 (EV=34.148, rank 6)
SHAPE_SMOOTH_WINDOW = 5

# ── School-year onset correction ───────────────────────────────────────────
# August school year starts → call patterns shift toward April (school-in)
# and away from June (summer). Applied AFTER smoothing; self-normalizing.
#
# Per-portfolio blend weight toward April-only shape in school windows.
# Based on Apr/Jun RoS ratio analysis (weekday business hours 7am-9pm):
#   A: mean ratio=0.753 across ALL biz hours (strong uniform seasonal shift)
#   B: mean ratio=0.992 (no signal — keep at 0)
#   C: mean ratio=0.996 but 7:00-8:30 ratio≈0.921 (weak morning dip)
#   D: mean ratio=1.000 but 7:00-7:30 ratio≈0.918 (very weak)
# Windows: 'all_biz' applies to 07:00-20:30 weekdays; 'morning' applies to
#   07:00-08:30 weekdays only. 'afternoon' applies to 14:00-16:30 weekdays.
# 0.0 = no correction (v31 baseline), 1.0 = fully April shape at those hours.
SCHOOL_YEAR_WEIGHT = {
    'A': {'window': 'all_biz',  'blend': 0.0},  # 0→0.3→0.5→0.7 in variants
    'B': {'window': 'morning',  'blend': 0.0},  # keep 0
    'C': {'window': 'morning',  'blend': 0.0},  # 0→0.2→0.3 in variants
    'D': {'window': 'morning',  'blend': 0.0},  # 0→0.1→0.2 in variants
}
# Window definitions (inclusive, 30-min slots)
_SCHOOL_WINDOWS = {
    'all_biz':  lambda h: 7.0  <= h <= 20.5,
    'morning':  lambda h: 7.0  <= h <= 8.5,
    'afternoon':lambda h: 14.0 <= h <= 16.5,
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
    '2025-06-15',  # Father's Day
    '2025-06-19',  # Juneteenth
}

DOM_CORRECTION_WEIGHT = 0.0   # bucket DOM correction — no signal at 3 months of data
DOM_MIN_OBS = 5               # min obs per bucket cell to apply correction

# Per-group morning boost (07:00-07:30) — set to 1.0 = no boost (reverted).
MORNING_BOOST_WEEKDAY = {'A': 1.0, 'B': 1.0, 'C': 1.0, 'D': 1.0}
MORNING_BOOST_WEEKEND = {'A': 1.0, 'B': 1.0, 'C': 1.0, 'D': 1.0}
MORNING_INTERVALS = {'07:00', '07:30'}
WEEKDAYS = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'}

# Afternoon boost (15:00-16:30) — reverted, no leaderboard improvement.
AFTERNOON_BOOST_WEEKDAY = {'A': 1.0, 'B': 1.0, 'C': 1.0, 'D': 1.0}
AFTERNOON_INTERVALS = {'15:00', '15:30', '16:00', '16:30'}

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

# --- Apply shape smoothing (circular, preserves sum per group×DOW) ---

if SHAPE_SMOOTH_ALPHA > 0:
    if SHAPE_SMOOTH_WINDOW == 3:
        kernel = np.array([0.25, 0.50, 0.25])
    else:  # 5
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
    print(f'Shape smoothing: alpha={SHAPE_SMOOTH_ALPHA}, window={SHAPE_SMOOTH_WINDOW}')

# --- School-year onset correction: blend smoothed shape toward April-only shape ---
# Applied AFTER smoothing. Self-normalizing per (group, DOW). Weekdays only.
# Rationale: August school start shifts intraday patterns back toward April (school-in)
# and away from June (summer-off). Blend is applied only in configured school windows.

if any(v['blend'] != 0.0 for v in SCHOOL_YEAR_WEIGHT.values()):
    # Build April-only RoS shape (ratio-of-sums, consistent with v17 approach)
    apr_intv_frames, apr_daily_frames = [], []
    for g in GROUPS:
        intv = pd.read_csv(f'cleaned_data/{g}_interval_cleaned.csv', encoding='utf-8-sig')
        intv['Date'] = pd.to_datetime(intv['Date'], format='%m/%d/%y')
        intv = intv[intv['Date'].dt.month == 4]
        intv = intv[~intv['Date'].dt.strftime('%Y-%m-%d').isin(_EXCLUDE_DATES)]
        intv['group'] = g.upper()
        intv['day_of_week'] = intv['Date'].dt.day_name()
        intv['interval'] = intv['Interval'].str.replace(r'^(\d):(\d{2})$', r'0\1:\2', regex=True)
        apr_intv_frames.append(intv[['group', 'day_of_week', 'interval', 'Call Volume']])

        d = pd.read_csv(f'cleaned_data/{g}_daily_cleaned.csv', encoding='utf-8-sig')
        d['Date'] = pd.to_datetime(d['Date'], format='%m/%d/%y')
        d = d[(d['Date'].dt.year == 2025) & (d['Date'].dt.month == 4)]
        d = d[~d['Date'].dt.strftime('%Y-%m-%d').isin(_EXCLUDE_DATES)]
        d['group'] = g.upper()
        d['day_of_week'] = d['Date'].dt.day_name()
        apr_daily_frames.append(d[['group', 'day_of_week', 'Call Volume']])

    apr_intv_all = pd.concat(apr_intv_frames)
    apr_daily_all = pd.concat(apr_daily_frames)

    # Ratio-of-sums: Σ(interval_cv) / Σ(daily_cv) per (group, DOW, interval)
    apr_intv_sum = (
        apr_intv_all.groupby(['group', 'day_of_week', 'interval'])['Call Volume']
        .sum().rename('intv_sum').reset_index()
    )
    apr_daily_sum = (
        apr_daily_all.groupby(['group', 'day_of_week'])['Call Volume']
        .sum().rename('daily_sum').reset_index()
    )
    apr_m = apr_intv_sum.merge(apr_daily_sum, on=['group', 'day_of_week'])
    apr_m['apr_shape'] = apr_m['intv_sum'] / apr_m['daily_sum']
    apr_shape_idx = apr_m.set_index(['group', 'day_of_week', 'interval'])['apr_shape']

    # Apply blending per (group, DOW) for weekdays only
    school_parts = []
    for (g, dow), grp in shape.groupby(['group', 'day_of_week']):
        grp = grp.sort_values('interval').copy()
        cfg = SCHOOL_YEAR_WEIGHT.get(g, {'window': 'morning', 'blend': 0.0})
        blend = cfg['blend']

        if blend == 0.0 or dow not in WEEKDAYS:
            school_parts.append(grp)
            continue

        win_fn = _SCHOOL_WINDOWS[cfg['window']]
        orig_sum = grp['shape_call_volume'].sum()

        new_vals = grp['shape_call_volume'].copy()
        for idx, row in grp.iterrows():
            # Parse hour from interval string (e.g. '07:30' → 7.5)
            h, m_str = row['interval'].split(':')
            hour_frac = int(h) + int(m_str) / 60.0
            if win_fn(hour_frac):
                key = (g, dow, row['interval'])
                if key in apr_shape_idx.index:
                    apr_val = apr_shape_idx[key]
                    new_vals[idx] = (1 - blend) * row['shape_call_volume'] + blend * apr_val

        # Renormalize to preserve daily total
        new_sum = new_vals.sum()
        if new_sum > 0:
            new_vals = new_vals * (orig_sum / new_sum)
        grp['shape_call_volume'] = new_vals
        school_parts.append(grp)

    shape = pd.concat(school_parts).reset_index(drop=True)
    active = {g: v for g, v in SCHOOL_YEAR_WEIGHT.items() if v['blend'] != 0.0}
    print(f'School-year correction applied: {active}')

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
