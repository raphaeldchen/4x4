import os
import pandas as pd

INTERVAL_YEAR = 2025
ALL_INTERVALS = [f"{h}:{m}" for h in range(24) for m in ('00', '30')]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, 'raw_data')
CLEANED_DIR = os.path.join(BASE_DIR, 'cleaned_data')

def raw(filename):
    return os.path.join(RAW_DIR, filename)

def cleaned(filename):
    return os.path.join(CLEANED_DIR, filename)

# =============================================================================
# Load
# =============================================================================

a_daily = pd.read_csv(raw('a_daily_table.csv'), encoding='utf-8-sig')
b_daily = pd.read_csv(raw('b_daily_table.csv'), encoding='utf-8-sig')
c_daily = pd.read_csv(raw('c_daily_table.csv'), encoding='utf-8-sig')
d_daily = pd.read_csv(raw('d_daily_table.csv'), encoding='utf-8-sig')

a_interval = pd.read_csv(raw('a_interval_table.csv'), encoding='utf-8-sig')
b_interval = pd.read_csv(raw('b_interval_table.csv'), encoding='utf-8-sig')
c_interval = pd.read_csv(raw('c_interval_table.csv'), encoding='utf-8-sig')
d_interval = pd.read_csv(raw('d_interval_table.csv'), encoding='utf-8-sig')

daily_staffing = pd.read_csv(raw('daily_staffing.csv'), encoding='utf-8-sig')

# =============================================================================
# Date normalization
# =============================================================================

for df in [a_daily, b_daily, c_daily, d_daily]:
    df['Date'] = pd.to_datetime(
        df['Date'].str.split(' ').str[0], format='%m/%d/%y'
    ).dt.strftime('%m/%d/%y')

for df in [a_interval, b_interval, c_interval, d_interval]:
    df['Date'] = pd.to_datetime(
        df['Month'] + ' ' + df['Day'].astype(str) + f' {INTERVAL_YEAR}',
        format='%B %d %Y'
    ).dt.strftime('%m/%d/%y')

daily_staffing = daily_staffing.rename(columns={'Unnamed: 0': 'Date'})
daily_staffing['Date'] = pd.to_datetime(
    daily_staffing['Date'], format='%m/%d/%y'
).dt.strftime('%m/%d/%y')

# =============================================================================
# Type cleaning
# =============================================================================

def clean_daily(df):
    df = df.copy()
    df['Call Volume'] = pd.to_numeric(df['Call Volume'].str.replace(',', ''), errors='coerce')
    df['Service Level'] = pd.to_numeric(df['Service Level'].str.rstrip('%'), errors='coerce') / 100
    df['Abandon Rate'] = pd.to_numeric(df['Abandon Rate'].str.rstrip('%'), errors='coerce') / 100
    return df

def clean_interval(df):
    df = df.copy()
    df['CCT'] = pd.to_numeric(df['CCT'].astype(str).str.replace(',', ''), errors='coerce')
    df['Service Level'] = pd.to_numeric(df['Service Level'].str.rstrip('%'), errors='coerce') / 100
    df['Abandoned Rate'] = pd.to_numeric(df['Abandoned Rate'].str.rstrip('%'), errors='coerce') / 100
    return df

def build_datetime(df):
    date_iso = pd.to_datetime(df['Date'], format='%m/%d/%y').dt.strftime('%Y-%m-%d')
    time_padded = df['Interval'].str.split(':').apply(
        lambda x: f"{int(x[0]):02d}:{x[1]}" if isinstance(x, list) else None
    )
    df['DateTime'] = date_iso + ' ' + time_padded
    return df

# =============================================================================
# Null handling
# =============================================================================

def handle_daily_nulls(df):
    """
    1. Delete rows where all 4 metrics are null (full outage rows).
    2. Impute partial rows with same-DOW same-year median;
       fall back to all-year DOW median if fewer than 3 peers.
    """
    metrics = ['Call Volume', 'CCT', 'Service Level', 'Abandon Rate']
    df = df.copy()

    dates = pd.to_datetime(df['Date'], format='%m/%d/%y')
    df['_dow'] = dates.dt.dayofweek
    df['_year'] = dates.dt.year

    # 1. Drop all-null rows
    df = df[~df[metrics].isnull().all(axis=1)].copy()

    # 2. Impute partial rows
    for metric in metrics:
        for idx in df.index[df[metric].isnull()]:
            dow, year = df.at[idx, '_dow'], df.at[idx, '_year']
            same_year = df[(df['_dow'] == dow) & (df['_year'] == year) & df[metric].notna()][metric]
            if len(same_year) >= 3:
                df.at[idx, metric] = same_year.median()
            else:
                all_years = df[(df['_dow'] == dow) & df[metric].notna()][metric]
                df.at[idx, metric] = all_years.median() if not all_years.empty else None

    return df.drop(columns=['_dow', '_year'])


def handle_interval_nulls(df):
    """
    1. Delete rows where Interval is null and all metrics are null.
    2. Recover Interval for rows where Interval is null but some metrics are
       present, by matching blank-row count to missing time slots for that day.
    3. Where Call Volume = 0, fill SL=100%, AC=0, AR=0%, CCT=0.
    4. Derive Abandoned Rate = AC / CV where possible.
    5. Derive Abandoned Calls = round(AR * CV) where possible.
    6. Impute remaining nulls with same-interval same-DOW same-month median;
       fall back to all-month DOW median for that interval.
    """
    metrics = ['Service Level', 'Call Volume', 'Abandoned Calls', 'Abandoned Rate', 'CCT']
    df = df.copy()

    # 1. Drop rows where Interval is null and all metrics are null
    drop_mask = df['Interval'].isnull() & df[metrics].isnull().all(axis=1)
    df = df[~drop_mask].copy()

    # 2. Recover Interval where Interval is null but some metrics are present
    blank_iv = df[df['Interval'].isnull() & df[metrics].notna().any(axis=1)]
    for (month, day), group in blank_iv.groupby(['Month', 'Day']):
        present = set(df[(df['Month'] == month) & (df['Day'] == day) & df['Interval'].notna()]['Interval'])
        missing = [iv for iv in ALL_INTERVALS if iv not in present]
        if len(missing) == len(group):
            for i, idx in enumerate(group.index):
                df.at[idx, 'Interval'] = missing[i]

    # 3. CV=0: all other metrics are derivable with certainty
    cv_zero = df['Call Volume'] == 0
    df.loc[cv_zero & df['Service Level'].isnull(),   'Service Level']   = 1.0
    df.loc[cv_zero & df['Abandoned Calls'].isnull(),  'Abandoned Calls']  = 0.0
    df.loc[cv_zero & df['Abandoned Rate'].isnull(),   'Abandoned Rate']   = 0.0
    df.loc[cv_zero & df['CCT'].isnull(),              'CCT']              = 0.0

    # 4. Derive Abandoned Rate = AC / CV
    can_derive_ar = (
        df['Abandoned Rate'].isnull() &
        df['Abandoned Calls'].notna() &
        df['Call Volume'].notna() &
        (df['Call Volume'] > 0)
    )
    df.loc[can_derive_ar, 'Abandoned Rate'] = (
        df.loc[can_derive_ar, 'Abandoned Calls'] / df.loc[can_derive_ar, 'Call Volume']
    )

    # 5. Derive Abandoned Calls = round(AR * CV)
    can_derive_ac = (
        df['Abandoned Calls'].isnull() &
        df['Abandoned Rate'].notna() &
        df['Call Volume'].notna()
    )
    df.loc[can_derive_ac, 'Abandoned Calls'] = (
        df.loc[can_derive_ac, 'Abandoned Rate'] * df.loc[can_derive_ac, 'Call Volume']
    ).round()

    # 6. Impute remaining nulls: same-interval same-DOW same-month median,
    #    falling back to all-month DOW median for that interval
    df['_dow'] = pd.to_datetime(df['Date'], format='%m/%d/%y').dt.dayofweek

    for metric in metrics:
        for idx in df.index[df[metric].isnull() & df['Interval'].notna()]:
            interval = df.at[idx, 'Interval']
            month    = df.at[idx, 'Month']
            dow      = df.at[idx, '_dow']

            peers = df[
                (df['Month'] == month) &
                (df['_dow'] == dow) &
                (df['Interval'] == interval) &
                df[metric].notna()
            ][metric]

            if not peers.empty:
                df.at[idx, metric] = peers.median()
            else:
                fallback = df[
                    (df['_dow'] == dow) &
                    (df['Interval'] == interval) &
                    df[metric].notna()
                ][metric]
                df.at[idx, metric] = fallback.median() if not fallback.empty else None

    df = df.drop(columns=['_dow'])
    df = build_datetime(df)
    return df


def handle_staffing_nulls(df):
    """Impute missing agent counts with same-DOW median per client."""
    df = df.copy()
    df['_dow'] = pd.to_datetime(df['Date'], format='%m/%d/%y').dt.dayofweek

    for col in ['A', 'B', 'C', 'D']:
        for idx in df.index[df[col].isnull()]:
            dow = df.at[idx, '_dow']
            peers = df[(df['_dow'] == dow) & df[col].notna()][col]
            df.at[idx, col] = peers.median() if not peers.empty else None

    return df.drop(columns=['_dow'])

# =============================================================================
# Apply
# =============================================================================

a_daily = handle_daily_nulls(clean_daily(a_daily))
b_daily = handle_daily_nulls(clean_daily(b_daily))
c_daily = handle_daily_nulls(clean_daily(c_daily))
d_daily = handle_daily_nulls(clean_daily(d_daily))

a_interval = handle_interval_nulls(clean_interval(a_interval))
b_interval = handle_interval_nulls(clean_interval(b_interval))
c_interval = handle_interval_nulls(clean_interval(c_interval))
d_interval = handle_interval_nulls(clean_interval(d_interval))

daily_staffing = handle_staffing_nulls(daily_staffing)

# =============================================================================
# Save
# =============================================================================

a_daily.to_csv(cleaned('a_daily_cleaned.csv'), index=False, encoding='utf-8-sig')
b_daily.to_csv(cleaned('b_daily_cleaned.csv'), index=False, encoding='utf-8-sig')
c_daily.to_csv(cleaned('c_daily_cleaned.csv'), index=False, encoding='utf-8-sig')
d_daily.to_csv(cleaned('d_daily_cleaned.csv'), index=False, encoding='utf-8-sig')

a_interval.to_csv(cleaned('a_interval_cleaned.csv'), index=False, encoding='utf-8-sig')
b_interval.to_csv(cleaned('b_interval_cleaned.csv'), index=False, encoding='utf-8-sig')
c_interval.to_csv(cleaned('c_interval_cleaned.csv'), index=False, encoding='utf-8-sig')
d_interval.to_csv(cleaned('d_interval_cleaned.csv'), index=False, encoding='utf-8-sig')

daily_staffing.to_csv(cleaned('daily_staffing_cleaned.csv'), index=False, encoding='utf-8-sig')

print("Cleaned files written to", CLEANED_DIR)