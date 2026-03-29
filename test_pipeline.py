"""
Test suite for the data cleaning / feature engineering pipeline.
Covers cleaning.py, agg.py, and intraday_shape.py functions.

Run with:  python -m pytest test_pipeline.py -v
           python test_pipeline.py   (also works via unittest)
"""

import math
import unittest
import pandas as pd
import numpy as np
import io
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR = os.path.join(BASE_DIR, 'cleaned_data')

# ---------------------------------------------------------------------------
# Import the pure functions from cleaning.py without executing its top-level
# side effects (file reads / writes).  We monkey-patch pd.read_csv to a no-op
# and redirect the save block so only the function definitions are imported.
# ---------------------------------------------------------------------------

import importlib, types, unittest.mock as mock

def _import_cleaning_functions():
    """
    Import pure functions from cleaning.py without triggering its module-level
    file I/O.  We stop execution right after the function definitions by raising
    a sentinel exception from the first write operation.
    """
    import importlib.util, unittest.mock as mock

    # Each read_csv call in cleaning.py expects a different schema.
    # Supply minimally-valid DataFrames in call order:
    #   calls 1-4  → daily files (a, b, c, d)
    #   calls 5-8  → interval files (a, b, c, d)
    #   call 9     → staffing file
    daily_df = pd.DataFrame({
        'Date': ['01/06/25'],
        'Call Volume': ['100'], 'CCT': ['300'],
        'Service Level': ['95.00%'], 'Abandon Rate': ['1.00%'],
    })
    # Call Volume and Abandoned Calls must be numeric: clean_interval() does NOT
    # convert them (relies on pandas auto-parsing from CSV).  In real use this is
    # fine because pd.read_csv infers numeric types; here we must set them explicitly.
    interval_df = pd.DataFrame({
        'Month': ['April'], 'Day': ['1'], 'Interval': ['0:00'],
        'Service Level': ['100.00%'], 'Call Volume': [5.0],
        'Abandoned Calls': [0.0], 'Abandoned Rate': ['0.00%'], 'CCT': ['200'],
    })
    staffing_df = pd.DataFrame({
        'Unnamed: 0': ['01/06/25'],
        'A': [10.0], 'B': [10.0], 'C': [10.0], 'D': [10.0],
    })

    read_csv_returns = (
        [daily_df] * 4 + [interval_df] * 4 + [staffing_df]
    )
    call_count = [0]

    def mock_read_csv(*args, **kwargs):
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(read_csv_returns):
            return read_csv_returns[idx].copy()
        return daily_df.copy()

    with mock.patch('pandas.read_csv', side_effect=mock_read_csv), \
         mock.patch.object(pd.DataFrame, 'to_csv', return_value=None):
        spec = importlib.util.spec_from_file_location(
            'cleaning_funcs',
            os.path.join(BASE_DIR, 'cleaning.py')
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


cleaning = _import_cleaning_functions()


# ---------------------------------------------------------------------------
# Helper: build a minimal daily DataFrame
# ---------------------------------------------------------------------------

def make_daily_df(rows):
    """rows = list of dicts with keys: Date, Call_Volume, CCT, Service_Level, Abandon_Rate"""
    data = {
        'Date':         [r['Date'] for r in rows],
        'Call Volume':  [r.get('Call Volume',  '100') for r in rows],
        'CCT':          [r.get('CCT',          '300') for r in rows],
        'Service Level':[r.get('Service Level','95.00%') for r in rows],
        'Abandon Rate': [r.get('Abandon Rate', '1.00%') for r in rows],
    }
    return pd.DataFrame(data)


def make_interval_df(rows):
    """rows = list of dicts with interval-level keys."""
    defaults = dict(
        Month='April', Day='1', Interval='0:00',
        **{'Service Level': '100.00%', 'Call Volume': '5',
           'Abandoned Calls': '0', 'Abandoned Rate': '0.00%',
           'CCT': '200', 'Date': '04/01/25'}
    )
    merged = [{**defaults, **r} for r in rows]
    return pd.DataFrame(merged)


# ===========================================================================
# 1. clean_daily
# ===========================================================================

class TestCleanDaily(unittest.TestCase):

    def _clean(self, rows):
        return cleaning.clean_daily(make_daily_df(rows))

    def test_call_volume_comma_removal(self):
        df = self._clean([{'Date': '01/01/24', 'Call Volume': '1,234'}])
        self.assertAlmostEqual(df['Call Volume'].iloc[0], 1234.0)

    def test_service_level_percent_strip(self):
        df = self._clean([{'Date': '01/01/24', 'Service Level': '95.50%'}])
        self.assertAlmostEqual(df['Service Level'].iloc[0], 0.955)

    def test_abandon_rate_percent_strip(self):
        df = self._clean([{'Date': '01/01/24', 'Abandon Rate': '2.50%'}])
        self.assertAlmostEqual(df['Abandon Rate'].iloc[0], 0.025)

    def test_cct_passes_through_unchanged(self):
        # clean_daily does NOT convert CCT (pandas auto-infers it as numeric from CSV).
        # Verify it stays as a numeric passthrough when already numeric.
        df = pd.DataFrame({
            'Date': ['01/01/24'], 'Call Volume': ['100'],
            'CCT': [350.0],  # already numeric as it would be from pd.read_csv
            'Service Level': ['95.00%'], 'Abandon Rate': ['1.00%'],
        })
        result = cleaning.clean_daily(df)
        self.assertAlmostEqual(result['CCT'].iloc[0], 350.0)

    def test_invalid_becomes_nan(self):
        df = self._clean([{'Date': '01/01/24', 'Call Volume': 'N/A'}])
        self.assertTrue(pd.isna(df['Call Volume'].iloc[0]))

    def test_zero_values_preserved(self):
        df = self._clean([{'Date': '01/01/24', 'Call Volume': '0', 'Abandon Rate': '0.00%'}])
        self.assertEqual(df['Call Volume'].iloc[0], 0.0)
        self.assertEqual(df['Abandon Rate'].iloc[0], 0.0)


# ===========================================================================
# 2. clean_interval
# ===========================================================================

class TestCleanInterval(unittest.TestCase):

    def _clean(self, rows):
        return cleaning.clean_interval(make_interval_df(rows))

    def test_service_level_percent_strip(self):
        df = self._clean([{'Service Level': '87.50%'}])
        self.assertAlmostEqual(df['Service Level'].iloc[0], 0.875)

    def test_abandoned_rate_percent_strip(self):
        df = self._clean([{'Abandoned Rate': '3.25%'}])
        self.assertAlmostEqual(df['Abandoned Rate'].iloc[0], 0.0325)

    def test_cct_numeric(self):
        df = self._clean([{'CCT': '412.5'}])
        self.assertAlmostEqual(df['CCT'].iloc[0], 412.5)

    def test_cct_comma_removal(self):
        # CCT can have commas (clean_interval strips them)
        df = self._clean([{'CCT': '1,200'}])
        self.assertAlmostEqual(df['CCT'].iloc[0], 1200.0)

    def test_zero_sl_allowed(self):
        df = self._clean([{'Service Level': '0.00%'}])
        self.assertAlmostEqual(df['Service Level'].iloc[0], 0.0)


# ===========================================================================
# 3. build_datetime
# ===========================================================================

class TestBuildDatetime(unittest.TestCase):

    def _build(self, date, interval):
        df = pd.DataFrame({'Date': [date], 'Interval': [interval]})
        return cleaning.build_datetime(df)

    def test_single_digit_hour_padded(self):
        df = self._build('04/01/25', '0:00')
        self.assertEqual(df['DateTime'].iloc[0], '2025-04-01 00:00')

    def test_double_digit_hour(self):
        df = self._build('04/01/25', '13:30')
        self.assertEqual(df['DateTime'].iloc[0], '2025-04-01 13:30')

    def test_midnight_half(self):
        df = self._build('06/15/25', '0:30')
        self.assertEqual(df['DateTime'].iloc[0], '2025-06-15 00:30')

    def test_last_interval(self):
        df = self._build('05/26/25', '23:30')
        self.assertEqual(df['DateTime'].iloc[0], '2025-05-26 23:30')


# ===========================================================================
# 4. handle_daily_nulls
# ===========================================================================

class TestHandleDailyNulls(unittest.TestCase):

    def _process(self, rows):
        df = cleaning.clean_daily(make_daily_df(rows))
        return cleaning.handle_daily_nulls(df)

    def test_all_null_row_dropped(self):
        rows = [
            {'Date': '01/06/25'},   # Sunday — all metrics will be NaN after clean_daily
            {'Date': '01/01/24', 'Call Volume': '100', 'CCT': '300',
             'Service Level': '95.00%', 'Abandon Rate': '1.00%'},
        ]
        # Force all-null in first row by passing NaN-producing strings
        df = pd.DataFrame({
            'Date':         ['01/06/25', '01/01/24'],
            'Call Volume':  [None, '100'],
            'CCT':          [None, '300'],
            'Service Level':[None, '95.00%'],
            'Abandon Rate': [None, '1.00%'],
        })
        df = cleaning.clean_daily(df)
        result = cleaning.handle_daily_nulls(df)
        self.assertEqual(len(result), 1)

    def test_partial_null_imputed_with_dow_median(self):
        # Two Mondays (DOW=0) with known values, one Monday with null CV
        rows_raw = pd.DataFrame({
            'Date':         ['01/06/25', '01/13/25', '01/20/25'],
            'Call Volume':  ['200', '400', None],
            'CCT':          ['300', '300', '300'],
            'Service Level':['95.00%', '95.00%', '95.00%'],
            'Abandon Rate': ['1.00%', '1.00%', '1.00%'],
        })
        df = cleaning.clean_daily(rows_raw)
        result = cleaning.handle_daily_nulls(df)
        # DOW-median of [200, 400] = 300
        imputed_cv = result.loc[result['Date'] == '01/20/25', 'Call Volume'].values[0]
        self.assertAlmostEqual(imputed_cv, 300.0)

    def test_helper_columns_removed(self):
        rows_raw = pd.DataFrame({
            'Date':         ['01/06/25'],
            'Call Volume':  ['100'],
            'CCT':          ['300'],
            'Service Level':['95.00%'],
            'Abandon Rate': ['1.00%'],
        })
        df = cleaning.clean_daily(rows_raw)
        result = cleaning.handle_daily_nulls(df)
        self.assertNotIn('_dow', result.columns)
        self.assertNotIn('_year', result.columns)


# ===========================================================================
# 5. handle_interval_nulls
# ===========================================================================

class TestHandleIntervalNulls(unittest.TestCase):

    def _base_row(self, **overrides):
        # Call Volume and Abandoned Calls must be numeric here because
        # clean_interval() does NOT convert them (it relies on pandas auto-parsing
        # from CSV).  In tests we must replicate what pd.read_csv produces.
        base = {
            'Month': 'April', 'Day': '1', 'Interval': '0:00',
            'Service Level': '100.00%', 'Call Volume': 5.0,
            'Abandoned Calls': 0.0, 'Abandoned Rate': '0.00%',
            'CCT': '200', 'Date': '04/01/25'
        }
        base.update(overrides)
        return base

    def _process(self, rows):
        df = cleaning.clean_interval(pd.DataFrame(rows))
        return cleaning.handle_interval_nulls(df)

    def test_all_null_interval_dropped(self):
        rows = [
            self._base_row(Interval=None,
                           **{'Service Level': None, 'Call Volume': None,
                              'Abandoned Calls': None, 'Abandoned Rate': None, 'CCT': None}),
            self._base_row(),
        ]
        result = self._process(rows)
        self.assertEqual(len(result), 1)

    def test_cv_zero_fills_derived_metrics(self):
        # Column dtype rules that mirror real pd.read_csv behavior:
        #   Service Level / Abandoned Rate: raw strings → use None (object dtype)
        #     so that clean_interval's .str.rstrip('%') succeeds.
        #   Abandoned Calls / Call Volume: raw numerics → use np.nan (float64)
        #     so handle_interval_nulls .loc-assignments work in pandas 3.x.
        #   CCT: uses .astype(str) first → either None or np.nan works.
        rows = [self._base_row(
            **{'Call Volume': 0.0,
               'Service Level': None,
               'Abandoned Calls': np.nan,
               'Abandoned Rate': None,
               'CCT': np.nan}
        )]
        result = self._process(rows)
        row = result.iloc[0]
        self.assertEqual(row['Service Level'],   1.0)
        self.assertEqual(row['Abandoned Calls'], 0.0)
        self.assertEqual(row['Abandoned Rate'],  0.0)
        self.assertEqual(row['CCT'],             0.0)

    def test_derives_abandoned_rate_from_ac_cv(self):
        rows = [self._base_row(
            **{'Call Volume': 100.0, 'Abandoned Calls': 10.0, 'Abandoned Rate': None}
        )]
        result = self._process(rows)
        self.assertAlmostEqual(result.iloc[0]['Abandoned Rate'], 0.10, places=4)

    def test_derives_abandoned_calls_from_ar_cv(self):
        rows = [self._base_row(
            **{'Call Volume': 100.0, 'Abandoned Calls': np.nan, 'Abandoned Rate': '5.00%'}
        )]
        result = self._process(rows)
        self.assertAlmostEqual(result.iloc[0]['Abandoned Calls'], 5.0, places=0)

    def test_derive_ar_skips_cv_zero(self):
        # When CV=0 and AR is missing, should be filled as 0 (step 3), not NaN
        rows = [self._base_row(
            **{'Call Volume': 0.0, 'Abandoned Calls': 0.0, 'Abandoned Rate': None}
        )]
        result = self._process(rows)
        self.assertEqual(result.iloc[0]['Abandoned Rate'], 0.0)

    def test_datetime_column_added(self):
        rows = [self._base_row()]
        result = self._process(rows)
        self.assertIn('DateTime', result.columns)
        self.assertEqual(result.iloc[0]['DateTime'], '2025-04-01 00:00')

    def test_helper_dow_column_removed(self):
        rows = [self._base_row()]
        result = self._process(rows)
        self.assertNotIn('_dow', result.columns)


# ===========================================================================
# 6. handle_staffing_nulls
# ===========================================================================

class TestHandleStaffingNulls(unittest.TestCase):

    def test_null_imputed_with_dow_median(self):
        df = pd.DataFrame({
            'Date': ['01/06/25', '01/13/25', '01/20/25'],
            'A': [10.0, 20.0, None],
            'B': [5.0, 5.0, 5.0],
            'C': [8.0, 8.0, 8.0],
            'D': [3.0, 3.0, 3.0],
        })
        result = cleaning.handle_staffing_nulls(df)
        # DOW-median of Monday A values [10, 20] = 15
        imputed = result.loc[result['Date'] == '01/20/25', 'A'].values[0]
        self.assertAlmostEqual(imputed, 15.0)

    def test_helper_column_removed(self):
        df = pd.DataFrame({
            'Date': ['01/06/25'],
            'A': [10.0], 'B': [5.0], 'C': [8.0], 'D': [3.0],
        })
        result = cleaning.handle_staffing_nulls(df)
        self.assertNotIn('_dow', result.columns)


# ===========================================================================
# 7. agg.py pure functions (imported directly)
# ===========================================================================

# Import agg functions without running its file I/O
def _import_agg_functions():
    import importlib.util, builtins, unittest.mock as mock

    spec = importlib.util.spec_from_file_location(
        'agg_funcs', os.path.join(BASE_DIR, 'agg.py')
    )
    mod = importlib.util.module_from_spec(spec)

    # Block file I/O: patch open and csv.DictWriter so no files touched
    real_open = builtins.open
    def patched_open(path, *args, **kwargs):
        # Allow opening agg.py itself (source read), block CSV I/O
        if str(path).endswith('.csv') or str(path).endswith('.py') is False:
            raise StopIteration('blocked')
        return real_open(path, *args, **kwargs)

    with mock.patch('builtins.open', side_effect=StopIteration):
        try:
            spec.loader.exec_module(mod)
        except StopIteration:
            pass  # halted at first open() call; functions are already defined

    return mod

agg = _import_agg_functions()


class TestAggParseDatetime(unittest.TestCase):

    def test_monday(self):
        dow, interval = agg.parse_datetime('2025-04-07 09:30')
        self.assertEqual(dow, 0)  # Monday
        self.assertEqual(interval, '09:30')

    def test_sunday(self):
        dow, interval = agg.parse_datetime('2025-04-06 00:00')
        self.assertEqual(dow, 6)  # Sunday
        self.assertEqual(interval, '00:00')

    def test_interval_preserved(self):
        _, interval = agg.parse_datetime('2025-06-01 23:30')
        self.assertEqual(interval, '23:30')

    def test_wednesday(self):
        dow, _ = agg.parse_datetime('2025-05-07 12:00')
        self.assertEqual(dow, 2)  # Wednesday


class TestAggTrimmedMean(unittest.TestCase):

    def test_basic_trim(self):
        # Drop 1 lowest (1) and 1 highest (9), mean([2,3,4,5,6,7,8]) = 5.0
        result = agg.trimmed_mean([1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertAlmostEqual(result, 5.0)

    def test_fallback_when_too_few(self):
        # len=2, 2*trim=2 → falls back to plain mean
        result = agg.trimmed_mean([10, 20])
        self.assertAlmostEqual(result, 15.0)

    def test_single_value(self):
        result = agg.trimmed_mean([42])
        self.assertAlmostEqual(result, 42.0)

    def test_exactly_two_trim_boundary(self):
        # len=2, trim=1: 2 <= 2*1 → fallback to plain mean
        result = agg.trimmed_mean([0, 100], trim=1)
        self.assertAlmostEqual(result, 50.0)

    def test_unsorted_input(self):
        result = agg.trimmed_mean([9, 1, 5, 3, 7])
        # After sort: [1,3,5,7,9], trim 1 each side → [3,5,7], mean=5
        self.assertAlmostEqual(result, 5.0)

    def test_all_equal(self):
        result = agg.trimmed_mean([7, 7, 7, 7, 7])
        self.assertAlmostEqual(result, 7.0)


class TestAggStdDev(unittest.TestCase):

    def test_known_result(self):
        # Population std of [2,4,4,4,5,5,7,9] = 2.0; sample std ≈ 2.138
        vals = [2, 4, 4, 4, 5, 5, 7, 9]
        mean = sum(vals) / len(vals)
        result = agg.std_dev(vals, mean)
        self.assertAlmostEqual(result, 2.138, places=2)

    def test_single_value_returns_zero(self):
        result = agg.std_dev([5], 5)
        self.assertEqual(result, 0.0)

    def test_identical_values(self):
        result = agg.std_dev([3, 3, 3], 3.0)
        self.assertAlmostEqual(result, 0.0)

    def test_non_negative(self):
        import random
        vals = [random.random() * 100 for _ in range(20)]
        mean = sum(vals) / len(vals)
        result = agg.std_dev(vals, mean)
        self.assertGreaterEqual(result, 0.0)


# ===========================================================================
# 8. intraday_shape.py trimmed_mean (slightly different implementation)
# ===========================================================================

def _import_shape_functions():
    import importlib.util, unittest.mock as mock

    spec = importlib.util.spec_from_file_location(
        'shape_funcs', os.path.join(BASE_DIR, 'intraday_shape.py')
    )
    mod = importlib.util.module_from_spec(spec)

    dummy_df = pd.DataFrame({
        'Date': ['01/01/24'],
        'Call Volume': [100.0], 'CCT': [300.0],
        'Service Level': [0.95], 'Abandon Rate': [0.01],
        'day_of_week': ['Monday'], 'group': ['A'],
        'A': [10.0], 'B': [10.0], 'C': [10.0], 'D': [10.0],
        'group_x': ['A'], 'group_y': ['A'],
    })

    with mock.patch('pandas.read_csv', return_value=dummy_df), \
         mock.patch('pandas.DataFrame.to_csv', return_value=None), \
         mock.patch('pandas.concat', return_value=dummy_df), \
         mock.patch('pandas.DataFrame.merge', return_value=dummy_df):
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass

    return mod

shape_mod = _import_shape_functions()


class TestShapeTrimmedMean(unittest.TestCase):

    def test_basic_trim(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])
        result = shape_mod.trimmed_mean(s)
        self.assertAlmostEqual(result, 5.0)

    def test_fallback_when_too_few(self):
        s = pd.Series([10.0, 20.0])
        result = shape_mod.trimmed_mean(s)
        self.assertAlmostEqual(result, 15.0)

    def test_single_non_null(self):
        s = pd.Series([42.0, None])
        result = shape_mod.trimmed_mean(s)
        self.assertAlmostEqual(result, 42.0)

    def test_ignores_nan(self):
        s = pd.Series([1, np.nan, 5, np.nan, 9])
        result = shape_mod.trimmed_mean(s)
        # After dropna: [1,5,9], trim 1 each side → [5], mean=5
        self.assertAlmostEqual(result, 5.0)


# ===========================================================================
# 9. Integration tests — actual output files
# ===========================================================================

@unittest.skipUnless(os.path.isdir(CLEANED_DIR), 'cleaned_data/ not found')
class TestIntervalAggregated(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(os.path.join(CLEANED_DIR, 'interval_aggregated.csv'))

    def test_row_count(self):
        # 4 groups × 7 days × 48 intervals
        self.assertEqual(len(self.df), 1344,
                         f"Expected 1344 rows, got {len(self.df)}")

    def test_no_null_key_columns(self):
        for col in ['group', 'day_of_week', 'interval']:
            self.assertEqual(self.df[col].isna().sum(), 0,
                             f"Nulls in key column: {col}")

    def test_no_null_mean_cv(self):
        self.assertEqual(self.df['mean_call_volume'].isna().sum(), 0)

    def test_no_negative_mean_cv(self):
        self.assertTrue((self.df['mean_call_volume'] >= 0).all())

    def test_no_negative_mean_cct(self):
        self.assertTrue((self.df['mean_cct'] >= 0).all())

    def test_groups_present(self):
        self.assertEqual(set(self.df['group']), {'A', 'B', 'C', 'D'})

    def test_all_intervals_present(self):
        expected = {f"{h:02d}:{m}" for h in range(24) for m in ('00', '30')}
        actual = set(self.df['interval'])
        self.assertEqual(actual, expected)

    def test_all_days_of_week_present(self):
        expected = {'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'}
        self.assertEqual(set(self.df['day_of_week']), expected)

    def test_n_observations_positive(self):
        self.assertTrue((self.df['n_observations'] > 0).all())

    def test_mean_abandoned_rate_between_0_and_1(self):
        self.assertTrue((self.df['mean_abandoned_rate'] >= 0).all())
        self.assertTrue((self.df['mean_abandoned_rate'] <= 1).all())

    def test_mean_service_level_between_0_and_1(self):
        self.assertTrue((self.df['mean_service_level'] >= 0).all())
        self.assertTrue((self.df['mean_service_level'] <= 1).all())

    def test_std_non_negative(self):
        for col in ['std_call_volume', 'std_cct', 'std_abandoned_rate']:
            self.assertTrue((self.df[col] >= 0).all(), f"Negative std in {col}")


@unittest.skipUnless(os.path.isdir(CLEANED_DIR), 'cleaned_data/ not found')
class TestIntradayShape(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(os.path.join(CLEANED_DIR, 'intraday_shape.csv'))

    def test_row_count(self):
        self.assertEqual(len(self.df), 1344,
                         f"Expected 1344 rows, got {len(self.df)}")

    def test_no_nulls(self):
        null_counts = self.df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        self.assertTrue(cols_with_nulls.empty,
                        f"Null values found:\n{cols_with_nulls}")

    def test_shape_cv_non_negative(self):
        self.assertTrue((self.df['shape_call_volume'] >= 0).all())

    def test_shape_cct_non_negative(self):
        self.assertTrue((self.df['shape_cct'] >= 0).all())

    def test_daily_cv_positive(self):
        self.assertTrue((self.df['daily_call_volume'] > 0).all())

    def test_shape_cv_sums_to_one_per_group_dow(self):
        """
        Sum of shape_call_volume across 48 intervals for a (group, DOW)
        should be ~1.0 (±5%) because interval means should sum to ~daily mean.
        Holiday exclusion can introduce small bias.
        """
        sums = self.df.groupby(['group', 'day_of_week'])['shape_call_volume'].sum()
        for key, total in sums.items():
            self.assertAlmostEqual(
                total, 1.0, delta=0.05,
                msg=f"shape_cv sum for {key} = {total:.4f} (expected ~1.0)"
            )

    def test_interval_cv_matches_agg(self):
        """interval_call_volume in shape must match mean_call_volume in aggregated."""
        agg_df = pd.read_csv(os.path.join(CLEANED_DIR, 'interval_aggregated.csv'))
        merged = self.df.merge(agg_df, on=['group', 'day_of_week', 'interval'])
        diff = (merged['interval_call_volume'] - merged['mean_call_volume']).abs()
        self.assertTrue((diff < 1e-3).all(),
                        f"interval_call_volume mismatch, max diff = {diff.max():.6f}")

    def test_shape_ratio_consistency(self):
        """shape = interval / daily; verify the ratio holds."""
        computed = self.df['interval_call_volume'] / self.df['daily_call_volume']
        stored   = self.df['shape_call_volume']
        diff = (computed - stored).abs()
        self.assertTrue((diff < 1e-9).all(),
                        f"shape_call_volume ratio inconsistent, max diff = {diff.max()}")

    def test_staffing_positive(self):
        self.assertTrue((self.df['daily_staffing'] > 0).all())


@unittest.skipUnless(os.path.isdir(CLEANED_DIR), 'cleaned_data/ not found')
class TestCleanedDailyFiles(unittest.TestCase):

    def test_all_files_exist(self):
        for g in ['a', 'b', 'c', 'd']:
            path = os.path.join(CLEANED_DIR, f'{g}_daily_cleaned.csv')
            self.assertTrue(os.path.isfile(path), f"Missing: {path}")

    def test_date_format(self):
        for g in ['a', 'b', 'c', 'd']:
            df = pd.read_csv(os.path.join(CLEANED_DIR, f'{g}_daily_cleaned.csv'),
                             encoding='utf-8-sig')
            # Should parse without error
            dates = pd.to_datetime(df['Date'], format='%m/%d/%y')
            self.assertFalse(dates.isna().any(), f"{g}: unparseable dates")

    def test_covers_2024_and_2025(self):
        for g in ['a']:
            df = pd.read_csv(os.path.join(CLEANED_DIR, f'{g}_daily_cleaned.csv'),
                             encoding='utf-8-sig')
            years = pd.to_datetime(df['Date'], format='%m/%d/%y').dt.year.unique()
            self.assertIn(2024, years, "Missing 2024 data")
            self.assertIn(2025, years, "Missing 2025 data")

    def test_cv_non_negative(self):
        for g in ['a', 'b', 'c', 'd']:
            df = pd.read_csv(os.path.join(CLEANED_DIR, f'{g}_daily_cleaned.csv'),
                             encoding='utf-8-sig')
            self.assertTrue((df['Call Volume'] >= 0).all(),
                            f"{g}: negative Call Volume")

    def test_no_all_null_rows(self):
        metrics = ['Call Volume', 'CCT', 'Service Level', 'Abandon Rate']
        for g in ['a', 'b', 'c', 'd']:
            df = pd.read_csv(os.path.join(CLEANED_DIR, f'{g}_daily_cleaned.csv'),
                             encoding='utf-8-sig')
            all_null = df[metrics].isnull().all(axis=1)
            self.assertFalse(all_null.any(), f"{g}: all-null rows remain")


@unittest.skipUnless(os.path.isdir(CLEANED_DIR), 'cleaned_data/ not found')
class TestCleanedIntervalFiles(unittest.TestCase):

    def test_all_files_exist(self):
        for g in ['a', 'b', 'c', 'd']:
            path = os.path.join(CLEANED_DIR, f'{g}_interval_cleaned.csv')
            self.assertTrue(os.path.isfile(path))

    def test_datetime_format(self):
        for g in ['a', 'b', 'c', 'd']:
            df = pd.read_csv(os.path.join(CLEANED_DIR, f'{g}_interval_cleaned.csv'),
                             encoding='utf-8-sig')
            dt = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M', errors='coerce')
            null_count = dt.isna().sum()
            self.assertEqual(null_count, 0,
                             f"{g}: {null_count} unparseable DateTime values")

    def test_interval_count_per_day(self):
        """
        Portfolios C and D are near-complete (48 intervals almost every day).
        Portfolio A has heavy overnight missingness (legitimate — closed overnight);
        portfolio B has moderate gaps.  Verify at least 38 intervals per day for
        all portfolios (ensures no whole-day data loss) and exactly 48 for C.
        """
        min_expected = {'a': 38, 'b': 42, 'c': 48, 'd': 47}
        for g in ['a', 'b', 'c', 'd']:
            df = pd.read_csv(os.path.join(CLEANED_DIR, f'{g}_interval_cleaned.csv'),
                             encoding='utf-8-sig')
            counts = df.groupby('Date')['Interval'].count()
            bad = counts[counts < min_expected[g]]
            self.assertTrue(bad.empty,
                            f"{g}: days with fewer than {min_expected[g]} intervals:\n{bad}")

    def test_cv_non_negative(self):
        for g in ['a', 'b', 'c', 'd']:
            df = pd.read_csv(os.path.join(CLEANED_DIR, f'{g}_interval_cleaned.csv'),
                             encoding='utf-8-sig')
            self.assertTrue((df['Call Volume'] >= 0).all(),
                            f"{g}: negative interval Call Volume")

    def test_abandoned_rate_between_0_and_1(self):
        for g in ['a', 'b', 'c', 'd']:
            df = pd.read_csv(os.path.join(CLEANED_DIR, f'{g}_interval_cleaned.csv'),
                             encoding='utf-8-sig')
            self.assertTrue((df['Abandoned Rate'] >= 0).all())
            self.assertTrue((df['Abandoned Rate'] <= 1).all())

    def test_apr_jun_2025_coverage(self):
        """Interval data must cover April, May, June 2025."""
        for g in ['a', 'b', 'c', 'd']:
            df = pd.read_csv(os.path.join(CLEANED_DIR, f'{g}_interval_cleaned.csv'),
                             encoding='utf-8-sig')
            dt = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M')
            months = dt.dt.month.unique()
            for m in [4, 5, 6]:
                self.assertIn(m, months, f"{g}: missing month {m}")

    def test_interval_cv_consistency_with_daily(self):
        """
        Sum of interval Call Volume per day vs daily Call Volume.

        NOTE: The two data sources don't always agree perfectly. Observed max diffs:
          a: 8.4%  b: 13.4%  c: 22.1%  d: 5.9%
        This is a known data-quality characteristic (likely different counting scopes
        between interval and daily exports). The test checks that the MEDIAN day is
        within 2% and that gross outliers (>25%) don't exist.
        """
        for g in ['a', 'b', 'c', 'd']:
            interval_df = pd.read_csv(
                os.path.join(CLEANED_DIR, f'{g}_interval_cleaned.csv'), encoding='utf-8-sig')
            daily_df = pd.read_csv(
                os.path.join(CLEANED_DIR, f'{g}_daily_cleaned.csv'), encoding='utf-8-sig')

            interval_df['date_only'] = pd.to_datetime(
                interval_df['DateTime'], format='%Y-%m-%d %H:%M').dt.strftime('%m/%d/%y')
            interval_daily_cv = interval_df.groupby('date_only')['Call Volume'].sum()

            daily_df = daily_df.set_index('Date')
            common_dates = interval_daily_cv.index.intersection(daily_df.index)
            self.assertGreater(len(common_dates), 0, f"{g}: no overlapping dates")

            diffs = []
            for d in common_dates:
                daily_total = daily_df.loc[d, 'Call Volume']
                if daily_total > 0:
                    pct_diff = abs(interval_daily_cv[d] - daily_total) / daily_total
                    diffs.append(pct_diff)
                    self.assertLess(
                        pct_diff, 0.25,
                        f"{g} date {d}: interval sum {interval_daily_cv[d]:.0f} vs "
                        f"daily {daily_total:.0f} ({pct_diff*100:.1f}% diff — gross outlier)"
                    )

            median_diff = float(pd.Series(diffs).median())
            self.assertLess(median_diff, 0.02,
                            f"{g}: median CV discrepancy {median_diff*100:.2f}% > 2%")


if __name__ == '__main__':
    unittest.main(verbosity=2)
