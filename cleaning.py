# combining null value logic, changing the data types, and joining the daily values to the original dataframes

# raphael im not touching any of the date stuff

# imports for types
import csv
import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

#imports for null value logic

#import for joins


# functions for type conversions 
def load_csv(filename):
    with open(filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return list(reader)

def parse_percentage(value):
    if isinstance(value, str):
        if '%' in value:
            value = value.replace('%', '').strip()
            try:
                return float(value) / 100
            except ValueError:
                return None
        try:
            return float(value)
        except ValueError:
            return None
    return value

def parse_call_volume(value):
    if isinstance(value, str):
        value = value.replace(',', '').strip()
        try:
            return float(value)
        except ValueError:
            return None
    return value

def normalize_daily(data):
    for row in data:
        if 'Call Volume' in row:
            row['Call Volume'] = parse_call_volume(row['Call Volume'])
        if 'CCT' in row:
            try:
                row['CCT'] = float(row['CCT']) if row['CCT'] else None
            except ValueError:
                row['CCT'] = None
        if 'Service Level' in row:
            row['Service Level'] = parse_percentage(row['Service Level'])
        if 'Abandon Rate' in row:
            row['Abandon Rate'] = parse_percentage(row['Abandon Rate'])
    return data

def normalize_interval(data):
    for row in data:
        if 'Interval' in row and row['Interval']:
            interval_str = row['Interval']
            if 'days' in interval_str:
                row['Interval'] = interval_str  # keep as str for now
            else:
                row['Interval'] = interval_str + ':00' if ':' not in interval_str else interval_str
        if 'Service Level' in row:
            row['Service Level'] = parse_percentage(row['Service Level'])
        if 'Abandoned Rate' in row:
            row['Abandoned Rate'] = parse_percentage(row['Abandoned Rate'])
        if 'Abandon Rate' in row:
            row['Abandon Rate'] = parse_percentage(row['Abandon Rate'])
        if 'CCT' in row:
            try:
                row['CCT'] = float(row['CCT']) if row['CCT'] else None
            except ValueError:
                row['CCT'] = None
    return data

def save_csv(filename, data):
    if not data:
        return
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        for row in data:
            # Convert dates back to str
            row_copy = row.copy()
            for k, v in row_copy.items():
                if isinstance(v, datetime.date):
                    row_copy[k] = v.isoformat()
                elif isinstance(v, datetime.datetime):
                    row_copy[k] = v.isoformat()
            writer.writerow(row_copy)


# loading data for the type conversion functions
a_daily_data = load_csv('a_daily_cleaned.csv')
a_interval_data = load_csv('a_interval_cleaned.csv')
b_daily_data = load_csv('b_daily_cleaned.csv')
b_interval_data = load_csv('b_interval_cleaned.csv')
c_daily_data = load_csv('c_daily_cleaned.csv')
c_interval_data = load_csv('c_interval_cleaned.csv')
d_daily_data = load_csv('d_daily_cleaned.csv')
d_interval_data = load_csv('d_interval_cleaned.csv')
daily_staffing_data = load_csv('daily_staffing_cleaned.csv')

datas = {
    'a_daily': a_daily_data, 'a_interval': a_interval_data,
    'b_daily': b_daily_data, 'b_interval': b_interval_data,
    'c_daily': c_daily_data, 'c_interval': c_interval_data,
    'd_daily': d_daily_data, 'd_interval': d_interval_data,
    'daily_staffing': daily_staffing_data,
}

for name, data in datas.items():
    print(f"\n{name} shape: {len(data)} rows")
    if data:
        print("Sample row:", dict(list(data[0].items())[:5]))

a_daily_data = normalize_daily(a_daily_data)
b_daily_data = normalize_daily(b_daily_data)
c_daily_data = normalize_daily(c_daily_data)
d_daily_data = normalize_daily(d_daily_data)

a_interval_data = normalize_interval(a_interval_data)
b_interval_data = normalize_interval(b_interval_data)
c_interval_data = normalize_interval(c_interval_data)
d_interval_data = normalize_interval(d_interval_data)

