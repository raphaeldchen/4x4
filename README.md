# 2026 Synchrony Datathon — Call Center Staffing Forecasting

Uses daily and interval CSV files for 4 clients (A–D), plus staffing data, to forecast call volume, abandoned calls/rate, and CCT at 30-minute intervals.

## Usage

Run the pipeline in order:

```
python cleaning.py
python agg.py
python intraday_shape.py
python forecast_cv.py
python forecast_cct.py
python forecast_abd.py
python assemble_forecast.py
```

Output is written to `forecasts/`.

## Demo

[Video walkthrough](https://drive.google.com/file/d/1CLR68AGvBmEGFr20AMNutlhiFKx53pkn/view?usp=sharing)
