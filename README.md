# demand-forecast

An end-to-end demand forecasting system built on the
[M5 (Walmart)](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
dataset: daily unit sales for ~3,000 products across 10 stores.

The goal is a forecasting service that is not just accurate but
trustworthy and production-ready:

- **Forecasts** daily demand per product and store, benchmarked against
  simple baselines on a time-based holdout.
- **Explains** its predictions, showing which factors (price, seasonality,
  events, recent sales) drive each forecast.
- **Tracks** every experiment reproducibly, with code, data and feature
  versions, metrics, and a model registry for promotion to production.
- **Serves** forecasts through an API.

## What the data says

An [exploratory analysis](docs/eda.md) shaped the modeling choices:

- **Most demand is intermittent.** 60% of days with the item on the shelf
  have zero sales, and 73% of series sell on only a few days. Hence
  scale-free metrics (WAPE, MASE, RMSSE; no MAPE) and a Tweedie objective.
- **Many zeros are not demand.** 64% of series launch after the data starts;
  their pre-launch days are 31% of all zeros and are excluded.
- **Calendar effects are strong and specific.** Weekends sell 12% to 26%
  above the weekly mean, and food assistance (SNAP) days lift food sales by
  6% to 20% beyond the start-of-month effect.
- **Holidays shift demand rather than add to it.** Sales drop on the day
  itself and rise one or two days before.

## Getting the data

The raw M5 files are downloaded from Kaggle and verified by SHA256.

1. Create a Kaggle API token (*Settings → API → Generate New Token*)
   and save it to `~/.kaggle/access_token` (or use a legacy
   `~/.kaggle/kaggle.json`).
2. Accept the [competition rules](https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules)
   — without this, the download returns 403.
3. From the repo root:

   ```bash
   uv sync
   uv run python -m demand_forecast.data.download
   ```

Files land in `data/raw/` (gitignored). Re-running is a no-op once the
files are present and valid; use `--force` to re-download.

## Status

🚧 Work in progress. Project infrastructure, the data download and the
exploratory analysis are in place; the feature pipeline comes next.
