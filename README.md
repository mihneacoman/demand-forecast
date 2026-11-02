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

## Baseline results

Every model is scored on three 28-day walk-forward folds and on the last
28 days as the test split. The primary metric is WRMSSE, the M5 competition
metric, here at the level of each product and store: each series' RMSSE,
weighted by its recent dollar sales.

| Baseline | WRMSSE, folds (mean ± std) | WRMSSE, test | WAPE, test |
|---|---|---|---|
| Naive (last day) | 1.211 ± 0.032 | 1.167 | 0.952 |
| Seasonal naive (same weekday, last week) | 1.131 ± 0.009 | 1.102 | 0.862 |
| **Moving average (last 28 days)** | **0.888 ± 0.015** | **0.870** | **0.739** |

- **The moving average is the bar to beat.** With most demand intermittent,
  copying one day or one week copies noise; averaging four weeks smooths it.
- **The folds predict the test split.** The ranking is the same on every
  fold and on the test split, and test scores sit close to the fold means,
  so the folds can be trusted for model selection.

To reproduce, with the MLflow stack running (`docker compose up -d`):

```bash
uv run --env-file .env python -m demand_forecast.models.evaluate_baselines
```

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

🚧 Work in progress. Project infrastructure, the data download, the
exploratory analysis and the feature pipeline are in place: the feature
table (sales lags and rolling stats, calendar, SNAP and price features) is
built from the CLI, and split by time into walk-forward folds and a test
split on the last 28 days. The baselines are scored and tracked in MLflow
(see [Baseline results](#baseline-results)); classical baselines (Croston,
ETS, ARIMA) and then a global LightGBM model come next.
