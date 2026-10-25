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

Modeling choices will be made experimentally once the data and feature
pipeline are in place.

## Status

🚧 Work in progress. Project infrastructure is set up; data validation
and exploration come next.
