"""Build the modeling tables from the raw M5 files."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from demand_forecast.data.download import RAW_DIR
from demand_forecast.data.load import SALES_ID_COLUMNS, RawData, load_raw

BASE_TABLE_PATH = Path("data/processed/base_table.parquet")
# Bump when the feature set changes, so runs stay tied to the features they used.
FEATURE_SET_VERSION = 1
FEATURES_PATH = Path(f"data/processed/features_v{FEATURE_SET_VERSION}.parquet")

CALENDAR_COLUMNS = [
    "d",
    "date",
    "wm_yr_wk",
    "event_name_1",
    "event_name_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
]
PRICE_KEYS = ["store_id", "item_id", "wm_yr_wk"]
# Multiples of 7, from the forecast horizon up: shorter lags are unknown for the
# later forecast days, and other lengths would compare different weekdays.
LAGS = [28, 35, 42, 56]
# Multiples of 7, so each weekday counts the same in every window.
ROLLING_WINDOWS = [7, 28, 91]
# The EDA finds shopping moves at most a few days ahead of an event.
MAX_DAYS_TO_EVENT = 7
COMPLETE_FEATURES = ["days_until_event", "day_of_week", "day_of_month", "month", "snap"]


def to_long(sales: pd.DataFrame) -> pd.DataFrame:
    """Reshape wide sales into one row per series and day."""
    day_columns = [column for column in sales.columns if column.startswith("d_")]
    days = np.array(
        [int(column.removeprefix("d_")) for column in day_columns], dtype="int16"
    )
    n_series, n_days = len(sales), len(days)
    return pd.DataFrame(
        {
            **{
                column: sales[column].repeat(n_days).array
                for column in SALES_ID_COLUMNS
            },
            "d": np.tile(days, n_series),
            "sales": sales[day_columns].to_numpy().ravel(),
        },
        copy=False,
    )


def day_numbers(calendar: pd.DataFrame) -> pd.Series:
    """Return the calendar's d_* labels as the day numbers of the long table."""
    return calendar["d"].astype(str).str.removeprefix("d_").astype("int16")


def add_calendar(long: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Add the date, Walmart week, events and SNAP flags to each sales row."""
    days = calendar[CALENDAR_COLUMNS].assign(d=day_numbers(calendar))
    merged = long.merge(days, on="d", how="left", validate="many_to_one")
    assert merged["date"].notna().all(), "sales days missing from the calendar"
    return merged


def add_prices(long: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Add the weekly price to each row, left missing before launch."""
    # Categoricals merged with different category sets turn into object columns.
    aligned = prices.astype({column: long[column].dtype for column in PRICE_KEYS[:2]})
    return long.merge(aligned, on=PRICE_KEYS, how="left", validate="many_to_one")


def drop_pre_launch(long: pd.DataFrame) -> pd.DataFrame:
    """Keep each series from its first priced day, which marks its launch."""
    # Rows must be in day order within a series: a row counts as launched once
    # any row up to it has a price. check_base_table catches later price gaps.
    launched = long["sell_price"].notna().groupby(long["id"]).cummax()
    return long[launched].reset_index(drop=True)


def check_base_table(base: pd.DataFrame, n_series: int) -> None:
    """Assert the invariants of the base table."""
    assert not base.duplicated(["id", "d"]).any(), "duplicate series-days"
    assert base["id"].nunique() == n_series, "series lost"
    first_rows = base.drop_duplicates("id")
    assert first_rows["sell_price"].notna().all(), "series starting without a price"
    assert base["sell_price"].notna().all(), "price gaps after launch"


def build_base_table(raw: RawData) -> pd.DataFrame:
    """Build one row per launched series and day, with its calendar and price."""
    long = add_calendar(to_long(raw.sales), raw.calendar)
    base = drop_pre_launch(add_prices(long, raw.prices))
    check_base_table(base, n_series=len(raw.sales))
    return base


def is_christmas(dates: pd.Series) -> pd.Series:
    """Return which dates are Christmas Day, when the stores are closed."""
    return (dates.dt.month == 12) & (dates.dt.day == 25)


def add_sales_history(base: pd.DataFrame) -> pd.DataFrame:
    """Add the sales seen by history features, missing on Christmas closures."""
    history = base["sales"].astype("float32").mask(is_christmas(base["date"]))
    return base.assign(sales_history=history)


def add_lags(table: pd.DataFrame, lags: Sequence[int] = LAGS) -> pd.DataFrame:
    """Add each series' sales from earlier days, by row within the series."""
    history = table.groupby("id")["sales_history"]
    return table.assign(**{f"sales_lag_{lag}": history.shift(lag) for lag in lags})


def add_rolling_stats(
    table: pd.DataFrame, windows: Sequence[int] = ROLLING_WINDOWS
) -> pd.DataFrame:
    """Add the mean and std of each series' sales over windows ending at lag 28."""
    columns = {}
    for window in windows:
        # Half a window is enough, so Christmas does not blank every window it is in.
        min_periods = (window + 1) // 2
        rolling = table.groupby("id")["sales_lag_28"].rolling(window, min_periods)
        for name, stat in (("mean", rolling.mean()), ("std", rolling.std())):
            stat = stat.reset_index(level="id", drop=True).astype("float32")
            columns[f"sales_{name}_{window}"] = stat
    return table.assign(**columns)


def add_days_since_sale(table: pd.DataFrame) -> pd.DataFrame:
    """Add the days from each series' last sale to lag 28, missing before any sale."""
    rows = table.groupby("id").cumcount()
    last_sale = rows.where(table["sales_lag_28"] > 0).groupby(table["id"]).ffill()
    return table.assign(days_since_sale=(rows - last_sale).astype("float32"))


def add_calendar_features(table: pd.DataFrame) -> pd.DataFrame:
    """Add the date parts, and replace the state SNAP flags with the series' own."""
    states = table["state_id"].cat
    snap_columns = [f"snap_{state}" for state in states.categories]
    snap = table[snap_columns].to_numpy()[np.arange(len(table)), states.codes]
    dates = table["date"].dt
    return table.drop(columns=snap_columns).assign(
        day_of_week=dates.day_of_week.astype("int8"),
        day_of_month=dates.day.astype("int8"),
        month=dates.month.astype("int8"),
        snap=snap,
    )


def add_next_event(table: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Add the days until the next event, capped, and its name when within the cap."""
    # Read from the full calendar: the table stops before events in the forecast days.
    events = calendar["event_name_1"]
    positions = pd.Series(np.arange(len(calendar)), index=calendar.index)
    days_until = positions.where(events.notna()).bfill() - positions
    is_near = days_until < MAX_DAYS_TO_EVENT
    upcoming = pd.DataFrame(
        {
            "d": day_numbers(calendar),
            "days_until_event": days_until.where(is_near, MAX_DAYS_TO_EVENT).astype(
                "int8"
            ),
            "next_event_name": events.bfill().where(is_near),
        }
    )
    return table.merge(upcoming, on="d", how="left", validate="many_to_one")


def check_feature_table(features: pd.DataFrame, n_series: int) -> None:
    """Assert the invariants of the feature table."""
    assert not features.duplicated(["id", "d"]).any(), "duplicate series-days"
    assert features["id"].nunique() == n_series, "series lost"
    assert not is_christmas(features["date"]).any(), "Christmas rows left"
    assert features[COMPLETE_FEATURES].notna().all(axis=None), "missing features"


def build_feature_table(base: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Build the model features, each known 28 days before its row's day."""
    # Christmas rows stay until here, so shifting by rows still means shifting by days.
    table = add_days_since_sale(add_rolling_stats(add_lags(add_sales_history(base))))
    table = add_next_event(add_calendar_features(table), calendar)
    features = table[~is_christmas(table["date"])].drop(columns="sales_history")
    features = features.reset_index(drop=True)
    check_feature_table(features, n_series=base["id"].nunique())
    return features


def write_table(table: pd.DataFrame, path: Path, name: str) -> None:
    """Write a table to Parquet and report its size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    print(f"{name}: {len(table):,} rows, {table['id'].nunique():,} series")
    print(f"Memory: {table.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    print(f"Written to {path}")


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Build the base sales and feature tables from the raw M5 files."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"directory with the raw CSV files (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_TABLE_PATH,
        help=f"path of the base table Parquet file (default: {BASE_TABLE_PATH})",
    )
    parser.add_argument(
        "--features-output",
        type=Path,
        default=FEATURES_PATH,
        help=f"path of the feature table Parquet file (default: {FEATURES_PATH})",
    )
    args = parser.parse_args(argv)

    raw = load_raw(args.raw_dir)
    base = build_base_table(raw)
    write_table(base, args.output, "Base table")
    features = build_feature_table(base, raw.calendar)
    write_table(features, args.features_output, "Feature table")


if __name__ == "__main__":
    main()
