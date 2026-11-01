"""Build the modeling tables from the raw M5 files."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from demand_forecast.data.download import RAW_DIR
from demand_forecast.data.load import SALES_ID_COLUMNS, RawData, load_raw

BASE_TABLE_PATH = Path("data/processed/base_table.parquet")

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


def add_calendar(long: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Add the date, Walmart week, events and SNAP flags to each sales row."""
    days = calendar[CALENDAR_COLUMNS].assign(
        d=lambda df: df["d"].astype(str).str.removeprefix("d_").astype("int16")
    )
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


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Build the base sales table from the raw M5 files."
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
        help=f"path of the Parquet file to write (default: {BASE_TABLE_PATH})",
    )
    args = parser.parse_args(argv)

    base = build_base_table(load_raw(args.raw_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(args.output, index=False)
    print(f"Base table: {len(base):,} rows, {base['id'].nunique():,} series")
    print(f"Memory: {base.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
