"""Load the raw M5 files with explicit, memory-efficient dtypes."""

from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from demand_forecast.data.download import RAW_DIR

CALENDAR_DTYPES = {
    "wm_yr_wk": "int16",
    "weekday": "category",
    "wday": "int8",
    "month": "int8",
    "year": "int16",
    "d": "category",
    "event_name_1": "category",
    "event_type_1": "category",
    "event_name_2": "category",
    "event_type_2": "category",
    "snap_CA": "int8",
    "snap_TX": "int8",
    "snap_WI": "int8",
}
PRICES_DTYPES = {
    "store_id": "category",
    "item_id": "category",
    "wm_yr_wk": "int16",
    "sell_price": "float32",
}
SALES_ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


class RawData(NamedTuple):
    calendar: pd.DataFrame
    prices: pd.DataFrame
    sales: pd.DataFrame


def load_calendar(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load calendar.csv: one row per day, with events and SNAP flags."""
    return pd.read_csv(
        raw_dir / "calendar.csv", dtype=CALENDAR_DTYPES, parse_dates=["date"]
    )


def load_prices(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load sell_prices.csv: one row per store, item and Walmart week."""
    return pd.read_csv(raw_dir / "sell_prices.csv", dtype=PRICES_DTYPES)


def load_sales(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load sales_train_evaluation.csv: one row per series, one d_* column per day."""
    # Every column not named here is a d_* column of daily unit sales.
    dtypes = defaultdict(
        lambda: "int16", {column: "category" for column in SALES_ID_COLUMNS}
    )
    return pd.read_csv(raw_dir / "sales_train_evaluation.csv", dtype=dtypes)


def load_raw(raw_dir: Path = RAW_DIR) -> RawData:
    """Load all three raw M5 files."""
    return RawData(
        calendar=load_calendar(raw_dir),
        prices=load_prices(raw_dir),
        sales=load_sales(raw_dir),
    )
