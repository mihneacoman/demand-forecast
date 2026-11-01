"""Build the modeling tables from the raw M5 files."""

import numpy as np
import pandas as pd

from demand_forecast.data.load import SALES_ID_COLUMNS


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
