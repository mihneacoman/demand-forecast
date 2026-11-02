from collections.abc import Callable
from functools import partial

import pandas as pd
import pytest

from demand_forecast.data.splits import Split
from demand_forecast.models.baselines import moving_average, naive, seasonal_naive

Baseline = Callable[[pd.DataFrame, Split], pd.Series]

SPLIT = Split(train_end=8, valid_end=16)
BASELINES: dict[str, Baseline] = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "moving_average": partial(moving_average, window=4),
}


def make_table() -> pd.DataFrame:
    # A sells from day 1; B launches on day 6; C launches on the first validation
    # day, so it has no history. Validation sales are 0 throughout.
    days = {"A": range(1, 17), "B": range(6, 17), "C": range(9, 17)}
    training_sales = {"A": [1, 2, 3, 4, 5, 6, 7, 8], "B": [3, 0, 6], "C": []}
    rows = [
        (series, d, (training_sales[series] + [0] * 16)[i])
        for series, series_days in days.items()
        for i, d in enumerate(series_days)
    ]
    table = pd.DataFrame(rows, columns=["id", "d", "sales"])
    return table.astype({"id": "category", "d": "int16", "sales": "int16"})


@pytest.mark.parametrize(
    ("name", "expected_a", "expected_b"),
    [
        ("naive", [8] * 8, [6] * 8),
        # Day 16 wraps to day 2; B has no sales row before day 6, so those are 0.
        ("seasonal_naive", [2, 3, 4, 5, 6, 7, 8, 2], [0, 0, 0, 0, 3, 0, 6, 0]),
        # Over days 5-8; B has only days 6-8.
        ("moving_average", [6.5] * 8, [3] * 8),
    ],
)
def test_baselines_forecast_the_validation_rows(
    name: str, expected_a: list[float], expected_b: list[float]
) -> None:
    table = make_table()

    predictions = BASELINES[name](table, SPLIT)

    assert predictions.index.equals(table.index[table["d"] > SPLIT.train_end])
    assert predictions.tolist() == expected_a + expected_b + [0] * 8


@pytest.mark.parametrize("name", BASELINES)
def test_baselines_ignore_validation_sales(name: str) -> None:
    table = make_table()
    changed = table.assign(sales=table["sales"].mask(table["d"] > SPLIT.train_end, 100))

    pd.testing.assert_series_equal(
        BASELINES[name](changed, SPLIT), BASELINES[name](table, SPLIT)
    )
