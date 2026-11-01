import pandas as pd
import pytest

from demand_forecast.data.build_features import add_calendar, add_prices, to_long
from demand_forecast.data.load import SALES_ID_COLUMNS


def make_sales() -> pd.DataFrame:
    sales = pd.DataFrame(
        {
            "id": ["A_CA_1_evaluation", "B_TX_1_evaluation"],
            "item_id": ["A", "B"],
            "dept_id": ["FOODS_1", "HOBBIES_1"],
            "cat_id": ["FOODS", "HOBBIES"],
            "store_id": ["CA_1", "TX_1"],
            "state_id": ["CA", "TX"],
            "d_1": [1, 4],
            "d_2": [2, 5],
            "d_3": [3, 6],
        }
    )
    return sales.astype(
        {column: "category" for column in SALES_ID_COLUMNS}
        | {"d_1": "int16", "d_2": "int16", "d_3": "int16"}
    )


def make_calendar() -> pd.DataFrame:
    # One more day than the sales, like the raw calendar's forecast days.
    calendar = pd.DataFrame(
        {
            "d": ["d_1", "d_2", "d_3", "d_4"],
            "date": pd.date_range("2011-01-29", periods=4),
            "wm_yr_wk": [11101, 11101, 11102, 11102],
            "event_name_1": [None, "SuperBowl", None, None],
            "event_name_2": [None, None, None, None],
            "snap_CA": [0, 1, 0, 1],
            "snap_TX": [1, 0, 0, 1],
            "snap_WI": [0, 0, 1, 1],
        }
    )
    return calendar.astype(
        {
            "d": "category",
            "wm_yr_wk": "int16",
            "event_name_1": "category",
            "event_name_2": "category",
            "snap_CA": "int8",
            "snap_TX": "int8",
            "snap_WI": "int8",
        }
    )


def make_prices() -> pd.DataFrame:
    # B has no price in week 11101; A in TX_1 is a different series from A in CA_1.
    prices = pd.DataFrame(
        {
            "store_id": ["CA_1", "CA_1", "TX_1", "TX_1"],
            "item_id": ["A", "A", "A", "B"],
            "wm_yr_wk": [11101, 11102, 11101, 11102],
            "sell_price": [1.5, 2.0, 9.99, 3.0],
        }
    )
    return prices.astype(
        {
            "store_id": "category",
            "item_id": "category",
            "wm_yr_wk": "int16",
            "sell_price": "float32",
        }
    )


def make_long() -> pd.DataFrame:
    return add_calendar(to_long(make_sales()), make_calendar())


def test_one_row_per_series_and_day() -> None:
    long = to_long(make_sales())
    assert len(long) == 6
    assert not long.duplicated(["id", "d"]).any()


def test_values_land_on_the_right_series_and_day() -> None:
    long = to_long(make_sales()).set_index(["item_id", "d"])["sales"]
    assert long["A", 2] == 2
    assert long["B", 3] == 6


def test_dtypes_stay_compact() -> None:
    long = to_long(make_sales())
    assert (long[SALES_ID_COLUMNS].dtypes == "category").all()
    assert long["d"].dtype == "int16"
    assert long["sales"].dtype == "int16"


def test_calendar_keeps_rows_and_their_order() -> None:
    long = to_long(make_sales())
    merged = add_calendar(long, make_calendar())
    pd.testing.assert_frame_equal(merged[long.columns], long)


def test_calendar_values_land_on_the_right_day() -> None:
    merged = add_calendar(to_long(make_sales()), make_calendar())
    day_2 = merged[merged["d"] == 2].iloc[0]
    assert day_2["date"] == pd.Timestamp("2011-01-30")
    assert day_2["event_name_1"] == "SuperBowl"
    assert day_2["snap_CA"] == 1


def test_calendar_dtypes_stay_compact() -> None:
    merged = add_calendar(to_long(make_sales()), make_calendar())
    assert merged["d"].dtype == "int16"
    assert merged["wm_yr_wk"].dtype == "int16"
    assert merged["event_name_1"].dtype == "category"
    assert merged["snap_CA"].dtype == "int8"


def test_sales_day_missing_from_calendar_fails() -> None:
    calendar = make_calendar().iloc[:2]
    with pytest.raises(AssertionError, match="missing from the calendar"):
        add_calendar(to_long(make_sales()), calendar)


def test_prices_keep_rows_and_their_order() -> None:
    long = make_long()
    merged = add_prices(long, make_prices())
    pd.testing.assert_frame_equal(merged[long.columns], long)


def test_prices_land_on_the_right_series_and_week() -> None:
    merged = add_prices(make_long(), make_prices())
    prices = merged.set_index(["item_id", "d"])["sell_price"]
    assert prices["A", 1] == 1.5
    assert prices["A", 3] == 2.0
    assert prices["B", 3] == 3.0


def test_days_of_a_week_share_its_price() -> None:
    merged = add_prices(make_long(), make_prices())
    prices = merged.set_index(["item_id", "d"])["sell_price"]
    assert prices["A", 1] == prices["A", 2]


def test_week_without_a_price_stays_missing() -> None:
    merged = add_prices(make_long(), make_prices())
    prices = merged.set_index(["item_id", "d"])["sell_price"]
    assert pd.isna(prices["B", 1])
    assert pd.isna(prices["B", 2])


def test_price_dtypes_stay_compact_when_categories_differ() -> None:
    prices = make_prices()
    prices["item_id"] = prices["item_id"].cat.set_categories(["Z", "B", "A"])
    prices["store_id"] = prices["store_id"].cat.set_categories(["TX_1", "CA_1"])
    merged = add_prices(make_long(), prices)
    assert merged["item_id"].dtype == "category"
    assert merged["store_id"].dtype == "category"
    assert merged["sell_price"].dtype == "float32"


def test_duplicate_price_fails() -> None:
    prices = make_prices()
    duplicated = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
    with pytest.raises(pd.errors.MergeError):
        add_prices(make_long(), duplicated)
