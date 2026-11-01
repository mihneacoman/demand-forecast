from pathlib import Path

import pandas as pd
import pytest

from demand_forecast.data.build_features import (
    add_calendar,
    add_calendar_features,
    add_days_since_sale,
    add_lags,
    add_next_event,
    add_prices,
    add_rolling_stats,
    add_sales_history,
    build_base_table,
    check_base_table,
    drop_pre_launch,
    main,
    to_long,
)
from demand_forecast.data.load import SALES_ID_COLUMNS, RawData, load_raw


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
            "event_name_2": [None, None, "Easter", None],
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


def make_priced() -> pd.DataFrame:
    # A is priced from day 1. B launches on day 3 and has a price gap on day 4.
    priced = pd.DataFrame(
        {
            "id": ["A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "d": [1, 2, 3, 4, 1, 2, 3, 4, 5],
            "sales": [1, 0, 2, 0, 0, 0, 0, 0, 3],
            "sell_price": [1.5, 1.5, 1.5, 1.5, None, None, 2.0, None, 2.0],
        }
    )
    return priced.astype(
        {"id": "category", "d": "int16", "sales": "int16", "sell_price": "float32"}
    )


def make_raw() -> RawData:
    return RawData(calendar=make_calendar(), prices=make_prices(), sales=make_sales())


def make_holiday_sales() -> pd.DataFrame:
    # Covers Christmas in two years, plus the same day of another month.
    holiday_sales = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2011-12-24", "2011-12-25", "2011-12-26", "2012-12-25", "2012-11-25"]
            ),
            "sales": [3, 0, 2, 1, 4],
        }
    )
    return holiday_sales.astype({"sales": "int16"})


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


def test_days_before_launch_are_dropped_per_series() -> None:
    launched = drop_pre_launch(make_priced())
    assert launched.loc[launched["id"] == "B", "d"].tolist() == [3, 4, 5]


def test_series_priced_from_the_start_stays_whole() -> None:
    launched = drop_pre_launch(make_priced())
    assert launched.loc[launched["id"] == "A", "d"].tolist() == [1, 2, 3, 4]


def test_zero_sales_after_launch_are_kept() -> None:
    launched = drop_pre_launch(make_priced()).set_index(["id", "d"])["sales"]
    assert launched["A", 2] == 0
    assert launched["B", 3] == 0


def test_price_gap_after_launch_is_kept() -> None:
    launched = drop_pre_launch(make_priced()).set_index(["id", "d"])["sell_price"]
    assert pd.isna(launched["B", 4])


def test_dropping_keeps_order_dtypes_and_a_clean_index() -> None:
    priced = make_priced()
    launched = drop_pre_launch(priced)
    expected = priced.drop(index=[4, 5]).reset_index(drop=True)
    pd.testing.assert_frame_equal(launched, expected)


def test_base_table_has_launched_rows_with_their_context() -> None:
    base = build_base_table(make_raw())
    assert list(zip(base["item_id"], base["d"])) == [
        ("A", 1),
        ("A", 2),
        ("A", 3),
        ("B", 3),
    ]
    assert base["sell_price"].tolist() == [1.5, 1.5, 2.0, 3.0]
    assert base["date"].iloc[-1] == pd.Timestamp("2011-01-31")


def test_base_table_passes_its_checks() -> None:
    check_base_table(make_priced().iloc[[0, 1, 2, 3, 6]], n_series=2)


@pytest.mark.parametrize(
    ("rows", "n_series", "message"),
    [
        ([0, 0, 6], 2, "duplicate series-days"),
        ([0, 1, 2], 2, "series lost"),
        ([0, 4, 6], 2, "series starting without a price"),
        ([0, 6, 7], 2, "price gaps after launch"),
    ],
)
def test_broken_base_table_fails_its_checks(
    rows: list[int], n_series: int, message: str
) -> None:
    with pytest.raises(AssertionError, match=message):
        check_base_table(make_priced().iloc[rows], n_series=n_series)


def test_cli_writes_the_base_table(tmp_path: Path) -> None:
    raw = make_raw()
    raw.calendar.to_csv(tmp_path / "calendar.csv", index=False)
    raw.prices.to_csv(tmp_path / "sell_prices.csv", index=False)
    raw.sales.to_csv(tmp_path / "sales_train_evaluation.csv", index=False)
    output = tmp_path / "processed" / "base_table.parquet"

    main(["--raw-dir", str(tmp_path), "--output", str(output)])

    expected = build_base_table(load_raw(tmp_path))
    pd.testing.assert_frame_equal(pd.read_parquet(output), expected)


def test_sales_history_is_missing_only_on_christmas() -> None:
    history = add_sales_history(make_holiday_sales())["sales_history"]
    assert history.isna().tolist() == [False, True, False, True, False]


def test_sales_history_copies_the_other_days() -> None:
    history = add_sales_history(make_holiday_sales())["sales_history"]
    assert history.dropna().tolist() == [3, 2, 4]
    assert history.dtype == "float32"


def test_sales_history_leaves_the_target_unchanged() -> None:
    holiday_sales = make_holiday_sales()
    result = add_sales_history(holiday_sales)
    pd.testing.assert_series_equal(result["sales"], holiday_sales["sales"])


def test_lags_stay_within_each_series_and_carry_missing_days() -> None:
    table = pd.DataFrame(
        {
            "id": ["A", "A", "A", "A", "B", "B"],
            "sales_history": [1, 2, None, 4, 5, 6],
        }
    ).astype({"id": "category", "sales_history": "float32"})

    expected = table.assign(
        sales_lag_1=pd.array([None, 1, 2, None, None, 5], dtype="float32"),
        sales_lag_2=pd.array([None, None, 1, 2, None, None], dtype="float32"),
    )
    pd.testing.assert_frame_equal(add_lags(table, lags=[1, 2]), expected)


def test_rolling_stats_stay_within_each_series_and_skip_missing_days() -> None:
    table = pd.DataFrame(
        {
            "id": ["A", "A", "A", "A", "A", "B", "B"],
            "sales_lag_28": [None, 1, 3, None, 5, 2, 4],
        }
    ).astype({"id": "category", "sales_lag_28": "float32"})

    spread = 2**0.5
    expected = table.assign(
        sales_mean_3=pd.array([None, None, 2, 2, 4, None, 3], dtype="float32"),
        sales_std_3=pd.array(
            [None, None, spread, spread, spread, None, spread], dtype="float32"
        ),
    )
    pd.testing.assert_frame_equal(add_rolling_stats(table, windows=[3]), expected)


def test_days_since_sale_counts_within_each_series_through_missing_days() -> None:
    table = pd.DataFrame(
        {
            "id": ["A", "A", "A", "A", "A", "A", "A", "B", "B", "B"],
            "sales_lag_28": [None, 0, 2, 0, None, 0, 1, 0, 0, 3],
        }
    ).astype({"id": "category", "sales_lag_28": "float32"})

    expected = table.assign(
        days_since_sale=pd.array(
            [None, None, 0, 1, 2, 3, 0, None, None, 0], dtype="float32"
        )
    )
    pd.testing.assert_frame_equal(add_days_since_sale(table), expected)


def test_calendar_features_use_the_date_and_the_series_own_state() -> None:
    table = pd.DataFrame(
        {
            "state_id": ["CA", "TX", "WI"],
            "date": pd.to_datetime(["2011-01-29", "2011-02-06", "2012-12-03"]),
            "snap_CA": [1, 1, 0],
            "snap_TX": [0, 0, 0],
            "snap_WI": [0, 1, 1],
        }
    ).astype(
        {
            "state_id": "category",
            "snap_CA": "int8",
            "snap_TX": "int8",
            "snap_WI": "int8",
        }
    )

    expected = table[["state_id", "date"]].assign(
        day_of_week=pd.array([5, 6, 0], dtype="int8"),
        day_of_month=pd.array([29, 6, 3], dtype="int8"),
        month=pd.array([1, 2, 12], dtype="int8"),
        snap=pd.array([1, 0, 1], dtype="int8"),
    )
    pd.testing.assert_frame_equal(add_calendar_features(table), expected)


def test_next_event_reads_the_full_calendar_and_applies_the_cap() -> None:
    # Day 12 has an event but no sales row, like events in the forecast days.
    names: list[str | None] = [None] * 13
    names[1], names[11] = "X", "Y"
    calendar = pd.DataFrame(
        {"d": [f"d_{day}" for day in range(1, 14)], "event_name_1": names}
    ).astype({"d": "category", "event_name_1": "category"})
    table = pd.DataFrame({"d": [1, 2, 3, 5, 6, 13]}).astype({"d": "int16"})

    expected = table.assign(
        days_until_event=pd.array([1, 0, 7, 7, 6, 7], dtype="int8"),
        next_event_name=pd.Categorical(
            ["X", "X", None, None, "Y", None], categories=["X", "Y"]
        ),
    )
    pd.testing.assert_frame_equal(add_next_event(table, calendar), expected)
