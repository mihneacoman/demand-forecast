import pandas as pd

from demand_forecast.data.build_features import to_long
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
