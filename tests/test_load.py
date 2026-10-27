from pathlib import Path

import pandas as pd
import pytest

from demand_forecast.data.load import load_raw

CALENDAR_CSV = """\
date,wm_yr_wk,weekday,wday,month,year,d,event_name_1,event_type_1,event_name_2,event_type_2,snap_CA,snap_TX,snap_WI
2011-01-29,11101,Saturday,1,1,2011,d_1,,,,,0,0,0
2011-01-30,11101,Sunday,2,1,2011,d_2,SuperBowl,Sporting,,,0,0,0
"""
PRICES_CSV = """\
store_id,item_id,wm_yr_wk,sell_price
CA_1,HOBBIES_1_001,11101,9.58
CA_1,HOBBIES_1_002,11101,3.97
"""
SALES_CSV = """\
id,item_id,dept_id,cat_id,store_id,state_id,d_1,d_2
HOBBIES_1_001_CA_1_evaluation,HOBBIES_1_001,HOBBIES_1,HOBBIES,CA_1,CA,0,3
HOBBIES_1_002_CA_1_evaluation,HOBBIES_1_002,HOBBIES_1,HOBBIES,CA_1,CA,1,0
"""


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    (tmp_path / "calendar.csv").write_text(CALENDAR_CSV)
    (tmp_path / "sell_prices.csv").write_text(PRICES_CSV)
    (tmp_path / "sales_train_evaluation.csv").write_text(SALES_CSV)
    return tmp_path


def test_calendar_dtypes(raw_dir: Path) -> None:
    calendar = load_raw(raw_dir).calendar
    assert pd.api.types.is_datetime64_any_dtype(calendar["date"])
    assert calendar["event_name_1"].dtype == "category"
    assert calendar["event_name_1"].isna().tolist() == [True, False]
    assert calendar["snap_CA"].dtype == "int8"


def test_prices_dtypes(raw_dir: Path) -> None:
    prices = load_raw(raw_dir).prices
    assert prices["item_id"].dtype == "category"
    assert prices["wm_yr_wk"].dtype == "int16"
    assert prices["sell_price"].dtype == "float32"


def test_sales_dtypes(raw_dir: Path) -> None:
    sales = load_raw(raw_dir).sales
    assert (sales[["id", "store_id"]].dtypes == "category").all()
    assert (sales[["d_1", "d_2"]].dtypes == "int16").all()
    assert sales["d_2"].tolist() == [3, 0]
