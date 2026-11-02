import pandas as pd

from demand_forecast.data.splits import Split, split_days, split_masks


def test_folds_walk_forward_up_to_the_test_split() -> None:
    folds, test = split_days(last_day=1941)

    assert folds == [
        Split(train_end=1829, valid_end=1857),
        Split(train_end=1857, valid_end=1885),
        Split(train_end=1885, valid_end=1913),
    ]
    assert test == Split(train_end=1913, valid_end=1941)
    assert test.valid_start == 1914


def test_masks_include_the_boundary_days_and_nothing_after() -> None:
    table = pd.DataFrame({"id": ["A"] * 6 + ["B"] * 6, "d": list(range(1, 7)) * 2})

    train, valid = split_masks(table, Split(train_end=3, valid_end=5))

    assert table.loc[train, "d"].tolist() == [1, 2, 3, 1, 2, 3]
    assert table.loc[valid, "d"].tolist() == [4, 5, 4, 5]
