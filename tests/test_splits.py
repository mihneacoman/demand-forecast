from demand_forecast.data.splits import Split, split_days


def test_folds_walk_forward_up_to_the_test_split() -> None:
    folds, test = split_days(last_day=1941)

    assert folds == [
        Split(train_end=1829, valid_end=1857),
        Split(train_end=1857, valid_end=1885),
        Split(train_end=1885, valid_end=1913),
    ]
    assert test == Split(train_end=1913, valid_end=1941)
    assert test.valid_start == 1914
