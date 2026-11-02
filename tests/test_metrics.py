import pandas as pd

from demand_forecast.data.splits import Split
from demand_forecast.models.metrics import series_scales


def test_scales_use_training_days_from_the_first_sale() -> None:
    # A sells first on day 3 and spikes after training; B skips day 3 (Christmas);
    # C sells first on the last training day, so it has nothing to scale by.
    table = pd.DataFrame(
        {
            "id": pd.Categorical(["A"] * 6 + ["B"] * 4 + ["C"] * 6),
            "d": [1, 2, 3, 4, 5, 6] + [1, 2, 4, 5] + [1, 2, 3, 4, 5, 6],
            "sales": [0, 0, 2, 0, 3, 9] + [1, 1, 4, 4] + [0, 0, 0, 0, 1, 5],
        }
    )

    scales = series_scales(table, Split(train_end=5, valid_end=6))

    expected = pd.DataFrame(
        {"mase_scale": [2.5, 1.0], "rmsse_scale": [6.5, 3.0]},
        index=pd.CategoricalIndex(["A", "B"], categories=["A", "B", "C"], name="id"),
    )
    pd.testing.assert_frame_equal(scales, expected)
