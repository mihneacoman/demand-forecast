import pandas as pd

from demand_forecast.data.splits import Split
from demand_forecast.models.metrics import series_errors, series_scales, wape


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


def make_scales(ids: list[str], mase: list[float], rmsse: list[float]) -> pd.DataFrame:
    index = pd.CategoricalIndex(ids, categories=["A", "B", "C"], name="id")
    return pd.DataFrame({"mase_scale": mase, "rmsse_scale": rmsse}, index=index)


def make_forecast(
    ids: list[str], sales: list[int], predictions: list[float]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": pd.Categorical(ids, categories=["A", "B", "C"]),
            "sales": sales,
            "prediction": predictions,
        }
    )


def test_errors_are_scaled_per_series() -> None:
    # Errors: A 4, 0; B 1, -3.
    forecast = make_forecast(["A", "A", "B", "B"], [1, 3, 2, 4], [5.0, 3.0, 3.0, 1.0])
    scales = make_scales(["A", "B"], mase=[2.0, 1.0], rmsse=[2.0, 5.0])

    assert wape(forecast) == 0.8
    pd.testing.assert_frame_equal(
        series_errors(forecast, scales),
        pd.DataFrame({"mase": [1.0, 2.0], "rmsse": [2.0, 1.0]}, index=scales.index),
    )


def test_series_without_a_usable_scale_are_left_out() -> None:
    # B never changes in training (zero scale); C has no scale at all.
    forecast = make_forecast(["A", "B", "C"], [1, 1, 1], [3.0, 2.0, 2.0])
    scales = make_scales(["A", "B"], mase=[2.0, 0.0], rmsse=[4.0, 0.0])

    assert series_errors(forecast, scales).index.tolist() == ["A"]
