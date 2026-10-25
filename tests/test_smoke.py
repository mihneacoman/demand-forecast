import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "demand_forecast.data",
        "demand_forecast.models",
        "demand_forecast.explainability",
        "demand_forecast.serving",
        "demand_forecast.utils",
    ],
)
def test_package_imports(module: str) -> None:
    importlib.import_module(module)
