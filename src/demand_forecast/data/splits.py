"""Split the feature table by time: walk-forward folds and a final test split."""

from dataclasses import dataclass

# The M5 forecast horizon: every validation and test window is this long.
HORIZON = 28
N_FOLDS = 3


@dataclass(frozen=True)
class Split:
    """A training window up to `train_end`, then the next days up to `valid_end`."""

    train_end: int
    valid_end: int

    @property
    def valid_start(self) -> int:
        return self.train_end + 1


def split_days(
    last_day: int, n_folds: int = N_FOLDS, horizon: int = HORIZON
) -> tuple[list[Split], Split]:
    """Return the walk-forward folds, oldest first, and the test split.

    Each training window expands from the first day, and each window is followed
    by the next one, so the test split covers the last `horizon` days. Features
    use sales at least `horizon` days old, so no gap is needed between windows.
    """
    ends = [last_day - k * horizon for k in range(n_folds, -1, -1)]
    assert ends[0] - horizon > 0, "not enough days for the folds"
    splits = [Split(train_end=end - horizon, valid_end=end) for end in ends]
    return splits[:-1], splits[-1]
