# CLAUDE.md

## Project
Demand forecasting on the M5 (Walmart) dataset. Pipeline: data validation →
feature engineering → LightGBM (+ TFT stretch) → SHAP explainability →
MLflow tracking/registry → FastAPI serving.

## Environment
Dependencies are managed with `uv`. Never use `pip install` directly.
```bash
uv sync                       # install/update from lockfile
uv add <package>              # add a new dependency
uv add --dev <package>        # add a dev-only dependency
uv run pytest                 # run any command inside the project env
docker compose up -d          # MLflow server + Postgres + MinIO
```
- `MLFLOW_TRACKING_URI` is read from `.env` — never hardcode it in code.
- Never read or write anything under `data/` or `mlruns/` — gitignored and
  access-denied in `.claude/settings.json`. Use `docker compose logs` or
  `mlflow runs list` to inspect state instead of reading files directly.

## Commands
```bash
uv run pytest                       # unit + integration tests
uv run ruff check . && uv run black .   # lint + format
uv run python -m src.data.build_features   # feature pipeline, CLI-callable
uv run mlflow ui                    # inspect runs locally
```

## Git Workflow
- `main` — always deployable, protected, PR-only.
- `feature/*` — one branch per unit of work (e.g. `feature/lightgbm-baseline`).
  No `dev` branch.
- Merge via PR (self-review is fine). One logical change per PR.

## Data & Modeling Rules
- Splits are always **time-based** (last 28 days as test, walk-forward for
  the rest). Never a random split.
- Rolling/lag features must reference only past data — check for leakage
  before merging any new feature.
- Feature engineering lives in `src/data/`, must be scriptable/CLI-callable,
  never notebook-only. Notebooks are for exploration only.
- Standard metrics for every model: **WAPE, MASE, RMSSE** (+ pinball loss for
  quantile models). Every new model is compared against the baseline tier in
  the same MLflow parent run.

## MLflow Conventions
Every run must log:
- git commit hash
- dataset version/hash + feature set version
- params, metrics (see above), environment
- artifacts: plots, feature importance, SHAP values where applicable

Promote via Model Registry stage transitions (Staging → Production), not by
naming convention alone.

## Code Standards
- black, ruff, isort, mypy — enforced via pre-commit; don't bypass with
  `--no-verify`.
- Type hints on all public functions in `src/`.
- No secrets or credentials in code or committed configs — use `.env`
  (gitignored) and `os.environ`.
