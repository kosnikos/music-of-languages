# The Music of Languages — Phase 0

Baseline + harness for prosody-based language proximity (feeds the Feature Exploration cycle).
See `docs/superpowers/specs/2026-06-24-music-of-languages-design.md` for the design.

## Setup (uv — recommended)

Install [uv](https://docs.astral.sh/uv/), then:

    uv sync --extra dev --extra feasibility

This creates `.venv` on a uv-managed Python (version in `.python-version`) with the exact
locked dependencies from `uv.lock`. torch/torchaudio are CPU builds pulled from the PyTorch
CPU index (configured in `pyproject.toml`).

> The project forces a uv-managed interpreter (`python-preference = "only-managed"`):
> Anaconda-managed Python interpreters make torch fail to load (`OSError: WinError 1114`),
> while a uv-managed Python imports it cleanly.

## Run

    uv run pytest                      # tests
    uv run jupyter notebook notebooks/ # notebooks
    uv run python scripts/collect_sample.py --clips-per-lang 5 --clip-seconds 60

Prefix project commands with `uv run` so they use the project venv (not a system/Anaconda
tool — bare `jupyter`/`python` may resolve to Anaconda and fail to import `musiclang`).
The collector (`collect_sample.py`) needs `ffmpeg` on `PATH`.

## Setup (pip fallback)

    python -m venv .venv
    .venv\Scripts\python -m pip install -e ".[dev,feasibility]"

pip ignores `uv.lock` (resolves latest within the `pyproject.toml` ranges) and does not pick
the managed Python, so on this machine the pip path can hit the torch `WinError 1114`. Prefer uv.
