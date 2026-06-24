# The Music of Languages — Phase 0

Baseline + harness for prosody-based language proximity (feeds the Feature Exploration cycle).
See `docs/superpowers/specs/2026-06-24-music-of-languages-design.md` for the design.

## Setup (Windows PowerShell)

    python -m venv .venv
    .venv\Scripts\python -m pip install -e ".[dev]"

## Run tests

    .venv\Scripts\python -m pytest -v

## Notebooks

    .venv\Scripts\jupyter notebook notebooks/
