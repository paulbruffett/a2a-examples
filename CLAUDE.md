# CLAUDE.md

## Repository overview

This is a collection of Agent-to-Agent (A2A) protocol examples. Each subdirectory at the root is a self-contained example project. All code is Python, with Jupyter notebooks as the primary medium.

## Project structure

- Each root-level folder (e.g. `chart_agent/`, `a2a_mcp/`) is an independent example
- Examples share the root-level UV environment (`pyproject.toml` + `uv.lock`) unless they have their own `pyproject.toml`
- Few or no dependencies between examples

## Environment

- Python 3.13, managed with `uv`
- Activate: `source .venv/bin/activate`
- Install deps: `uv sync`
- Environment variables are in `.env` (gitignored)

## Conventions

- Notebooks (`.ipynb`) are the primary deliverable — prefer editing notebooks over creating standalone `.py` files
- Each example should be runnable independently
- Keep shared root dependencies in the root `pyproject.toml`
