# Contributing to heataxis

## Development setup

```bash
pip install -e ".[dev]"
pre-commit install                       # run ruff on every commit
pre-commit install --hook-type pre-push  # run pytest before every push
```

## Checks (identical to CI)

```bash
ruff check src tests      # lint
pytest                    # tests
```

Continuous integration runs the same two commands across an OS × Python
matrix (`.github/workflows/tests.yml`). The `release` workflow runs them
again as a gate before building and publishing to PyPI.

`ruff format` is **not** enforced: the aligned palette and unit tables in
`constants.py` and the box-drawing module banners are laid out by hand, and a
formatter would reflow them. Keep linting (`ruff check`) green; leave layout
to the author.
