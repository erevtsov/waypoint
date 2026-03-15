# waypoint

A Python library for financial portfolio analysis, focused on returns-based time series.

## Installation

```bash
uv add waypoint
```

## Development

```bash
git clone https://github.com/erevtsov/waypoint
cd waypoint
uv sync --extra dev
```

```bash
uv run pytest                        # tests + coverage
uv run ruff check src/ tests/        # lint
uv run mypy                          # type-check
```

## License

MIT — see [LICENSE](LICENSE).
