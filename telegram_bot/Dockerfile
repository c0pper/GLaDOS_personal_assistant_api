FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim

# Disable development dependencies, precompile bytecode for faster startup
ENV UV_NO_DEV=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

ENV TZ=Europe/Rome

WORKDIR /app

# Install dependencies first so this layer is reused when only the source code changes
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

COPY bot.py ./

# Run the venv's python directly: `uv run` would re-check the environment on every start
CMD ["/app/.venv/bin/python", "bot.py"]
