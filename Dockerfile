# Use the official Python 3.12 image as the base
FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim

# Copy the project into the image
COPY . /app

# Disable development dependencies
ENV UV_NO_DEV=1

ENV PYTHONPATH=/app
ENV TZ=Europe/Rome

# Sync the project into a new environment, asserting the lockfile is up to date
WORKDIR /app
RUN uv sync --locked

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

