# Stage 1: Build the Python virtual environment.
FROM python:3.12-slim AS builder

WORKDIR /app

# Install the same uv version used by the project.
RUN pip install --no-cache-dir uv==0.12.10

# Copy dependency files first so Docker can cache this layer.
COPY pyproject.toml uv.lock ./

# Install locked production dependencies.
# The project itself is not installed yet because src/ is copied later.
RUN uv sync --frozen --no-dev --no-install-project


# Stage 2: Create the smaller runtime image.
FROM python:3.12-slim AS runtime

WORKDIR /app

# PyTorch may require the GNU OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Use Python and packages from the virtual environment.
ENV PATH="/app/.venv/bin:$PATH"

# Copy the completed virtual environment from the builder stage.
COPY --from=builder /app/.venv /app/.venv

# Copy only the application source code.
COPY src/ /app/src/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.food11.serve:app", "--host", "0.0.0.0", "--port", "8000"]