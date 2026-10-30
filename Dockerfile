FROM python:3.14-slim

WORKDIR /app

# Install dependencies from pyproject.toml first to leverage Docker layer caching
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy the application code
COPY api/ ./api/

# Create and switch to a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
