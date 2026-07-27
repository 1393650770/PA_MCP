# [AI:BEGIN]
# PA_MCP - Single-stage Dockerfile (pure Python, zero C compilation)
# Works on any platform: Windows WSL, Linux, Mac
# [AI:END]

FROM python:3.12-slim-bookworm

# Create non-root user
RUN useradd --create-home --shell /bin/bash pa_mcp

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/
COPY config/ ./config/

# Install dependencies (pure Python, no C compilation needed)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[dev]"

# Create data directory
RUN mkdir -p /data && chown -R pa_mcp:pa_mcp /data /app

USER pa_mcp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "from pa_mcp.config import get_settings; get_settings()" || exit 1

EXPOSE 8080

ENV PA_MCP_SERVER__TRANSPORT=http
ENV PA_MCP_DATABASE__PATH=/data/pa_mcp.duckdb

ENTRYPOINT ["python", "-m", "pa_mcp.server"]
