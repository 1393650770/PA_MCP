# [AI:BEGIN]
# PA_MCP - Multi-stage Dockerfile
# Stage 1: Build with TA-Lib compilation
# Stage 2: Minimal runtime image
# [AI:END]

# ---- Build Stage ----
FROM python:3.12-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Build TA-Lib from source
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        mcp \
        pydantic pydantic-settings pyyaml \
        akshare duckdb numpy pandas \
        TA-Lib pandas-ta \
        httpx tenacity redis croniter structlog

# ---- Runtime Stage ----
FROM python:3.12-slim-bookworm

# Copy TA-Lib shared library
COPY --from=builder /usr/lib/libta_lib* /usr/lib/
RUN ldconfig

# Create non-root user
RUN useradd --create-home --shell /bin/bash pa_mcp

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY src/ ./src/
COPY config/ ./config/
COPY pyproject.toml .

# Create data directory
RUN mkdir -p /data && chown -R pa_mcp:pa_mcp /data /app

USER pa_mcp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, 'src'); from pa_mcp.config import get_settings; get_settings()" || exit 1

EXPOSE 8080

ENV PA_MCP_SERVER__TRANSPORT=http
ENV PA_MCP_DATABASE__PATH=/data/pa_mcp.duckdb

ENTRYPOINT ["python", "-m", "pa_mcp.server"]
