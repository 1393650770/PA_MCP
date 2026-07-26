# [AI:BEGIN]
# PA_MCP - Cron Scheduler Dockerfile
# Runs daily data update at 17:30 Beijing time
# [AI:END]

FROM pa-mcp:latest

USER root

# Install cron
RUN apt-get update && apt-get install -y --no-install-recommends cron \
    && rm -rf /var/lib/apt/lists/*

# Copy crontab
COPY crontab /etc/cron.d/pa-mcp-cron
RUN chmod 0644 /etc/cron.d/pa-mcp-cron \
    && crontab /etc/cron.d/pa-mcp-cron

# Create log file
RUN touch /var/log/cron.log && chown pa_mcp:pa_mcp /var/log/cron.log

USER pa_mcp

CMD ["cron", "-f"]
