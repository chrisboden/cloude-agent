FROM python:3.11-slim

# Install Node.js 20 and required system tools
RUN apt-get update && \
    apt-get install -y curl gnupg procps && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI globally
RUN npm install -g @anthropic-ai/claude-code

# Create non-root user (required for bypassPermissions mode)
RUN useradd -m -s /bin/bash appuser

# Set up app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create workspace directory and set base permissions
# Note: Skills are managed on the volume at runtime, not baked into the build
RUN mkdir -p /app/workspace && \
    chown -R appuser:appuser /app && \
    chmod +x /app/entrypoint.sh

# Default port (Railway sets PORT env var)
ENV PORT=8080
ENV WORKSPACE_DIR=/app/workspace
EXPOSE 8080

# Use entrypoint to handle volume permissions then drop to appuser
ENTRYPOINT ["/app/entrypoint.sh"]
