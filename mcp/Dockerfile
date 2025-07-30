# MCP Services Container - Multi-stage build
FROM node:20-alpine AS base

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN addgroup -g 1001 -S appuser && \
    adduser -u 1001 -S appuser -G appuser

# Copy package.json first for better caching
COPY package.json ./

# Install dependencies including ts-node for TypeScript execution
RUN npm install

# Create config directory
RUN mkdir -p /app/config

# Copy configuration files
COPY config/ /app/config/

# Copy the filesystem server code
COPY . .

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose internal port for filesystem server
EXPOSE 3001

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD node -e "process.exit(require('http').get('http://localhost:3001/health').statusCode === 200 ? 0 : 1)"

# Default command - will be overridden by docker-compose
CMD ["node", "index.js", "--config", "/app/config/filesystem.json"]
