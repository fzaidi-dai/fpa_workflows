# Docker Development Setup for FPA Agents

This document provides comprehensive instructions for using Docker with the FPA Agents project.

## Overview

The Docker setup provides:
- **Python 3.13** environment matching your local setup
- **UV package manager** for fast dependency management
- **Live code synchronization** between host and container
- **Internet connectivity** for API calls
- **Environment variable management** via .env files
- **VSCode Dev Container** support for seamless development

## Prerequisites

- Docker Desktop installed and running
- VSCode with Dev Containers extension (optional but recommended)

## Quick Start

### 1. Build the Docker Image
```bash
./docker-dev.sh build
```

### 2. Start the Development Environment
```bash
./docker-dev.sh up
```

### 3. Run Your Code
```bash
./docker-dev.sh run main.py
```

### 4. Stop the Environment
```bash
./docker-dev.sh down
```

## Available Commands

The `docker-dev.sh` script provides convenient commands for Docker operations:

| Command | Description |
|---------|-------------|
| `./docker-dev.sh build` | Build the Docker image |
| `./docker-dev.sh up` | Start the development environment |
| `./docker-dev.sh down` | Stop the development environment |
| `./docker-dev.sh restart` | Restart the development environment |
| `./docker-dev.sh logs` | View container logs |
| `./docker-dev.sh exec [command]` | Execute command in container (or open shell) |
| `./docker-dev.sh run [file.py]` | Run Python code with uv |
| `./docker-dev.sh install <packages>` | Install new packages with uv add |
| `./docker-dev.sh status` | Show container status |
| `./docker-dev.sh clean` | Remove all containers, images, and volumes |
| `./docker-dev.sh help` | Show help message |

## Development Workflow

### Live Code Development
Your local code changes are automatically synchronized with the container through volume mounting. Simply:

1. Start the container: `./docker-dev.sh up`
2. Edit code in VSCode locally
3. Run updated code: `./docker-dev.sh run main.py`

### Installing New Dependencies
```bash
# Install new packages
./docker-dev.sh install fastapi uvicorn

# The changes will be reflected in pyproject.toml and uv.lock
```

### Running Interactive Shell
```bash
# Open bash shell in container
./docker-dev.sh exec

# Or run specific commands
./docker-dev.sh exec python --version
./docker-dev.sh exec uv list
```

## VSCode Dev Container Integration

The project includes comprehensive VSCode integration with Dev Containers, debugging, tasks, and extensions.

### Quick VSCode Setup
1. **Install Extensions**: VSCode will prompt to install recommended extensions
2. **Dev Container**: Press `Cmd+Shift+P` → "Dev Containers: Reopen in Container"
3. **Start Coding**: Everything is automatically configured!

### VSCode Features Included
- ✅ **Dev Container** with Python 3.13 + FastAPI environment
- ✅ **Debug Configurations** for FastAPI development and debugging
- ✅ **Tasks** for Docker, FastAPI, testing, and package management
- ✅ **Extensions** curated for Python, Docker, and web development
- ✅ **Settings** optimized for Python development with Ruff formatting
- ✅ **Port Forwarding** automatic forwarding of FastAPI server (port 8000)

### Available VSCode Tasks
Access via `Cmd+Shift+P` → "Tasks: Run Task":
- **Docker**: Build, start, stop, logs, shell access
- **FastAPI**: Run locally or in container, test endpoints
- **Package Management**: Install dependencies, add packages
- **Production**: Build and deploy production environment
- **Code Quality**: Format and lint with Ruff

### Debug Configurations
Access via Debug panel (`Cmd+Shift+D`):
- **FastAPI Development Server** - Run with auto-reload
- **Debug FastAPI with Breakpoints** - Full debugging support
- **Docker: Attach to Container** - Debug inside container

📖 **See [VSCODE_INTEGRATION.md](./VSCODE_INTEGRATION.md) for complete VSCode setup guide**

## Environment Variables

Environment variables are managed through the `.env` file and automatically loaded into the container. The setup includes:

- Google Cloud credentials
- API keys (Google AI, OpenRouter)
- FastAPI configuration
- Development environment settings

## Network Configuration

The container has full internet access and can:
- Make API calls to external services
- Download packages and dependencies
- Access cloud services (Google Cloud, etc.)

## File Structure

```
.
├── Dockerfile                 # Container definition
├── docker-compose.yml         # Service orchestration
├── .dockerignore             # Files to exclude from build
├── docker-dev.sh             # Development script
├── .devcontainer/            # VSCode Dev Container config
│   └── devcontainer.json
├── .env                      # Environment variables
├── pyproject.toml            # Python dependencies
├── uv.lock                   # Locked dependencies
└── main.py                   # Your application code
```

## Troubleshooting

### Container Won't Start
```bash
# Check Docker is running
docker --version

# Check container status
./docker-dev.sh status

# View logs for errors
./docker-dev.sh logs
```

### Permission Issues
```bash
# Rebuild the image
./docker-dev.sh build

# Clean and restart
./docker-dev.sh clean
./docker-dev.sh build
./docker-dev.sh up
```

### Package Installation Issues
```bash
# Update dependencies in container
./docker-dev.sh exec uv sync

# Or rebuild the image
./docker-dev.sh build
```

### Internet Connectivity Issues
```bash
# Test connectivity
./docker-dev.sh exec curl -s https://www.google.com

# Check DNS resolution
./docker-dev.sh exec nslookup google.com
```

## Performance Tips

1. **Layer Caching**: Dependencies are installed before copying source code for better build performance
2. **Volume Caching**: UV cache is persisted across container restarts
3. **Minimal Base Image**: Uses Python 3.13 slim for smaller image size
4. **Optimized .dockerignore**: Excludes unnecessary files from build context

## Security Considerations

- Environment variables are loaded from `.env` file
- API keys are not hardcoded in the Dockerfile
- Container runs with appropriate permissions
- Network access is controlled through Docker's bridge network

## Advanced Usage

### Custom Commands
You can extend the `docker-dev.sh` script with custom commands for your specific workflow.

### Multiple Services
The `docker-compose.yml` can be extended to include additional services like databases, Redis, etc.

### Production Deployment
For production, create a separate Dockerfile that doesn't include development tools and volume mounts.

## Production Deployment

For production deployment with enterprise-grade security, monitoring, and scalability:

### Quick Production Setup
```bash
# 1. Configure production environment
cp .env.prod.template .env.prod
# Edit .env.prod with your production values

# 2. Deploy to production
./deploy-prod.sh build
./deploy-prod.sh start

# 3. Verify deployment
./deploy-prod.sh health
```

### Production Features
- ✅ **Security**: Non-root user, read-only filesystem, security headers
- ✅ **SSL/TLS**: HTTPS with automatic certificate management
- ✅ **Monitoring**: Health checks, structured logging, resource limits
- ✅ **Scalability**: Nginx reverse proxy, rate limiting, performance optimization
- ✅ **Backup**: Automated backup and recovery procedures

### Production Commands
```bash
./deploy-prod.sh build       # Build production image
./deploy-prod.sh start       # Start production environment
./deploy-prod.sh status      # Show status and resource usage
./deploy-prod.sh health      # Run health checks
./deploy-prod.sh logs        # View production logs
./deploy-prod.sh backup      # Create backup
./deploy-prod.sh update      # Update with backup
./deploy-prod.sh cleanup     # Remove all resources
```

📖 **See [PRODUCTION_DEPLOYMENT.md](./PRODUCTION_DEPLOYMENT.md) for complete production deployment guide**

## Getting Help

- Run `./docker-dev.sh help` for development command reference
- Run `./deploy-prod.sh help` for production command reference
- Check container logs with `./docker-dev.sh logs` or `./deploy-prod.sh logs`
- Use `./docker-dev.sh exec` to debug inside the container
