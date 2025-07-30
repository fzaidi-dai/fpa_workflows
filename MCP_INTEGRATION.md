# MCP (Model Context Protocol) Integration

This document describes the MCP integration setup for the FPA Agents project, providing filesystem access and other services through containerized MCP servers.

## Overview

The MCP integration consists of:
- **MCP Services Container**: Runs MCP filesystem server for file operations
- **Main Application Container**: FPA Agents application with MCP client capabilities
- **Shared Data Volumes**: Persistent storage for data, scratch_pad, and memory

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network                           │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │  MCP Services   │    │      FPA Agents                 │ │
│  │  Container      │    │      Container                  │ │
│  │                 │    │                                 │ │
│  │ - Filesystem    │◄───┤ - MCP Client                    │ │
│  │   Server        │    │ - FastAPI App                   │ │
│  │ - Port 3001     │    │ - Port 8000                     │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
│           │                           │                     │
│           └───────────┬───────────────┘                     │
│                       │                                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Shared Volumes                             │ │
│  │  - ./data:/mcp-data/data                               │ │
│  │  - ./scratch_pad:/mcp-data/scratch_pad                 │ │
│  │  - ./memory:/mcp-data/memory                           │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
fpa_agents/
├── mcp/                          # MCP services configuration
│   ├── Dockerfile               # MCP services container
│   ├── config/
│   │   └── filesystem.json      # Filesystem server config
│   ├── index.js                 # MCP filesystem server
│   └── docker-compose.mcp.yml   # Standalone MCP compose
├── data/                        # Persistent data storage
├── scratch_pad/                 # Temporary workspace
├── memory/                      # Agent memory storage
├── tools/                       # Tool definitions
├── prompts/                     # Prompt templates
├── agents/                      # Agent configurations
├── ui/                          # User interface
├── docker-compose.yml           # Development environment
├── docker-compose.prod.yml      # Production environment
└── docker-dev.sh               # Development script
```

## Configuration Files

### MCP Filesystem Configuration (`mcp/config/filesystem.json`)

```json
{
    "server": {
        "port": 3001,
        "host": "0.0.0.0"
    },
    "filesystem": {
        "allowedDirectories": [
            "/mcp-data/data",
            "/mcp-data/scratch_pad",
            "/mcp-data/memory"
        ],
        "readOnly": false
    },
    "logging": {
        "level": "info",
        "format": "json"
    }
}
```

### Docker Compose Services

#### MCP Services Container
- **Image**: Built from `mcp/Dockerfile`
- **Port**: 3001 (internal only)
- **Volumes**: Mounts data directories with read/write access
- **Health Check**: HTTP endpoint monitoring
- **Security**: Non-root user, resource limits

#### Main Application Container
- **Image**: Built from main `Dockerfile`
- **Port**: 8000 (exposed to host)
- **Volumes**: Source code, data directories
- **Environment**: Development/production variables

## Usage

### Development Environment

```bash
# Build all containers
./docker-dev.sh build

# Start development environment
./docker-dev.sh up

# View logs for all services
./docker-dev.sh logs

# View logs for specific service
./docker-dev.sh logs mcp-services
./docker-dev.sh logs fpa-agents

# Check container status
./docker-dev.sh status

# Stop environment
./docker-dev.sh down
```

### Production Environment

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Start production environment
docker-compose -f docker-compose.prod.yml up -d

# View production logs
docker-compose -f docker-compose.prod.yml logs -f
```

### Standalone MCP Services

```bash
# Run only MCP services
cd mcp
docker-compose -f docker-compose.mcp.yml up -d
```

## MCP Server Endpoints

### Filesystem Server
- **URL**: `http://mcp-services:3001` (internal)
- **Health Check**: `http://mcp-services:3001/health`
- **Capabilities**:
  - File read/write operations
  - Directory listing
  - File metadata access
  - Path validation and security

### Available Operations
- `read_file`: Read file contents
- `write_file`: Write file contents
- `list_directory`: List directory contents
- `create_directory`: Create new directories
- `delete_file`: Delete files
- `file_info`: Get file metadata

## Security Features

### Container Security
- Non-root user execution
- Read-only root filesystem (production)
- Resource limits (CPU/Memory)
- Network isolation
- Health monitoring

### Filesystem Security
- Restricted directory access
- Path traversal protection
- Permission validation
- Audit logging

## Monitoring and Logging

### Health Checks
- MCP services health endpoint
- Container restart policies
- Resource usage monitoring

### Logging
- JSON structured logs
- Log rotation (10MB, 3 files)
- Centralized log collection
- Debug/info/error levels

## Troubleshooting

### Common Issues

1. **MCP Services Not Starting**
   ```bash
   # Check container logs
   ./docker-dev.sh logs mcp-services

   # Verify health status
   docker exec fpa-mcp-services curl http://localhost:3001/health
   ```

2. **Permission Denied Errors**
   ```bash
   # Check volume permissions
   ls -la data/ scratch_pad/ memory/

   # Fix permissions if needed
   sudo chown -R $USER:$USER data/ scratch_pad/ memory/
   ```

3. **Network Connectivity Issues**
   ```bash
   # Test internal connectivity
   docker exec fpa-agents-dev curl http://mcp-services:3001/health

   # Check network configuration
   docker network ls
   docker network inspect fpa_agents_fpa-network
   ```

### Debug Mode

Enable debug logging by setting environment variables:

```bash
# In .env file
MCP_LOG_LEVEL=debug
FASTAPI_DEBUG=true
```

## Development Guidelines

### Adding New MCP Servers
1. Create server directory under `mcp/`
2. Add Dockerfile and configuration
3. Update docker-compose.yml
4. Add health checks and monitoring
5. Update documentation

### Testing MCP Integration
1. Unit tests for MCP client
2. Integration tests with containers
3. End-to-end workflow tests
4. Performance benchmarks

## Performance Considerations

### Resource Allocation
- MCP Services: 256MB RAM, 0.5 CPU
- Main Application: 1GB RAM, 1.0 CPU
- Adjust based on workload requirements

### Volume Performance
- Use local volumes for development
- Consider network storage for production
- Monitor I/O performance metrics

## Backup and Recovery

### Data Persistence
- Regular backups of data volumes
- Version control for configurations
- Container image versioning
- Disaster recovery procedures

### Backup Script Example
```bash
#!/bin/bash
# Backup data volumes
docker run --rm -v fpa_agents_data-volume:/data -v $(pwd):/backup alpine tar czf /backup/data-backup-$(date +%Y%m%d).tar.gz /data
```

## Future Enhancements

### Planned Features
- Additional MCP servers (database, API)
- Advanced security policies
- Distributed deployment support
- Monitoring dashboard
- Automated scaling

### Integration Roadmap
- Kubernetes deployment
- Service mesh integration
- Advanced observability
- Multi-tenant support
