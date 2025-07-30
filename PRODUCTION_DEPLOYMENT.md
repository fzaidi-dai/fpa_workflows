# Production Deployment Guide for FPA Agents

This guide covers deploying FPA Agents to production using Docker with enterprise-grade security, monitoring, and scalability.

## 🚀 Quick Production Deployment

### Prerequisites
- Docker Desktop installed and running
- Production environment variables configured
- SSL certificates (for HTTPS)
- Domain name configured (optional)

### 1. Configure Production Environment
```bash
# Copy the template and configure your production values
cp .env.prod.template .env.prod

# Edit .env.prod with your production credentials
nano .env.prod
```

### 2. Deploy to Production
```bash
# Build and start production environment
./deploy-prod.sh build
./deploy-prod.sh start

# Check deployment status
./deploy-prod.sh status
./deploy-prod.sh health
```

## 📋 Production Configuration

### Environment Variables (.env.prod)
```bash
# Google Cloud Configuration
GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/service-account.json
GOOGLE_PROJECT_ID=your-production-project
GOOGLE_API_KEY=your-production-api-key

# API Keys
OPEN_ROUTER_API_KEY=your-production-openrouter-key

# Application Settings
ENVIRONMENT=production
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
FASTAPI_DEBUG=false

# Security
SECRET_KEY=your-super-secure-secret-key-here
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### SSL Certificates
Place your SSL certificates in the `ssl/` directory:
```
ssl/
├── cert.pem    # SSL certificate
└── key.pem     # Private key
```

For testing, the deployment script will create self-signed certificates automatically.

## 🏗️ Production Architecture

### Container Stack
- **fpa-agents**: Main application container (Python 3.13 + FastAPI)
- **nginx**: Reverse proxy with SSL termination and security headers
- **Networks**: Isolated bridge network for container communication

### Security Features
- ✅ Non-root user execution
- ✅ Read-only root filesystem
- ✅ Security headers (HSTS, XSS protection, etc.)
- ✅ Rate limiting (10 requests/second)
- ✅ SSL/TLS encryption
- ✅ No new privileges security option

### Resource Management
- **CPU Limits**: 1.0 CPU max, 0.5 CPU reserved
- **Memory Limits**: 1GB max, 512MB reserved
- **Log Rotation**: 10MB max file size, 3 files retained

## 🛠️ Production Commands

### Deployment Management
```bash
./deploy-prod.sh build       # Build production image
./deploy-prod.sh start       # Start production environment
./deploy-prod.sh stop        # Stop production environment
./deploy-prod.sh restart     # Restart production environment
./deploy-prod.sh status      # Show container status and resources
```

### Monitoring & Maintenance
```bash
./deploy-prod.sh logs        # View all logs
./deploy-prod.sh logs nginx  # View specific service logs
./deploy-prod.sh health      # Run health checks
./deploy-prod.sh backup      # Create backup
./deploy-prod.sh update      # Update deployment (with backup)
```

### Cleanup
```bash
./deploy-prod.sh cleanup     # Remove all production resources
```

## 📊 Monitoring & Health Checks

### Built-in Health Endpoints
- **Health Check**: `https://yourdomain.com/health`
- **API Status**: `https://yourdomain.com/api/status`
- **Root Endpoint**: `https://yourdomain.com/`

### Container Health Monitoring
The production container includes automatic health checks:
- **Interval**: Every 30 seconds
- **Timeout**: 30 seconds
- **Retries**: 3 attempts
- **Start Period**: 5 seconds

### Log Management
- **Format**: JSON structured logging
- **Rotation**: Automatic log rotation (10MB files, 3 retained)
- **Location**: Container logs accessible via `docker logs`

## 🔒 Security Best Practices

### Container Security
- Runs as non-root user (`appuser`)
- Read-only root filesystem with writable tmpfs
- No new privileges allowed
- Minimal base image (Python 3.13 slim)

### Network Security
- Isolated Docker network
- Nginx reverse proxy with security headers
- Rate limiting to prevent abuse
- SSL/TLS encryption for all traffic

### Application Security
- Environment-based configuration
- Secrets management via environment variables
- Debug mode disabled in production
- Secure headers (HSTS, XSS protection, etc.)

## 🌐 Domain & DNS Configuration

### DNS Setup
Point your domain to your server's IP address:
```
A    yourdomain.com      -> YOUR_SERVER_IP
A    www.yourdomain.com  -> YOUR_SERVER_IP
```

### SSL Certificate Setup
For production, replace self-signed certificates with real ones:

#### Using Let's Encrypt (Recommended)
```bash
# Install certbot
sudo apt-get install certbot

# Generate certificates
sudo certbot certonly --standalone -d yourdomain.com -d www.yourdomain.com

# Copy certificates
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ssl/cert.pem
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ssl/key.pem
```

#### Using Custom Certificates
```bash
# Copy your certificates to the ssl directory
cp your-certificate.crt ssl/cert.pem
cp your-private-key.key ssl/key.pem
```

## 📈 Scaling & Performance

### Horizontal Scaling
To scale the application:

1. **Load Balancer**: Add a load balancer in front of multiple instances
2. **Database**: Use external database for shared state
3. **Redis**: Add Redis for session management and caching

### Vertical Scaling
Adjust resource limits in `docker-compose.prod.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'        # Increase CPU limit
      memory: 2G         # Increase memory limit
    reservations:
      cpus: '1.0'
      memory: 1G
```

### Performance Optimization
- **Nginx Caching**: Enable caching for static content
- **Gzip Compression**: Already enabled for text content
- **Connection Pooling**: Configure database connection pooling
- **CDN**: Use CDN for static assets

## 🔧 Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check logs
./deploy-prod.sh logs

# Check container status
./deploy-prod.sh status

# Rebuild image
./deploy-prod.sh build
```

#### SSL Certificate Issues
```bash
# Check certificate validity
openssl x509 -in ssl/cert.pem -text -noout

# Verify certificate matches private key
openssl x509 -noout -modulus -in ssl/cert.pem | openssl md5
openssl rsa -noout -modulus -in ssl/key.pem | openssl md5
```

#### Health Check Failures
```bash
# Test health endpoint directly
curl -f http://localhost:8000/health

# Check application logs
./deploy-prod.sh logs fpa-agents

# Restart application
./deploy-prod.sh restart
```

#### Performance Issues
```bash
# Check resource usage
./deploy-prod.sh status

# Monitor container stats
docker stats

# Check nginx access logs
./deploy-prod.sh logs nginx
```

## 🔄 Backup & Recovery

### Automated Backups
The deployment script includes backup functionality:
```bash
# Create manual backup
./deploy-prod.sh backup

# Backups are stored in backups/ directory with timestamp
```

### Backup Contents
- Environment configuration (.env.prod)
- SSL certificates
- Container logs
- Application state (if applicable)

### Recovery Process
1. Stop current deployment: `./deploy-prod.sh stop`
2. Restore configuration from backup
3. Rebuild and restart: `./deploy-prod.sh build && ./deploy-prod.sh start`
4. Verify health: `./deploy-prod.sh health`

## 📞 Support & Maintenance

### Regular Maintenance Tasks
- **Weekly**: Review logs and performance metrics
- **Monthly**: Update base images and dependencies
- **Quarterly**: Security audit and certificate renewal

### Update Process
```bash
# Automated update with backup
./deploy-prod.sh update

# Manual update process
./deploy-prod.sh backup
git pull  # If using git
./deploy-prod.sh build
./deploy-prod.sh restart
./deploy-prod.sh health
```

### Monitoring Integration
Consider integrating with:
- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **Sentry**: Error tracking
- **New Relic**: Application performance monitoring

## 🎯 Production Checklist

Before going live, ensure:

- [ ] Production environment variables configured
- [ ] SSL certificates installed and valid
- [ ] Domain DNS configured correctly
- [ ] Health checks passing
- [ ] Security headers configured
- [ ] Rate limiting tested
- [ ] Backup strategy implemented
- [ ] Monitoring setup
- [ ] Log aggregation configured
- [ ] Performance testing completed
- [ ] Security audit performed
- [ ] Documentation updated
- [ ] Team trained on deployment procedures

## 📚 Additional Resources

- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [Nginx Security Configuration](https://nginx.org/en/docs/http/configuring_https_servers.html)
- [FastAPI Production Deployment](https://fastapi.tiangolo.com/deployment/)
- [Let's Encrypt Documentation](https://letsencrypt.org/docs/)
