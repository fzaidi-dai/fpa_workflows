#!/bin/bash

# FPA Agents Production Deployment Script
# This script handles production deployment with proper security and monitoring

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker."
        exit 1
    fi
}

# Function to check if production environment file exists
check_prod_env() {
    if [ ! -f ".env.prod" ]; then
        print_error "Production environment file .env.prod not found!"
        print_status "Please copy .env.prod.template to .env.prod and configure it:"
        print_status "cp .env.prod.template .env.prod"
        print_status "Then edit .env.prod with your production values"
        exit 1
    fi
}

# Function to validate SSL certificates
check_ssl() {
    if [ ! -d "ssl" ]; then
        print_warning "SSL directory not found. Creating self-signed certificates for testing..."
        mkdir -p ssl
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout ssl/key.pem \
            -out ssl/cert.pem \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        print_warning "Self-signed certificates created. Replace with real certificates for production!"
    fi
}

# Function to build production image
build_prod() {
    print_status "Building production Docker image..."
    check_docker
    check_prod_env
    
    docker-compose -f docker-compose.prod.yml build --no-cache
    print_success "Production image built successfully!"
}

# Function to start production environment
start_prod() {
    print_status "Starting production environment..."
    check_docker
    check_prod_env
    check_ssl
    
    # Pull latest images
    docker-compose -f docker-compose.prod.yml pull nginx
    
    # Start services
    docker-compose -f docker-compose.prod.yml up -d
    
    print_success "Production environment started!"
    print_status "Services available at:"
    print_status "  - HTTP:  http://localhost"
    print_status "  - HTTPS: https://localhost"
    print_status "  - API:   https://localhost/api/status"
    print_status "  - Health: https://localhost/health"
}

# Function to stop production environment
stop_prod() {
    print_status "Stopping production environment..."
    check_docker
    docker-compose -f docker-compose.prod.yml down
    print_success "Production environment stopped!"
}

# Function to restart production environment
restart_prod() {
    print_status "Restarting production environment..."
    stop_prod
    start_prod
}

# Function to view production logs
logs_prod() {
    check_docker
    if [ $# -eq 0 ]; then
        docker-compose -f docker-compose.prod.yml logs -f
    else
        docker-compose -f docker-compose.prod.yml logs -f "$1"
    fi
}

# Function to show production status
status_prod() {
    check_docker
    print_status "Production environment status:"
    docker-compose -f docker-compose.prod.yml ps
    
    print_status "\nContainer resource usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
}

# Function to run health checks
health_check() {
    check_docker
    print_status "Running health checks..."
    
    # Check if containers are running
    if ! docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
        print_error "Production containers are not running!"
        return 1
    fi
    
    # Check health endpoint
    print_status "Checking health endpoint..."
    if curl -f -s http://localhost:8000/health > /dev/null; then
        print_success "Health check passed!"
    else
        print_error "Health check failed!"
        return 1
    fi
    
    # Check API endpoint
    print_status "Checking API endpoint..."
    if curl -f -s http://localhost:8000/api/status > /dev/null; then
        print_success "API check passed!"
    else
        print_error "API check failed!"
        return 1
    fi
}

# Function to backup production data
backup() {
    print_status "Creating production backup..."
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup environment files
    cp .env.prod "$BACKUP_DIR/"
    
    # Backup SSL certificates
    if [ -d "ssl" ]; then
        cp -r ssl "$BACKUP_DIR/"
    fi
    
    # Export container logs
    docker-compose -f docker-compose.prod.yml logs > "$BACKUP_DIR/container_logs.txt"
    
    print_success "Backup created in $BACKUP_DIR"
}

# Function to update production deployment
update() {
    print_status "Updating production deployment..."
    
    # Create backup first
    backup
    
    # Pull latest code (if using git)
    if [ -d ".git" ]; then
        print_status "Pulling latest code..."
        git pull
    fi
    
    # Rebuild and restart
    build_prod
    restart_prod
    
    # Run health checks
    sleep 10
    health_check
    
    print_success "Production update completed!"
}

# Function to clean up production resources
cleanup() {
    print_warning "This will remove all production containers, images, and volumes."
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Cleaning up production resources..."
        docker-compose -f docker-compose.prod.yml down -v --rmi all
        docker system prune -f
        print_success "Cleanup completed!"
    else
        print_status "Cleanup cancelled."
    fi
}

# Function to show help
help() {
    echo "FPA Agents Production Deployment Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build       Build production Docker image"
    echo "  start       Start production environment"
    echo "  stop        Stop production environment"
    echo "  restart     Restart production environment"
    echo "  logs        View production logs"
    echo "  status      Show production status and resource usage"
    echo "  health      Run health checks"
    echo "  backup      Create production backup"
    echo "  update      Update production deployment"
    echo "  cleanup     Remove all production resources"
    echo "  help        Show this help message"
    echo ""
    echo "Prerequisites:"
    echo "  - Copy .env.prod.template to .env.prod and configure"
    echo "  - Place SSL certificates in ssl/ directory"
    echo "  - Ensure Docker is running"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 start"
    echo "  $0 logs fpa-agents"
    echo "  $0 health"
}

# Main script logic
case "${1:-help}" in
    build)
        build_prod
        ;;
    start)
        start_prod
        ;;
    stop)
        stop_prod
        ;;
    restart)
        restart_prod
        ;;
    logs)
        shift
        logs_prod "$@"
        ;;
    status)
        status_prod
        ;;
    health)
        health_check
        ;;
    backup)
        backup
        ;;
    update)
        update
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        help
        exit 1
        ;;
esac
