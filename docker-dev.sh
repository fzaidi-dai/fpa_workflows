#!/bin/bash

# FPA Agents Docker Development Script
# This script provides easy commands for Docker-based development

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
        print_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi
}

# Function to build the Docker images
build() {
    print_status "Building Docker images..."
    check_docker
    docker-compose build
    print_success "Docker images built successfully!"
}

# Function to start the development environment
up() {
    print_status "Starting development environment..."
    check_docker
    docker-compose up -d
    print_success "Development environment started!"
    print_status "Application will be available at: http://localhost:8000"
    print_status "MCP services will be available internally at: http://mcp-services:3001"
    print_status "Use 'docker-dev.sh logs' to view logs"
}

# Function to stop the development environment
down() {
    print_status "Stopping development environment..."
    check_docker
    docker-compose down
    print_success "Development environment stopped!"
}

# Function to restart the development environment
restart() {
    print_status "Restarting development environment..."
    down
    up
}

# Function to view logs
logs() {
    check_docker
    if [ $# -eq 0 ]; then
        print_status "Viewing logs for all services..."
        docker-compose logs -f
    else
        print_status "Viewing logs for: $*"
        docker-compose logs -f "$@"
    fi
}

# Function to execute commands in the container
exec_cmd() {
    check_docker
    if [ $# -eq 0 ]; then
        print_status "Opening interactive shell in container..."
        docker-compose exec fpa-agents /bin/bash
    else
        print_status "Executing command in container: $*"
        docker-compose exec fpa-agents "$@"
    fi
}

# Function to run Python code
run() {
    check_docker
    if [ $# -eq 0 ]; then
        print_status "Running main.py..."
        docker-compose exec fpa-agents uv run main.py
    else
        print_status "Running: uv run $*"
        docker-compose exec fpa-agents uv run "$@"
    fi
}

# Function to install new dependencies
install() {
    if [ $# -eq 0 ]; then
        print_error "Please specify package name(s) to install"
        exit 1
    fi
    check_docker
    print_status "Installing packages: $*"
    docker-compose exec fpa-agents uv add "$@"
    print_success "Packages installed successfully!"
}

# Function to show status
status() {
    check_docker
    print_status "Container status:"
    docker-compose ps
    print_status "MCP services: http://mcp-services:3001"
    print_status "Main application: http://localhost:8000"
}

# Function to clean up
clean() {
    print_warning "This will remove all containers, images, and volumes related to this project."
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Cleaning up..."
        docker-compose down -v --rmi all
        print_success "Cleanup completed!"
    else
        print_status "Cleanup cancelled."
    fi
}

# Function to show help
help() {
    echo "FPA Agents Docker Development Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build     Build the Docker images"
    echo "  up        Start the development environment"
    echo "  down      Stop the development environment"
    echo "  restart   Restart the development environment"
    echo "  logs      View container logs (use 'mcp-services' or 'fpa-agents' to view specific service)"
    echo "  exec      Execute command in container (or open shell if no command)"
    echo "  run       Run Python code with uv (defaults to main.py)"
    echo "  install   Install new packages with uv add"
    echo "  status    Show container status"
    echo "  clean     Remove all containers, images, and volumes"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 up"
    echo "  $0 run main.py"
    echo "  $0 install fastapi uvicorn"
    echo "  $0 exec python --version"
    echo "  $0 logs mcp-services"
}

# Main script logic
case "${1:-help}" in
    build)
        build
        ;;
    up)
        up
        ;;
    down)
        down
        ;;
    restart)
        restart
        ;;
    logs)
        logs
        ;;
    exec)
        shift
        exec_cmd "$@"
        ;;
    run)
        shift
        run "$@"
        ;;
    install)
        shift
        install "$@"
        ;;
    status)
        status
        ;;
    clean)
        clean
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
