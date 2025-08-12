# VSCode Integration Guide for FPA Agents

This guide covers the complete VSCode integration for the FPA Agents project, including Dev Containers, debugging, tasks, and extensions.

## 🚀 Quick Start with VSCode

### Option 1: Dev Container (Recommended)
1. Install the "Dev Containers" extension in VSCode
2. Open the project folder in VSCode
3. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
4. Type "Dev Containers: Reopen in Container"
5. VSCode will automatically build and connect to the development container

### Option 2: Local Development with Docker Support
1. Open the project folder in VSCode
2. Install recommended extensions (VSCode will prompt you)
3. Use the built-in tasks to manage Docker containers
4. Press `Cmd+Shift+P` and type "Tasks: Run Task" to see available options

## 📁 VSCode Configuration Files

The project includes comprehensive VSCode configuration:

```
.vscode/
├── settings.json          # Editor settings and Python configuration
├── launch.json           # Debug configurations
├── tasks.json            # Build and development tasks
└── extensions.json       # Recommended extensions

.devcontainer/
└── devcontainer.json     # Dev Container configuration
```

## 🐳 Dev Container Features

### Automatic Setup
- **Python 3.13** with UV package manager
- **FastAPI** development environment
- **All dependencies** pre-installed
- **Port forwarding** for FastAPI server (8000)
- **Git integration** with GitHub CLI
- **Extensions** automatically installed

### Container Configuration
- **Workspace**: `/app` (mapped to your project root)
- **User**: `root` (for development flexibility)
- **Environment**: Development mode with debug enabled
- **Volumes**: Live code synchronization
- **Network**: Full internet access

## 🛠️ Available Tasks

Access tasks via `Cmd+Shift+P` → "Tasks: Run Task":

### Docker Development Tasks
- **Docker: Build Development Image** - Build the development container
- **Docker: Start Development Environment** - Start the container
- **Docker: Stop Development Environment** - Stop the container
- **Docker: View Logs** - View container logs
- **Docker: Open Shell in Container** - Open bash shell in container

### FastAPI Tasks
- **FastAPI: Run in Container** - Run the FastAPI app in Docker
- **FastAPI: Run Locally** - Run the FastAPI app locally
- **Test: Health Endpoint** - Test the health endpoint
- **Test: API Status** - Test the API status endpoint

### Package Management
- **UV: Install Dependencies** - Sync dependencies with UV
- **UV: Add Package** - Add a new package (prompts for name)
- **Docker: Install Package in Container** - Install package in container

### Production Tasks
- **Production: Build Image** - Build production Docker image
- **Production: Start Environment** - Start production environment
- **Production: Health Check** - Run production health checks

### Code Quality
- **Format: Ruff Format** - Format code with Ruff
- **Lint: Ruff Check** - Check code with Ruff linter

## 🐛 Debug Configurations

Access via the Debug panel (`Cmd+Shift+D`) or Run menu:

### Available Configurations
1. **FastAPI Development Server** - Run FastAPI with auto-reload
2. **FastAPI Production Mode** - Run FastAPI in production mode
3. **Debug FastAPI with Breakpoints** - Full debugging with breakpoints
4. **Run Python Script** - Debug any Python file
5. **Docker: Attach to Container** - Attach debugger to running container

### Debugging Features
- **Breakpoints** - Set breakpoints in your code
- **Variable inspection** - Inspect variables during debugging
- **Call stack** - View the call stack
- **Watch expressions** - Monitor specific expressions
- **Debug console** - Interactive debugging console

## 🔧 Extensions

### Automatically Installed Extensions
The project includes curated extension recommendations:

#### Python Development
- **Python** - Core Python support
- **Pylance** - Advanced Python language server
- **Ruff** - Fast Python linter and formatter

#### Docker & Containers
- **Dev Containers** - Container development support
- **Docker** - Docker management and support

#### Web Development
- **JSON** - JSON language support
- **YAML** - YAML language support
- **Debugpy** - Python debugging support

#### Productivity
- **GitLens** - Enhanced Git capabilities
- **TODO Highlight** - Highlight TODO comments
- **Code Spell Checker** - Spell checking for code
- **Live Share** - Collaborative development

#### AI Assistance (Optional)
- **GitHub Copilot** - AI code completion
- **GitHub Copilot Chat** - AI chat assistance

#### Documentation
- **Markdown All in One** - Markdown support
- **Markdown Lint** - Markdown linting

## ⚙️ Settings Configuration

### Python Settings
- **Interpreter**: Automatically configured for UV virtual environment
- **Formatting**: Ruff formatter (following your preferences)
- **Linting**: Ruff linter enabled
- **Type Checking**: Basic type checking with Pylance

### Editor Settings
- **Format on Save**: Enabled
- **Auto Import Organization**: Enabled
- **Rulers**: Set at 88 characters (Python standard)
- **Tab Size**: 4 spaces
- **Trim Whitespace**: Enabled

### File Exclusions
- **__pycache__** directories hidden
- **.pytest_cache** directories hidden
- **.venv** excluded from file watching
- **node_modules** excluded from file watching

## 🚀 Development Workflow

### 1. Start Development
```bash
# Option A: Use Dev Container
Cmd+Shift+P → "Dev Containers: Reopen in Container"

# Option B: Use Tasks
Cmd+Shift+P → "Tasks: Run Task" → "Docker: Start Development Environment"
```

### 2. Run FastAPI Application
```bash
# Option A: Use Debug Configuration
F5 → Select "FastAPI Development Server"

# Option B: Use Task
Cmd+Shift+P → "Tasks: Run Task" → "FastAPI: Run in Container"

# Option C: Use Terminal
./docker-dev.sh run main.py
```

### 3. Debug Your Code
1. Set breakpoints in your Python code
2. Press `F5` and select "Debug FastAPI with Breakpoints"
3. Make requests to your API endpoints
4. Debug interactively when breakpoints are hit

### 4. Install New Packages
```bash
# Option A: Use Task
Cmd+Shift+P → "Tasks: Run Task" → "UV: Add Package"

# Option B: Use Terminal
./docker-dev.sh install package-name
```

### 5. Test Your API
```bash
# Option A: Use Tasks
Cmd+Shift+P → "Tasks: Run Task" → "Test: Health Endpoint"

# Option B: Use Browser
# Navigate to http://localhost:8000/health
```

## 🔍 Troubleshooting

### Dev Container Issues

#### Container Won't Start
1. Check Docker is running: `docker --version`
2. Rebuild container: `Cmd+Shift+P` → "Dev Containers: Rebuild Container"
3. Check logs in VSCode terminal

#### Extensions Not Loading
1. Reload window: `Cmd+Shift+P` → "Developer: Reload Window"
2. Check extension recommendations: `Cmd+Shift+P` → "Extensions: Show Recommended Extensions"

#### Port Forwarding Issues
1. Check if port 8000 is forwarded in the Ports panel
2. Try accessing http://localhost:8000 directly
3. Restart the Dev Container

### Python Environment Issues

#### Wrong Python Interpreter
1. `Cmd+Shift+P` → "Python: Select Interpreter"
2. Choose the UV virtual environment: `./.venv/bin/python`

#### Import Errors
1. Ensure dependencies are installed: Run "UV: Install Dependencies" task
2. Check PYTHONPATH in terminal: `echo $PYTHONPATH`
3. Reload window: `Cmd+Shift+P` → "Developer: Reload Window"

### Debugging Issues

#### Breakpoints Not Working
1. Ensure you're using the correct debug configuration
2. Check that the Python interpreter is correct
3. Verify the file is being executed (not imported)

#### Can't Connect to Container
1. Ensure the container is running
2. Check port forwarding configuration
3. Restart the debug session

## 🎯 Best Practices

### Development Workflow
1. **Always use Dev Container** for consistent environment
2. **Set breakpoints** for effective debugging
3. **Use tasks** for common operations
4. **Format on save** to maintain code quality
5. **Run health checks** regularly

### Code Quality
1. **Enable format on save** (already configured)
2. **Use Ruff** for linting and formatting
3. **Follow type hints** with Pylance
4. **Write docstrings** for functions and classes
5. **Use meaningful variable names**

### Container Management
1. **Rebuild containers** when dependencies change
2. **Use volume mounts** for persistent data
3. **Monitor resource usage** in Docker Desktop
4. **Clean up** unused containers regularly

## 📚 Additional Resources

### VSCode Documentation
- [Dev Containers Documentation](https://code.visualstudio.com/docs/devcontainers/containers)
- [Python in VSCode](https://code.visualstudio.com/docs/python/python-tutorial)
- [Debugging in VSCode](https://code.visualstudio.com/docs/editor/debugging)

### Docker Documentation
- [Docker Desktop](https://docs.docker.com/desktop/)
- [Docker Compose](https://docs.docker.com/compose/)

### FastAPI Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FastAPI Debugging](https://fastapi.tiangolo.com/tutorial/debugging/)

## 🆘 Getting Help

### VSCode Issues
1. Check the VSCode output panel for errors
2. Use `Cmd+Shift+P` → "Developer: Toggle Developer Tools"
3. Check the Dev Container logs

### Docker Issues
1. Check Docker Desktop status
2. View container logs: `./docker-dev.sh logs`
3. Restart Docker Desktop if needed

### Python Issues
1. Check the Python output panel
2. Verify virtual environment activation
3. Run dependency sync: `uv sync`

### FastAPI Issues
1. Check the terminal output for errors
2. Test endpoints with curl or browser
3. Review the application logs
