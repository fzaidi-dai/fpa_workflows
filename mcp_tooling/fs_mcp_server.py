#!/usr/bin/env python3
"""
Filesystem MCP Server using FastMCP

This server provides file system operations through the Model Context Protocol (MCP).
It includes tools for reading, writing, and listing files with proper security controls.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server and FastAPI app
mcp = FastMCP("Filesystem Server")
app = FastAPI(title="Filesystem MCP Server", version="1.0.0")

def load_config():
    """Load server configuration from filesystem.json."""
    import json
    from pathlib import Path

    config_path = Path(__file__).parent / "config" / "filesystem.json"
    with open(config_path, 'r') as f:
        return json.load(f)

# Load configuration
CONFIG = load_config()
ALLOWED_DIRECTORIES = CONFIG["filesystem"]["allowedDirectories"]

class FileReadRequest(BaseModel):
    """Request model for reading files."""
    path: str = Field(..., description="Path to the file to read")

class FileWriteRequest(BaseModel):
    """Request model for writing files."""
    path: str = Field(..., description="Path to the file to write")
    content: str = Field(..., description="Content to write to the file")

class DirectoryListRequest(BaseModel):
    """Request model for listing directories."""
    path: str = Field(..., description="Path to the directory to list")

class FileInfo(BaseModel):
    """Model for file information."""
    name: str
    type: str  # 'file' or 'directory'
    size: Optional[int] = None
    modified: Optional[str] = None

def resolve_path(input_path: str) -> str:
    """
    Resolve a relative or absolute path to a full server path.

    This function handles the path prefixing logic that was previously
    hardcoded in the MCP client, making the client generic while keeping
    server-specific path logic on the server side.

    Args:
        input_path: Input path from client (relative or absolute)

    Returns:
        Resolved full path with appropriate prefixes
    """
    # If already a full path starting with /mcp-data/, return as-is
    if str(input_path).startswith("/mcp-data/"):
        return input_path

    # Handle relative paths by applying appropriate prefixes
    input_str = str(input_path)

    # Handle directory-specific prefixes
    if input_str.startswith("data/"):
        return f"/mcp-data/{input_str}"
    elif input_str.startswith("scratch_pad/"):
        return f"/mcp-data/{input_str}"
    elif input_str.startswith("memory/"):
        return f"/mcp-data/{input_str}"
    elif input_str in ["data", "scratch_pad", "memory"]:
        return f"/mcp-data/{input_str}"
    elif input_str == ".":
        return "/mcp-data"
    else:
        # Default to data directory for simple filenames
        return f"/mcp-data/data/{input_str}"


def validate_path(file_path: str) -> bool:
    """
    Validate that the file path is within allowed directories.

    Args:
        file_path: The file path to validate

    Returns:
        True if path is allowed, False otherwise
    """
    try:
        resolved_path = Path(file_path).resolve()
        return any(
            str(resolved_path).startswith(str(Path(allowed_dir).resolve()))
            for allowed_dir in ALLOWED_DIRECTORIES
        )
    except Exception as e:
        logger.error(f"Path validation error for {file_path}: {e}")
        return False

def ensure_directory_exists(file_path: str) -> None:
    """
    Ensure the parent directory of the file path exists.

    Args:
        file_path: The file path whose parent directory should exist
    """
    parent_dir = Path(file_path).parent
    parent_dir.mkdir(parents=True, exist_ok=True)

def read_file_tool(path: str) -> Dict[str, Any]:
    """
    Read the contents of a file.

    Args:
        path: Path to the file to read

    Returns:
        Dictionary containing the file content or error message
    """
    # Resolve the path using server-side logic
    resolved_path = resolve_path(path)

    if not validate_path(resolved_path):
        return {
            "success": False,
            "error": f"Access denied: Path '{resolved_path}' is not in allowed directories"
        }

    try:
        file_path = Path(resolved_path)

        if not file_path.exists():
            return {
                "success": False,
                "error": f"File not found: {resolved_path}"
            }

        if not file_path.is_file():
            return {
                "success": False,
                "error": f"Path is not a file: {resolved_path}"
            }

        content = file_path.read_text(encoding='utf-8')

        logger.info(f"Successfully read file: {resolved_path} (original: {path})")
        return {
            "success": True,
            "content": content,
            "path": resolved_path,
            "size": len(content)
        }

    except Exception as e:
        error_msg = f"Failed to read file '{resolved_path}': {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg
        }

def write_file_tool(path: str, content: str) -> Dict[str, Any]:
    """
    Write content to a file.

    Args:
        path: Path to the file to write
        content: Content to write to the file

    Returns:
        Dictionary containing success status and file information
    """
    # Resolve the path using server-side logic
    resolved_path = resolve_path(path)

    if not validate_path(resolved_path):
        return {
            "success": False,
            "error": f"Access denied: Path '{resolved_path}' is not in allowed directories"
        }

    try:
        file_path = Path(resolved_path)

        # Ensure parent directory exists
        ensure_directory_exists(resolved_path)

        # Write the content
        file_path.write_text(content, encoding='utf-8')

        # Get file stats
        stat = file_path.stat()

        logger.info(f"Successfully wrote file: {resolved_path} (original: {path}) ({len(content)} bytes)")
        return {
            "success": True,
            "path": resolved_path,
            "bytes_written": len(content),
            "size": stat.st_size,
            "message": f"Successfully wrote {len(content)} bytes to {resolved_path}"
        }

    except Exception as e:
        error_msg = f"Failed to write file '{resolved_path}': {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg
        }

@mcp.tool()
def list_directory(request: DirectoryListRequest) -> Dict[str, Any]:
    """
    List the contents of a directory.

    Args:
        request: Directory list request containing the path

    Returns:
        Dictionary containing directory contents and metadata

    Raises:
        ValueError: If path is not allowed or directory cannot be listed
    """
    # Resolve the path using server-side logic
    resolved_path = resolve_path(request.path)

    if not validate_path(resolved_path):
        raise ValueError(f"Access denied: Path '{resolved_path}' is not in allowed directories")

    try:
        dir_path = Path(resolved_path)

        if not dir_path.exists():
            raise ValueError(f"Directory not found: {request.path}")

        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {request.path}")

        items = []
        for item in sorted(dir_path.iterdir()):
            try:
                stat = item.stat()
                file_info = FileInfo(
                    name=item.name,
                    type="directory" if item.is_dir() else "file",
                    size=stat.st_size if item.is_file() else None,
                    modified=str(stat.st_mtime)
                )
                items.append(file_info.dict())
            except Exception as e:
                logger.warning(f"Could not get info for {item}: {e}")
                # Still include the item with basic info
                items.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file"
                })

        logger.info(f"Successfully listed directory: {request.path} ({len(items)} items)")
        return {
            "success": True,
            "path": request.path,
            "items": items,
            "count": len(items)
        }

    except Exception as e:
        error_msg = f"Failed to list directory '{request.path}': {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

@mcp.tool()
def create_directory(path: str) -> Dict[str, Any]:
    """
    Create a directory and any necessary parent directories.

    Args:
        path: Path to the directory to create

    Returns:
        Dictionary containing success status and directory information

    Raises:
        ValueError: If path is not allowed or directory cannot be created
    """
    # Resolve the path using server-side logic
    resolved_path = resolve_path(path)

    if not validate_path(resolved_path):
        raise ValueError(f"Access denied: Path '{resolved_path}' is not in allowed directories")

    try:
        dir_path = Path(resolved_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Successfully created directory: {resolved_path} (original: {path})")
        return {
            "success": True,
            "path": resolved_path,
            "message": f"Successfully created directory: {resolved_path}"
        }

    except Exception as e:
        error_msg = f"Failed to create directory '{resolved_path}': {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

@mcp.tool()
def delete_file(path: str) -> Dict[str, Any]:
    """
    Delete a file.

    Args:
        path: Path to the file to delete

    Returns:
        Dictionary containing success status

    Raises:
        ValueError: If path is not allowed or file cannot be deleted
    """
    # Resolve the path using server-side logic
    resolved_path = resolve_path(path)

    if not validate_path(resolved_path):
        raise ValueError(f"Access denied: Path '{resolved_path}' is not in allowed directories")

    try:
        file_path = Path(resolved_path)

        if not file_path.exists():
            raise ValueError(f"File not found: {resolved_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {resolved_path}")

        file_path.unlink()

        logger.info(f"Successfully deleted file: {resolved_path} (original: {path})")
        return {
            "success": True,
            "path": resolved_path,
            "message": f"Successfully deleted file: {resolved_path}"
        }

    except Exception as e:
        error_msg = f"Failed to delete file '{resolved_path}': {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

@mcp.tool()
def get_file_info(path: str) -> Dict[str, Any]:
    """
    Get information about a file or directory.

    Args:
        path: Path to the file or directory

    Returns:
        Dictionary containing file/directory information

    Raises:
        ValueError: If path is not allowed or info cannot be retrieved
    """
    # Resolve the path using server-side logic
    resolved_path = resolve_path(path)

    if not validate_path(resolved_path):
        raise ValueError(f"Access denied: Path '{resolved_path}' is not in allowed directories")

    try:
        file_path = Path(resolved_path)

        if not file_path.exists():
            raise ValueError(f"Path not found: {resolved_path}")

        stat = file_path.stat()

        info = {
            "success": True,
            "path": resolved_path,
            "name": file_path.name,
            "type": "directory" if file_path.is_dir() else "file",
            "size": stat.st_size,
            "modified": str(stat.st_mtime),
            "created": str(stat.st_ctime),
            "permissions": oct(stat.st_mode)[-3:],
            "exists": True
        }

        logger.info(f"Successfully retrieved info for: {resolved_path} (original: {path})")
        return info

    except Exception as e:
        error_msg = f"Failed to get info for '{resolved_path}': {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

# FastAPI HTTP endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Filesystem MCP Server",
        "version": "1.0.0",
        "allowed_directories": ALLOWED_DIRECTORIES
    }

@app.post("/fs_mcp")
async def fs_mcp_endpoint(request_data: dict):
    """Filesystem MCP tool endpoint."""
    try:
        method = request_data.get("method")
        params = request_data.get("params", {})

        if method == "tools/list":
            # Return tools configuration from JSON file
            return {
                "tools": CONFIG.get("tools", [])
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if tool_name == "read_file":
                result = read_file_tool(arguments.get("path"))
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }
                    ]
                }

            elif tool_name == "write_file":
                result = write_file_tool(
                    arguments.get("path"),
                    arguments.get("content")
                )
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }
                    ]
                }

            elif tool_name == "list_directory":
                try:
                    request = DirectoryListRequest(path=arguments.get("path"))
                    result = list_directory(request)
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, indent=2)
                            }
                        ]
                    }
                except Exception as e:
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps({"success": False, "error": str(e)}, indent=2)
                            }
                        ]
                    }

            else:
                raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")

        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {method}")

    except Exception as e:
        logger.error(f"Error handling MCP request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/sse")
async def sse_endpoint():
    """SSE endpoint for MCP communication."""
    return {"message": "MCP SSE endpoint - use MCP client to connect"}

def setup_data_directories():
    """Create the allowed data directories if they don't exist."""
    for directory in ALLOWED_DIRECTORIES:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")

# Create MCP tools without mounting - we'll handle requests manually

async def main():
    """Main entry point for the MCP server."""
    logger.info("Starting Filesystem MCP Server...")

    # Setup data directories
    setup_data_directories()

    # Log available tools
    logger.info("Available tools:")
    for tool_name in ["read_file", "write_file", "list_directory", "create_directory", "delete_file", "get_file_info"]:
        logger.info(f"  - {tool_name}")

    logger.info(f"Allowed directories: {ALLOWED_DIRECTORIES}")

    # Run the FastAPI server with uvicorn
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=int(os.getenv("FILESYSTEM_SERVER_PORT", 3001)),
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
