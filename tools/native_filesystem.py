#!/usr/bin/env python3
"""
Native Filesystem Tools for FPA Agents

Simple filesystem functions following ADK patterns.
ADK will automatically convert these functions to tools.
"""

import logging
from pathlib import Path
from typing import Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

# Define allowed directories for filesystem operations
ALLOWED_DIRECTORIES = ["data", "scratch_pad", "memory"]

def get_project_root() -> Path:
    """Get the project root directory."""
    # Assuming this file is in tools/ folder, parent.parent gets to project root
    return Path(__file__).parent.parent

def normalize_path(input_path: str) -> Path:
    """
    Normalize the input path to work with local filesystem.
    
    Args:
        input_path: Input path from user (can be relative or just filename)
    
    Returns:
        Normalized Path object
    """
    project_root = get_project_root()
    path_str = str(input_path).strip()
    
    # If path starts with one of our allowed directories, it's relative to project root
    if any(path_str.startswith(d) for d in ALLOWED_DIRECTORIES):
        return project_root / path_str
    
    # If it's just a filename or relative path, default to data directory
    if not Path(path_str).is_absolute():
        return project_root / "data" / path_str
    
    # Otherwise return as Path object
    return Path(path_str)

def validate_path(file_path: Path) -> bool:
    """
    Validate that the file path is within allowed directories.

    Args:
        file_path: The file path to validate

    Returns:
        True if path is allowed, False otherwise
    """
    try:
        project_root = get_project_root()
        resolved_path = file_path.resolve()
        
        # Check if path is within any allowed directory
        for allowed_dir in ALLOWED_DIRECTORIES:
            allowed_path = (project_root / allowed_dir).resolve()
            if str(resolved_path).startswith(str(allowed_path)):
                return True
        return False
    except Exception as e:
        logger.error(f"Path validation error for {file_path}: {e}")
        return False

def ensure_directory_exists(file_path: Path) -> None:
    """
    Ensure the parent directory of the file path exists.

    Args:
        file_path: The file path whose parent directory should exist
    """
    parent_dir = file_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

def read_file(path: str) -> Dict[str, Any]:
    """
    Read the contents of a file from the local filesystem.

    Args:
        path: Path to the file to read (supports data/, scratch_pad/, memory/ directories)

    Returns:
        Dictionary containing the file content or error message
    """
    # Normalize the path for local filesystem
    file_path = normalize_path(path)

    if not validate_path(file_path):
        return {
            "success": False,
            "error": f"Access denied: Path '{file_path}' is not in allowed directories",
            "path": str(file_path)
        }

    try:
        if not file_path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "path": str(file_path)
            }

        if not file_path.is_file():
            return {
                "success": False,
                "error": f"Path is not a file: {file_path}",
                "path": str(file_path)
            }

        content = file_path.read_text(encoding='utf-8')

        logger.info(f"Successfully read file: {file_path}")
        return {
            "success": True,
            "content": content,
            "path": str(file_path),
            "size": len(content),
            "bytes_read": len(content.encode('utf-8'))
        }

    except Exception as e:
        error_msg = f"Failed to read file '{file_path}': {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "path": str(file_path)
        }

def write_file(path: str, content: str) -> Dict[str, Any]:
    """
    Write content to a file in the local filesystem.

    Args:
        path: Path to the file to write (supports data/, scratch_pad/, memory/ directories)
        content: Content to write to the file

    Returns:
        Dictionary containing success status and file information
    """
    # Normalize the path for local filesystem
    file_path = normalize_path(path)

    if not validate_path(file_path):
        return {
            "success": False,
            "error": f"Access denied: Path '{file_path}' is not in allowed directories",
            "path": str(file_path)
        }

    try:
        # Ensure parent directory exists
        ensure_directory_exists(file_path)

        # Write the content
        file_path.write_text(content, encoding='utf-8')

        # Get file stats
        stat = file_path.stat()

        logger.info(f"Successfully wrote file: {file_path} ({len(content)} bytes)")
        return {
            "success": True,
            "message": f"Successfully wrote {len(content)} bytes to {file_path}",
            "path": str(file_path),
            "bytes_written": len(content),
            "size": stat.st_size
        }

    except Exception as e:
        error_msg = f"Failed to write file '{file_path}': {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "path": str(file_path)
        }

def list_files(path: str) -> Dict[str, Any]:
    """
    List the contents of a local directory.

    Args:
        path: Path to the directory to list (supports data/, scratch_pad/, memory/ directories)

    Returns:
        Dictionary containing directory contents and metadata
    """
    # Normalize the path for local filesystem
    dir_path = normalize_path(path)

    if not validate_path(dir_path):
        return {
            "success": False,
            "error": f"Access denied: Path '{dir_path}' is not in allowed directories",
            "path": str(dir_path)
        }

    try:
        if not dir_path.exists():
            return {
                "success": False,
                "error": f"Directory not found: {dir_path}",
                "path": str(dir_path)
            }

        if not dir_path.is_dir():
            return {
                "success": False,
                "error": f"Path is not a directory: {dir_path}",
                "path": str(dir_path)
            }

        items = []
        for item in sorted(dir_path.iterdir()):
            try:
                stat = item.stat()
                file_info = {
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": stat.st_size if item.is_file() else None,
                    "modified": str(stat.st_mtime)
                }
                items.append(file_info)
            except Exception as e:
                logger.warning(f"Could not get info for {item}: {e}")
                # Still include the item with basic info
                items.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file"
                })

        logger.info(f"Successfully listed directory: {dir_path} ({len(items)} items)")
        return {
            "success": True,
            "message": f"Listed {len(items)} items in {dir_path}",
            "path": str(dir_path),
            "items": items,
            "count": len(items)
        }

    except Exception as e:
        error_msg = f"Failed to list directory '{dir_path}': {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "path": str(dir_path)
        }

def create_directory(path: str) -> Dict[str, Any]:
    """
    Create a directory and any necessary parent directories in local filesystem.

    Args:
        path: Path to the directory to create (supports data/, scratch_pad/, memory/ directories)

    Returns:
        Dictionary containing success status and directory information
    """
    # Normalize the path for local filesystem
    dir_path = normalize_path(path)

    if not validate_path(dir_path):
        return {
            "success": False,
            "error": f"Access denied: Path '{dir_path}' is not in allowed directories",
            "path": str(dir_path)
        }

    try:
        dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Successfully created directory: {dir_path}")
        return {
            "success": True,
            "message": f"Successfully created directory: {dir_path}",
            "path": str(dir_path)
        }

    except Exception as e:
        error_msg = f"Failed to create directory '{dir_path}': {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "path": str(dir_path)
        }

def delete_file(path: str) -> Dict[str, Any]:
    """
    Delete a file from the local filesystem.

    Args:
        path: Path to the file to delete (supports data/, scratch_pad/, memory/ directories)

    Returns:
        Dictionary containing success status
    """
    # Normalize the path for local filesystem
    file_path = normalize_path(path)

    if not validate_path(file_path):
        return {
            "success": False,
            "error": f"Access denied: Path '{file_path}' is not in allowed directories",
            "path": str(file_path)
        }

    try:
        if not file_path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "path": str(file_path)
            }

        if not file_path.is_file():
            return {
                "success": False,
                "error": f"Path is not a file: {file_path}",
                "path": str(file_path)
            }

        file_path.unlink()

        logger.info(f"Successfully deleted file: {file_path}")
        return {
            "success": True,
            "message": f"Successfully deleted file: {file_path}",
            "path": str(file_path)
        }

    except Exception as e:
        error_msg = f"Failed to delete file '{file_path}': {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "path": str(file_path)
        }

def get_file_info(path: str) -> Dict[str, Any]:
    """
    Get information about a file or directory in local filesystem.

    Args:
        path: Path to the file or directory to get info for (supports data/, scratch_pad/, memory/ directories)

    Returns:
        Dictionary containing file/directory information
    """
    # Normalize the path for local filesystem
    file_path = normalize_path(path)

    if not validate_path(file_path):
        return {
            "success": False,
            "error": f"Access denied: Path '{file_path}' is not in allowed directories",
            "path": str(file_path)
        }

    try:
        if not file_path.exists():
            return {
                "success": False,
                "error": f"Path not found: {file_path}",
                "path": str(file_path)
            }

        stat = file_path.stat()

        info = {
            "name": file_path.name,
            "type": "directory" if file_path.is_dir() else "file",
            "size": stat.st_size,
            "modified": str(stat.st_mtime),
            "created": str(stat.st_ctime),
            "permissions": oct(stat.st_mode)[-3:],
            "exists": True
        }

        logger.info(f"Successfully retrieved info for: {file_path}")
        return {
            "success": True,
            "message": f"Retrieved info for: {file_path}",
            "path": str(file_path),
            "info": info
        }

    except Exception as e:
        error_msg = f"Failed to get info for '{file_path}': {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "path": str(file_path)
        }

# List of all filesystem functions for easy import
FILESYSTEM_FUNCTIONS = [
    read_file,
    write_file,
    list_files,
    create_directory,
    delete_file,
    get_file_info
]