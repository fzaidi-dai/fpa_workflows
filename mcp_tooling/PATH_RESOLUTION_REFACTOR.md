# Path Resolution Refactor: Moving Logic from Client to Server

## Overview

This refactor addresses the architectural issue where the MCP client (`mcp_tools_adk.py`) was hardcoding path prefixes like `/mcp-data/` for filesystem operations. The solution moves all path resolution logic to the server side, making the client truly generic.

## Problem Statement

### Before (Problematic Architecture)
- **Client-side hardcoding**: The `_create_tool_function` method in `mcp_tools_adk.py` contained hardcoded logic to prefix paths with `/mcp-data/`
- **Non-generic client**: The client was tightly coupled to the filesystem server's specific path structure
- **Inflexible**: Different servers couldn't have different path resolution strategies

### Code that was removed from client:
```python
# Handle path prefixing for filesystem operations
if param_name == "path" and param_value:
    if not str(param_value).startswith("/mcp-data/"):
        # Add the MCP data prefix for relative paths
        if str(param_value).startswith("data/") or str(param_value).startswith("scratch_pad/") or str(param_value).startswith("memory/"):
            param_value = f"/mcp-data/{param_value}"
        elif param_value in ["data", "scratch_pad", "memory"]:
            param_value = f"/mcp-data/{param_value}"
        elif param_value == ".":
            param_value = "/mcp-data"
        else:
            # Default to data directory for simple filenames
            param_value = f"/mcp-data/data/{param_value}"
```

## Solution Implemented

### 1. Generic MCP Client (`mcp_tools_adk.py`)
- **Removed hardcoded path prefixing**: The client now passes paths as-is to the server
- **Generic parameter handling**: All parameters are passed through without modification
- **Server-agnostic**: Can work with any MCP server regardless of its path structure

### 2. Server-Side Path Resolution (`fs_mcp_server.py`)
- **Added `resolve_path()` function**: Handles all path resolution logic
- **Configurable mapping**: Path resolution rules are controlled by server configuration
- **Backward compatibility**: Still accepts full paths starting with `/mcp-data/`

### New `resolve_path()` function:
```python
def resolve_path(input_path: str) -> str:
    """
    Resolve a relative or absolute path to a full server path.

    This function handles the path prefixing logic that was previously
    hardcoded in the MCP client, making the client generic while keeping
    server-specific path logic on the server side.
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
```

### 3. Updated All Server Functions
All filesystem functions now use `resolve_path()`:
- `read_file_tool()`
- `write_file_tool()`
- `list_directory()`
- `create_directory()`
- `delete_file()`
- `get_file_info()`

## Benefits Achieved

### 1. **Generic MCP Client**
- Can work with any MCP server
- No hardcoded assumptions about server filesystem structure
- Easier to maintain and extend

### 2. **Server-Controlled Path Logic**
- Each server can implement its own path resolution strategy
- Configuration-driven path mapping
- Server maintains full control over security and access

### 3. **Improved Flexibility**
- Different servers can have different directory structures
- Easy to modify path resolution rules without changing client code
- Better separation of concerns

### 4. **Backward Compatibility**
- Existing code using full paths continues to work
- Gradual migration path for existing implementations

## Path Resolution Examples

| Client Input | Server Resolves To | Description |
|-------------|-------------------|-------------|
| `"data/file.txt"` | `"/mcp-data/data/file.txt"` | Relative path with directory |
| `"scratch_pad/notes.txt"` | `"/mcp-data/scratch_pad/notes.txt"` | Scratch pad access |
| `"memory/cache.json"` | `"/mcp-data/memory/cache.json"` | Memory directory |
| `"simple.txt"` | `"/mcp-data/data/simple.txt"` | Simple filename defaults to data |
| `"data"` | `"/mcp-data/data"` | Directory listing |
| `"/mcp-data/data/file.txt"` | `"/mcp-data/data/file.txt"` | Full path unchanged |

## Testing

A test script (`test_path_resolution.py`) has been created to verify the functionality:

```bash
cd mcp_tooling
python test_path_resolution.py
```

This test demonstrates:
- Writing files with relative paths
- Reading files using relative paths
- Directory listing with relative paths
- Proper path resolution logging

## Migration Guide

### For Client Code
No changes needed! The client now works more generically:

```python
# This now works without hardcoded prefixes
await client.call_tool("write_file", {
    "path": "data/my_file.txt",  # Relative path
    "content": "Hello World"
})
```

### For Server Implementations
Other MCP servers can implement their own `resolve_path()` function to handle path resolution according to their specific requirements.

## Configuration

Path resolution behavior is controlled by the server's configuration in `config/filesystem.json`:

```json
{
    "filesystem": {
        "allowedDirectories": [
            "/mcp-data/data",
            "/mcp-data/scratch_pad",
            "/mcp-data/memory"
        ]
    }
}
```

## Future Enhancements

1. **Configurable Path Mapping**: Add configuration options to customize path resolution rules
2. **Multiple Root Directories**: Support for multiple root directories with different access levels
3. **Path Aliases**: Allow custom aliases for common directory paths
4. **Security Policies**: Enhanced path validation and access control rules

## Conclusion

This refactor successfully achieves the goal of making the MCP client generic while maintaining all existing functionality. The path resolution logic is now properly encapsulated on the server side, leading to better architecture, improved maintainability, and enhanced flexibility for future development.
