#!/usr/bin/env python3
"""
Context Dumping Tool for FPA Agents

Simple context dumping function following ADK patterns.
ADK will automatically convert this function to a tool.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    """Get the project root directory."""
    # Assuming this file is in tools/ folder, parent.parent gets to project root
    return Path(__file__).parent.parent

def dump_session_context(session_state: Dict[str, Any], task_folder: str) -> Dict[str, Any]:
    """
    Dump session state to current_context.md in task folder.
    
    This function saves the current session context to a markdown file
    for recovery and continuity between agent executions.
    
    Args:
        session_state: Dictionary containing session variables and context
        task_folder: Name of the task folder where context should be saved
        
    Returns:
        Dictionary containing dump operation status and file information
    """
    project_root = get_project_root()
    
    try:
        # Construct task folder path
        task_path = project_root / "tasks" / task_folder
        
        if not task_path.exists():
            return {
                "success": False,
                "error": f"Task folder '{task_folder}' does not exist",
                "task_folder": task_folder
            }
        
        if not task_path.is_dir():
            return {
                "success": False,
                "error": f"Path '{task_folder}' is not a directory",
                "task_folder": task_folder
            }
        
        # Create current_context.md content
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        context_content = f"""# Current Session Context

## Last Updated
{timestamp}

## Session State
"""
        
        # Add session state variables
        for key, value in session_state.items():
            context_content += f"\n### {key.replace('_', ' ').title()}\n"
            
            if isinstance(value, dict):
                context_content += "```json\n"
                context_content += json.dumps(value, indent=2, ensure_ascii=False)
                context_content += "\n```\n"
            elif isinstance(value, list):
                context_content += "```json\n"
                context_content += json.dumps(value, indent=2, ensure_ascii=False)
                context_content += "\n```\n"
            elif isinstance(value, str) and len(value) > 100:
                context_content += "```\n"
                context_content += value
                context_content += "\n```\n"
            else:
                context_content += f"```\n{value}\n```\n"
        
        context_content += f"""
## Context Summary
- **Total Variables**: {len(session_state)}
- **Context Size**: {len(str(session_state))} characters
- **Generated**: {timestamp}

## Notes
This context file is automatically generated and updated by the FP&A agent system.
It provides session state recovery capability for continuous workflows.
"""
        
        # Write context file
        context_file_path = task_path / "current_context.md"
        context_file_path.write_text(context_content, encoding='utf-8')
        
        logger.info(f"Successfully dumped session context to task: {task_folder}")
        
        return {
            "success": True,
            "message": f"Successfully saved session context to '{task_folder}/current_context.md'",
            "task_folder": task_folder,
            "context_file": str(context_file_path),
            "variables_saved": len(session_state),
            "timestamp": timestamp,
            "file_size": context_file_path.stat().st_size
        }
        
    except Exception as e:
        error_msg = f"Failed to dump session context to '{task_folder}': {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "task_folder": task_folder
        }

def load_session_context(task_folder: str) -> Dict[str, Any]:
    """
    Load session context from current_context.md in task folder.
    
    This function attempts to recover session state from a previously
    saved context file.
    
    Args:
        task_folder: Name of the task folder to load context from
        
    Returns:
        Dictionary containing loaded session state or error information
    """
    project_root = get_project_root()
    
    try:
        # Construct task folder path
        task_path = project_root / "tasks" / task_folder
        context_file_path = task_path / "current_context.md"
        
        if not context_file_path.exists():
            return {
                "success": False,
                "error": f"Context file not found in task '{task_folder}'",
                "task_folder": task_folder
            }
        
        # Read context file
        context_content = context_file_path.read_text(encoding='utf-8')
        
        # This is a simplified loader - in practice, you might want to parse
        # the markdown and reconstruct the session state dictionary
        # For now, return the raw content for inspection
        
        logger.info(f"Successfully loaded session context from task: {task_folder}")
        
        return {
            "success": True,
            "message": f"Successfully loaded session context from '{task_folder}/current_context.md'",
            "task_folder": task_folder,
            "context_file": str(context_file_path),
            "content": context_content,
            "file_size": context_file_path.stat().st_size
        }
        
    except Exception as e:
        error_msg = f"Failed to load session context from '{task_folder}': {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "task_folder": task_folder
        }

# List of context functions for easy import
CONTEXT_FUNCTIONS = [
    dump_session_context,
    load_session_context
]