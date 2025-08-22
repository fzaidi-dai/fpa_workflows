#!/usr/bin/env python3
"""
Create Task Tool for FPA Agents

Simple task creation function following ADK patterns.
ADK will automatically convert this function to a tool.
"""

import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    """Get the project root directory."""
    # Assuming this file is in tools/ folder, parent.parent gets to project root
    return Path(__file__).parent.parent

def create_task(task_name: str, task_description: str) -> Dict[str, Any]:
    """
    Create a task folder with initial structure for FP&A analysis.
    
    Creates:
    - /tasks/{task_name}/
    - /tasks/{task_name}/task.md
    - /tasks/{task_name}/memory/
    
    Args:
        task_name: Canonical name for the task folder (provided by Root Agent)
        task_description: Description of the task to be performed
        
    Returns:
        Dictionary containing task creation status and folder information
    """
    project_root = get_project_root()
    
    try:
        # Sanitize task name (remove special characters, spaces to underscores)
        sanitized_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in task_name.lower())
        sanitized_name = sanitized_name.strip("_")
        
        if not sanitized_name:
            return {
                "success": False,
                "error": "Task name cannot be empty after sanitization",
                "task_name": task_name
            }
        
        # Create task folder path
        task_folder = project_root / "tasks" / sanitized_name
        
        # Check if task already exists
        if task_folder.exists():
            return {
                "success": False,
                "error": f"Task '{sanitized_name}' already exists",
                "task_name": sanitized_name,
                "task_folder": str(task_folder)
            }
        
        # Create task folder structure
        task_folder.mkdir(parents=True, exist_ok=True)
        memory_folder = task_folder / "memory"
        memory_folder.mkdir(exist_ok=True)
        
        # Create task.md file with description
        task_md_content = f"""# Task: {task_name}

## Description
{task_description}

## Created
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Status
- Created: ✓
- Planning: Pending
- Execution: Pending
- Completed: Pending

## Files
- task.md: This file (task description)
- high_level_plan.md: Strategic plan (will be created by planner)
- current_context.md: Session state (updated by agents)
- memory/: Folder for plan history and feedback

## Data Sources
Will be populated based on analysis requirements.

## Notes
Add any additional notes about the task here.
"""
        
        task_md_path = task_folder / "task.md"
        task_md_path.write_text(task_md_content, encoding='utf-8')
        
        logger.info(f"Successfully created task: {sanitized_name}")
        
        return {
            "success": True,
            "message": f"Successfully created task '{sanitized_name}'",
            "task_name": sanitized_name,
            "original_name": task_name,
            "task_folder": str(task_folder),
            "task_description": task_description,
            "files_created": [
                str(task_md_path),
                str(memory_folder)
            ],
            "next_steps": [
                "Task folder created successfully",
                "Ready for high-level planning",
                "Session context will be managed automatically"
            ]
        }
        
    except Exception as e:
        error_msg = f"Failed to create task '{task_name}': {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "task_name": task_name
        }

# List of task creation functions for easy import
TASK_FUNCTIONS = [
    create_task
]