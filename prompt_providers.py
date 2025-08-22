#!/usr/bin/env python3
"""
Prompt Provider Functions for FPA Multi-Agent System

Functions to load and provide agent prompt configurations for ADK.
These functions read prompt files and return agent configurations with placeholders.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent

def load_prompt_file(prompt_card_dir: str, filename: str) -> Optional[str]:
    """
    Load content from a prompt file.
    
    Args:
        prompt_card_dir: Directory name under prompts/ folder
        filename: Name of the file to load (e.g., 'system_prompt.md')
        
    Returns:
        File content as string, or None if file doesn't exist
    """
    try:
        project_root = get_project_root()
        file_path = project_root / "prompts" / prompt_card_dir / filename
        
        if not file_path.exists():
            logger.warning(f"Prompt file not found: {file_path}")
            return None
            
        content = file_path.read_text(encoding='utf-8')
        logger.debug(f"Loaded prompt file: {file_path}")
        return content
        
    except Exception as e:
        logger.error(f"Failed to load prompt file '{prompt_card_dir}/{filename}': {e}")
        return None

def parse_specs_file(specs_content: str) -> Dict[str, Any]:
    """
    Parse the specs.md file content to extract agent configuration.
    
    Args:
        specs_content: Content of the specs.md file
        
    Returns:
        Dictionary containing parsed configuration
    """
    config = {}
    
    try:
        lines = specs_content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if line.startswith('- **') and '**:' in line:
                # Parse lines like: - **Model Provider**: google
                key_part = line.split('**:')[0].replace('- **', '').replace('**', '').strip()
                value_part = line.split('**:')[1].strip()
                
                # Convert key to lowercase with underscores
                key = key_part.lower().replace(' ', '_')
                
                # Parse value (remove quotes if present)
                value = value_part.strip('"\'')
                
                # Try to convert numeric values
                if value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)
                    
                config[key] = value
                
    except Exception as e:
        logger.warning(f"Error parsing specs file: {e}")
        
    return config

def root_agent_prompt_provider() -> Dict[str, Any]:
    """
    Read prompt files and return Root Agent configuration for ADK.
    
    Returns:
        Dictionary containing agent configuration with prompts and settings
    """
    try:
        # Load all prompt files
        specs_content = load_prompt_file("root_agent_card", "specs.md")
        description_content = load_prompt_file("root_agent_card", "description.md")
        system_prompt_content = load_prompt_file("root_agent_card", "system_prompt.md")
        instruction_content = load_prompt_file("root_agent_card", "instruction.md")
        
        if not all([specs_content, description_content, system_prompt_content, instruction_content]):
            raise ValueError("Missing required prompt files for Root Agent")
        
        # Parse specifications
        config = parse_specs_file(specs_content)
        
        # Build agent configuration
        agent_config = {
            "name": config.get("name", "FPA Assistant"),
            "model": {
                "provider": config.get("model_provider", "google"),
                "model": config.get("model", "gemini-2.5-flash"),
                "temperature": config.get("temperature", 0),
                "max_tokens": config.get("max_tokens", 4000),
                "top_p": config.get("top_p", 0.9)
            },
            "prompts": {
                "description": description_content,
                "system_prompt": system_prompt_content,
                "instruction": instruction_content
            },
            "tools": [
                "create_task",
                "high_level_planner", 
                "filesystem_operations",
                "dump_session_context"
            ],
            "capabilities": [
                "conversational_interface",
                "task_creation",
                "planning_coordination",
                "feedback_management"
            ]
        }
        
        logger.info("Successfully loaded Root Agent configuration")
        return agent_config
        
    except Exception as e:
        logger.error(f"Failed to load Root Agent configuration: {e}")
        return {
            "error": f"Failed to load Root Agent configuration: {e}",
            "name": "FPA Assistant (Error)",
            "model": {"provider": "google", "model": "gemini-2.5-flash", "temperature": 0}
        }

def high_level_planner_prompt_provider() -> Dict[str, Any]:
    """
    Read prompt files and return High-Level Planner configuration for ADK.
    
    Returns:
        Dictionary containing planner configuration with prompts and settings
    """
    try:
        # Load all prompt files
        specs_content = load_prompt_file("high_level_planner_card", "specs.md")
        description_content = load_prompt_file("high_level_planner_card", "description.md")
        system_prompt_content = load_prompt_file("high_level_planner_card", "system_prompt.md")
        instruction_content = load_prompt_file("high_level_planner_card", "instruction.md")
        
        if not all([specs_content, description_content, system_prompt_content, instruction_content]):
            raise ValueError("Missing required prompt files for High-Level Planner")
        
        # Parse specifications
        config = parse_specs_file(specs_content)
        
        # Build planner configuration
        planner_config = {
            "name": config.get("name", "Strategic Planner"),
            "model": {
                "provider": config.get("model_provider", "google"),
                "model": config.get("model", "gemini-2.5-flash"),
                "temperature": config.get("temperature", 0),
                "max_tokens": config.get("max_tokens", 4000),
                "top_p": config.get("top_p", 0.9)
            },
            "prompts": {
                "description": description_content,
                "system_prompt": system_prompt_content,
                "instruction": instruction_content
            },
            "tools": [
                "filesystem_operations"
            ],
            "capabilities": [
                "data_source_analysis",
                "toolset_assessment", 
                "strategic_planning",
                "feedback_integration"
            ]
        }
        
        logger.info("Successfully loaded High-Level Planner configuration")
        return planner_config
        
    except Exception as e:
        logger.error(f"Failed to load High-Level Planner configuration: {e}")
        return {
            "error": f"Failed to load High-Level Planner configuration: {e}",
            "name": "Strategic Planner (Error)",
            "model": {"provider": "google", "model": "gemini-2.5-flash", "temperature": 0}
        }

def get_all_agent_configurations() -> Dict[str, Dict[str, Any]]:
    """
    Get all agent configurations for the FP&A system.
    
    Returns:
        Dictionary containing all agent configurations keyed by agent type
    """
    return {
        "root_agent": root_agent_prompt_provider(),
        "high_level_planner": high_level_planner_prompt_provider()
    }

def validate_agent_configuration(config: Dict[str, Any]) -> bool:
    """
    Validate that an agent configuration has all required fields.
    
    Args:
        config: Agent configuration dictionary
        
    Returns:
        True if configuration is valid, False otherwise
    """
    required_fields = ["name", "model", "prompts"]
    
    for field in required_fields:
        if field not in config:
            logger.error(f"Missing required field '{field}' in agent configuration")
            return False
    
    if "error" in config:
        logger.error(f"Agent configuration contains error: {config['error']}")
        return False
        
    return True

# List of prompt provider functions for easy import
PROMPT_PROVIDERS = [
    root_agent_prompt_provider,
    high_level_planner_prompt_provider,
    get_all_agent_configurations,
    validate_agent_configuration
]