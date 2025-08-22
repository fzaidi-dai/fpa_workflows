#!/usr/bin/env python3
"""
Native Metadata Analysis Tools for FPA Agents

Simple metadata analysis functions using Polars following ADK patterns.
ADK will automatically convert these functions to tools.
"""

import logging
from pathlib import Path
from typing import Dict, Any
import polars as pl

# Configure logging
logger = logging.getLogger(__name__)

# Define allowed directories for metadata operations
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
    
    # If it's just a directory name, use it relative to project root
    if not Path(path_str).is_absolute():
        return project_root / path_str
    
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

def analyze_csv_file(file_path: Path) -> Dict[str, Any]:
    """
    Analyze a single CSV file using Polars.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Dictionary containing schema and metadata information
    """
    try:
        # Read the CSV file with Polars
        df = pl.read_csv(file_path)
        
        # Get schema information (convert Polars dtypes to strings)
        schema = {col: str(dtype) for col, dtype in df.schema.items()}
        
        # Get descriptive statistics
        describe_df = df.describe()
        
        # Convert describe DataFrame to dictionary for easier consumption
        describe_dict = {}
        for row in describe_df.iter_rows(named=True):
            stat_name = row['statistic']
            # Convert all values to strings to ensure JSON serialization
            describe_dict[stat_name] = {col: str(row[col]) if row[col] is not None else None 
                                      for col in df.columns if col != 'statistic'}
        
        # Get null counts
        null_counts = {col: df.select(pl.col(col).null_count()).item() for col in df.columns}
        
        # Get unique counts for string/categorical columns
        unique_counts = {}
        for col in df.columns:
            if df[col].dtype in [pl.Utf8, pl.Categorical]:
                try:
                    unique_counts[col] = df[col].n_unique()
                except Exception as e:
                    logger.warning(f"Could not get unique count for column {col}: {e}")
                    unique_counts[col] = "unknown"
        
        # Get sample data (first 5 rows by default)
        sample_data = []
        head_df = df.head(5)
        for row in head_df.iter_rows(named=True):
            # Convert all values to strings for JSON serialization
            sample_row = {col: str(val) if val is not None else None for col, val in row.items()}
            sample_data.append(sample_row)

        # Additional metadata
        metadata = {
            "schema": schema,
            "stats": {
                "row_count": df.height,
                "column_count": df.width,
                "null_counts": null_counts,
                "describe": describe_dict,
                "unique_counts": unique_counts
            },
            "sample_data": {
                "head_rows": sample_data,
                "sample_size": len(sample_data)
            },
            "file_info": {
                "file_size_bytes": file_path.stat().st_size,
                "columns": list(df.columns),
                "dtypes": [str(dtype) for dtype in df.dtypes]
            }
        }
        
        logger.info(f"Successfully analyzed CSV file: {file_path}")
        return metadata
        
    except Exception as e:
        error_msg = f"Failed to analyze CSV file '{file_path}': {str(e)}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "file_path": str(file_path)
        }

def get_metadata_one_file(file_path: str) -> Dict[str, Any]:
    """
    Get metadata for a single CSV file using Polars analysis.

    Args:
        file_path: Path to the CSV file to analyze

    Returns:
        Dictionary containing schema and metadata for the CSV file
    """
    # Normalize the path for local filesystem
    path = normalize_path(file_path)

    if not validate_path(path):
        return {
            "success": False,
            "error": f"Access denied: Path '{path}' is not in allowed directories",
            "path": str(path)
        }

    try:
        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {path}",
                "path": str(path)
            }

        if not path.is_file():
            return {
                "success": False,
                "error": f"Path is not a file: {path}",
                "path": str(path)
            }

        if path.suffix.lower() != '.csv':
            return {
                "success": False,
                "error": f"File is not a CSV file: {path}",
                "path": str(path)
            }

        # Analyze the CSV file
        file_metadata = analyze_csv_file(path)
        
        if "error" in file_metadata:
            return {
                "success": False,
                "error": file_metadata["error"],
                "path": str(path)
            }

        logger.info(f"Successfully analyzed CSV file: {path}")
        
        return {
            "success": True,
            "message": f"Successfully analyzed CSV file: {path.name}",
            "file_path": str(path),
            "file_name": path.name,
            "metadata": file_metadata
        }

    except Exception as e:
        error_msg = f"Failed to analyze file '{path}': {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "path": str(path)
        }

def get_metadata(folder_path: str = "data") -> Dict[str, Any]:
    """
    Get consolidated metadata for all CSV files in a folder using Polars analysis.

    Args:
        folder_path: Path to the folder to analyze (supports data/, scratch_pad/, memory/ directories)

    Returns:
        Dictionary containing consolidated schema and metadata for all CSV files
    """
    # Normalize the path for local filesystem
    dir_path = normalize_path(folder_path)

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

        # Find all CSV files in the directory
        csv_files = list(dir_path.glob("*.csv"))
        
        if not csv_files:
            return {
                "success": True,
                "message": f"No CSV files found in {dir_path}",
                "folder_path": str(dir_path),
                "files_analyzed": 0,
                "metadata": {}
            }

        # Analyze each CSV file
        metadata_results = {}
        successful_analyses = 0
        
        for csv_file in csv_files:
            file_metadata = analyze_csv_file(csv_file)
            metadata_results[csv_file.name] = file_metadata
            
            if "error" not in file_metadata:
                successful_analyses += 1

        logger.info(f"Successfully analyzed {successful_analyses}/{len(csv_files)} CSV files in {dir_path}")
        
        return {
            "success": True,
            "message": f"Analyzed {successful_analyses}/{len(csv_files)} CSV files in {dir_path}",
            "folder_path": str(dir_path),
            "files_analyzed": successful_analyses,
            "total_files_found": len(csv_files),
            "metadata": metadata_results
        }

    except Exception as e:
        error_msg = f"Failed to analyze folder '{dir_path}': {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "path": str(dir_path)
        }

# List of all metadata functions for easy import
METADATA_FUNCTIONS = [
    get_metadata,
    get_metadata_one_file
]