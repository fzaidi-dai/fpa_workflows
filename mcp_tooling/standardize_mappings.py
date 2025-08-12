#!/usr/bin/env python3
"""
Standardize Formula Mappings

This script ensures all formula mapping JSON files have:
1. Consistent key structure (all required fields present)
2. Correct implementation_status based on actual function existence
3. Proper empty values for optional fields
4. Alphabetical ordering of functions within files
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Set, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MappingStandardizer:
    """Standardizes formula mapping files"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.mappings_dir = self.base_dir / "formula_mappings"
        self.sheets_functions_file = self.base_dir / "sheets_compatible_functions.py"
        
        # Standard schema for all mapping entries
        self.standard_schema = {
            "polars": "",
            "sheets": "",
            "validation": "exact_match",
            "complexity_level": "simple",
            "description": "",
            "implementation_status": "pending",
            "syntax": "",
            "parameters": {},
            "examples": {},
            "use_cases": [],
            "category": "",
            "notes": "",
            "version_added": "",
            "polars_implementation": "",
            "sheets_function": "",
            "array_context": False,
            "helper_columns": []
        }
        
        self.implemented_functions = set()
        self.stats = {
            "files_processed": 0,
            "functions_standardized": 0,
            "implementation_status_updated": 0,
            "missing_keys_added": 0
        }
    
    def standardize_all_mappings(self):
        """Main method to standardize all mapping files"""
        logger.info("🚀 Starting Formula Mappings Standardization")
        
        # Step 1: Get implemented functions from sheets_compatible_functions.py
        self._load_implemented_functions()
        
        # Step 2: Process each mapping file
        self._process_all_mapping_files()
        
        # Step 3: Report results
        self._report_results()
        
        logger.info("✅ Standardization completed successfully!")
    
    def _load_implemented_functions(self):
        """Extract function names from sheets_compatible_functions.py"""
        logger.info("📖 Loading implemented functions...")
        
        try:
            with open(self.sheets_functions_file, 'r') as f:
                content = f.read()
            
            # Find all function definitions (def FUNCTION_NAME)
            pattern = r'def\s+([A-Z][A-Z_]*)\s*\('
            matches = re.findall(pattern, content)
            
            # Convert to lowercase for comparison
            self.implemented_functions = {func.lower() for func in matches}
            
            logger.info(f"   Found {len(self.implemented_functions)} implemented functions")
            logger.info(f"   Sample functions: {list(sorted(self.implemented_functions))[:10]}")
            
        except Exception as e:
            logger.error(f"❌ Error loading implemented functions: {e}")
            # Add known Stage 2 & 3 functions manually as fallback
            self.implemented_functions = {
                'sum', 'average', 'count', 'min', 'max', 'stdev', 'var',
                'npv', 'irr', 'pv', 'fv', 'pmt', 'nper', 'rate', 'xnpv', 'xirr',
                'pivot_sum', 'pivot_count', 'pivot_average', 'running_total'
            }
    
    def _process_all_mapping_files(self):
        """Process all JSON files in the mappings directory"""
        json_files = list(self.mappings_dir.glob("*.json"))
        logger.info(f"📁 Processing {len(json_files)} mapping files...")
        
        for json_file in json_files:
            self._process_mapping_file(json_file)
    
    def _process_mapping_file(self, file_path: Path):
        """Process a single mapping file"""
        logger.info(f"📄 Processing {file_path.name}...")
        
        try:
            # Load current data
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Standardize each function
            standardized_data = {}
            category = file_path.stem  # Use filename as category
            
            for func_name, func_data in data.items():
                if isinstance(func_data, dict):
                    standardized_func = self._standardize_function(
                        func_name, func_data, category
                    )
                    standardized_data[func_name] = standardized_func
                    self.stats["functions_standardized"] += 1
            
            # Sort functions alphabetically
            sorted_data = dict(sorted(standardized_data.items()))
            
            # Write back to file with proper formatting
            with open(file_path, 'w') as f:
                json.dump(sorted_data, f, indent=2, sort_keys=False)
            
            self.stats["files_processed"] += 1
            logger.info(f"   ✅ Standardized {len(sorted_data)} functions in {file_path.name}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
    
    def _standardize_function(self, func_name: str, func_data: Dict[str, Any], category: str) -> Dict[str, Any]:
        """Standardize a single function mapping"""
        # Start with standard schema
        standardized = self.standard_schema.copy()
        
        # Update with existing data, preserving user values
        for key, value in func_data.items():
            if key in standardized:
                standardized[key] = value
        
        # Set category if not present
        if not standardized["category"]:
            standardized["category"] = category
        
        # Determine implementation status
        old_status = standardized["implementation_status"]
        new_status = self._determine_implementation_status(func_name)
        if old_status != new_status:
            standardized["implementation_status"] = new_status
            self.stats["implementation_status_updated"] += 1
        
        # Ensure sheets_function is set if not present
        if not standardized["sheets_function"] and standardized["sheets"]:
            # Extract function name from sheets formula
            sheets_func = self._extract_sheets_function(standardized["sheets"])
            if sheets_func:
                standardized["sheets_function"] = sheets_func
        
        # Ensure polars_implementation is set if not present
        if not standardized["polars_implementation"] and standardized["polars"]:
            standardized["polars_implementation"] = standardized["polars"]
        
        # Count missing keys that were added
        for key in self.standard_schema:
            if key not in func_data:
                self.stats["missing_keys_added"] += 1
        
        return standardized
    
    def _determine_implementation_status(self, func_name: str) -> str:
        """Determine implementation status based on actual function existence"""
        # Clean function name for comparison
        clean_name = func_name.lower().strip()
        
        # Check if function exists in sheets_compatible_functions.py
        if clean_name in self.implemented_functions:
            return "completed"
        
        # Special cases for known aliases or variations
        if clean_name in ["counta", "count_non_empty"]:
            return "completed" if "counta" in self.implemented_functions else "pending"
        
        if clean_name in ["stdev", "stdev_s"]:
            return "completed" if "stdev" in self.implemented_functions else "pending"
        
        if clean_name in ["var", "var_s"]:
            return "completed" if "var" in self.implemented_functions else "pending"
        
        return "pending"
    
    def _extract_sheets_function(self, sheets_formula: str) -> str:
        """Extract the main function name from a sheets formula"""
        if not sheets_formula or not sheets_formula.startswith('='):
            return ""
        
        # Remove = and extract first function name
        formula = sheets_formula[1:]  # Remove =
        
        # Match function name pattern
        pattern = r'^([A-Z][A-Z0-9_]*)\s*\('
        match = re.match(pattern, formula)
        
        if match:
            return match.group(1)
        
        # Handle ARRAYFORMULA case
        if formula.startswith('ARRAYFORMULA'):
            return "ARRAYFORMULA"
        
        return ""
    
    def _report_results(self):
        """Report standardization results"""
        logger.info("\n📊 Standardization Results:")
        logger.info(f"   Files processed: {self.stats['files_processed']}")
        logger.info(f"   Functions standardized: {self.stats['functions_standardized']}")
        logger.info(f"   Implementation status updated: {self.stats['implementation_status_updated']}")
        logger.info(f"   Missing keys added: {self.stats['missing_keys_added']}")
        
        # Show implementation status breakdown
        completed_count = 0
        pending_count = 0
        
        for json_file in self.mappings_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                for func_data in data.values():
                    if isinstance(func_data, dict):
                        status = func_data.get("implementation_status", "pending")
                        if status == "completed":
                            completed_count += 1
                        else:
                            pending_count += 1
            except:
                pass
        
        logger.info(f"\n📈 Implementation Status Summary:")
        logger.info(f"   ✅ Completed: {completed_count} functions")
        logger.info(f"   ⏳ Pending: {pending_count} functions")
        logger.info(f"   🎯 Completion Rate: {completed_count/(completed_count+pending_count)*100:.1f}%")

if __name__ == "__main__":
    standardizer = MappingStandardizer()
    standardizer.standardize_all_mappings()