"""
Formula Translator for Dual-Layer Execution

Translates Polars operations to Google Sheets formulas while maintaining compatibility
and transparency in the dual-layer architecture.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl

from .data_models import FormulaMapping, OperationType, PlanStep


class FormulaTranslator:
    """Translates Polars operations to Google Sheets formulas"""
    
    def __init__(self, mappings_dir: Optional[Union[str, Path]] = None):
        """Initialize the formula translator with mapping files"""
        if mappings_dir is None:
            mappings_dir = Path(__file__).parent.parent / "formula_mappings"
        
        self.mappings_dir = Path(mappings_dir)
        self.mappings: Dict[str, FormulaMapping] = {}
        self.complex_handler = ComplexFormulaHandler()
        self._load_mappings()
    
    def _load_mappings(self):
        """Load formula mappings from JSON files"""
        mapping_files = [
            "simple_formulas.json",
            "array_formulas.json", 
            "pivot_formulas.json",
            "financial_formulas.json",
            "complex_chains.json"
        ]
        
        for file_name in mapping_files:
            file_path = self.mappings_dir / file_name
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        self._process_mapping_data(data, file_name)
                except Exception as e:
                    print(f"Warning: Could not load {file_name}: {e}")
    
    def _process_mapping_data(self, data: Dict[str, Any], source_file: str):
        """Process and store mapping data with enhanced metadata support"""
        category = source_file.replace('.json', '').replace('_', ' ').title()
        
        for operation_name, mapping_data in data.items():
            if isinstance(mapping_data, dict):
                # Use the new from_dict method for backwards compatibility
                mapping_data_copy = mapping_data.copy()
                mapping_data_copy["operation_name"] = operation_name
                
                # Set category if not present
                if "category" not in mapping_data_copy:
                    mapping_data_copy["category"] = category.lower().replace(' ', '_')
                
                mapping = FormulaMapping.from_dict(mapping_data_copy)
                self.mappings[operation_name] = mapping
    
    def translate_operation(self, 
                           polars_op: Dict[str, Any], 
                           sheet_context: Dict[str, Any]) -> str:
        """
        Convert Polars operation to Google Sheets formula
        
        Args:
            polars_op: Dictionary describing the Polars operation
            sheet_context: Context about the sheet (ranges, columns, etc.)
            
        Returns:
            Google Sheets formula string
        """
        operation_type = polars_op.get("type", "").lower()
        
        # Check if we have a direct mapping
        if operation_type in self.mappings:
            return self._apply_mapping(operation_type, polars_op, sheet_context)
        
        # Handle common operations dynamically
        if operation_type == "sum":
            return self._translate_sum(polars_op, sheet_context)
        elif operation_type == "average":
            return self._translate_average(polars_op, sheet_context)
        elif operation_type == "count":
            return self._translate_count(polars_op, sheet_context)
        elif operation_type == "max":
            return self._translate_max(polars_op, sheet_context)
        elif operation_type == "min":
            return self._translate_min(polars_op, sheet_context)
        elif operation_type == "filter":
            return self._translate_filter(polars_op, sheet_context)
        elif operation_type == "groupby_sum":
            return self._translate_groupby_sum(polars_op, sheet_context)
        elif operation_type == "vlookup":
            return self._translate_vlookup(polars_op, sheet_context)
        elif operation_type == "sumif":
            return self._translate_sumif(polars_op, sheet_context)
        else:
            # Try complex formula handler
            return self.complex_handler.handle_complex_operation(polars_op, sheet_context)
    
    def _apply_mapping(self, operation_type: str, polars_op: Dict[str, Any], sheet_context: Dict[str, Any]) -> str:
        """Apply a stored mapping to generate formula"""
        mapping = self.mappings[operation_type]
        formula_template = mapping.sheets_formula
        
        # Replace placeholders in the formula
        formula = self._replace_placeholders(formula_template, polars_op, sheet_context)
        
        return formula
    
    def _replace_placeholders(self, template: str, polars_op: Dict[str, Any], sheet_context: Dict[str, Any]) -> str:
        """Replace placeholders in formula templates"""
        # Common placeholders
        replacements = {
            "{range}": self._get_data_range(polars_op, sheet_context),
            "{column}": polars_op.get("column", "A"),
            "{criteria}": polars_op.get("criteria", ""),
            "{cell}": sheet_context.get("current_cell", "A1"),
            "{sheet}": sheet_context.get("sheet_name", "Sheet1")
        }
        
        # Add custom replacements from operation
        replacements.update(polars_op.get("replacements", {}))
        
        result = template
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, str(value))
        
        return result
    
    def _get_data_range(self, polars_op: Dict[str, Any], sheet_context: Dict[str, Any]) -> str:
        """Determine the data range for the operation"""
        # Check if range is explicitly provided
        if "range" in polars_op:
            return polars_op["range"]
        
        if "data_range" in sheet_context:
            return sheet_context["data_range"]
        
        # Default range based on column
        column = polars_op.get("column", "A")
        return f"{column}:{column}"
    
    def _translate_sum(self, polars_op: Dict[str, Any], sheet_context: Dict[str, Any]) -> str:
        """Translate SUM operation"""
        range_spec = self._get_data_range(polars_op, sheet_context)
        return f"=SUM({range_spec})"
    
    def _translate_average(self, polars_op: Dict[str, Any], sheet_context: Dict[str, Any]) -> str:
        """Translate AVERAGE operation"""
        range_spec = self._get_data_range(polars_op, sheet_context)
        return f"=AVERAGE({range_spec})"
    
    def _translate_count(self, polars_op: Dict[str, Any], sheet_context: Dict[str, Any]) -> str:
        """Translate COUNT operation"""
        range_spec = self._get_data_range(polars_op, sheet_context)
        return f"=COUNT({range_spec})"
    
    def _translate_max(self, polars_op: Dict[str, Any], sheet_context: Dict[str, Any]) -> str:
        """Translate MAX operation"""
        range_spec = self._get_data_range(polars_op, sheet_context)
        return f"=MAX({range_spec})"
    
    def _translate_min(self, polars_op: Dict[str, Any], sheet_context: Dict[str, Any]) -> str:
        """Translate MIN operation"""
        range_spec = self._get_data_range(polars_op, sheet_context)
        return f"=MIN({range_spec})"
    
    def _translate_filter(self, polars_op: Dict[str, Any], sheet_context: Dict[str, Any]) -> str:
        """Translate FILTER operation"""
        range_spec = self._get_data_range(polars_op, sheet_context)
        condition = polars_op.get("condition", {})
        
        if condition:
            column = condition.get("column", "A")
            value = condition.get("value", "")
            operator = condition.get("operator", "=")
            
            condition_range = f"{column}:{column}"
            return f"=FILTER({range_spec}, {condition_range}{operator}\"{value}\")"
        
        return f"=FILTER({range_spec}, TRUE)"
    
    def _translate_groupby_sum(self, polars_op: Dict[str, Any], sheet_context: Dict[str, Any]) -> str:
        """Translate GROUP BY SUM operation using SUMIF"""
        group_column = polars_op.get("group_by", "A")
        sum_column = polars_op.get("sum_column", "B")
        group_value = polars_op.get("group_value", sheet_context.get("current_group_value", ""))
        
        return f"=SUMIF({group_column}:{group_column}, \"{group_value}\", {sum_column}:{sum_column})"
    
    def _translate_vlookup(self, polars_op: Dict[str, Any], sheet_context: Dict[str, Any]) -> str:
        """Translate VLOOKUP operation"""
        lookup_value = polars_op.get("lookup_value", "")
        table_range = polars_op.get("table_range", "A:Z")
        col_index = polars_op.get("col_index", 2)
        range_lookup = polars_op.get("range_lookup", False)
        
        range_lookup_str = "TRUE" if range_lookup else "FALSE"
        return f"=VLOOKUP(\"{lookup_value}\", {table_range}, {col_index}, {range_lookup_str})"
    
    def _translate_sumif(self, polars_op: Dict[str, Any], sheet_context: Dict[str, Any]) -> str:
        """Translate SUMIF operation"""
        criteria_range = polars_op.get("criteria_range", "A:A")
        criteria = polars_op.get("criteria", "")
        sum_range = polars_op.get("sum_range", "B:B")
        
        return f"=SUMIF({criteria_range}, \"{criteria}\", {sum_range})"
    
    def validate_translation(self, 
                            polars_result: Any, 
                            sheets_formula: str,
                            tolerance: float = 0.001) -> bool:
        """
        Verify translation produces same result
        
        Note: This is a placeholder for actual validation logic.
        In practice, this would execute the sheets formula and compare results.
        """
        # For now, return True as we cannot execute Sheets formulas without API
        return True
    
    def get_formula_complexity(self, formula: str) -> str:
        """Determine the complexity level of a formula"""
        # Count nested functions and array operations
        nested_count = formula.count("(") - 1
        has_array = "ARRAYFORMULA" in formula.upper()
        has_multiple_functions = len(re.findall(r'[A-Z]+\(', formula)) > 1
        
        if has_array or nested_count >= 3:
            return "complex"
        elif has_multiple_functions or nested_count >= 2:
            return "moderate"
        else:
            return "simple"


class ComplexFormulaHandler:
    """Handles complex formula operations that require special logic"""
    
    def handle_complex_operation(self, polars_op: Dict[str, Any], sheet_context: Dict[str, Any]) -> str:
        """Handle operations that don't have simple mappings"""
        operation_type = polars_op.get("type", "").lower()
        
        if operation_type == "moving_average":
            return self._handle_moving_average(polars_op, sheet_context)
        elif operation_type == "cumulative_sum":
            return self._handle_cumulative_sum(polars_op, sheet_context)
        elif operation_type == "rank":
            return self._handle_rank(polars_op, sheet_context)
        elif operation_type == "percentile":
            return self._handle_percentile(polars_op, sheet_context)
        else:
            return f"=ERROR(\"Unsupported operation: {operation_type}\")"
    
    def _handle_moving_average(self, polars_op: Dict[str, Any], sheet_context: Dict[str, Any]) -> str:
        """Handle moving average calculation"""
        window = polars_op.get("window", 3)
        current_row = sheet_context.get("current_row", 1)
        column = polars_op.get("column", "A")
        
        # Create OFFSET formula for moving average
        start_row = max(1, current_row - window + 1)
        return f"=AVERAGE(OFFSET({column}{current_row},-{window-1},0,{window},1))"
    
    def _handle_cumulative_sum(self, polars_op: Dict[str, Any], sheet_context: Dict[str, Any]) -> str:
        """Handle cumulative sum calculation"""
        current_row = sheet_context.get("current_row", 1)
        column = polars_op.get("column", "A")
        
        return f"=SUM({column}$1:{column}{current_row})"
    
    def _handle_rank(self, polars_op: Dict[str, Any], sheet_context: Dict[str, Any]) -> str:
        """Handle RANK calculation"""
        current_cell = sheet_context.get("current_cell", "A1")
        data_range = polars_op.get("range", "A:A")
        descending = polars_op.get("descending", True)
        
        order = 0 if descending else 1
        return f"=RANK({current_cell}, {data_range}, {order})"
    
    def _handle_percentile(self, polars_op: Dict[str, Any], sheet_context: Dict[str, Any]) -> str:
        """Handle PERCENTILE calculation"""
        data_range = polars_op.get("range", "A:A")
        percentile = polars_op.get("percentile", 0.5)
        
        return f"=PERCENTILE({data_range}, {percentile})"
    
    # Enhanced metadata access methods
    def get_function_metadata(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Get enhanced metadata for a function"""
        if operation_name in self.mappings:
            return self.mappings[operation_name].to_dict()
        return None
    
    def get_function_examples(self, operation_name: str) -> Dict[str, str]:
        """Get examples for a function"""
        if operation_name in self.mappings:
            return self.mappings[operation_name].examples
        return {}
    
    def get_function_parameters(self, operation_name: str) -> Dict[str, Any]:
        """Get parameter definitions for a function"""
        if operation_name in self.mappings:
            return self.mappings[operation_name].parameters
        return {}
    
    def get_functions_by_category(self, category: str) -> List[str]:
        """Get all function names in a category"""
        return [name for name, mapping in self.mappings.items() 
                if mapping.category == category]
    
    def get_completed_functions(self) -> List[str]:
        """Get all functions with completed implementation"""
        return [name for name, mapping in self.mappings.items() 
                if mapping.implementation_status == "completed"]
    
    def search_functions(self, query: str) -> List[str]:
        """Search functions by name or description"""
        query_lower = query.lower()
        results = []
        
        for name, mapping in self.mappings.items():
            if (query_lower in name.lower() or 
                query_lower in mapping.description.lower() or
                any(query_lower in use_case.lower() for use_case in mapping.use_cases)):
                results.append(name)
        
        return results