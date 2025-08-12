"""Formula translation between Polars and Google Sheets"""
import json
from typing import Dict, Any, Optional, List
import polars as pl
from pathlib import Path

class FormulaTranslator:
    """Translate between Polars operations and Google Sheets formulas"""
    
    def __init__(self):
        self.load_mappings()
    
    def load_mappings(self):
        """Load formula mappings from JSON files"""
        # Simple mappings for initial implementation
        self.simple_mappings = {
            "sum": {"sheets": "SUM", "polars": "sum"},
            "mean": {"sheets": "AVERAGE", "polars": "mean"},
            "average": {"sheets": "AVERAGE", "polars": "mean"},
            "count": {"sheets": "COUNT", "polars": "count"},
            "max": {"sheets": "MAX", "polars": "max"},
            "min": {"sheets": "MIN", "polars": "min"},
            "std": {"sheets": "STDEV", "polars": "std"},
            "var": {"sheets": "VAR", "polars": "var"},
            "median": {"sheets": "MEDIAN", "polars": "median"},
        }
        
        # Array formula mappings
        self.array_mappings = {
            "multiply": {"sheets": "ARRAYFORMULA", "operation": "*"},
            "divide": {"sheets": "ARRAYFORMULA", "operation": "/"},
            "add": {"sheets": "ARRAYFORMULA", "operation": "+"},
            "subtract": {"sheets": "ARRAYFORMULA", "operation": "-"},
        }
        
        # Lookup formula mappings
        self.lookup_mappings = {
            "vlookup": {"sheets": "VLOOKUP", "polars": "join"},
            "hlookup": {"sheets": "HLOOKUP", "polars": "join"},
            "index_match": {"sheets": "INDEX(MATCH())", "polars": "filter"},
            "xlookup": {"sheets": "XLOOKUP", "polars": "join"},
        }
        
        # Financial formula mappings
        self.financial_mappings = {
            "npv": {"sheets": "NPV", "polars": "custom"},
            "irr": {"sheets": "IRR", "polars": "custom"},
            "pmt": {"sheets": "PMT", "polars": "custom"},
            "fv": {"sheets": "FV", "polars": "custom"},
            "pv": {"sheets": "PV", "polars": "custom"},
        }
    
    def polars_to_sheets_formula(self, 
                                 operation: str,
                                 range_notation: str,
                                 **kwargs) -> str:
        """Convert Polars operation to Google Sheets formula"""
        operation_lower = operation.lower()
        
        # Check simple mappings
        if operation_lower in self.simple_mappings:
            sheets_func = self.simple_mappings[operation_lower]["sheets"]
            return f"={sheets_func}({range_notation})"
        
        # Check array mappings
        if operation_lower in self.array_mappings:
            mapping = self.array_mappings[operation_lower]
            if 'range2' in kwargs:
                return f"=ARRAYFORMULA({range_notation}{mapping['operation']}{kwargs['range2']})"
            else:
                return f"=ARRAYFORMULA({range_notation})"
        
        # Check lookup mappings
        if operation_lower in self.lookup_mappings:
            sheets_func = self.lookup_mappings[operation_lower]["sheets"]
            if operation_lower == "vlookup":
                lookup_value = kwargs.get('lookup_value', 'A1')
                col_index = kwargs.get('col_index', 2)
                exact_match = kwargs.get('exact_match', 'FALSE')
                return f"={sheets_func}({lookup_value},{range_notation},{col_index},{exact_match})"
            elif operation_lower == "index_match":
                match_value = kwargs.get('match_value', 'A1')
                match_range = kwargs.get('match_range', range_notation)
                return f"=INDEX({range_notation},MATCH({match_value},{match_range},0))"
        
        # Check financial mappings
        if operation_lower in self.financial_mappings:
            sheets_func = self.financial_mappings[operation_lower]["sheets"]
            if operation_lower == "npv":
                rate = kwargs.get('rate', 0.1)
                return f"={sheets_func}({rate},{range_notation})"
            elif operation_lower == "pmt":
                rate = kwargs.get('rate', 0.05/12)
                nper = kwargs.get('nper', 360)
                pv = kwargs.get('pv', 100000)
                return f"={sheets_func}({rate},{nper},{pv})"
        
        # Default: return as comment
        return f"'Unsupported operation: {operation}"
    
    def sheets_formula_to_polars(self, formula: str) -> Dict[str, Any]:
        """Convert Google Sheets formula to Polars operation"""
        # Remove = sign if present
        if formula.startswith('='):
            formula = formula[1:]
        
        # Parse formula to extract function and arguments
        import re
        match = re.match(r'([A-Z]+)\((.*)\)', formula)
        if not match:
            return {"error": "Invalid formula format"}
        
        func_name = match.group(1)
        args = match.group(2)
        
        # Find corresponding Polars operation
        for mapping_dict in [self.simple_mappings, self.lookup_mappings, self.financial_mappings]:
            for key, value in mapping_dict.items():
                if value.get("sheets") == func_name:
                    return {
                        "polars_operation": value.get("polars", "custom"),
                        "original_formula": formula,
                        "arguments": args
                    }
        
        return {"polars_operation": "custom", "original_formula": formula}
    
    def validate_formula_syntax(self, formula: str) -> bool:
        """Validate Google Sheets formula syntax"""
        if not formula.startswith('='):
            return False
        
        # Basic validation: check parentheses balance
        open_count = formula.count('(')
        close_count = formula.count(')')
        if open_count != close_count:
            return False
        
        # Check for valid function name
        import re
        if not re.match(r'=[A-Z][A-Z0-9_]*\(', formula):
            return False
        
        return True
    
    def generate_formula_chain(self, operations: List[Dict[str, Any]]) -> str:
        """Generate a chain of formulas for complex operations"""
        formulas = []
        
        for op in operations:
            formula = self.polars_to_sheets_formula(
                op['operation'],
                op['range'],
                **op.get('params', {})
            )
            formulas.append(formula)
        
        # For now, return the last formula
        # In future, this could create intermediate cells
        return formulas[-1] if formulas else "'No formula generated"
    
    def get_formula_documentation(self, operation: str) -> Dict[str, str]:
        """Get documentation for a specific formula"""
        docs = {
            "sum": "SUM(range) - Returns the sum of a series of numbers",
            "average": "AVERAGE(range) - Returns the average of a series of numbers",
            "count": "COUNT(range) - Returns count of numeric values",
            "vlookup": "VLOOKUP(search_key, range, index, [is_sorted]) - Vertical lookup",
            "npv": "NPV(discount_rate, cashflow1, [cashflow2, ...]) - Net present value",
        }
        
        return {
            "formula": operation,
            "description": docs.get(operation.lower(), "No documentation available"),
            "category": self._get_formula_category(operation)
        }
    
    def _get_formula_category(self, operation: str) -> str:
        """Determine the category of a formula"""
        op_lower = operation.lower()
        
        if op_lower in self.simple_mappings:
            if op_lower in ['sum', 'average', 'count', 'max', 'min']:
                return "Math"
            elif op_lower in ['std', 'var', 'median']:
                return "Statistical"
        elif op_lower in self.lookup_mappings:
            return "Lookup"
        elif op_lower in self.financial_mappings:
            return "Financial"
        elif op_lower in self.array_mappings:
            return "Array"
        
        return "Other"