"""Complex Formula Handler for advanced Google Sheets formula operations

This module handles complex formula chains, multi-step calculations, and advanced
formula patterns that require sophisticated translation between Polars and Google Sheets.
"""
import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ComplexFormulaHandler:
    """Handles complex formula chains and advanced formula patterns"""
    
    def __init__(self, mappings_dir: Optional[str] = None):
        """
        Initialize the complex formula handler.
        
        Args:
            mappings_dir: Directory containing formula mapping JSON files
        """
        if mappings_dir is None:
            # Use consolidated mappings directory
            mappings_dir = Path(__file__).parent.parent.parent / "formula_mappings"
        
        self.mappings_dir = Path(mappings_dir)
        self.formula_mappings = {}
        self._load_all_mappings()
    
    def _load_all_mappings(self):
        """Load all formula mapping JSON files"""
        mapping_files = [
            'simple_formulas.json',
            'array_formulas.json',
            'pivot_formulas.json',
            'financial_formulas.json',
            'complex_chains.json'
        ]
        
        for file_name in mapping_files:
            file_path = self.mappings_dir / file_name
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        mapping_data = json.load(f)
                    
                    # Store mappings by category
                    category = file_name.replace('.json', '')
                    self.formula_mappings[category] = mapping_data
                    logger.info(f"Loaded formula mappings from {file_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to load {file_name}: {e}")
            else:
                logger.warning(f"Mapping file not found: {file_path}")
    
    def identify_formula_pattern(self, formula: str) -> Dict[str, Any]:
        """
        Identify the pattern and complexity of a formula.
        
        Args:
            formula: Google Sheets formula to analyze
            
        Returns:
            Dictionary with pattern information
        """
        formula = formula.strip()
        if not formula.startswith('='):
            return {'type': 'value', 'complexity': 'simple', 'pattern': 'literal'}
        
        # Remove the = sign for analysis
        formula_body = formula[1:]
        
        # Check for array formulas
        if 'ARRAYFORMULA' in formula_body.upper():
            return {
                'type': 'array',
                'complexity': 'advanced',
                'pattern': 'array_formula',
                'functions': self._extract_functions(formula_body)
            }
        
        # Check for nested functions
        function_count = len(self._extract_functions(formula_body))
        nesting_depth = self._calculate_nesting_depth(formula_body)
        
        # Determine complexity
        if function_count == 1 and nesting_depth <= 1:
            complexity = 'simple'
        elif function_count <= 3 and nesting_depth <= 2:
            complexity = 'intermediate'
        elif function_count <= 6 and nesting_depth <= 3:
            complexity = 'advanced'
        else:
            complexity = 'expert'
        
        # Check for specific patterns
        pattern = self._identify_specific_pattern(formula_body)
        
        return {
            'type': 'formula',
            'complexity': complexity,
            'pattern': pattern,
            'functions': self._extract_functions(formula_body),
            'nesting_depth': nesting_depth,
            'function_count': function_count
        }
    
    def _extract_functions(self, formula: str) -> List[str]:
        """Extract all function names from a formula"""
        # Pattern to match function names (letters followed by opening parenthesis)
        function_pattern = r'\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)?)\s*\('
        functions = re.findall(function_pattern, formula.upper())
        return list(set(functions))  # Remove duplicates
    
    def _calculate_nesting_depth(self, formula: str) -> int:
        """Calculate the maximum nesting depth of parentheses"""
        max_depth = 0
        current_depth = 0
        
        for char in formula:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        
        return max_depth
    
    def _identify_specific_pattern(self, formula: str) -> str:
        """Identify specific formula patterns"""
        formula_upper = formula.upper()
        
        # Financial patterns
        if any(func in formula_upper for func in ['NPV', 'IRR', 'PMT', 'PV', 'FV']):
            return 'financial_calculation'
        
        # Lookup patterns
        if any(func in formula_upper for func in ['VLOOKUP', 'HLOOKUP', 'INDEX', 'MATCH']):
            if 'IFERROR' in formula_upper:
                return 'lookup_with_error_handling'
            else:
                return 'lookup_operation'
        
        # Conditional aggregation
        if any(func in formula_upper for func in ['SUMIFS', 'COUNTIFS', 'AVERAGEIFS']):
            return 'conditional_aggregation'
        
        # Text processing
        if any(func in formula_upper for func in ['LEFT', 'RIGHT', 'MID', 'SUBSTITUTE', 'FIND']):
            return 'text_processing'
        
        # Array operations
        if 'ARRAYFORMULA' in formula_upper:
            return 'array_operation'
        
        # Date/time operations
        if any(func in formula_upper for func in ['DATE', 'EOMONTH', 'WORKDAY', 'TODAY']):
            return 'date_time_calculation'
        
        return 'general_formula'
    
    def translate_complex_formula(self, formula: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Translate a complex formula to Polars equivalent.
        
        Args:
            formula: Google Sheets formula to translate
            context: Additional context for translation
            
        Returns:
            Dictionary with translation information
        """
        pattern_info = self.identify_formula_pattern(formula)
        
        # Try to find specific translation in mappings
        translation = self._find_formula_translation(formula, pattern_info)
        
        if translation:
            return {
                'original_formula': formula,
                'pattern_info': pattern_info,
                'polars_equivalent': translation.get('polars_equivalent'),
                'implementation_notes': translation.get('implementation_notes', []),
                'complexity_analysis': translation.get('complexity_analysis'),
                'success': True
            }
        
        # Fallback to component-wise translation
        return self._translate_by_components(formula, pattern_info, context)
    
    def _find_formula_translation(self, formula: str, pattern_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find specific translation for a formula pattern"""
        pattern_type = pattern_info.get('pattern', '')
        
        # Search through all mapping categories
        for category, mappings in self.formula_mappings.items():
            if self._search_in_mapping(formula, mappings, pattern_type):
                return self._search_in_mapping(formula, mappings, pattern_type)
        
        return None
    
    def _search_in_mapping(self, formula: str, mappings: Dict[str, Any], pattern_type: str) -> Optional[Dict[str, Any]]:
        """Search for formula in a specific mapping category"""
        for key, value in mappings.items():
            if isinstance(value, dict):
                # Check both 'sheets' and 'sheets_formula' fields for compatibility
                sheets_formula = value.get('sheets', value.get('sheets_formula', ''))
                if sheets_formula and self._formula_matches_pattern(formula, sheets_formula):
                    return value
                
                # Also check examples for matches
                if 'examples' in value and isinstance(value['examples'], dict):
                    for example_formula in value['examples'].values():
                        if self._formula_matches_pattern(formula, example_formula):
                            return value
        
        return None
    
    def _formula_matches_pattern(self, formula: str, pattern: str) -> bool:
        """Check if a formula matches a pattern template"""
        # Simple pattern matching - can be enhanced with more sophisticated logic
        formula_functions = set(self._extract_functions(formula))
        pattern_functions = set(self._extract_functions(pattern))
        
        # Check if main functions match
        return bool(formula_functions.intersection(pattern_functions))
    
    def _translate_by_components(self, formula: str, pattern_info: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Translate formula by breaking it into components"""
        functions = pattern_info.get('functions', [])
        
        component_translations = []
        polars_components = []
        
        for function in functions:
            component_info = self._get_function_translation(function)
            if component_info:
                component_translations.append({
                    'function': function,
                    'polars_equivalent': component_info.get('polars_mapping'),
                    'description': component_info.get('description')
                })
                
                if component_info.get('polars_mapping'):
                    polars_components.append(component_info['polars_mapping'])
        
        # Generate combined Polars expression
        if polars_components:
            polars_equivalent = self._combine_polars_components(polars_components, pattern_info)
        else:
            polars_equivalent = "# Complex formula requires manual translation"
        
        return {
            'original_formula': formula,
            'pattern_info': pattern_info,
            'component_translations': component_translations,
            'polars_equivalent': polars_equivalent,
            'implementation_notes': [
                "This is a component-wise translation",
                "Manual refinement may be needed for complex logic",
                f"Pattern type: {pattern_info.get('pattern', 'unknown')}"
            ],
            'success': len(component_translations) > 0
        }
    
    def _get_function_translation(self, function: str) -> Optional[Dict[str, Any]]:
        """Get translation for a specific function"""
        # Search through simple formulas first
        if 'simple_formulas' in self.formula_mappings:
            simple_mappings = self.formula_mappings['simple_formulas']
            
            def search_function(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                for key, value in data.items():
                    if isinstance(value, dict):
                        if value.get('sheets_function') == function:
                            return value
                        # Search nested
                        result = search_function(value)
                        if result:
                            return result
                return None
            
            return search_function(simple_mappings)
        
        return None
    
    def _combine_polars_components(self, components: List[str], pattern_info: Dict[str, Any]) -> str:
        """Combine multiple Polars components into a coherent expression"""
        pattern = pattern_info.get('pattern', '')
        
        if pattern == 'lookup_operation':
            # Prioritize join operations for lookups
            join_components = [c for c in components if 'join' in c.lower()]
            if join_components:
                return join_components[0]
        
        elif pattern == 'conditional_aggregation':
            # Combine filter and aggregation
            filter_components = [c for c in components if 'filter' in c.lower()]
            agg_components = [c for c in components if any(agg in c.lower() for agg in ['sum', 'count', 'mean'])]
            
            if filter_components and agg_components:
                return f"df.{filter_components[0].replace('df.', '')}.{agg_components[0].split('.')[-1]}"
        
        elif pattern == 'array_operation':
            # Use vectorized operations
            return "# Use vectorized column operations: " + " | ".join(components)
        
        # Default combination
        return " | ".join(components)
    
    def generate_formula_chain(self, chain_definition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a complex formula chain from a definition.
        
        Args:
            chain_definition: Definition of the formula chain
            
        Returns:
            Dictionary with formula chain information
        """
        chain_type = chain_definition.get('type', 'sequential')
        steps = chain_definition.get('steps', [])
        
        if chain_type == 'sequential':
            return self._generate_sequential_chain(steps)
        elif chain_type == 'parallel':
            return self._generate_parallel_chain(steps)
        elif chain_type == 'conditional':
            return self._generate_conditional_chain(steps)
        else:
            return {
                'error': f'Unknown chain type: {chain_type}',
                'success': False
            }
    
    def _generate_sequential_chain(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a sequential formula chain"""
        formula_steps = []
        polars_steps = []
        
        for i, step in enumerate(steps):
            step_name = step.get('name', f'Step_{i+1}')
            step_formula = step.get('formula', '')
            
            # Translate each step
            translation = self.translate_complex_formula(step_formula)
            
            formula_steps.append({
                'step': step_name,
                'formula': step_formula,
                'description': step.get('description', '')
            })
            
            if translation.get('success'):
                polars_steps.append({
                    'step': step_name,
                    'polars_code': translation.get('polars_equivalent'),
                    'notes': translation.get('implementation_notes', [])
                })
        
        return {
            'chain_type': 'sequential',
            'formula_steps': formula_steps,
            'polars_steps': polars_steps,
            'total_steps': len(steps),
            'success': True
        }
    
    def _generate_parallel_chain(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a parallel formula chain"""
        # Similar to sequential but can be executed independently
        return self._generate_sequential_chain(steps)  # Simplified for now
    
    def _generate_conditional_chain(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a conditional formula chain"""
        conditions = []
        
        for step in steps:
            condition = step.get('condition', '')
            formula = step.get('formula', '')
            
            if condition and formula:
                translation = self.translate_complex_formula(formula)
                
                conditions.append({
                    'condition': condition,
                    'formula': formula,
                    'polars_equivalent': translation.get('polars_equivalent') if translation.get('success') else None
                })
        
        return {
            'chain_type': 'conditional',
            'conditions': conditions,
            'success': True
        }
    
    def get_formula_documentation(self, formula_or_pattern: str) -> Dict[str, Any]:
        """
        Get documentation for a formula or pattern.
        
        Args:
            formula_or_pattern: Formula or pattern name to document
            
        Returns:
            Dictionary with documentation
        """
        # Try to identify the pattern first
        if formula_or_pattern.startswith('='):
            pattern_info = self.identify_formula_pattern(formula_or_pattern)
            pattern_name = pattern_info.get('pattern', '')
        else:
            pattern_name = formula_or_pattern
        
        # Search for documentation in mappings
        documentation = self._find_pattern_documentation(pattern_name)
        
        if documentation:
            return {
                'pattern': pattern_name,
                'documentation': documentation,
                'success': True
            }
        
        return {
            'pattern': pattern_name,
            'error': 'Documentation not found',
            'success': False
        }
    
    def _find_pattern_documentation(self, pattern_name: str) -> Optional[Dict[str, Any]]:
        """Find documentation for a specific pattern"""
        for category, mappings in self.formula_mappings.items():
            doc = self._search_documentation_recursive(mappings, pattern_name)
            if doc:
                return doc
        return None
    
    def _search_documentation_recursive(self, data: Dict[str, Any], pattern_name: str) -> Optional[Dict[str, Any]]:
        """Recursively search for documentation"""
        for key, value in data.items():
            if key == pattern_name and isinstance(value, dict):
                return value
            elif isinstance(value, dict):
                result = self._search_documentation_recursive(value, pattern_name)
                if result:
                    return result
        return None
    
    def validate_formula_chain(self, chain: List[str]) -> Dict[str, Any]:
        """
        Validate a chain of formulas for consistency and correctness.
        
        Args:
            chain: List of formulas in execution order
            
        Returns:
            Dictionary with validation results
        """
        validation_results = []
        overall_valid = True
        
        for i, formula in enumerate(chain):
            step_validation = {
                'step': i + 1,
                'formula': formula,
                'valid': True,
                'issues': []
            }
            
            # Basic syntax validation
            if not formula.startswith('='):
                step_validation['valid'] = False
                step_validation['issues'].append("Formula must start with '='")
            
            # Check parentheses balance
            if formula.count('(') != formula.count(')'):
                step_validation['valid'] = False
                step_validation['issues'].append("Unbalanced parentheses")
            
            # Check for common errors
            if '##' in formula:
                step_validation['issues'].append("Contains error indicators")
            
            if not step_validation['valid']:
                overall_valid = False
            
            validation_results.append(step_validation)
        
        return {
            'chain_valid': overall_valid,
            'step_results': validation_results,
            'total_steps': len(chain),
            'valid_steps': sum(1 for r in validation_results if r['valid'])
        }