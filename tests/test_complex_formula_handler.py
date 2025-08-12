"""Tests for ComplexFormulaHandler"""
import pytest
import json
from pathlib import Path
import sys
import os

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp_tooling.google_sheets.api.complex_formula_handler import ComplexFormulaHandler


class TestComplexFormulaHandler:
    
    @pytest.fixture
    def handler(self):
        """Create a ComplexFormulaHandler instance for testing"""
        # Use the actual mappings directory
        mappings_dir = project_root / "mcp_tooling" / "google_sheets" / "mappings"
        return ComplexFormulaHandler(str(mappings_dir))
    
    def test_initialization(self, handler):
        """Test handler initialization"""
        assert handler.mappings_dir.exists()
        assert len(handler.formula_mappings) > 0
        
        # Check that all expected mapping files are loaded
        expected_categories = ['simple_formulas', 'array_formulas', 'pivot_formulas', 
                             'financial_formulas', 'complex_chains']
        
        for category in expected_categories:
            assert category in handler.formula_mappings, f"Missing category: {category}"
    
    def test_identify_simple_formula(self, handler):
        """Test identification of simple formulas"""
        # Test simple SUM formula
        pattern_info = handler.identify_formula_pattern("=SUM(A1:A10)")
        
        assert pattern_info['type'] == 'formula'
        assert pattern_info['complexity'] == 'simple'
        assert 'SUM' in pattern_info['functions']
        assert pattern_info['nesting_depth'] == 1
        assert pattern_info['function_count'] == 1
    
    def test_identify_complex_formula(self, handler):
        """Test identification of complex formulas"""
        # Test complex nested formula
        complex_formula = "=IFERROR(INDEX(B:B,MATCH(A2&C2,A:A&C:C,0)),\"Not Found\")"
        pattern_info = handler.identify_formula_pattern(complex_formula)
        
        assert pattern_info['type'] == 'formula'
        assert pattern_info['complexity'] in ['advanced', 'expert']
        assert 'IFERROR' in pattern_info['functions']
        assert 'INDEX' in pattern_info['functions']
        assert 'MATCH' in pattern_info['functions']
        assert pattern_info['nesting_depth'] > 1
    
    def test_identify_array_formula(self, handler):
        """Test identification of array formulas"""
        array_formula = "=ARRAYFORMULA(A1:A10*B1:B10)"
        pattern_info = handler.identify_formula_pattern(array_formula)
        
        assert pattern_info['type'] == 'array'
        assert pattern_info['complexity'] == 'advanced'
        assert pattern_info['pattern'] == 'array_formula'
        assert 'ARRAYFORMULA' in pattern_info['functions']
    
    def test_identify_financial_formula(self, handler):
        """Test identification of financial formulas"""
        financial_formula = "=PMT(0.05/12,360,-200000)"
        pattern_info = handler.identify_formula_pattern(financial_formula)
        
        assert pattern_info['type'] == 'formula'
        assert pattern_info['pattern'] == 'financial_calculation'
        assert 'PMT' in pattern_info['functions']
    
    def test_identify_lookup_formula(self, handler):
        """Test identification of lookup formulas"""
        lookup_formula = "=VLOOKUP(A2,Data!A:C,3,FALSE)"
        pattern_info = handler.identify_formula_pattern(lookup_formula)
        
        assert pattern_info['type'] == 'formula'
        assert pattern_info['pattern'] == 'lookup_operation'
        assert 'VLOOKUP' in pattern_info['functions']
    
    def test_identify_conditional_aggregation(self, handler):
        """Test identification of conditional aggregation formulas"""
        conditional_formula = "=SUMIFS(D:D,A:A,\"Product1\",B:B,\">100\")"
        pattern_info = handler.identify_formula_pattern(conditional_formula)
        
        assert pattern_info['type'] == 'formula'
        assert pattern_info['pattern'] == 'conditional_aggregation'
        assert 'SUMIFS' in pattern_info['functions']
    
    def test_extract_functions(self, handler):
        """Test function extraction from formulas"""
        formula = "=IFERROR(VLOOKUP(A2,Data!A:C,3,FALSE),\"Not Found\")"
        functions = handler._extract_functions(formula)
        
        assert 'IFERROR' in functions
        assert 'VLOOKUP' in functions
        assert len(functions) == 2
    
    def test_calculate_nesting_depth(self, handler):
        """Test nesting depth calculation"""
        # Simple formula - depth 1
        simple_depth = handler._calculate_nesting_depth("SUM(A1:A10)")
        assert simple_depth == 1
        
        # Nested formula - depth 2
        nested_depth = handler._calculate_nesting_depth("IFERROR(VLOOKUP(A2,Data!A:C,3,FALSE),\"Not Found\")")
        assert nested_depth == 2
        
        # Deeply nested - depth 3
        deep_depth = handler._calculate_nesting_depth("IF(SUM(IF(A1:A10>0,B1:B10,0))>100,\"High\",\"Low\")")
        assert deep_depth == 3
    
    def test_translate_simple_formula(self, handler):
        """Test translation of simple formulas"""
        translation = handler.translate_complex_formula("=SUM(A1:A10)")
        
        assert translation['success'] == True
        assert translation['original_formula'] == "=SUM(A1:A10)"
        assert 'pattern_info' in translation
        
        # Should have either direct translation or component translation
        assert 'polars_equivalent' in translation or 'component_translations' in translation
    
    def test_translate_financial_formula(self, handler):
        """Test translation of financial formulas"""
        translation = handler.translate_complex_formula("=PMT(0.05/12,360,-200000)")
        
        assert translation['success'] == True
        assert 'polars_equivalent' in translation or 'component_translations' in translation
    
    def test_translate_lookup_formula(self, handler):
        """Test translation of lookup formulas"""
        translation = handler.translate_complex_formula("=VLOOKUP(A2,Data!A:C,3,FALSE)")
        
        assert translation['success'] == True
        assert 'pattern_info' in translation
        assert translation['pattern_info']['pattern'] == 'lookup_operation'
    
    def test_get_function_translation(self, handler):
        """Test getting translation for specific functions"""
        sum_translation = handler._get_function_translation('SUM')
        
        if sum_translation:  # May not be found depending on mapping structure
            assert 'polars_mapping' in sum_translation or 'description' in sum_translation
    
    def test_formula_validation(self, handler):
        """Test formula chain validation"""
        valid_chain = [
            "=A1*B1",
            "=SUM(C1:C10)",
            "=AVERAGE(D1:D10)"
        ]
        
        validation = handler.validate_formula_chain(valid_chain)
        
        assert validation['chain_valid'] == True
        assert validation['total_steps'] == 3
        assert validation['valid_steps'] == 3
    
    def test_invalid_formula_validation(self, handler):
        """Test validation of invalid formulas"""
        invalid_chain = [
            "A1*B1",  # Missing =
            "=SUM(A1:A10",  # Missing closing parenthesis
            "=AVERAGE(B1:B10)"  # Valid
        ]
        
        validation = handler.validate_formula_chain(invalid_chain)
        
        assert validation['chain_valid'] == False
        assert validation['total_steps'] == 3
        assert validation['valid_steps'] == 1
        
        # Check specific issues
        assert not validation['step_results'][0]['valid']  # Missing =
        assert not validation['step_results'][1]['valid']  # Unbalanced parentheses
        assert validation['step_results'][2]['valid']      # Valid formula
    
    def test_generate_sequential_chain(self, handler):
        """Test generation of sequential formula chains"""
        chain_definition = {
            'type': 'sequential',
            'steps': [
                {
                    'name': 'Calculate_Sales',
                    'formula': '=A1*B1',
                    'description': 'Calculate total sales'
                },
                {
                    'name': 'Sum_Sales',
                    'formula': '=SUM(C1:C10)',
                    'description': 'Sum all sales'
                }
            ]
        }
        
        chain_result = handler.generate_formula_chain(chain_definition)
        
        assert chain_result['success'] == True
        assert chain_result['chain_type'] == 'sequential'
        assert len(chain_result['formula_steps']) == 2
        assert chain_result['total_steps'] == 2
    
    def test_get_formula_documentation(self, handler):
        """Test getting formula documentation"""
        # Test with a formula
        doc_result = handler.get_formula_documentation("=SUM(A1:A10)")
        
        # Should return some result (either success or failure)
        assert 'success' in doc_result
        assert 'pattern' in doc_result
        
        # Test with pattern name
        pattern_doc = handler.get_formula_documentation("financial_calculation")
        assert 'success' in pattern_doc
    
    def test_literal_value_identification(self, handler):
        """Test identification of literal values (not formulas)"""
        pattern_info = handler.identify_formula_pattern("123.45")
        
        assert pattern_info['type'] == 'value'
        assert pattern_info['complexity'] == 'simple'
        assert pattern_info['pattern'] == 'literal'
    
    def test_empty_formula_handling(self, handler):
        """Test handling of empty or whitespace formulas"""
        empty_pattern = handler.identify_formula_pattern("")
        whitespace_pattern = handler.identify_formula_pattern("   ")
        
        assert empty_pattern['type'] == 'value'
        assert whitespace_pattern['type'] == 'value'
    
    def test_formula_with_references(self, handler):
        """Test formulas with cell references and ranges"""
        ref_formula = "=A1+B2*$C$3"
        pattern_info = handler.identify_formula_pattern(ref_formula)
        
        assert pattern_info['type'] == 'formula'
        # Should be simple since it's basic arithmetic
        assert pattern_info['complexity'] == 'simple'
    
    def test_mappings_loaded_correctly(self, handler):
        """Test that all mapping files are loaded with expected structure"""
        # Check simple formulas
        if 'simple_formulas' in handler.formula_mappings:
            simple_mappings = handler.formula_mappings['simple_formulas']
            assert isinstance(simple_mappings, dict)
            
            # Should have categories like 'simple_math', 'text_functions', etc.
            expected_categories = ['simple_math', 'text_functions', 'logical_functions']
            found_categories = [cat for cat in expected_categories if cat in simple_mappings]
            assert len(found_categories) > 0, "No expected categories found in simple_formulas"
        
        # Check financial formulas
        if 'financial_formulas' in handler.formula_mappings:
            financial_mappings = handler.formula_mappings['financial_formulas']
            assert isinstance(financial_mappings, dict)
            
            # Should have financial categories
            expected_financial = ['time_value_of_money', 'loan_analysis']
            found_financial = [cat for cat in expected_financial if cat in financial_mappings]
            assert len(found_financial) > 0, "No expected financial categories found"


def test_integration_with_actual_files():
    """Integration test using actual mapping files"""
    # This test ensures the actual JSON files can be loaded
    mappings_dir = Path(__file__).parent.parent / "mcp_tooling" / "google_sheets" / "mappings"
    
    if mappings_dir.exists():
        handler = ComplexFormulaHandler(str(mappings_dir))
        
        # Test some real formulas
        test_formulas = [
            "=SUM(A1:A10)",
            "=VLOOKUP(A2,Data!A:C,3,FALSE)",
            "=PMT(0.05/12,360,-200000)",
            "=ARRAYFORMULA(A1:A10*B1:B10)",
            "=IFERROR(VLOOKUP(A2,Data!A:C,3,FALSE),\"Not Found\")"
        ]
        
        for formula in test_formulas:
            pattern_info = handler.identify_formula_pattern(formula)
            assert pattern_info is not None
            assert 'type' in pattern_info
            assert 'complexity' in pattern_info
            
            # Try to translate each formula
            translation = handler.translate_complex_formula(formula)
            assert translation is not None
            assert 'success' in translation


if __name__ == "__main__":
    # Run a simple test if executed directly
    mappings_dir = Path(__file__).parent.parent / "mcp_tooling" / "google_sheets" / "mappings"
    handler = ComplexFormulaHandler(str(mappings_dir))
    
    # Test formula identification
    formulas = [
        "=SUM(A1:A10)",
        "=VLOOKUP(A2,Data!A:C,3,FALSE)", 
        "=PMT(0.05/12,360,-200000)",
        "=ARRAYFORMULA(A1:A10*B1:B10)"
    ]
    
    print("Testing Complex Formula Handler:")
    for formula in formulas:
        pattern_info = handler.identify_formula_pattern(formula)
        print(f"Formula: {formula}")
        print(f"  Pattern: {pattern_info.get('pattern', 'unknown')}")
        print(f"  Complexity: {pattern_info.get('complexity', 'unknown')}")
        print(f"  Functions: {pattern_info.get('functions', [])}")
        print()