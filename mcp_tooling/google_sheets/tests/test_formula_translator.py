"""Unit tests for FormulaTranslator module"""
import pytest
from mcp_tooling.google_sheets.api.formula_translator import FormulaTranslator

class TestFormulaTranslator:
    """Test cases for FormulaTranslator functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.translator = FormulaTranslator()
    
    def test_simple_formula_mappings(self):
        """Test basic formula translations"""
        # Test SUM
        result = self.translator.polars_to_sheets_formula('sum', 'A1:A10')
        assert result == '=SUM(A1:A10)'
        
        # Test AVERAGE
        result = self.translator.polars_to_sheets_formula('mean', 'B1:B10')
        assert result == '=AVERAGE(B1:B10)'
        
        # Test COUNT
        result = self.translator.polars_to_sheets_formula('count', 'C1:C10')
        assert result == '=COUNT(C1:C10)'
        
        # Test MAX
        result = self.translator.polars_to_sheets_formula('max', 'D1:D10')
        assert result == '=MAX(D1:D10)'
        
        # Test MIN
        result = self.translator.polars_to_sheets_formula('min', 'E1:E10')
        assert result == '=MIN(E1:E10)'
    
    def test_array_formula_mappings(self):
        """Test array formula translations"""
        # Test multiplication
        result = self.translator.polars_to_sheets_formula('multiply', 'A1:A10', range2='B1:B10')
        assert result == '=ARRAYFORMULA(A1:A10*B1:B10)'
        
        # Test division
        result = self.translator.polars_to_sheets_formula('divide', 'A1:A10', range2='B1:B10')
        assert result == '=ARRAYFORMULA(A1:A10/B1:B10)'
        
        # Test addition
        result = self.translator.polars_to_sheets_formula('add', 'A1:A10', range2='B1:B10')
        assert result == '=ARRAYFORMULA(A1:A10+B1:B10)'
    
    def test_lookup_formula_mappings(self):
        """Test lookup formula translations"""
        # Test VLOOKUP
        result = self.translator.polars_to_sheets_formula(
            'vlookup', 
            'A1:C10', 
            lookup_value='E1',
            col_index=2,
            exact_match='FALSE'
        )
        assert result == '=VLOOKUP(E1,A1:C10,2,FALSE)'
        
        # Test INDEX/MATCH
        result = self.translator.polars_to_sheets_formula(
            'index_match',
            'B1:B10',
            match_value='D1',
            match_range='A1:A10'
        )
        assert result == '=INDEX(B1:B10,MATCH(D1,A1:A10,0))'
    
    def test_financial_formula_mappings(self):
        """Test financial formula translations"""
        # Test NPV
        result = self.translator.polars_to_sheets_formula('npv', 'B1:B10', rate=0.12)
        assert result == '=NPV(0.12,B1:B10)'
        
        # Test PMT
        result = self.translator.polars_to_sheets_formula(
            'pmt',
            'unused_range',
            rate=0.05/12,
            nper=360,
            pv=100000
        )
        assert result == '=PMT(0.041666666666666664,360,100000)'
    
    def test_case_insensitive(self):
        """Test that function names are case-insensitive"""
        result1 = self.translator.polars_to_sheets_formula('SUM', 'A1:A10')
        result2 = self.translator.polars_to_sheets_formula('sum', 'A1:A10')
        result3 = self.translator.polars_to_sheets_formula('Sum', 'A1:A10')
        
        assert result1 == result2 == result3 == '=SUM(A1:A10)'
    
    def test_unsupported_operations(self):
        """Test handling of unsupported operations"""
        result = self.translator.polars_to_sheets_formula('unsupported_op', 'A1:A10')
        assert result == "'Unsupported operation: unsupported_op"
    
    def test_sheets_to_polars_conversion(self):
        """Test converting Sheets formulas back to Polars operations"""
        # Test simple formula
        result = self.translator.sheets_formula_to_polars('=SUM(A1:A10)')
        assert result['polars_operation'] == 'sum'
        assert result['arguments'] == 'A1:A10'
        
        # Test with equals sign
        result = self.translator.sheets_formula_to_polars('SUM(A1:A10)')
        assert result['polars_operation'] == 'sum'
        
        # Test AVERAGE
        result = self.translator.sheets_formula_to_polars('=AVERAGE(B1:B10)')
        assert result['polars_operation'] == 'mean'
    
    def test_formula_validation(self):
        """Test formula syntax validation"""
        # Valid formulas
        assert self.translator.validate_formula_syntax('=SUM(A1:A10)') == True
        assert self.translator.validate_formula_syntax('=AVERAGE(B1:B10)') == True
        assert self.translator.validate_formula_syntax('=VLOOKUP(D1,A1:C10,2,FALSE)') == True
        
        # Invalid formulas
        assert self.translator.validate_formula_syntax('SUM(A1:A10)') == False  # Missing =
        assert self.translator.validate_formula_syntax('=SUM(A1:A10') == False  # Unbalanced parentheses
        assert self.translator.validate_formula_syntax('=sum(A1:A10)') == False  # Lowercase function name
        assert self.translator.validate_formula_syntax('=123ABC(A1:A10)') == False  # Invalid function name
    
    def test_formula_documentation(self):
        """Test formula documentation retrieval"""
        doc = self.translator.get_formula_documentation('sum')
        assert 'SUM' in doc['description']
        assert doc['category'] == 'Math'
        
        doc = self.translator.get_formula_documentation('vlookup')
        assert 'VLOOKUP' in doc['description']
        assert doc['category'] == 'Lookup'
        
        doc = self.translator.get_formula_documentation('npv')
        assert 'NPV' in doc['description']
        assert doc['category'] == 'Financial'
    
    def test_formula_chain_generation(self):
        """Test generating formula chains for complex operations"""
        operations = [
            {
                'operation': 'sum',
                'range': 'A1:A10',
                'params': {}
            },
            {
                'operation': 'multiply',
                'range': 'B1:B10',
                'params': {'range2': 'C1:C10'}
            }
        ]
        
        result = self.translator.generate_formula_chain(operations)
        # Should return the last formula in the chain
        assert result == '=ARRAYFORMULA(B1:B10*C1:C10)'
    
    def test_category_classification(self):
        """Test formula category classification"""
        assert self.translator._get_formula_category('sum') == 'Math'
        assert self.translator._get_formula_category('average') == 'Math'
        assert self.translator._get_formula_category('std') == 'Statistical'
        assert self.translator._get_formula_category('vlookup') == 'Lookup'
        assert self.translator._get_formula_category('npv') == 'Financial'
        assert self.translator._get_formula_category('multiply') == 'Array'
        assert self.translator._get_formula_category('unknown') == 'Other'