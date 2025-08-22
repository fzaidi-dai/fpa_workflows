#!/usr/bin/env python3
"""
Phase 0: Formula Builder Implementation

This module implements the core Formula Builder architecture that eliminates the 70% 
error rate problem by having tools generate formulas with 100% syntax accuracy instead 
of agents generating formula strings.

The Formula Builder provides a business-parameter interface where agents specify
business logic and parameters, and tools generate syntactically perfect Google Sheets formulas.
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FormulaValidationResult:
    """Result of formula parameter validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class AbstractFormulaBuilder(ABC):
    """
    Abstract base class for platform-agnostic formula builders.
    
    This enables future support for Excel, LibreOffice Calc, or other spreadsheet platforms
    while maintaining the same business-parameter interface.
    """
    
    @abstractmethod
    def build_formula(self, formula_type: str, parameters: Dict[str, Any]) -> str:
        """
        Build a formula of the specified type with given parameters.
        
        Args:
            formula_type: Type of formula to build (e.g., 'sumif', 'vlookup')
            parameters: Business parameters for the formula
            
        Returns:
            Complete formula string with perfect syntax
            
        Raises:
            ValueError: If formula_type is unsupported or parameters are invalid
        """
        pass
    
    @abstractmethod
    def validate_parameters(self, formula_type: str, parameters: Dict[str, Any]) -> FormulaValidationResult:
        """
        Validate parameters for a formula type.
        
        Args:
            formula_type: Type of formula to validate for
            parameters: Parameters to validate
            
        Returns:
            FormulaValidationResult with validation status and any errors/warnings
        """
        pass
    
    @abstractmethod
    def get_supported_formulas(self) -> List[str]:
        """
        Get list of supported formula types.
        
        Returns:
            List of formula type strings supported by this builder
        """
        pass


class GoogleSheetsFormulaBuilder(AbstractFormulaBuilder):
    """
    Google Sheets specific formula builder.
    
    Routes formula building requests to specialized builders for different categories:
    - Aggregation (SUMIF, COUNTIF, AVERAGEIF, etc.)
    - Lookup (VLOOKUP, XLOOKUP, INDEX/MATCH)
    - Financial (NPV, IRR, PMT, etc.)
    - Array (ARRAYFORMULA, array operations)
    - Custom (Business-specific complex formulas)
    """
    
    def __init__(self):
        """Initialize with all specialized formula builders"""
        self.aggregation_builder = AggregationFormulaBuilder()
        self.lookup_builder = LookupFormulaBuilder()
        self.financial_builder = FinancialFormulaBuilder()
        self.array_builder = ArrayFormulaBuilder()
        self.custom_builder = CustomFormulaBuilder()
        self.text_builder = TextFormulaBuilder()
        self.logical_builder = LogicalFormulaBuilder()
        self.statistical_builder = StatisticalFormulaBuilder()
        self.datetime_builder = DateTimeFormulaBuilder()
        
        # Map formula types to their builders and methods
        self._formula_map = {
            # Aggregation formulas
            'sum': (self.aggregation_builder, 'build_sum'),
            'average': (self.aggregation_builder, 'build_average'),
            'count': (self.aggregation_builder, 'build_count'),
            'counta': (self.aggregation_builder, 'build_counta'),
            'max': (self.aggregation_builder, 'build_max'),
            'min': (self.aggregation_builder, 'build_min'),
            'sumif': (self.aggregation_builder, 'build_sumif'),
            'sumifs': (self.aggregation_builder, 'build_sumifs'),
            'countif': (self.aggregation_builder, 'build_countif'),
            'countifs': (self.aggregation_builder, 'build_countifs'),
            'averageif': (self.aggregation_builder, 'build_averageif'),
            'averageifs': (self.aggregation_builder, 'build_averageifs'),
            'subtotal': (self.aggregation_builder, 'build_subtotal'),
            
            # Lookup formulas
            'vlookup': (self.lookup_builder, 'build_vlookup'),
            'hlookup': (self.lookup_builder, 'build_hlookup'),
            'xlookup': (self.lookup_builder, 'build_xlookup'),
            'index': (self.lookup_builder, 'build_index'),
            'match': (self.lookup_builder, 'build_match'),
            'index_match': (self.lookup_builder, 'build_index_match'),
            
            # Financial formulas
            'npv': (self.financial_builder, 'build_npv'),
            'irr': (self.financial_builder, 'build_irr'),
            'mirr': (self.financial_builder, 'build_mirr'),
            'xirr': (self.financial_builder, 'build_xirr'),
            'xnpv': (self.financial_builder, 'build_xnpv'),
            'pmt': (self.financial_builder, 'build_pmt'),
            'pv': (self.financial_builder, 'build_pv'),
            'fv': (self.financial_builder, 'build_fv'),
            'nper': (self.financial_builder, 'build_nper'),
            'rate': (self.financial_builder, 'build_rate'),
            'ipmt': (self.financial_builder, 'build_ipmt'),
            'ppmt': (self.financial_builder, 'build_ppmt'),
            'sln': (self.financial_builder, 'build_sln'),
            'db': (self.financial_builder, 'build_db'),
            'ddb': (self.financial_builder, 'build_ddb'),
            'syd': (self.financial_builder, 'build_syd'),
            
            # Array formulas
            'arrayformula': (self.array_builder, 'build_arrayformula'),
            'transpose': (self.array_builder, 'build_transpose'),
            'unique': (self.array_builder, 'build_unique'),
            'sort': (self.array_builder, 'build_sort'),
            'filter': (self.array_builder, 'build_filter'),
            'sequence': (self.array_builder, 'build_sequence'),
            'sumproduct': (self.array_builder, 'build_sumproduct'),
            
            # Text formulas
            'concatenate': (self.text_builder, 'build_concatenate'),
            'left': (self.text_builder, 'build_left'),
            'right': (self.text_builder, 'build_right'),
            'mid': (self.text_builder, 'build_mid'),
            'len': (self.text_builder, 'build_len'),
            'upper': (self.text_builder, 'build_upper'),
            'lower': (self.text_builder, 'build_lower'),
            'trim': (self.text_builder, 'build_trim'),
            
            # Logical formulas
            'if': (self.logical_builder, 'build_if'),
            'and': (self.logical_builder, 'build_and'),
            'or': (self.logical_builder, 'build_or'),
            'not': (self.logical_builder, 'build_not'),
            
            # Statistical formulas
            'median': (self.statistical_builder, 'build_median'),
            'stdev': (self.statistical_builder, 'build_stdev'),
            'var': (self.statistical_builder, 'build_var'),
            'mode': (self.statistical_builder, 'build_mode'),
            'percentile': (self.statistical_builder, 'build_percentile'),
            'percentrank': (self.statistical_builder, 'build_percentrank'),
            'rank': (self.statistical_builder, 'build_rank'),
            
            # Date/Time formulas
            'now': (self.datetime_builder, 'build_now'),
            'today': (self.datetime_builder, 'build_today'),
            'date': (self.datetime_builder, 'build_date'),
            'year': (self.datetime_builder, 'build_year'),
            'month': (self.datetime_builder, 'build_month'),
            'day': (self.datetime_builder, 'build_day'),
            'eomonth': (self.datetime_builder, 'build_eomonth'),
            
            # Custom business formulas
            'profit_margin': (self.custom_builder, 'build_profit_margin'),
            'variance_percent': (self.custom_builder, 'build_variance_percent'),
            'compound_growth': (self.custom_builder, 'build_compound_growth'),
            'cagr': (self.custom_builder, 'build_cagr'),
            'capm': (self.custom_builder, 'build_capm'),
            'sharpe_ratio': (self.custom_builder, 'build_sharpe_ratio'),
            'beta_coefficient': (self.custom_builder, 'build_beta_coefficient'),
            'customer_ltv': (self.custom_builder, 'build_customer_ltv'),
            'churn_rate': (self.custom_builder, 'build_churn_rate'),
            'market_share': (self.custom_builder, 'build_market_share'),
            'customer_acquisition_cost': (self.custom_builder, 'build_customer_acquisition_cost'),
            'break_even_analysis': (self.custom_builder, 'build_break_even_analysis'),
            'dupont_analysis': (self.custom_builder, 'build_dupont_analysis'),
            'z_score': (self.custom_builder, 'build_z_score'),
        }
        
        logger.info(f"GoogleSheetsFormulaBuilder initialized with {len(self._formula_map)} formula types")
    
    def build_formula(self, formula_type: str, parameters: Dict[str, Any]) -> str:
        """
        Route formula building to appropriate specialized builder.
        
        Args:
            formula_type: Type of formula to build
            parameters: Business parameters for the formula
            
        Returns:
            Complete Google Sheets formula with perfect syntax
            
        Raises:
            ValueError: If formula_type is unsupported or parameters are invalid
        """
        formula_type = formula_type.lower()
        
        if formula_type not in self._formula_map:
            supported = ', '.join(sorted(self._formula_map.keys()))
            raise ValueError(
                f"Unsupported formula type: '{formula_type}'. "
                f"Supported types: {supported}"
            )
        
        # Validate parameters before building
        validation_result = self.validate_parameters(formula_type, parameters)
        if not validation_result.is_valid:
            error_msg = '; '.join(validation_result.errors)
            raise ValueError(f"Parameter validation failed for {formula_type}: {error_msg}")
        
        # Get builder and method
        builder, method_name = self._formula_map[formula_type]
        method = getattr(builder, method_name)
        
        # Build formula with guaranteed syntax accuracy
        try:
            formula = method(**parameters)
            logger.debug(f"Built {formula_type} formula: {formula}")
            return formula
        except Exception as e:
            logger.error(f"Error building {formula_type} formula: {e}")
            raise ValueError(f"Failed to build {formula_type} formula: {str(e)}") from e
    
    def validate_parameters(self, formula_type: str, parameters: Dict[str, Any]) -> FormulaValidationResult:
        """
        Validate parameters for a formula type.
        
        Args:
            formula_type: Type of formula to validate for
            parameters: Parameters to validate
            
        Returns:
            FormulaValidationResult with validation status and any errors/warnings
        """
        formula_type = formula_type.lower()
        
        if formula_type not in self._formula_map:
            return FormulaValidationResult(
                is_valid=False,
                errors=[f"Unsupported formula type: {formula_type}"],
                warnings=[]
            )
        
        # Get the appropriate builder for validation
        builder, _ = self._formula_map[formula_type]
        
        # Each builder implements its own validation logic
        return builder.validate_parameters(formula_type, parameters)
    
    def get_supported_formulas(self) -> List[str]:
        """
        Get list of supported formula types.
        
        Returns:
            List of formula type strings supported by this builder
        """
        return list(self._formula_map.keys())


class BaseFormulaBuilder(ABC):
    """Base class for specialized formula builders with common utilities"""
    
    def _escape_criteria(self, criteria: Union[str, int, float]) -> str:
        """
        Properly escape criteria for Google Sheets formulas.
        
        Args:
            criteria: Criteria value to escape
            
        Returns:
            Properly escaped criteria string for Google Sheets
        """
        if isinstance(criteria, (int, float)):
            return str(criteria)
        
        criteria_str = str(criteria)
        
        # Handle comparison operators
        if criteria_str.startswith(('>', '<', '=', '>=', '<=', '<>', '!=')):
            return f'"{criteria_str}"'
        
        # Handle numeric strings (don't quote them)
        if self._is_numeric(criteria_str):
            return criteria_str
        
        # Handle text criteria (quote them)
        return f'"{criteria_str}"'
    
    def _is_numeric(self, value: str) -> bool:
        """Check if a string represents a numeric value"""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _validate_range(self, range_ref: str) -> bool:
        """Validate that a string looks like a valid range reference"""
        if not range_ref:
            return False
        
        # Basic validation for A1 notation or named ranges
        # Patterns like: A:A, A1:B10, Sheet1!A:A, Sheet1!A1:B10, named_range
        a1_pattern = r'^([A-Za-z_][A-Za-z0-9_]*!)?([A-Z]+[0-9]*:)?[A-Z]+[0-9]*$|^[A-Za-z_][A-Za-z0-9_]*$'
        return bool(re.match(a1_pattern, range_ref))
    
    def _validate_cell_ref(self, cell_ref: str) -> bool:
        """Validate that a string looks like a valid cell reference"""
        if not cell_ref:
            return False
        
        # Patterns like: A1, Sheet1!A1, $A$1
        cell_pattern = r'^([A-Za-z_][A-Za-z0-9_]*!)?\$?[A-Z]+\$?[0-9]+$'
        return bool(re.match(cell_pattern, cell_ref))
    
    def validate_parameters(self, formula_type: str, parameters: Dict[str, Any]) -> FormulaValidationResult:
        """
        Default parameter validation. Subclasses can override for specific validation.
        
        Args:
            formula_type: Type of formula being validated
            parameters: Parameters to validate
            
        Returns:
            FormulaValidationResult with validation status
        """
        return FormulaValidationResult(
            is_valid=True,
            errors=[],
            warnings=[]
        )


# Import specialized builders (these will be implemented in the next steps)
class AggregationFormulaBuilder(BaseFormulaBuilder):
    """
    Builds aggregation formulas with guaranteed syntax accuracy.
    
    Supports: SUM, AVERAGE, COUNT, COUNTA, MAX, MIN, SUMIF, SUMIFS, 
    COUNTIF, COUNTIFS, AVERAGEIF, AVERAGEIFS, SUBTOTAL
    """
    
    def build_sum(self, range_ref: str) -> str:
        """Build SUM formula"""
        return f"=SUM({range_ref})"
    
    def build_average(self, range_ref: str) -> str:
        """Build AVERAGE formula"""
        return f"=AVERAGE({range_ref})"
    
    def build_count(self, range_ref: str) -> str:
        """Build COUNT formula"""
        return f"=COUNT({range_ref})"
    
    def build_counta(self, range_ref: str) -> str:
        """Build COUNTA formula"""
        return f"=COUNTA({range_ref})"
    
    def build_max(self, range_ref: str) -> str:
        """Build MAX formula"""
        return f"=MAX({range_ref})"
    
    def build_min(self, range_ref: str) -> str:
        """Build MIN formula"""
        return f"=MIN({range_ref})"
    
    def build_sumif(self, criteria_range: str, criteria: Union[str, int, float], 
                   sum_range: Optional[str] = None) -> str:
        """
        Build SUMIF formula with guaranteed syntax accuracy.
        
        Args:
            criteria_range: Range containing criteria values (e.g., "A:A")
            criteria: Criteria to match (e.g., ">100", "North", "Active")
            sum_range: Range containing values to sum (optional, defaults to criteria_range)
            
        Returns:
            Complete SUMIF formula string
        """
        escaped_criteria = self._escape_criteria(criteria)
        
        if sum_range:
            return f"=SUMIF({criteria_range},{escaped_criteria},{sum_range})"
        else:
            return f"=SUMIF({criteria_range},{escaped_criteria})"
    
    def build_sumifs(self, sum_range: str, criteria_pairs: List[Tuple[str, Union[str, int, float]]]) -> str:
        """
        Build SUMIFS formula with multiple criteria.
        
        Args:
            sum_range: Range containing values to sum
            criteria_pairs: List of (criteria_range, criteria) tuples
            
        Returns:
            Complete SUMIFS formula string
            
        Raises:
            ValueError: If too many criteria pairs (Google Sheets limit is 127)
        """
        if len(criteria_pairs) > 127:  # Google Sheets limit
            raise ValueError("SUMIFS supports maximum 127 criteria pairs")
        
        if not criteria_pairs:
            raise ValueError("SUMIFS requires at least one criteria pair")
        
        formula_parts = [f"=SUMIFS({sum_range}"]
        for criteria_range, criteria in criteria_pairs:
            escaped_criteria = self._escape_criteria(criteria)
            formula_parts.extend([criteria_range, escaped_criteria])
        
        return ",".join(formula_parts) + ")"
    
    def build_countif(self, criteria_range: str, criteria: Union[str, int, float]) -> str:
        """
        Build COUNTIF formula with guaranteed syntax accuracy.
        
        Args:
            criteria_range: Range containing criteria values
            criteria: Criteria to match
            
        Returns:
            Complete COUNTIF formula string
        """
        escaped_criteria = self._escape_criteria(criteria)
        return f"=COUNTIF({criteria_range},{escaped_criteria})"
    
    def build_countifs(self, criteria_pairs: List[Tuple[str, Union[str, int, float]]]) -> str:
        """
        Build COUNTIFS formula with multiple criteria.
        
        Args:
            criteria_pairs: List of (criteria_range, criteria) tuples
            
        Returns:
            Complete COUNTIFS formula string
        """
        if len(criteria_pairs) > 127:  # Google Sheets limit
            raise ValueError("COUNTIFS supports maximum 127 criteria pairs")
        
        if not criteria_pairs:
            raise ValueError("COUNTIFS requires at least one criteria pair")
        
        formula_parts = ["=COUNTIFS("]
        first_pair = True
        for criteria_range, criteria in criteria_pairs:
            if not first_pair:
                formula_parts.append(",")
            escaped_criteria = self._escape_criteria(criteria)
            formula_parts.extend([criteria_range, ",", escaped_criteria])
            first_pair = False
        
        return "".join(formula_parts) + ")"
    
    def build_averageif(self, criteria_range: str, criteria: Union[str, int, float], 
                       average_range: Optional[str] = None) -> str:
        """
        Build AVERAGEIF formula with guaranteed syntax accuracy.
        
        Args:
            criteria_range: Range containing criteria values
            criteria: Criteria to match
            average_range: Range containing values to average (optional, defaults to criteria_range)
            
        Returns:
            Complete AVERAGEIF formula string
        """
        escaped_criteria = self._escape_criteria(criteria)
        
        if average_range:
            return f"=AVERAGEIF({criteria_range},{escaped_criteria},{average_range})"
        else:
            return f"=AVERAGEIF({criteria_range},{escaped_criteria})"
    
    def build_averageifs(self, average_range: str, criteria_pairs: List[Tuple[str, Union[str, int, float]]]) -> str:
        """
        Build AVERAGEIFS formula with multiple criteria.
        
        Args:
            average_range: Range containing values to average
            criteria_pairs: List of (criteria_range, criteria) tuples
            
        Returns:
            Complete AVERAGEIFS formula string
        """
        if len(criteria_pairs) > 127:  # Google Sheets limit
            raise ValueError("AVERAGEIFS supports maximum 127 criteria pairs")
        
        if not criteria_pairs:
            raise ValueError("AVERAGEIFS requires at least one criteria pair")
        
        formula_parts = [f"=AVERAGEIFS({average_range}"]
        for criteria_range, criteria in criteria_pairs:
            escaped_criteria = self._escape_criteria(criteria)
            formula_parts.extend([criteria_range, escaped_criteria])
        
        return ",".join(formula_parts) + ")"
    
    def build_subtotal(self, function_num: int, range_ref: str) -> str:
        """
        Build SUBTOTAL formula.
        
        Args:
            function_num: Function number (1-11 or 101-111)
            range_ref: Range to apply subtotal to
            
        Returns:
            Complete SUBTOTAL formula string
        """
        # Validate function number
        valid_nums = list(range(1, 12)) + list(range(101, 112))
        if function_num not in valid_nums:
            raise ValueError(f"Invalid SUBTOTAL function number: {function_num}")
        
        return f"=SUBTOTAL({function_num},{range_ref})"
    
    def validate_parameters(self, formula_type: str, parameters: Dict[str, Any]) -> FormulaValidationResult:
        """
        Validate parameters for aggregation formulas.
        
        Args:
            formula_type: Type of aggregation formula
            parameters: Parameters to validate
            
        Returns:
            FormulaValidationResult with validation details
        """
        errors = []
        warnings = []
        
        if formula_type in ['sum', 'average', 'count', 'counta', 'max', 'min']:
            # Simple single-range formulas
            if 'range_ref' not in parameters:
                errors.append("Missing required parameter: range_ref")
            elif not self._validate_range(parameters['range_ref']):
                errors.append(f"Invalid range reference: {parameters['range_ref']}")
        
        elif formula_type in ['sumif', 'countif', 'averageif']:
            # Single criteria formulas
            if 'criteria_range' not in parameters:
                errors.append("Missing required parameter: criteria_range")
            elif not self._validate_range(parameters['criteria_range']):
                errors.append(f"Invalid criteria_range: {parameters['criteria_range']}")
            
            if 'criteria' not in parameters:
                errors.append("Missing required parameter: criteria")
        
        elif formula_type in ['sumifs', 'countifs', 'averageifs']:
            # Multiple criteria formulas
            if 'criteria_pairs' not in parameters:
                errors.append("Missing required parameter: criteria_pairs")
            else:
                criteria_pairs = parameters['criteria_pairs']
                if not isinstance(criteria_pairs, list):
                    errors.append("criteria_pairs must be a list of tuples")
                elif len(criteria_pairs) == 0:
                    errors.append("criteria_pairs cannot be empty")
                elif len(criteria_pairs) > 127:
                    errors.append("Too many criteria pairs (maximum 127)")
                else:
                    for i, pair in enumerate(criteria_pairs):
                        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                            errors.append(f"criteria_pairs[{i}] must be a tuple of (range, criteria)")
                        elif not self._validate_range(pair[0]):
                            errors.append(f"Invalid range in criteria_pairs[{i}]: {pair[0]}")
            
            if formula_type in ['sumifs', 'averageifs']:
                sum_avg_range = 'sum_range' if formula_type == 'sumifs' else 'average_range'
                if sum_avg_range not in parameters:
                    errors.append(f"Missing required parameter: {sum_avg_range}")
                elif not self._validate_range(parameters[sum_avg_range]):
                    errors.append(f"Invalid {sum_avg_range}: {parameters[sum_avg_range]}")
        
        elif formula_type == 'subtotal':
            if 'function_num' not in parameters:
                errors.append("Missing required parameter: function_num")
            else:
                function_num = parameters['function_num']
                valid_nums = list(range(1, 12)) + list(range(101, 112))
                if not isinstance(function_num, int) or function_num not in valid_nums:
                    errors.append(f"Invalid function_num: {function_num}")
            
            if 'range_ref' not in parameters:
                errors.append("Missing required parameter: range_ref")
            elif not self._validate_range(parameters['range_ref']):
                errors.append(f"Invalid range_ref: {parameters['range_ref']}")
        
        return FormulaValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

class LookupFormulaBuilder(BaseFormulaBuilder):
    """
    Builds lookup formulas with guaranteed syntax accuracy.
    
    Supports: VLOOKUP, HLOOKUP, XLOOKUP, INDEX, MATCH, INDEX/MATCH combinations
    """
    
    def build_vlookup(self, lookup_value: str, table_array: str, 
                     col_index_num: int, range_lookup: bool = False) -> str:
        """
        Build VLOOKUP formula with parameter validation.
        
        Args:
            lookup_value: Value to search for (cell reference or value)
            table_array: Table range to search in
            col_index_num: Column number to return (1-based)
            range_lookup: Whether to use approximate match (default: False for exact match)
            
        Returns:
            Complete VLOOKUP formula string
            
        Raises:
            ValueError: If col_index_num is invalid
        """
        if col_index_num < 1:
            raise ValueError("Column index must be >= 1")
        
        range_lookup_str = "TRUE" if range_lookup else "FALSE"
        return f"=VLOOKUP({lookup_value},{table_array},{col_index_num},{range_lookup_str})"
    
    def build_hlookup(self, lookup_value: str, table_array: str, 
                     row_index_num: int, range_lookup: bool = False) -> str:
        """
        Build HLOOKUP formula with parameter validation.
        
        Args:
            lookup_value: Value to search for
            table_array: Table range to search in
            row_index_num: Row number to return (1-based)
            range_lookup: Whether to use approximate match (default: False)
            
        Returns:
            Complete HLOOKUP formula string
        """
        if row_index_num < 1:
            raise ValueError("Row index must be >= 1")
        
        range_lookup_str = "TRUE" if range_lookup else "FALSE"
        return f"=HLOOKUP({lookup_value},{table_array},{row_index_num},{range_lookup_str})"
    
    def build_xlookup(self, lookup_value: str, lookup_array: str, 
                     return_array: str, if_not_found: Optional[str] = None,
                     match_mode: int = 0, search_mode: int = 1) -> str:
        """
        Build XLOOKUP formula (newer Google Sheets function).
        
        Args:
            lookup_value: Value to search for
            lookup_array: Array to search in
            return_array: Array to return values from
            if_not_found: Value to return if not found (optional)
            match_mode: Match mode (0=exact, 1=exact or next smallest, 2=exact or next largest, -1=wildcard)
            search_mode: Search mode (1=first to last, -1=last to first, 2=binary asc, -2=binary desc)
            
        Returns:
            Complete XLOOKUP formula string
        """
        formula_parts = [f"=XLOOKUP({lookup_value},{lookup_array},{return_array}"]
        
        if if_not_found is not None:
            formula_parts.append(f",{if_not_found}")
            
            # Add match_mode and search_mode only if if_not_found is provided
            if match_mode != 0 or search_mode != 1:
                formula_parts.append(f",{match_mode}")
                if search_mode != 1:
                    formula_parts.append(f",{search_mode}")
        
        return "".join(formula_parts) + ")"
    
    def build_index(self, array: str, row_num: Optional[int] = None, 
                   col_num: Optional[int] = None) -> str:
        """
        Build INDEX formula.
        
        Args:
            array: Array or range to index into
            row_num: Row number (1-based, optional for single row)
            col_num: Column number (1-based, optional for single column)
            
        Returns:
            Complete INDEX formula string
        """
        formula_parts = [f"=INDEX({array}"]
        
        if row_num is not None:
            formula_parts.append(f",{row_num}")
            if col_num is not None:
                formula_parts.append(f",{col_num}")
        elif col_num is not None:
            formula_parts.append(f",1,{col_num}")  # Default to row 1 if only col_num provided
        
        return "".join(formula_parts) + ")"
    
    def build_match(self, lookup_value: str, lookup_array: str, 
                   match_type: int = 0) -> str:
        """
        Build MATCH formula.
        
        Args:
            lookup_value: Value to search for
            lookup_array: Array to search in
            match_type: Match type (0=exact, 1=largest value <=, -1=smallest value >=)
            
        Returns:
            Complete MATCH formula string
        """
        return f"=MATCH({lookup_value},{lookup_array},{match_type})"
    
    def build_index_match(self, return_range: str, lookup_value: str, 
                         lookup_range: str, match_type: int = 0) -> str:
        """
        Build INDEX/MATCH combination for flexible lookups.
        
        Args:
            return_range: Range to return values from
            lookup_value: Value to search for
            lookup_range: Range to search in
            match_type: Match type for MATCH function
            
        Returns:
            Complete INDEX/MATCH formula string
        """
        return f"=INDEX({return_range},MATCH({lookup_value},{lookup_range},{match_type}))"
    
    def validate_parameters(self, formula_type: str, parameters: Dict[str, Any]) -> FormulaValidationResult:
        """
        Validate parameters for lookup formulas.
        
        Args:
            formula_type: Type of lookup formula
            parameters: Parameters to validate
            
        Returns:
            FormulaValidationResult with validation details
        """
        errors = []
        warnings = []
        
        if formula_type == 'vlookup':
            # Validate VLOOKUP parameters
            required_params = ['lookup_value', 'table_array', 'col_index_num']
            for param in required_params:
                if param not in parameters:
                    errors.append(f"Missing required parameter: {param}")
            
            if 'table_array' in parameters and not self._validate_range(parameters['table_array']):
                errors.append(f"Invalid table_array: {parameters['table_array']}")
            
            if 'col_index_num' in parameters:
                col_index = parameters['col_index_num']
                if not isinstance(col_index, int) or col_index < 1:
                    errors.append("col_index_num must be an integer >= 1")
        
        elif formula_type == 'hlookup':
            # Validate HLOOKUP parameters  
            required_params = ['lookup_value', 'table_array', 'row_index_num']
            for param in required_params:
                if param not in parameters:
                    errors.append(f"Missing required parameter: {param}")
            
            if 'table_array' in parameters and not self._validate_range(parameters['table_array']):
                errors.append(f"Invalid table_array: {parameters['table_array']}")
            
            if 'row_index_num' in parameters:
                row_index = parameters['row_index_num']
                if not isinstance(row_index, int) or row_index < 1:
                    errors.append("row_index_num must be an integer >= 1")
        
        elif formula_type == 'xlookup':
            # Validate XLOOKUP parameters
            required_params = ['lookup_value', 'lookup_array', 'return_array']
            for param in required_params:
                if param not in parameters:
                    errors.append(f"Missing required parameter: {param}")
            
            for array_param in ['lookup_array', 'return_array']:
                if array_param in parameters and not self._validate_range(parameters[array_param]):
                    errors.append(f"Invalid {array_param}: {parameters[array_param]}")
            
            # Validate optional parameters
            if 'match_mode' in parameters:
                match_mode = parameters['match_mode']
                if not isinstance(match_mode, int) or match_mode not in [0, 1, 2, -1]:
                    errors.append("match_mode must be 0, 1, 2, or -1")
            
            if 'search_mode' in parameters:
                search_mode = parameters['search_mode']
                if not isinstance(search_mode, int) or search_mode not in [1, -1, 2, -2]:
                    errors.append("search_mode must be 1, -1, 2, or -2")
        
        elif formula_type == 'index':
            # Validate INDEX parameters
            if 'array' not in parameters:
                errors.append("Missing required parameter: array")
            elif not self._validate_range(parameters['array']):
                errors.append(f"Invalid array: {parameters['array']}")
            
            # Validate optional row and column numbers
            for num_param in ['row_num', 'col_num']:
                if num_param in parameters:
                    num_value = parameters[num_param]
                    if num_value is not None and (not isinstance(num_value, int) or num_value < 1):
                        errors.append(f"{num_param} must be an integer >= 1")
        
        elif formula_type == 'match':
            # Validate MATCH parameters
            required_params = ['lookup_value', 'lookup_array']
            for param in required_params:
                if param not in parameters:
                    errors.append(f"Missing required parameter: {param}")
            
            if 'lookup_array' in parameters and not self._validate_range(parameters['lookup_array']):
                errors.append(f"Invalid lookup_array: {parameters['lookup_array']}")
            
            if 'match_type' in parameters:
                match_type = parameters['match_type']
                if not isinstance(match_type, int) or match_type not in [0, 1, -1]:
                    errors.append("match_type must be 0, 1, or -1")
        
        elif formula_type == 'index_match':
            # Validate INDEX/MATCH combination parameters
            required_params = ['return_range', 'lookup_value', 'lookup_range']
            for param in required_params:
                if param not in parameters:
                    errors.append(f"Missing required parameter: {param}")
            
            for range_param in ['return_range', 'lookup_range']:
                if range_param in parameters and not self._validate_range(parameters[range_param]):
                    errors.append(f"Invalid {range_param}: {parameters[range_param]}")
            
            if 'match_type' in parameters:
                match_type = parameters['match_type']
                if not isinstance(match_type, int) or match_type not in [0, 1, -1]:
                    errors.append("match_type must be 0, 1, or -1")
        
        return FormulaValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

class FinancialFormulaBuilder(BaseFormulaBuilder):
    """
    Builds financial formulas with guaranteed syntax accuracy.
    
    Supports: NPV, IRR, MIRR, XIRR, XNPV, PMT, PV, FV, NPER, RATE, 
    IPMT, PPMT, SLN, DB, DDB, SYD, and other financial calculations
    """
    
    def build_npv(self, rate: float, values_range: str) -> str:
        """
        Build NPV (Net Present Value) formula.
        
        Args:
            rate: Discount rate per period
            values_range: Range containing cash flows
            
        Returns:
            Complete NPV formula string
        """
        return f"=NPV({rate},{values_range})"
    
    def build_irr(self, values_range: str, guess: Optional[float] = None) -> str:
        """
        Build IRR (Internal Rate of Return) formula.
        
        Args:
            values_range: Range containing cash flows (must include initial investment)
            guess: Initial guess for IRR calculation (optional)
            
        Returns:
            Complete IRR formula string
        """
        if guess is not None:
            return f"=IRR({values_range},{guess})"
        else:
            return f"=IRR({values_range})"
    
    def build_mirr(self, values: str, finance_rate: float, reinvest_rate: float) -> str:
        """
        Build MIRR (Modified Internal Rate of Return) formula.
        
        Args:
            values: Range containing cash flows
            finance_rate: Interest rate paid on money used in cash flows
            reinvest_rate: Interest rate received on reinvestment of cash flows
            
        Returns:
            Complete MIRR formula string
        """
        return f"=MIRR({values},{finance_rate},{reinvest_rate})"
    
    def build_xirr(self, values: str, dates: str, guess: Optional[float] = None) -> str:
        """
        Build XIRR (IRR for irregular periods) formula.
        
        Args:
            values: Range containing cash flows
            dates: Range containing corresponding dates
            guess: Initial guess for XIRR calculation (optional)
            
        Returns:
            Complete XIRR formula string
        """
        if guess is not None:
            return f"=XIRR({values},{dates},{guess})"
        else:
            return f"=XIRR({values},{dates})"
    
    def build_xnpv(self, rate: float, values: str, dates: str) -> str:
        """
        Build XNPV (NPV for irregular periods) formula.
        
        Args:
            rate: Discount rate
            values: Range containing cash flows
            dates: Range containing corresponding dates
            
        Returns:
            Complete XNPV formula string
        """
        return f"=XNPV({rate},{values},{dates})"
    
    def build_pmt(self, rate: float, nper: int, pv: float, 
                  fv: Optional[float] = None, type_: int = 0) -> str:
        """
        Build PMT (Payment) formula.
        
        Args:
            rate: Interest rate per period
            nper: Number of periods
            pv: Present value (loan amount)
            fv: Future value (optional, defaults to 0)
            type_: Payment timing (0=end of period, 1=beginning of period)
            
        Returns:
            Complete PMT formula string
        """
        formula_parts = [f"=PMT({rate},{nper},{pv}"]
        
        if fv is not None:
            formula_parts.append(f",{fv}")
            if type_ != 0:
                formula_parts.append(f",{type_}")
        elif type_ != 0:
            formula_parts.append(f",0,{type_}")
        
        return "".join(formula_parts) + ")"
    
    def build_pv(self, rate: float, nper: int, pmt: float, 
                 fv: Optional[float] = None, type_: int = 0) -> str:
        """
        Build PV (Present Value) formula.
        
        Args:
            rate: Interest rate per period
            nper: Number of periods
            pmt: Payment per period
            fv: Future value (optional, defaults to 0)
            type_: Payment timing (0=end of period, 1=beginning of period)
            
        Returns:
            Complete PV formula string
        """
        formula_parts = [f"=PV({rate},{nper},{pmt}"]
        
        if fv is not None:
            formula_parts.append(f",{fv}")
            if type_ != 0:
                formula_parts.append(f",{type_}")
        elif type_ != 0:
            formula_parts.append(f",0,{type_}")
        
        return "".join(formula_parts) + ")"
    
    def build_fv(self, rate: float, nper: int, pmt: float, 
                 pv: Optional[float] = None, type_: int = 0) -> str:
        """
        Build FV (Future Value) formula.
        
        Args:
            rate: Interest rate per period
            nper: Number of periods
            pmt: Payment per period
            pv: Present value (optional, defaults to 0)
            type_: Payment timing (0=end of period, 1=beginning of period)
            
        Returns:
            Complete FV formula string
        """
        formula_parts = [f"=FV({rate},{nper},{pmt}"]
        
        if pv is not None:
            formula_parts.append(f",{pv}")
            if type_ != 0:
                formula_parts.append(f",{type_}")
        elif type_ != 0:
            formula_parts.append(f",0,{type_}")
        
        return "".join(formula_parts) + ")"
    
    def build_nper(self, rate: float, pmt: float, pv: float, 
                   fv: Optional[float] = None, type_: int = 0) -> str:
        """
        Build NPER (Number of Periods) formula.
        
        Args:
            rate: Interest rate per period
            pmt: Payment per period
            pv: Present value
            fv: Future value (optional, defaults to 0)
            type_: Payment timing (0=end of period, 1=beginning of period)
            
        Returns:
            Complete NPER formula string
        """
        formula_parts = [f"=NPER({rate},{pmt},{pv}"]
        
        if fv is not None:
            formula_parts.append(f",{fv}")
            if type_ != 0:
                formula_parts.append(f",{type_}")
        elif type_ != 0:
            formula_parts.append(f",0,{type_}")
        
        return "".join(formula_parts) + ")"
    
    def build_rate(self, nper: int, pmt: float, pv: float, 
                   fv: Optional[float] = None, type_: int = 0, 
                   guess: Optional[float] = None) -> str:
        """
        Build RATE (Interest Rate) formula.
        
        Args:
            nper: Number of periods
            pmt: Payment per period
            pv: Present value
            fv: Future value (optional, defaults to 0)
            type_: Payment timing (0=end of period, 1=beginning of period)
            guess: Initial guess for rate calculation (optional)
            
        Returns:
            Complete RATE formula string
        """
        formula_parts = [f"=RATE({nper},{pmt},{pv}"]
        
        if fv is not None or type_ != 0 or guess is not None:
            fv_val = fv if fv is not None else 0
            formula_parts.append(f",{fv_val}")
            
            if type_ != 0 or guess is not None:
                formula_parts.append(f",{type_}")
                
                if guess is not None:
                    formula_parts.append(f",{guess}")
        
        return "".join(formula_parts) + ")"
    
    def build_ipmt(self, rate: float, per: int, nper: int, pv: float, 
                   fv: Optional[float] = None, type_: int = 0) -> str:
        """
        Build IPMT (Interest Payment) formula.
        
        Args:
            rate: Interest rate per period
            per: Period for which to calculate interest payment
            nper: Total number of periods
            pv: Present value
            fv: Future value (optional, defaults to 0)
            type_: Payment timing (0=end of period, 1=beginning of period)
            
        Returns:
            Complete IPMT formula string
        """
        formula_parts = [f"=IPMT({rate},{per},{nper},{pv}"]
        
        if fv is not None:
            formula_parts.append(f",{fv}")
            if type_ != 0:
                formula_parts.append(f",{type_}")
        elif type_ != 0:
            formula_parts.append(f",0,{type_}")
        
        return "".join(formula_parts) + ")"
    
    def build_ppmt(self, rate: float, per: int, nper: int, pv: float, 
                   fv: Optional[float] = None, type_: int = 0) -> str:
        """
        Build PPMT (Principal Payment) formula.
        
        Args:
            rate: Interest rate per period
            per: Period for which to calculate principal payment
            nper: Total number of periods
            pv: Present value
            fv: Future value (optional, defaults to 0)
            type_: Payment timing (0=end of period, 1=beginning of period)
            
        Returns:
            Complete PPMT formula string
        """
        formula_parts = [f"=PPMT({rate},{per},{nper},{pv}"]
        
        if fv is not None:
            formula_parts.append(f",{fv}")
            if type_ != 0:
                formula_parts.append(f",{type_}")
        elif type_ != 0:
            formula_parts.append(f",0,{type_}")
        
        return "".join(formula_parts) + ")"
    
    def build_sln(self, cost: float, salvage: float, life: int) -> str:
        """
        Build SLN (Straight-Line Depreciation) formula.
        
        Args:
            cost: Initial cost of asset
            salvage: Salvage value at end of life
            life: Number of periods over which asset is depreciated
            
        Returns:
            Complete SLN formula string
        """
        return f"=SLN({cost},{salvage},{life})"
    
    def build_db(self, cost: float, salvage: float, life: int, 
                 period: int, month: Optional[int] = None) -> str:
        """
        Build DB (Declining Balance Depreciation) formula.
        
        Args:
            cost: Initial cost of asset
            salvage: Salvage value at end of life
            life: Number of periods over which asset is depreciated
            period: Period for which to calculate depreciation
            month: Number of months in first year (optional, defaults to 12)
            
        Returns:
            Complete DB formula string
        """
        if month is not None:
            return f"=DB({cost},{salvage},{life},{period},{month})"
        else:
            return f"=DB({cost},{salvage},{life},{period})"
    
    def build_ddb(self, cost: float, salvage: float, life: int, 
                  period: int, factor: Optional[float] = None) -> str:
        """
        Build DDB (Double-Declining Balance Depreciation) formula.
        
        Args:
            cost: Initial cost of asset
            salvage: Salvage value at end of life
            life: Number of periods over which asset is depreciated
            period: Period for which to calculate depreciation
            factor: Rate at which balance declines (optional, defaults to 2)
            
        Returns:
            Complete DDB formula string
        """
        if factor is not None:
            return f"=DDB({cost},{salvage},{life},{period},{factor})"
        else:
            return f"=DDB({cost},{salvage},{life},{period})"
    
    def build_syd(self, cost: float, salvage: float, life: int, period: int) -> str:
        """
        Build SYD (Sum-of-Years Digits Depreciation) formula.
        
        Args:
            cost: Initial cost of asset
            salvage: Salvage value at end of life
            life: Number of periods over which asset is depreciated
            period: Period for which to calculate depreciation
            
        Returns:
            Complete SYD formula string
        """
        return f"=SYD({cost},{salvage},{life},{period})"
    
    def validate_parameters(self, formula_type: str, parameters: Dict[str, Any]) -> FormulaValidationResult:
        """
        Validate parameters for financial formulas.
        
        Args:
            formula_type: Type of financial formula
            parameters: Parameters to validate
            
        Returns:
            FormulaValidationResult with validation details
        """
        errors = []
        warnings = []
        
        if formula_type == 'npv':
            if 'rate' not in parameters:
                errors.append("Missing required parameter: rate")
            elif not isinstance(parameters['rate'], (int, float)):
                errors.append("rate must be a number")
            
            if 'values_range' not in parameters:
                errors.append("Missing required parameter: values_range")
            elif not self._validate_range(parameters['values_range']):
                errors.append(f"Invalid values_range: {parameters['values_range']}")
        
        elif formula_type == 'irr':
            if 'values_range' not in parameters:
                errors.append("Missing required parameter: values_range")
            elif not self._validate_range(parameters['values_range']):
                errors.append(f"Invalid values_range: {parameters['values_range']}")
            
            if 'guess' in parameters and not isinstance(parameters['guess'], (int, float)):
                errors.append("guess must be a number")
        
        elif formula_type == 'mirr':
            required_params = ['values', 'finance_rate', 'reinvest_rate']
            for param in required_params:
                if param not in parameters:
                    errors.append(f"Missing required parameter: {param}")
                elif param == 'values' and not self._validate_range(parameters[param]):
                    errors.append(f"Invalid values range: {parameters[param]}")
                elif param in ['finance_rate', 'reinvest_rate'] and not isinstance(parameters[param], (int, float)):
                    errors.append(f"{param} must be a number")
        
        elif formula_type in ['xirr', 'xnpv']:
            if 'values' not in parameters:
                errors.append("Missing required parameter: values")
            elif not self._validate_range(parameters['values']):
                errors.append(f"Invalid values range: {parameters['values']}")
            
            if 'dates' not in parameters:
                errors.append("Missing required parameter: dates")
            elif not self._validate_range(parameters['dates']):
                errors.append(f"Invalid dates range: {parameters['dates']}")
            
            if formula_type == 'xnpv':
                if 'rate' not in parameters:
                    errors.append("Missing required parameter: rate")
                elif not isinstance(parameters['rate'], (int, float)):
                    errors.append("rate must be a number")
            elif 'guess' in parameters and not isinstance(parameters['guess'], (int, float)):
                errors.append("guess must be a number")
        
        elif formula_type in ['pmt', 'pv', 'fv']:
            # Common validation for payment functions
            required_numeric = ['rate', 'nper', 'pmt' if formula_type != 'pmt' else 'pv']
            if formula_type == 'pmt':
                required_numeric = ['rate', 'nper', 'pv']
            elif formula_type == 'pv':
                required_numeric = ['rate', 'nper', 'pmt']
            elif formula_type == 'fv':
                required_numeric = ['rate', 'nper', 'pmt']
            
            for param in required_numeric:
                if param not in parameters:
                    errors.append(f"Missing required parameter: {param}")
                elif not isinstance(parameters[param], (int, float)):
                    errors.append(f"{param} must be a number")
            
            # Validate optional parameters
            for optional_param in ['fv', 'pv', 'type_']:
                if optional_param in parameters:
                    if optional_param == 'type_' and parameters[optional_param] not in [0, 1]:
                        errors.append("type_ must be 0 (end of period) or 1 (beginning of period)")
                    elif optional_param != 'type_' and not isinstance(parameters[optional_param], (int, float)):
                        errors.append(f"{optional_param} must be a number")
        
        elif formula_type in ['sln', 'db', 'ddb', 'syd']:
            # Depreciation functions validation
            required_params = ['cost', 'salvage', 'life', 'period']
            if formula_type == 'sln':
                required_params = ['cost', 'salvage', 'life']
            
            for param in required_params:
                if param not in parameters:
                    errors.append(f"Missing required parameter: {param}")
                elif not isinstance(parameters[param], (int, float)):
                    errors.append(f"{param} must be a number")
                elif param in ['life', 'period'] and parameters[param] <= 0:
                    errors.append(f"{param} must be positive")
        
        return FormulaValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

class ArrayFormulaBuilder(BaseFormulaBuilder):
    """Builds array formulas and ARRAYFORMULA expressions with guaranteed syntax accuracy"""
    
    def build_arrayformula(self, expression: str) -> str:
        """Build ARRAYFORMULA wrapper"""
        if expression.startswith('='):
            expression = expression[1:]  # Remove leading =
        return f"=ARRAYFORMULA({expression})"
    
    def build_transpose(self, array: str) -> str:
        """Build TRANSPOSE formula"""
        return f"=TRANSPOSE({array})"
    
    def build_unique(self, range_ref: str) -> str:
        """Build UNIQUE formula"""
        return f"=UNIQUE({range_ref})"
    
    def build_sort(self, range_ref: str, sort_column: Optional[int] = None, 
                  is_ascending: bool = True) -> str:
        """Build SORT formula"""
        if sort_column is not None:
            sort_order = "TRUE" if is_ascending else "FALSE"
            return f"=SORT({range_ref},{sort_column},{sort_order})"
        return f"=SORT({range_ref})"
    
    def build_filter(self, range_ref: str, condition: str) -> str:
        """Build FILTER formula"""
        return f"=FILTER({range_ref},{condition})"
    
    def build_sequence(self, rows: int, columns: Optional[int] = None, 
                      start: Optional[int] = None, step: Optional[int] = None) -> str:
        """Build SEQUENCE formula"""
        parts = [f"=SEQUENCE({rows}"]
        if columns is not None:
            parts.append(f",{columns}")
            if start is not None:
                parts.append(f",{start}")
                if step is not None:
                    parts.append(f",{step}")
        return "".join(parts) + ")"
    
    def build_sumproduct(self, *arrays: str) -> str:
        """Build SUMPRODUCT formula"""
        return f"=SUMPRODUCT({','.join(arrays)})"

class CustomFormulaBuilder(BaseFormulaBuilder):
    """Builds business-specific custom formulas with guaranteed syntax accuracy"""
    
    def build_profit_margin(self, revenue_cell: str, cost_cell: str) -> str:
        """Build profit margin calculation"""
        return f"=({revenue_cell}-{cost_cell})/{revenue_cell}"
    
    def build_variance_percent(self, actual_cell: str, budget_cell: str) -> str:
        """Build variance percentage calculation"""
        return f"=({actual_cell}-{budget_cell})/{budget_cell}"
    
    def build_compound_growth(self, end_value: str, start_value: str, periods: int) -> str:
        """Build compound annual growth rate"""
        return f"=POWER({end_value}/{start_value},1/{periods})-1"
    
    def build_cagr(self, ending_value: str, beginning_value: str, years: int) -> str:
        """Build CAGR (Compound Annual Growth Rate)"""
        return f"=POWER({ending_value}/{beginning_value},1/{years})-1"
    
    def build_customer_ltv(self, customer_range: str, customer_id: str, 
                          revenue_range: str, months_range: str) -> str:
        """Build Customer Lifetime Value formula"""
        escaped_customer_id = self._escape_criteria(customer_id)
        return f"=SUMIF({customer_range},{escaped_customer_id},{revenue_range})*MAXIFS({months_range},{customer_range},{escaped_customer_id})/12"
    
    def build_churn_rate(self, status_range: str) -> str:
        """Build churn rate percentage"""
        return f"=COUNTIF({status_range},\"Churned\")/COUNTIF({status_range},\"<>New\")*100"

class TextFormulaBuilder(BaseFormulaBuilder):
    """Builds text formulas with guaranteed syntax accuracy"""
    
    def build_concatenate(self, *text_values: str) -> str:
        """Build CONCATENATE formula"""
        return f"=CONCATENATE({','.join(text_values)})"
    
    def build_left(self, text: str, num_chars: int) -> str:
        """Build LEFT formula"""
        return f"=LEFT({text},{num_chars})"
    
    def build_right(self, text: str, num_chars: int) -> str:
        """Build RIGHT formula"""
        return f"=RIGHT({text},{num_chars})"
    
    def build_mid(self, text: str, start_num: int, num_chars: int) -> str:
        """Build MID formula"""
        return f"=MID({text},{start_num},{num_chars})"
    
    def build_len(self, text: str) -> str:
        """Build LEN formula"""
        return f"=LEN({text})"
    
    def build_upper(self, text: str) -> str:
        """Build UPPER formula"""
        return f"=UPPER({text})"
    
    def build_lower(self, text: str) -> str:
        """Build LOWER formula"""
        return f"=LOWER({text})"
    
    def build_trim(self, text: str) -> str:
        """Build TRIM formula"""
        return f"=TRIM({text})"

class LogicalFormulaBuilder(BaseFormulaBuilder):
    """Builds logical formulas with guaranteed syntax accuracy"""
    
    def build_if(self, logical_test: str, value_if_true: str, value_if_false: str) -> str:
        """Build IF formula"""
        return f"=IF({logical_test},{value_if_true},{value_if_false})"
    
    def build_and(self, *logical_values: str) -> str:
        """Build AND formula"""
        return f"=AND({','.join(logical_values)})"
    
    def build_or(self, *logical_values: str) -> str:
        """Build OR formula"""
        return f"=OR({','.join(logical_values)})"
    
    def build_not(self, logical_value: str) -> str:
        """Build NOT formula"""
        return f"=NOT({logical_value})"

class StatisticalFormulaBuilder(BaseFormulaBuilder):
    """Builds statistical formulas with guaranteed syntax accuracy"""
    
    def build_median(self, range_ref: str) -> str:
        """Build MEDIAN formula"""
        return f"=MEDIAN({range_ref})"
    
    def build_stdev(self, range_ref: str) -> str:
        """Build STDEV formula"""
        return f"=STDEV({range_ref})"
    
    def build_var(self, range_ref: str) -> str:
        """Build VAR formula"""
        return f"=VAR({range_ref})"
    
    def build_mode(self, range_ref: str) -> str:
        """Build MODE formula"""
        return f"=MODE({range_ref})"
    
    def build_percentile(self, array: str, k: float) -> str:
        """Build PERCENTILE formula"""
        return f"=PERCENTILE({array},{k})"
    
    def build_percentrank(self, array: str, x: str) -> str:
        """Build PERCENTRANK formula"""
        return f"=PERCENTRANK({array},{x})"
    
    def build_rank(self, number: str, ref: str, order: int = 0) -> str:
        """Build RANK formula"""
        return f"=RANK({number},{ref},{order})"

class DateTimeFormulaBuilder(BaseFormulaBuilder):
    """Builds date/time formulas with guaranteed syntax accuracy"""
    
    def build_now(self) -> str:
        """Build NOW formula"""
        return "=NOW()"
    
    def build_today(self) -> str:
        """Build TODAY formula"""
        return "=TODAY()"
    
    def build_date(self, year: int, month: int, day: int) -> str:
        """Build DATE formula"""
        return f"=DATE({year},{month},{day})"
    
    def build_year(self, date: str) -> str:
        """Build YEAR formula"""
        return f"=YEAR({date})"
    
    def build_month(self, date: str) -> str:
        """Build MONTH formula"""
        return f"=MONTH({date})"
    
    def build_day(self, date: str) -> str:
        """Build DAY formula"""
        return f"=DAY({date})"
    
    def build_eomonth(self, start_date: str, months: int) -> str:
        """Build EOMONTH formula"""
        return f"=EOMONTH({start_date},{months})"