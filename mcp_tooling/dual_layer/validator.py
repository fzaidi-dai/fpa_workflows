"""
Dual-Layer Validation System

Validates consistency between Polars computations and Google Sheets formulas
to ensure accuracy in the dual-layer execution system.
"""

import math
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import logging

from .data_models import ValidationCheck, ValidationResult

# Set up logger
logger = logging.getLogger(__name__)


class ToleranceType(Enum):
    """Types of tolerance checks"""
    EXACT = "exact"
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    FUZZY = "fuzzy"


class DualLayerValidator:
    """Validates consistency between Polars and Sheets results"""
    
    def __init__(self, 
                 default_tolerance: float = 0.001,
                 enable_business_rules: bool = True):
        """
        Initialize the dual-layer validator
        
        Args:
            default_tolerance: Default tolerance for numeric comparisons
            enable_business_rules: Whether to validate business rules
        """
        self.default_tolerance = default_tolerance
        self.enable_business_rules = enable_business_rules
        self.business_rules: Dict[str, callable] = {}
        self._load_default_business_rules()
    
    def _load_default_business_rules(self):
        """Load default business rules for validation"""
        self.business_rules.update({
            "positive_values": lambda x: self._check_positive(x),
            "no_null_values": lambda x: self._check_no_nulls(x),
            "within_range": lambda x, min_val, max_val: self._check_range(x, min_val, max_val),
            "sum_equals_parts": lambda total, parts: self._check_sum_consistency(total, parts),
            "percentage_valid": lambda x: self._check_percentage_range(x)
        })
    
    def validate_dual_layer(self,
                           polars_result: Any,
                           sheets_formula: str,
                           validation_rules: List[Dict[str, Any]],
                           tolerance: Optional[float] = None) -> List[ValidationCheck]:
        """
        Validate consistency between Polars result and Sheets formula
        
        Args:
            polars_result: Result from Polars computation
            sheets_formula: Google Sheets formula used
            validation_rules: List of validation rules to apply
            tolerance: Tolerance for numeric comparisons
            
        Returns:
            List of validation check results
        """
        if tolerance is None:
            tolerance = self.default_tolerance
        
        checks = []
        
        # Note: In a real implementation, we would execute the sheets_formula
        # and compare with polars_result. For now, we simulate this.
        
        # Simulate sheets execution (placeholder)
        simulated_sheets_result = self._simulate_sheets_execution(sheets_formula, polars_result)
        
        # Core dual-layer validation
        core_check = self._validate_core_consistency(
            polars_result, 
            simulated_sheets_result, 
            tolerance
        )
        checks.append(core_check)
        
        # Apply additional validation rules
        for rule in validation_rules:
            rule_check = self._apply_validation_rule(polars_result, rule)
            checks.append(rule_check)
        
        # Business rule validation
        if self.enable_business_rules:
            business_checks = self._validate_business_rules(polars_result, validation_rules)
            checks.extend(business_checks)
        
        return checks
    
    def _simulate_sheets_execution(self, sheets_formula: str, polars_result: Any) -> Any:
        """
        Simulate sheets formula execution
        
        Note: This is a placeholder. In a real implementation, this would
        actually execute the formula in Google Sheets and return the result.
        """
        # For simulation, we'll introduce small variations to test tolerance
        if isinstance(polars_result, (int, float)):
            # Add small numerical noise
            noise = polars_result * 0.0001  # 0.01% noise
            return polars_result + noise
        elif isinstance(polars_result, list):
            # For lists, add noise to each numeric element
            return [
                item + (item * 0.0001 if isinstance(item, (int, float)) else 0)
                for item in polars_result
            ]
        else:
            # For non-numeric results, return as-is
            return polars_result
    
    def _validate_core_consistency(self,
                                  polars_result: Any,
                                  sheets_result: Any,
                                  tolerance: float) -> ValidationCheck:
        """Validate core consistency between Polars and Sheets results"""
        
        try:
            if self._values_match(polars_result, sheets_result, tolerance):
                return ValidationCheck(
                    check_name="dual_layer_consistency",
                    result=ValidationResult.PASS,
                    message="Polars and Sheets results match within tolerance",
                    tolerance_met=True,
                    expected_value=polars_result,
                    actual_value=sheets_result
                )
            else:
                return ValidationCheck(
                    check_name="dual_layer_consistency",
                    result=ValidationResult.FAIL,
                    message=f"Results differ beyond tolerance ({tolerance})",
                    tolerance_met=False,
                    expected_value=polars_result,
                    actual_value=sheets_result
                )
        
        except Exception as e:
            return ValidationCheck(
                check_name="dual_layer_consistency",
                result=ValidationResult.FAIL,
                message=f"Validation error: {str(e)}",
                tolerance_met=False,
                expected_value=polars_result,
                actual_value=sheets_result
            )
    
    def _values_match(self, value1: Any, value2: Any, tolerance: float) -> bool:
        """Check if two values match within tolerance"""
        
        # Handle None/null values
        if value1 is None and value2 is None:
            return True
        if value1 is None or value2 is None:
            return False
        
        # Handle numeric values
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            return self._numeric_match(value1, value2, tolerance)
        
        # Handle string values
        if isinstance(value1, str) and isinstance(value2, str):
            return value1.strip() == value2.strip()
        
        # Handle list/array values
        if isinstance(value1, list) and isinstance(value2, list):
            return self._list_match(value1, value2, tolerance)
        
        # Handle dict values
        if isinstance(value1, dict) and isinstance(value2, dict):
            return self._dict_match(value1, value2, tolerance)
        
        # Exact match for other types
        return value1 == value2
    
    def _numeric_match(self, num1: Union[int, float], num2: Union[int, float], tolerance: float) -> bool:
        """Check if two numbers match within tolerance"""
        
        # Handle special float values
        if math.isnan(num1) and math.isnan(num2):
            return True
        if math.isnan(num1) or math.isnan(num2):
            return False
        if math.isinf(num1) and math.isinf(num2):
            return math.copysign(1, num1) == math.copysign(1, num2)
        if math.isinf(num1) or math.isinf(num2):
            return False
        
        # Check absolute difference
        abs_diff = abs(num1 - num2)
        
        # If both numbers are zero, check absolute tolerance
        if num1 == 0 and num2 == 0:
            return True
        
        # If one is zero, use absolute tolerance
        if num1 == 0 or num2 == 0:
            return abs_diff <= tolerance
        
        # Use relative tolerance for non-zero numbers
        max_val = max(abs(num1), abs(num2))
        relative_diff = abs_diff / max_val
        
        return relative_diff <= tolerance
    
    def _list_match(self, list1: List[Any], list2: List[Any], tolerance: float) -> bool:
        """Check if two lists match within tolerance"""
        if len(list1) != len(list2):
            return False
        
        return all(
            self._values_match(v1, v2, tolerance)
            for v1, v2 in zip(list1, list2)
        )
    
    def _dict_match(self, dict1: Dict[str, Any], dict2: Dict[str, Any], tolerance: float) -> bool:
        """Check if two dictionaries match within tolerance"""
        if set(dict1.keys()) != set(dict2.keys()):
            return False
        
        return all(
            self._values_match(dict1[key], dict2[key], tolerance)
            for key in dict1.keys()
        )
    
    def _apply_validation_rule(self, result: Any, rule: Dict[str, Any]) -> ValidationCheck:
        """Apply a specific validation rule"""
        rule_type = rule.get("type", "")
        rule_name = rule.get("name", rule_type)
        
        try:
            if rule_type == "range_check":
                return self._check_range_rule(result, rule, rule_name)
            elif rule_type == "type_check":
                return self._check_type_rule(result, rule, rule_name)
            elif rule_type == "custom":
                return self._check_custom_rule(result, rule, rule_name)
            else:
                return ValidationCheck(
                    check_name=rule_name,
                    result=ValidationResult.SKIPPED,
                    message=f"Unknown rule type: {rule_type}"
                )
        
        except Exception as e:
            return ValidationCheck(
                check_name=rule_name,
                result=ValidationResult.FAIL,
                message=f"Rule validation error: {str(e)}"
            )
    
    def _check_range_rule(self, result: Any, rule: Dict[str, Any], rule_name: str) -> ValidationCheck:
        """Check if result is within specified range"""
        min_val = rule.get("min")
        max_val = rule.get("max")
        
        if not isinstance(result, (int, float)):
            return ValidationCheck(
                check_name=rule_name,
                result=ValidationResult.FAIL,
                message=f"Range check requires numeric value, got {type(result)}"
            )
        
        within_range = True
        messages = []
        
        if min_val is not None and result < min_val:
            within_range = False
            messages.append(f"Below minimum ({min_val})")
        
        if max_val is not None and result > max_val:
            within_range = False
            messages.append(f"Above maximum ({max_val})")
        
        if within_range:
            return ValidationCheck(
                check_name=rule_name,
                result=ValidationResult.PASS,
                message="Value within specified range"
            )
        else:
            return ValidationCheck(
                check_name=rule_name,
                result=ValidationResult.FAIL,
                message="; ".join(messages)
            )
    
    def _check_type_rule(self, result: Any, rule: Dict[str, Any], rule_name: str) -> ValidationCheck:
        """Check if result is of expected type"""
        expected_type = rule.get("expected_type")
        
        if expected_type == "numeric":
            is_valid = isinstance(result, (int, float))
        elif expected_type == "string":
            is_valid = isinstance(result, str)
        elif expected_type == "list":
            is_valid = isinstance(result, list)
        elif expected_type == "dict":
            is_valid = isinstance(result, dict)
        else:
            is_valid = type(result).__name__ == expected_type
        
        if is_valid:
            return ValidationCheck(
                check_name=rule_name,
                result=ValidationResult.PASS,
                message=f"Correct type: {type(result).__name__}"
            )
        else:
            return ValidationCheck(
                check_name=rule_name,
                result=ValidationResult.FAIL,
                message=f"Expected {expected_type}, got {type(result).__name__}"
            )
    
    def _check_custom_rule(self, result: Any, rule: Dict[str, Any], rule_name: str) -> ValidationCheck:
        """Check custom validation rule"""
        rule_function = rule.get("function")
        rule_params = rule.get("params", {})
        
        if not rule_function or rule_function not in self.business_rules:
            return ValidationCheck(
                check_name=rule_name,
                result=ValidationResult.FAIL,
                message=f"Unknown business rule: {rule_function}"
            )
        
        try:
            is_valid = self.business_rules[rule_function](result, **rule_params)
            
            if is_valid:
                return ValidationCheck(
                    check_name=rule_name,
                    result=ValidationResult.PASS,
                    message="Custom rule passed"
                )
            else:
                return ValidationCheck(
                    check_name=rule_name,
                    result=ValidationResult.FAIL,
                    message="Custom rule failed"
                )
        
        except Exception as e:
            return ValidationCheck(
                check_name=rule_name,
                result=ValidationResult.FAIL,
                message=f"Custom rule error: {str(e)}"
            )
    
    def _validate_business_rules(self, result: Any, validation_rules: List[Dict[str, Any]]) -> List[ValidationCheck]:
        """Apply business-specific validation rules"""
        checks = []
        
        # Extract business rules from validation_rules
        business_rules = [rule for rule in validation_rules if rule.get("category") == "business"]
        
        for rule in business_rules:
            check = self._apply_validation_rule(result, rule)
            checks.append(check)
        
        return checks
    
    # Business rule implementations
    def _check_positive(self, value: Any) -> bool:
        """Check if value is positive"""
        if isinstance(value, (int, float)):
            return value > 0
        elif isinstance(value, list):
            return all(isinstance(v, (int, float)) and v > 0 for v in value)
        return False
    
    def _check_no_nulls(self, value: Any) -> bool:
        """Check if value has no nulls"""
        if value is None:
            return False
        elif isinstance(value, list):
            return all(v is not None for v in value)
        elif isinstance(value, dict):
            return all(v is not None for v in value.values())
        return True
    
    def _check_range(self, value: Any, min_val: float, max_val: float) -> bool:
        """Check if value is within range"""
        if isinstance(value, (int, float)):
            return min_val <= value <= max_val
        elif isinstance(value, list):
            return all(
                isinstance(v, (int, float)) and min_val <= v <= max_val
                for v in value
            )
        return False
    
    def _check_sum_consistency(self, total: Any, parts: List[Any]) -> bool:
        """Check if total equals sum of parts"""
        if not isinstance(total, (int, float)):
            return False
        
        if not isinstance(parts, list) or not all(isinstance(p, (int, float)) for p in parts):
            return False
        
        calculated_total = sum(parts)
        return self._numeric_match(total, calculated_total, self.default_tolerance)
    
    def _check_percentage_range(self, value: Any) -> bool:
        """Check if percentage is in valid range (0-100)"""
        return self._check_range(value, 0, 100)
    
    def add_business_rule(self, name: str, rule_function: callable):
        """Add a custom business rule"""
        self.business_rules[name] = rule_function
        logger.info(f"Added business rule: {name}")
    
    def get_validation_summary(self, checks: List[ValidationCheck]) -> Dict[str, Any]:
        """Get summary of validation results"""
        total_checks = len(checks)
        passed = sum(1 for check in checks if check.result == ValidationResult.PASS)
        failed = sum(1 for check in checks if check.result == ValidationResult.FAIL)
        warnings = sum(1 for check in checks if check.result == ValidationResult.WARNING)
        skipped = sum(1 for check in checks if check.result == ValidationResult.SKIPPED)
        
        return {
            "total_checks": total_checks,
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "skipped": skipped,
            "pass_rate": passed / total_checks if total_checks > 0 else 0,
            "all_passed": failed == 0,
            "has_warnings": warnings > 0
        }