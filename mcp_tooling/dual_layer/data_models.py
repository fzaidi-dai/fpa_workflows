"""
Data Models for Dual-Layer Execution System

Defines the core data structures used throughout the dual-layer execution system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl


class CellType(Enum):
    """Types of cells in the dual-layer system"""
    SOURCE_DATA = "source_data"
    CALCULATION = "calculation" 
    COMPLEX_CALCULATION = "complex_calculation"
    PIVOT_TABLE = "pivot_table"
    CHART_DATA = "chart_data"


class OperationType(Enum):
    """Types of operations that can be executed"""
    AGGREGATION = "aggregation"
    TRANSFORMATION = "transformation"
    LOOKUP = "lookup"
    CONDITIONAL = "conditional"
    ARRAY_FORMULA = "array_formula"
    PIVOT = "pivot"


class ValidationResult(Enum):
    """Results of validation checks"""
    PASS = "pass"
    FAIL = "fail" 
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class CellContext:
    """Context information for a cell in the dual-layer system"""
    range_spec: str
    cell_type: CellType
    values: Optional[Union[List[Any], Any]] = None
    formula: Optional[str] = None
    computed_value: Optional[Any] = None
    pivot_config: Optional[Dict[str, Any]] = None
    validation_note: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "range_spec": self.range_spec,
            "cell_type": self.cell_type.value,
            "values": self.values,
            "formula": self.formula,
            "computed_value": self.computed_value,
            "pivot_config": self.pivot_config,
            "validation_note": self.validation_note
        }


@dataclass
class PlanStep:
    """A single step in an execution plan"""
    step_id: str
    description: str
    operation_type: OperationType
    operation: Dict[str, Any]
    input_data: Union[pl.DataFrame, str, Path]
    output_range: str
    sheet_context: Dict[str, Any]
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "step_id": self.step_id,
            "description": self.description,
            "operation_type": self.operation_type.value,
            "operation": self.operation,
            "input_data": str(self.input_data) if not isinstance(self.input_data, dict) else self.input_data,
            "output_range": self.output_range,
            "sheet_context": self.sheet_context,
            "validation_rules": self.validation_rules,
            "dependencies": self.dependencies,
            "timeout_seconds": self.timeout_seconds
        }


@dataclass
class ValidationCheck:
    """Result of a single validation check"""
    check_name: str
    result: ValidationResult
    message: str
    tolerance_met: Optional[bool] = None
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "check_name": self.check_name,
            "result": self.result.value,
            "message": self.message,
            "tolerance_met": self.tolerance_met,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value
        }


@dataclass 
class StepResult:
    """Result of executing a plan step"""
    step_id: str
    success: bool
    polars_result: Any
    sheets_formula: Optional[str] = None
    validation: Optional[List[ValidationCheck]] = None
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None
    sheets_range: Optional[str] = None
    cell_contexts: List[CellContext] = field(default_factory=list)
    
    @property
    def all_validations_passed(self) -> bool:
        """Check if all validations passed"""
        if not self.validation:
            return True
        return all(check.result == ValidationResult.PASS for check in self.validation)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "step_id": self.step_id,
            "success": self.success,
            "polars_result": str(self.polars_result),
            "sheets_formula": self.sheets_formula,
            "validation": [check.to_dict() for check in (self.validation or [])],
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "sheets_range": self.sheets_range,
            "cell_contexts": [ctx.to_dict() for ctx in self.cell_contexts]
        }


@dataclass
class ExecutionPlan:
    """Complete execution plan with multiple steps"""
    plan_id: str
    name: str
    description: str
    steps: List[PlanStep]
    created_at: datetime = field(default_factory=datetime.now)
    target_spreadsheet_id: Optional[str] = None
    
    def get_step(self, step_id: str) -> Optional[PlanStep]:
        """Get a step by its ID"""
        return next((step for step in self.steps if step.step_id == step_id), None)
    
    def get_dependencies(self, step_id: str) -> List[PlanStep]:
        """Get all dependencies for a step"""
        step = self.get_step(step_id)
        if not step:
            return []
        
        return [self.get_step(dep_id) for dep_id in step.dependencies if self.get_step(dep_id)]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "plan_id": self.plan_id,
            "name": self.name,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
            "created_at": self.created_at.isoformat(),
            "target_spreadsheet_id": self.target_spreadsheet_id
        }


@dataclass
class FormulaMapping:
    """Enhanced mapping between Polars operations and Google Sheets formulas with rich metadata"""
    operation_name: str
    polars_code: str
    sheets_formula: str
    validation_type: str = "exact_match"
    helper_columns: List[str] = field(default_factory=list)
    complexity_level: str = "simple"  # simple, moderate, complex
    description: str = ""
    implementation_status: str = "pending"  # pending, completed, deprecated
    
    # Enhanced metadata fields for comprehensive documentation
    syntax: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    examples: Dict[str, str] = field(default_factory=dict)
    use_cases: List[str] = field(default_factory=list)
    category: str = ""
    notes: str = ""
    version_added: str = ""
    polars_implementation: str = ""  # Detailed implementation notes
    sheets_function: str = ""  # Primary Google Sheets function name
    array_context: bool = False  # Whether function works in array context
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "operation_name": self.operation_name,
            "polars_code": self.polars_code,
            "sheets_formula": self.sheets_formula,
            "validation_type": self.validation_type,
            "helper_columns": self.helper_columns,
            "complexity_level": self.complexity_level,
            "description": self.description,
            "implementation_status": self.implementation_status,
            "syntax": self.syntax,
            "parameters": self.parameters,
            "examples": self.examples,
            "use_cases": self.use_cases,
            "category": self.category,
            "notes": self.notes,
            "version_added": self.version_added,
            "polars_implementation": self.polars_implementation,
            "sheets_function": self.sheets_function,
            "array_context": self.array_context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FormulaMapping':
        """Create FormulaMapping from dictionary with backwards compatibility"""
        # Required fields
        operation_name = data.get("operation_name", "")
        polars_code = data.get("polars", data.get("polars_code", ""))
        sheets_formula = data.get("sheets", data.get("sheets_formula", ""))
        
        # Optional fields with backwards compatibility
        return cls(
            operation_name=operation_name,
            polars_code=polars_code,
            sheets_formula=sheets_formula,
            validation_type=data.get("validation", data.get("validation_type", "exact_match")),
            helper_columns=data.get("helper_columns", []),
            complexity_level=data.get("complexity_level", "simple"),
            description=data.get("description", ""),
            implementation_status=data.get("implementation_status", "pending"),
            syntax=data.get("syntax", ""),
            parameters=data.get("parameters", {}),
            examples=data.get("examples", {}),
            use_cases=data.get("use_cases", []),
            category=data.get("category", ""),
            notes=data.get("notes", ""),
            version_added=data.get("version_added", ""),
            polars_implementation=data.get("polars_implementation", data.get("polars_mapping", "")),
            sheets_function=data.get("sheets_function", ""),
            array_context=data.get("array_context", False)
        )


@dataclass
class PolarsExecutor:
    """Configuration for Polars execution"""
    
    def execute(self, operation: Dict[str, Any], input_data: Union[pl.DataFrame, str, Path]) -> Any:
        """Execute a Polars operation"""
        # Load data if needed
        if isinstance(input_data, (str, Path)):
            if str(input_data).endswith('.parquet'):
                df = pl.read_parquet(input_data)
            elif str(input_data).endswith('.csv'):
                df = pl.read_csv(input_data)
            else:
                raise ValueError(f"Unsupported file format: {input_data}")
        else:
            df = input_data
        
        operation_type = operation.get("type")
        
        if operation_type == "sum":
            column = operation.get("column")
            if column:
                return df[column].sum()
            else:
                # Sum all numeric columns
                numeric_cols = [col for col in df.columns if df[col].dtype.is_numeric()]
                return df.select(numeric_cols).sum().to_dicts()[0] if numeric_cols else 0
                
        elif operation_type == "average":
            column = operation.get("column")
            if column:
                return df[column].mean()
            else:
                numeric_cols = [col for col in df.columns if df[col].dtype.is_numeric()]
                return df.select(numeric_cols).mean().to_dicts()[0] if numeric_cols else 0
                
        elif operation_type == "filter":
            condition = operation.get("condition")
            if condition:
                # Simple equality filter for now
                column = condition.get("column")
                value = condition.get("value")
                return df.filter(pl.col(column) == value)
            
        elif operation_type == "groupby_sum":
            group_by = operation.get("group_by")
            sum_column = operation.get("sum_column")
            if group_by and sum_column:
                return df.group_by(group_by).agg(pl.col(sum_column).sum())
                
        elif operation_type == "stdev":
            column = operation.get("column")
            if column:
                return df[column].std(ddof=1)
            else:
                # STDEV of all numeric columns
                numeric_cols = [col for col in df.columns if df[col].dtype.is_numeric()]
                if numeric_cols:
                    # Combine all numeric values and calculate stdev
                    all_values = []
                    for col in numeric_cols:
                        all_values.extend(df[col].drop_nulls().to_list())
                    if all_values:
                        temp_df = pl.DataFrame({"values": all_values})
                        return temp_df["values"].std(ddof=1)
                return 0
                
        elif operation_type == "var":
            column = operation.get("column")
            if column:
                return df[column].var(ddof=1)
            else:
                # VAR of all numeric columns
                numeric_cols = [col for col in df.columns if df[col].dtype.is_numeric()]
                if numeric_cols:
                    # Combine all numeric values and calculate variance
                    all_values = []
                    for col in numeric_cols:
                        all_values.extend(df[col].drop_nulls().to_list())
                    if all_values:
                        temp_df = pl.DataFrame({"values": all_values})
                        return temp_df["values"].var(ddof=1)
                return 0
                
        elif operation_type == "npv":
            rate = operation.get("rate")
            cash_flow_column = operation.get("cash_flow_column")
            if rate is not None and cash_flow_column and cash_flow_column in df.columns:
                cash_flows = df[cash_flow_column].drop_nulls().to_list()
                npv = sum(cf / ((1 + rate) ** (i + 1)) for i, cf in enumerate(cash_flows))
                return float(npv)
            return 0
            
        elif operation_type == "irr":
            cash_flow_column = operation.get("cash_flow_column")
            guess = operation.get("guess", 0.1)
            if cash_flow_column and cash_flow_column in df.columns:
                from scipy.optimize import fsolve
                cash_flows = df[cash_flow_column].drop_nulls().to_list()
                
                def npv_func(rate):
                    return sum(cf / ((1 + rate) ** (i + 1)) for i, cf in enumerate(cash_flows))
                
                try:
                    result = fsolve(npv_func, guess)[0]
                    return float(result)
                except:
                    return guess
            return 0
            
        elif operation_type == "pv":
            rate = operation.get("rate")
            nper = operation.get("nper")
            pmt = operation.get("pmt")
            fv = operation.get("fv", 0)
            type_ = operation.get("type", 0)
            
            if rate is not None and nper is not None and pmt is not None:
                if rate == 0:
                    return -(pmt * nper + fv)
                
                pv_annuity = pmt * (1 - (1 + rate) ** (-nper)) / rate
                pv_lump_sum = fv / ((1 + rate) ** nper)
                
                if type_ == 1:
                    pv_annuity *= (1 + rate)
                
                return -(pv_annuity + pv_lump_sum)
            return 0
            
        elif operation_type == "fv":
            rate = operation.get("rate")
            nper = operation.get("nper") 
            pmt = operation.get("pmt")
            pv = operation.get("pv", 0)
            type_ = operation.get("type", 0)
            
            if rate is not None and nper is not None and pmt is not None:
                if rate == 0:
                    return -(pv + pmt * nper)
                
                fv_annuity = pmt * (((1 + rate) ** nper - 1) / rate)
                fv_lump_sum = pv * ((1 + rate) ** nper)
                
                if type_ == 1:
                    fv_annuity *= (1 + rate)
                
                return -(fv_lump_sum + fv_annuity)
            return 0
            
        elif operation_type == "pmt":
            rate = operation.get("rate")
            nper = operation.get("nper")
            pv = operation.get("pv")
            fv = operation.get("fv", 0)
            type_ = operation.get("type", 0)
            
            if rate is not None and nper is not None and pv is not None:
                if rate == 0:
                    return -(pv + fv) / nper
                
                factor = (1 + rate) ** nper
                pmt = -(pv * factor + fv) * rate / (factor - 1)
                
                if type_ == 1:
                    pmt /= (1 + rate)
                
                return pmt
            return 0
            
        elif operation_type == "xnpv":
            rate = operation.get("rate")
            cash_flow_column = operation.get("cash_flow_column")
            date_column = operation.get("date_column")
            if rate is not None and cash_flow_column and date_column:
                # Simplified XNPV for demo - assume regular periods for now
                if cash_flow_column in df.columns:
                    cash_flows = df[cash_flow_column].drop_nulls().to_list()
                    xnpv = sum(cf / ((1 + rate) ** (i + 1)) for i, cf in enumerate(cash_flows))
                    return float(xnpv)
            return 0
            
        elif operation_type == "xirr":
            cash_flow_column = operation.get("cash_flow_column")
            guess = operation.get("guess", 0.1)
            if cash_flow_column and cash_flow_column in df.columns:
                # Simplified XIRR - use regular IRR logic for demo
                try:
                    from scipy.optimize import fsolve
                    cash_flows = df[cash_flow_column].drop_nulls().to_list()
                    
                    def npv_func(rate):
                        return sum(cf / ((1 + rate) ** (i + 1)) for i, cf in enumerate(cash_flows))
                    
                    result = fsolve(npv_func, guess)[0]
                    return float(result)
                except:
                    return guess
            return 0
            
        elif operation_type == "nper":
            rate = operation.get("rate")
            pmt = operation.get("pmt")
            pv = operation.get("pv")
            fv = operation.get("fv", 0)
            type_ = operation.get("type", 0)
            
            if rate is not None and pmt is not None and pv is not None:
                if rate == 0:
                    return -(pv + fv) / pmt
                
                if type_ == 1:
                    pmt = pmt * (1 + rate)
                
                import math
                numerator = pmt - fv * rate
                denominator = pmt + pv * rate
                
                if numerator > 0 and denominator > 0:
                    nper = math.log(numerator / denominator) / math.log(1 + rate)
                    return float(nper)
            return 0
            
        elif operation_type == "rate":
            nper = operation.get("nper")
            pmt = operation.get("pmt")
            pv = operation.get("pv")
            fv = operation.get("fv", 0)
            guess = operation.get("guess", 0.1)
            
            if nper is not None and pmt is not None and pv is not None:
                # Simplified RATE calculation using approximation
                rate = guess
                for _ in range(10):
                    factor = (1 + rate) ** nper
                    f = pv * factor + pmt * (factor - 1) / rate + fv
                    if abs(f) < 1e-6:
                        break
                    df = pv * nper * factor / (1 + rate) + pmt * (nper * factor - (factor - 1) / rate) / (rate * (1 + rate))
                    if abs(df) > 1e-12:
                        rate = rate - f / df
                return float(rate)
            return 0
            
        elif operation_type == "pivot_sum":
            index_column = operation.get("index_column")
            values_column = operation.get("values_column")
            aggfunc_column = operation.get("aggfunc_column")
            
            if index_column and values_column and index_column in df.columns and values_column in df.columns:
                if aggfunc_column and aggfunc_column in df.columns:
                    # Cross-tabulation pivot
                    result = df.group_by([index_column, aggfunc_column]).agg(
                        pl.col(values_column).sum().alias("sum_" + values_column)
                    )
                    return result.pivot(
                        index=index_column,
                        columns=aggfunc_column,
                        values="sum_" + values_column,
                        aggregate_function="first"
                    )
                else:
                    # Simple groupby sum
                    return df.group_by(index_column).agg(
                        pl.col(values_column).sum().alias("sum_" + values_column)
                    )
            return pl.DataFrame()
            
        elif operation_type == "group_by_sum":
            group_column = operation.get("group_column")
            sum_column = operation.get("sum_column")
            
            if group_column and sum_column and group_column in df.columns and sum_column in df.columns:
                return df.group_by(group_column).agg(
                    pl.col(sum_column).sum().alias("sum_" + sum_column)
                ).sort(group_column)
            return pl.DataFrame()
            
        elif operation_type == "running_total":
            value_column = operation.get("value_column")
            order_column = operation.get("order_column")
            partition_column = operation.get("partition_column")
            
            if value_column and value_column in df.columns:
                # Sort if order column specified
                if order_column and order_column in df.columns:
                    if partition_column and partition_column in df.columns:
                        df = df.sort([partition_column, order_column])
                    else:
                        df = df.sort(order_column)
                
                # Calculate running total
                if partition_column and partition_column in df.columns:
                    return df.with_columns(
                        pl.col(value_column).cum_sum().over(partition_column).alias("running_total")
                    )
                else:
                    return df.with_columns(
                        pl.col(value_column).cum_sum().alias("running_total")
                    )
            return df
            
        elif operation_type == "percent_of_total":
            value_column = operation.get("value_column")
            partition_column = operation.get("partition_column")
            
            if value_column and value_column in df.columns:
                if partition_column and partition_column in df.columns:
                    return df.with_columns(
                        (pl.col(value_column) / pl.col(value_column).sum().over(partition_column) * 100).alias("percent_of_total")
                    )
                else:
                    total = df[value_column].sum()
                    return df.with_columns(
                        (pl.col(value_column) / total * 100).alias("percent_of_total")
                    )
            return df
            
        elif operation_type == "moving_sum":
            value_column = operation.get("value_column")
            window_size = operation.get("window_size", 3)
            order_column = operation.get("order_column")
            partition_column = operation.get("partition_column")
            
            if value_column and value_column in df.columns:
                # Sort if order column specified
                if order_column and order_column in df.columns:
                    if partition_column and partition_column in df.columns:
                        df = df.sort([partition_column, order_column])
                    else:
                        df = df.sort(order_column)
                
                # Calculate moving sum
                if partition_column and partition_column in df.columns:
                    return df.with_columns(
                        pl.col(value_column).rolling_sum(window_size=window_size).over(partition_column).alias("moving_sum")
                    )
                else:
                    return df.with_columns(
                        pl.col(value_column).rolling_sum(window_size=window_size).alias("moving_sum")
                    )
            return df
            
        elif operation_type == "rank":
            value_column = operation.get("value_column")
            order_desc = operation.get("order_desc", True)
            
            if value_column and value_column in df.columns:
                if order_desc:
                    return df.with_columns(
                        pl.col(value_column).rank(method="ordinal", descending=True).alias("rank")
                    ).sort("rank")
                else:
                    return df.with_columns(
                        pl.col(value_column).rank(method="ordinal", descending=False).alias("rank")
                    ).sort("rank")
            return df
            
        elif operation_type == "subtotal":
            function_num = operation.get("function_num", 9)  # Default to SUM
            column = operation.get("column")
            
            if column and column in df.columns:
                values = df[column].drop_nulls()
                
                if function_num == 1:  # AVERAGE
                    return float(values.mean())
                elif function_num == 2:  # COUNT
                    return float(values.count())
                elif function_num == 4:  # MAX
                    return float(values.max())
                elif function_num == 5:  # MIN
                    return float(values.min())
                elif function_num == 7:  # STDEV
                    return float(values.std(ddof=1))
                elif function_num == 9:  # SUM
                    return float(values.sum())
            return 0
        
        # Add more operation types as needed
        raise ValueError(f"Unsupported operation type: {operation_type}")


@dataclass  
class SheetsPusher:
    """Configuration for pushing data to Google Sheets"""
    
    def push_with_formula(self, value: Any, formula: str, range_spec: str) -> bool:
        """Push both computed value and formula to Google Sheets"""
        # This would interface with Google Sheets API
        # For now, return success simulation
        print(f"Pushing to {range_spec}: formula={formula}, value={value}")
        return True
    
    def push_values(self, range_spec: str, values: List[Any]) -> bool:
        """Push raw values to Google Sheets"""
        print(f"Pushing values to {range_spec}: {values}")
        return True
        
    def push_formula(self, range_spec: str, formula: str) -> bool:
        """Push formula to Google Sheets"""
        print(f"Pushing formula to {range_spec}: {formula}")
        return True
        
    def add_note(self, range_spec: str, note: str) -> bool:
        """Add a note to a cell"""
        print(f"Adding note to {range_spec}: {note}")
        return True
        
    def create_pivot_table(self, pivot_config: Dict[str, Any]) -> bool:
        """Create a native Google Sheets pivot table"""
        print(f"Creating pivot table: {pivot_config}")
        return True