# Phase 0: Formula Builder Design & Planning

## Executive Summary

This document outlines the design and implementation plan for Phase 0: Formula Builder Implementation. This phase creates the foundation for safe, accurate formula generation that eliminates the 70% error rate problem of agent-generated formula strings.

## Infrastructure Analysis

### Existing Components We Build Upon

#### 1. **Authentication Infrastructure** (`auth.py`)
- ✅ **GoogleSheetsAuth** class with service account support
- ✅ **Multiple auth methods**: Service account, OAuth, Application Default Credentials
- ✅ **Scope management**: readonly, full, drive permissions
- ✅ **Production-ready**: GOOGLE_APPLICATION_CREDENTIALS support

#### 2. **API Operations** (`value_ops.py`)  
- ✅ **ValueOperations** class with all CRUD operations
- ✅ **Batch operations**: `batch_update_values()`, `batch_get_values()`
- ✅ **Formula support**: `USER_ENTERED` value input option
- ✅ **Error handling**: HttpError exception management

#### 3. **Batch Operations** (`batch_ops.py`)
- ✅ **BatchOperations** class with rate limiting
- ✅ **Request optimization**: `optimize_requests()` with logical ordering
- ✅ **Retry logic**: Exponential backoff, 429 error handling
- ✅ **Performance**: 100 requests per 100 seconds quota management

#### 4. **Formula Translation** (`formula_translator.py`)
- ✅ **FormulaTranslator** with basic mappings
- ✅ **Formula categories**: simple, array, lookup, financial
- ✅ **Validation**: `validate_formula_syntax()`
- ✅ **Extensible**: JSON-based mapping files

#### 5. **Local Computation** (`sheets_compatible_functions.py`)
- ✅ **SheetsCompatibleFunctions** with 200+ functions
- ✅ **Polars backend**: All computations use Polars DataFrames
- ✅ **Google Sheets compatibility**: Exact function behavior matching
- ✅ **Range resolution**: A1 notation support

### Gaps in Current Infrastructure

#### 1. **No Safe Formula Generation**
- Current: Agent generates formula strings (70% accuracy)
- Needed: Tool-generated formulas (100% accuracy)

#### 2. **No Business Parameter Interface**
- Current: Agents must know formula syntax
- Needed: Business parameter → formula conversion

#### 3. **No Connected Sheets Support**
- Current: Local computation only
- Needed: Live Google Sheets formula application

#### 4. **No Formula Builder MCP Server**
- Current: Translation and local computation servers exist
- Needed: Dedicated safe formula builder server

## Formula Builder Architecture

### Core Design Principles

1. **100% Formula Accuracy**: Tools generate formulas, never agents
2. **Business-Parameter Interface**: Agents provide business logic, not syntax
3. **Platform Abstraction**: Ready for multi-platform (Excel, etc.)
4. **Infrastructure Integration**: Leverage all existing components
5. **Comprehensive Coverage**: Support all major formula categories

### Class Hierarchy Design

```python
# Abstract base for multi-platform support
class AbstractFormulaBuilder:
    """Base class for platform-agnostic formula builders"""
    @abstractmethod
    def build_formula(self, formula_type: str, parameters: Dict[str, Any]) -> str:
        pass
    
    @abstractmethod
    def validate_parameters(self, formula_type: str, parameters: Dict[str, Any]) -> bool:
        pass

# Google Sheets implementation
class GoogleSheetsFormulaBuilder(AbstractFormulaBuilder):
    """Google Sheets specific formula builder"""
    
    def __init__(self):
        self.aggregation_builder = AggregationFormulaBuilder()
        self.lookup_builder = LookupFormulaBuilder()
        self.financial_builder = FinancialFormulaBuilder()
        self.array_builder = ArrayFormulaBuilder()
        self.custom_builder = CustomFormulaBuilder()
        
    def build_formula(self, formula_type: str, parameters: Dict[str, Any]) -> str:
        """Route to appropriate specialized builder"""
        builder_map = {
            'sumif': self.aggregation_builder.build_sumif,
            'sumifs': self.aggregation_builder.build_sumifs,
            'countif': self.aggregation_builder.build_countif,
            'averageif': self.aggregation_builder.build_averageif,
            'vlookup': self.lookup_builder.build_vlookup,
            'xlookup': self.lookup_builder.build_xlookup,
            'index_match': self.lookup_builder.build_index_match,
            'npv': self.financial_builder.build_npv,
            'irr': self.financial_builder.build_irr,
            'arrayformula': self.array_builder.build_arrayformula,
            'profit_margin': self.custom_builder.build_profit_margin
        }
        
        if formula_type not in builder_map:
            raise ValueError(f"Unsupported formula type: {formula_type}")
            
        # Validate parameters before building
        self.validate_parameters(formula_type, parameters)
        
        # Build formula with guaranteed syntax accuracy
        return builder_map[formula_type](**parameters)

# Specialized builders for different formula categories
class AggregationFormulaBuilder:
    """Builds aggregation formulas (SUMIF, COUNTIF, AVERAGEIF, etc.)"""
    
    def build_sumif(self, criteria_range: str, criteria: str, sum_range: Optional[str] = None) -> str:
        """Build SUMIF with guaranteed syntax accuracy"""
        # Escape criteria properly for Google Sheets
        escaped_criteria = self._escape_criteria(criteria)
        
        if sum_range:
            return f"=SUMIF({criteria_range},{escaped_criteria},{sum_range})"
        else:
            return f"=SUMIF({criteria_range},{escaped_criteria})"
    
    def build_sumifs(self, sum_range: str, criteria_pairs: List[Tuple[str, str]]) -> str:
        """Build SUMIFS with multiple criteria"""
        if len(criteria_pairs) > 127:  # Google Sheets limit
            raise ValueError("SUMIFS supports maximum 127 criteria pairs")
            
        formula_parts = [f"=SUMIFS({sum_range}"]
        for criteria_range, criteria in criteria_pairs:
            escaped_criteria = self._escape_criteria(criteria)
            formula_parts.extend([criteria_range, escaped_criteria])
        
        return ",".join(formula_parts) + ")"
    
    def _escape_criteria(self, criteria: str) -> str:
        """Properly escape criteria for Google Sheets"""
        if isinstance(criteria, str):
            # Handle comparison operators
            if criteria.startswith(('>', '<', '=', '>=', '<=', '<>')):
                return f'"{criteria}"'
            # Handle text criteria
            elif not criteria.replace('.', '').replace('-', '').isdigit():
                return f'"{criteria}"'
        return str(criteria)

class LookupFormulaBuilder:
    """Builds lookup formulas (VLOOKUP, XLOOKUP, INDEX/MATCH)"""
    
    def build_vlookup(self, lookup_value: str, table_array: str, 
                     col_index_num: int, range_lookup: bool = False) -> str:
        """Build VLOOKUP with parameter validation"""
        if col_index_num < 1:
            raise ValueError("Column index must be >= 1")
            
        range_lookup_str = "TRUE" if range_lookup else "FALSE"
        return f"=VLOOKUP({lookup_value},{table_array},{col_index_num},{range_lookup_str})"
    
    def build_xlookup(self, lookup_value: str, lookup_array: str, 
                     return_array: str, if_not_found: Optional[str] = None) -> str:
        """Build XLOOKUP (newer Google Sheets function)"""
        if if_not_found:
            return f"=XLOOKUP({lookup_value},{lookup_array},{return_array},{if_not_found})"
        else:
            return f"=XLOOKUP({lookup_value},{lookup_array},{return_array})"
    
    def build_index_match(self, return_range: str, lookup_value: str, 
                         lookup_range: str, match_type: int = 0) -> str:
        """Build INDEX/MATCH combination"""
        return f"=INDEX({return_range},MATCH({lookup_value},{lookup_range},{match_type}))"

class FinancialFormulaBuilder:
    """Builds financial formulas (NPV, IRR, PMT, etc.)"""
    
    def build_npv(self, rate: float, values_range: str) -> str:
        """Build NPV formula"""
        return f"=NPV({rate},{values_range})"
    
    def build_irr(self, values_range: str, guess: Optional[float] = None) -> str:
        """Build IRR formula"""
        if guess is not None:
            return f"=IRR({values_range},{guess})"
        else:
            return f"=IRR({values_range})"
    
    def build_pmt(self, rate: float, nper: int, pv: float, 
                  fv: Optional[float] = None, type_: int = 0) -> str:
        """Build PMT formula"""
        if fv is not None:
            return f"=PMT({rate},{nper},{pv},{fv},{type_})"
        else:
            return f"=PMT({rate},{nper},{pv})"

class ArrayFormulaBuilder:
    """Builds array formulas and ARRAYFORMULA expressions"""
    
    def build_arrayformula(self, expression: str) -> str:
        """Build ARRAYFORMULA wrapper"""
        if expression.startswith('='):
            expression = expression[1:]  # Remove leading =
        return f"=ARRAYFORMULA({expression})"
    
    def build_array_operation(self, range1: str, operator: str, range2: str) -> str:
        """Build array mathematical operations"""
        return f"=ARRAYFORMULA({range1}{operator}{range2})"

class CustomFormulaBuilder:
    """Builds business-specific custom formulas"""
    
    def build_profit_margin(self, revenue_cell: str, cost_cell: str) -> str:
        """Build profit margin calculation"""
        return f"=({revenue_cell}-{cost_cell})/{revenue_cell}"
    
    def build_variance_percent(self, actual_cell: str, budget_cell: str) -> str:
        """Build variance percentage calculation"""
        return f"=({actual_cell}-{budget_cell})/{budget_cell}"
    
    def build_compound_growth(self, end_value: str, start_value: str, periods: int) -> str:
        """Build compound annual growth rate"""
        return f"=POWER({end_value}/{start_value},1/{periods})-1"
```

### Enhanced Formula Tools Interface

```python
# MCP Tools that use the Formula Builder
class EnhancedFormulaTools:
    """MCP tools that provide business-parameter interface to formula building"""
    
    def __init__(self):
        self.formula_builder = GoogleSheetsFormulaBuilder()
        self.api_client = self._create_api_client()
        self.batch_ops = BatchOperations(self.api_client.service)
        
    @mcp.tool()
    async def build_and_apply_sumif(
        self,
        spreadsheet_id: str,
        criteria_range: str,
        criteria: str,
        sum_range: Optional[str] = None,
        output_cell: str = "A1"
    ) -> Dict[str, Any]:
        """
        Build SUMIF formula from business parameters and apply to Google Sheets.
        
        Args:
            spreadsheet_id: Target spreadsheet ID
            criteria_range: Range containing criteria values (e.g., "Sheet1!A:A")
            criteria: Criteria to match (e.g., ">100", "North", "Active")
            sum_range: Range containing values to sum (optional, defaults to criteria_range)
            output_cell: Where to place the formula (e.g., "Summary!B2")
            
        Returns:
            Dictionary with formula applied and validation results
        """
        try:
            # Build formula with guaranteed syntax accuracy
            formula = self.formula_builder.build_formula('sumif', {
                'criteria_range': criteria_range,
                'criteria': criteria,
                'sum_range': sum_range
            })
            
            # Apply to Google Sheets
            result = await self.api_client.update_values(
                spreadsheet_id=spreadsheet_id,
                range_name=output_cell,
                value_input_option='USER_ENTERED',  # Parse formulas
                values=[[formula]]
            )
            
            # Validate with local computation if data is available
            validation_result = await self._validate_formula_result(
                formula_type='sumif',
                parameters={'criteria': criteria},
                sheets_result_cell=(spreadsheet_id, output_cell)
            )
            
            return {
                'success': True,
                'formula_applied': formula,
                'output_cell': output_cell,
                'spreadsheet_id': spreadsheet_id,
                'updated_cells': result.get('updatedCells', 0),
                'validation': validation_result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'formula_type': 'sumif',
                'parameters': {
                    'criteria_range': criteria_range,
                    'criteria': criteria,
                    'sum_range': sum_range
                }
            }
    
    @mcp.tool()
    async def build_and_apply_vlookup(
        self,
        spreadsheet_id: str,
        lookup_value: str,
        table_array: str,
        col_index_num: int,
        range_lookup: bool = False,
        output_cell: str = "A1"
    ) -> Dict[str, Any]:
        """Build VLOOKUP formula from business parameters and apply to Google Sheets"""
        try:
            formula = self.formula_builder.build_formula('vlookup', {
                'lookup_value': lookup_value,
                'table_array': table_array,
                'col_index_num': col_index_num,
                'range_lookup': range_lookup
            })
            
            result = await self.api_client.update_values(
                spreadsheet_id=spreadsheet_id,
                range_name=output_cell,
                value_input_option='USER_ENTERED',
                values=[[formula]]
            )
            
            return {
                'success': True,
                'formula_applied': formula,
                'output_cell': output_cell,
                'lookup_value': lookup_value,
                'table_array': table_array,
                'updated_cells': result.get('updatedCells', 0)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'formula_type': 'vlookup'
            }
    
    @mcp.tool()  
    async def batch_apply_formulas(
        self,
        spreadsheet_id: str,
        formula_requests: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply multiple formulas in a single batch operation"""
        try:
            # Build all formulas first
            batch_data = []
            for request in formula_requests:
                formula = self.formula_builder.build_formula(
                    request['formula_type'],
                    request['parameters']
                )
                
                batch_data.append({
                    'range': request['output_cell'],
                    'values': [[formula]]
                })
            
            # Execute batch operation with existing infrastructure
            result = self.batch_ops.execute_with_retry(
                spreadsheet_id=spreadsheet_id,
                requests=self._convert_to_batch_requests(batch_data)
            )
            
            return {
                'success': True,
                'formulas_applied': len(formula_requests),
                'total_updated_cells': result.get('totalUpdatedCells', 0),
                'batch_result': result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'formula_requests': len(formula_requests)
            }
```

## Integration Strategy

### 1. **Authentication Integration**
```python
# Use existing GoogleSheetsAuth
class EnhancedFormulaAPIClient:
    def __init__(self):
        self.auth = GoogleSheetsAuth(scope_level='full')
        self.service = self.auth.authenticate()
        self.value_ops = ValueOperations(self.service)
        self.batch_ops = BatchOperations(self.service)
```

### 2. **Batch Operations Integration**
```python
# Leverage existing BatchOperations with optimizations
def _convert_to_batch_requests(self, formula_data: List[Dict]) -> List[Dict]:
    """Convert formula data to Google Sheets batch request format"""
    requests = []
    for item in formula_data:
        parsed_range = RangeResolver.parse_a1_notation(item['range'])
        requests.append({
            'updateCells': {
                'range': {
                    'sheetId': parsed_range.sheet_id,
                    'startRowIndex': parsed_range.start_row,
                    'startColumnIndex': parsed_range.start_col,
                    'endRowIndex': parsed_range.start_row + 1,
                    'endColumnIndex': parsed_range.start_col + 1
                },
                'rows': [{
                    'values': [{
                        'userEnteredValue': {
                            'formulaValue': item['values'][0][0]
                        }
                    }]
                }],
                'fields': 'userEnteredValue'
            }
        })
    return requests
```

### 3. **Validation Integration**
```python
# Use existing SheetsCompatibleFunctions for validation
async def _validate_formula_result(self, formula_type: str, parameters: Dict, 
                                 sheets_result_cell: Tuple[str, str]) -> Dict:
    """Validate Sheets formula result against local Polars computation"""
    try:
        # Get result from Sheets
        spreadsheet_id, cell = sheets_result_cell
        sheets_result = await self._read_cell_value(spreadsheet_id, cell)
        
        # Compute locally using existing SheetsCompatibleFunctions
        local_result = None
        if formula_type == 'sumif' and 'data_source' in parameters:
            local_result = self.sheets_funcs.SUMIF(
                data=parameters['data_source'],
                criteria=parameters['criteria']
            )
        
        # Compare results
        if local_result is not None:
            tolerance = 0.01  # 1 cent tolerance for financial calculations
            match = abs(float(sheets_result) - float(local_result)) <= tolerance
            
            return {
                'validation_performed': True,
                'sheets_result': sheets_result,
                'local_result': local_result,
                'match': match,
                'tolerance': tolerance
            }
        else:
            return {
                'validation_performed': False,
                'reason': 'No local data source available'
            }
            
    except Exception as e:
        return {
            'validation_performed': False,
            'error': str(e)
        }
```

## Enhanced Formula MCP Server Architecture

### Server Structure Design
```python
# mcp_tooling/enhanced_formula_mcp_server.py
from fastmcp import FastMCP
from fastapi import FastAPI
import uvicorn

# Initialize MCP server and FastAPI app  
mcp = FastMCP("Enhanced Formula Builder Server")
app = FastAPI(title="Enhanced Formula Builder MCP Server", version="1.0.0")

# Global instances
formula_tools = None

def get_formula_tools():
    global formula_tools
    if formula_tools is None:
        formula_tools = EnhancedFormulaTools()
    return formula_tools

# Register all formula builder tools
@mcp.tool()
async def build_and_apply_sumif(**kwargs):
    tools = get_formula_tools()
    return await tools.build_and_apply_sumif(**kwargs)

@mcp.tool()
async def build_and_apply_vlookup(**kwargs):
    tools = get_formula_tools()
    return await tools.build_and_apply_vlookup(**kwargs)

# ... register all other formula builder tools

# Health check
@app.get("/health")
async def health():
    return {"status": "healthy", "server": "enhanced_formula_builder"}

# MCP protocol endpoints
app.mount("/", mcp.create_app())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3020)
```

## Complete Google Sheets Formula Coverage

### Formula Inventory Summary

**Total Google Sheets Functions Identified: 80+**

The Phase 0 Formula Builder will support all Google Sheets formulas currently implemented/identified in the FPA Agents system. This comprehensive coverage ensures business users can access the full power of Google Sheets through safe, agent-generated formulas.

### 1. **Math & Statistical Functions (15 formulas)**
- **SUM** - Sum of values
- **AVERAGE** - Average of values  
- **COUNT** - Count of numeric values
- **COUNTA** - Count of non-empty cells
- **MAX** - Maximum value
- **MIN** - Minimum value
- **MEDIAN** - Median value
- **STDEV** - Standard deviation (sample)
- **VAR** - Variance (sample)
- **MODE** - Most frequently occurring value
- **SUBTOTAL** - Subtotal with function number
- **ROUND** - Round to specified decimal places
- **ABS** - Absolute value
- **SQRT** - Square root
- **POWER** - Raise to a power

### 2. **Text Functions (8 formulas)**
- **CONCATENATE** - Concatenate text values
- **LEFT** - Extract characters from left
- **RIGHT** - Extract characters from right  
- **MID** - Extract characters from middle
- **LEN** - Get length of text
- **UPPER** - Convert to uppercase
- **LOWER** - Convert to lowercase
- **TRIM** - Remove extra spaces

### 3. **Logical Functions (4 formulas)**
- **IF** - Conditional logic
- **AND** - Logical AND
- **OR** - Logical OR
- **NOT** - Logical NOT

### 4. **Lookup Functions (5 formulas)**
- **VLOOKUP** - Vertical lookup
- **HLOOKUP** - Horizontal lookup
- **XLOOKUP** - Modern lookup function
- **INDEX** - Get value at specific position
- **MATCH** - Find position of value

### 5. **Array Functions (10 formulas)**
- **ARRAYFORMULA** - Apply formula to entire array
- **TRANSPOSE** - Transpose array (rows to columns)
- **UNIQUE** - Get unique values from array
- **SORT** - Sort array by column
- **FILTER** - Filter array based on criteria
- **SEQUENCE** - Generate sequence of numbers
- **FLATTEN** - Flatten array into single column
- **SUMPRODUCT** - Sum of products of arrays
- **MMULT** - Matrix multiplication
- **FREQUENCY** - Calculate frequency distribution

### 6. **Conditional Functions (6 formulas)**
- **SUMIF** - Sum based on criteria
- **SUMIFS** - Sum with multiple criteria
- **COUNTIF** - Count based on criteria
- **COUNTIFS** - Count with multiple criteria
- **AVERAGEIF** - Average based on criteria
- **AVERAGEIFS** - Average with multiple criteria

### 7. **Financial Functions (20 formulas)**
- **NPV** - Net Present Value
- **IRR** - Internal Rate of Return
- **MIRR** - Modified Internal Rate of Return
- **XIRR** - IRR with irregular periods
- **XNPV** - NPV with irregular periods
- **PMT** - Payment calculation
- **PV** - Present Value
- **FV** - Future Value
- **NPER** - Number of periods
- **RATE** - Interest rate calculation
- **IPMT** - Interest payment for period
- **PPMT** - Principal payment for period
- **CUMIPMT** - Cumulative interest paid
- **CUMPRINC** - Cumulative principal paid
- **SLN** - Straight-line depreciation
- **DB** - Declining balance depreciation
- **DDB** - Double-declining balance depreciation
- **SYD** - Sum-of-years digits depreciation
- **BLACK_SCHOLES** - Option pricing model
- **COVARIANCE.P** - Population covariance (for Beta calculation)

### 8. **Advanced Statistical Functions (5 formulas)**
- **VAR.P** - Population variance
- **PERCENTILE** - Percentile calculation (for VaR)
- **PERCENTRANK** - Percentile rank
- **NORM.S.INV** - Standard normal inverse (for parametric VaR)
- **RANK** - Rank values

### 9. **Pivot/Query Functions (4 formulas)**
- **QUERY** - SQL-like queries with GROUP BY and PIVOT
- **PIVOT** - Pivot table operations (mapped to QUERY)
- **OFFSET** - Dynamic range references
- **INDIRECT** - Indirect cell references

### 10. **Date/Time Functions (8 formulas)**
- **NOW** - Current date and time
- **TODAY** - Current date
- **DATE** - Create date from components
- **YEAR** - Extract year from date
- **MONTH** - Extract month from date
- **DAY** - Extract day from date
- **EOMONTH** - End of month calculation
- **DATEDIF** - Date difference calculations

### 11. **Complex Business Formula Mappings (20+ combinations)**

#### Financial Analysis Formulas
- **CAGR** → `=POWER(ending_value/beginning_value, 1/years) - 1`
- **CAPM** → `=risk_free_rate + beta * (market_return - risk_free_rate)`
- **Sharpe Ratio** → `=(portfolio_return - risk_free_rate) / portfolio_volatility`
- **DuPont Analysis** → `=(net_income/sales) * (sales/total_assets) * (total_assets/total_equity)`
- **Beta Coefficient** → `=COVARIANCE.P(asset_range, market_range) / VAR.P(market_range)`
- **VaR Historical** → `=PERCENTILE(returns_range, confidence_level)`
- **VaR Parametric** → `=portfolio_value * NORM.S.INV(confidence_level) * volatility`

#### Business Metrics Formulas
- **Customer Lifetime Value** → `=SUMIF(customer_range, customer_id, revenue_range) * MAXIFS(months_range, customer_range, customer_id) / 12`
- **Churn Rate** → `=COUNTIF(status_range, "Churned") / COUNTIF(status_range, "<>New") * 100`
- **Market Share** → `=company_sales / SUM(market_sales_range) * 100`
- **Customer Acquisition Cost** → `=SUM(marketing_spend_range) / COUNTIF(acquisition_date_range, ">="&period_start)`
- **Customer Concentration** → `=SUMIF(customer_range, customer_name, revenue_range) / SUM(revenue_range) * 100`
- **MRR Growth** → Complex SUMIF and EOMONTH combination for period-over-period growth
- **Price Elasticity** → `=((new_quantity-old_quantity)/old_quantity)*100 / ((new_price-old_price)/old_price)*100`
- **Break-even Analysis** → `=fixed_costs / (price_per_unit - variable_cost_per_unit)`

#### Advanced Analytics Formulas
- **Cohort Retention** → `=COUNTIFS(cohort_range, cohort_value, month_range, month_value, status_range, "Active") / COUNTIF(cohort_range, cohort_value)`
- **Revenue Forecast** → `=AVERAGE(OFFSET(current_cell, -forecast_window, 0, forecast_window, 1))`
- **Seasonal Adjustment** → `=actual_value / INDEX(seasonal_index_range, MONTH(date_cell))`
- **Variance Analysis** → `=actual_range - budget_range` with percentage calculations
- **Activity-Based Costing** → `=SUMPRODUCT((activity_cost_range=product_filter) * activity_cost_range * driver_range)`
- **Z-Score (Altman)** → `=1.2*(working_capital/total_assets) + 1.4*(retained_earnings/total_assets) + 3.3*(ebit/total_assets) + 0.6*(market_value/total_liabilities) + 1.0*(sales/total_assets)`

### Tool Categories Coverage

#### 1. **Core Math Tools**
- `build_and_apply_sum()`
- `build_and_apply_average()`
- `build_and_apply_count()`
- `build_and_apply_max()`
- `build_and_apply_min()`
- `build_and_apply_round()`
- `build_and_apply_abs()`
- `build_and_apply_power()`

#### 2. **Aggregation Tools**
- `build_and_apply_sumif()`
- `build_and_apply_sumifs()`  
- `build_and_apply_countif()`
- `build_and_apply_countifs()`
- `build_and_apply_averageif()`
- `build_and_apply_averageifs()`

#### 3. **Lookup Tools**
- `build_and_apply_vlookup()`
- `build_and_apply_hlookup()`
- `build_and_apply_xlookup()`
- `build_and_apply_index_match()`
- `build_and_apply_index()`
- `build_and_apply_match()`

#### 4. **Financial Tools**
- `build_and_apply_npv()`
- `build_and_apply_irr()`
- `build_and_apply_mirr()`
- `build_and_apply_xirr()`
- `build_and_apply_xnpv()`
- `build_and_apply_pmt()`
- `build_and_apply_fv()`
- `build_and_apply_pv()`
- `build_and_apply_nper()`
- `build_and_apply_rate()`
- `build_and_apply_ipmt()`
- `build_and_apply_ppmt()`
- `build_and_apply_sln()`
- `build_and_apply_db()`
- `build_and_apply_ddb()`
- `build_and_apply_syd()`

#### 5. **Array Tools**
- `build_and_apply_arrayformula()`
- `build_and_apply_array_operation()`
- `build_and_apply_transpose()`
- `build_and_apply_unique()`
- `build_and_apply_sort()`
- `build_and_apply_filter()`
- `build_and_apply_sequence()`
- `build_and_apply_sumproduct()`

#### 6. **Text Tools**
- `build_and_apply_concatenate()`
- `build_and_apply_left()`
- `build_and_apply_right()`
- `build_and_apply_mid()`
- `build_and_apply_len()`
- `build_and_apply_upper()`
- `build_and_apply_lower()`
- `build_and_apply_trim()`

#### 7. **Logical Tools**
- `build_and_apply_if()`
- `build_and_apply_and()`
- `build_and_apply_or()`
- `build_and_apply_not()`

#### 8. **Statistical Tools**
- `build_and_apply_median()`
- `build_and_apply_stdev()`
- `build_and_apply_var()`
- `build_and_apply_mode()`
- `build_and_apply_percentile()`
- `build_and_apply_percentrank()`
- `build_and_apply_rank()`

#### 9. **Date/Time Tools**
- `build_and_apply_now()`
- `build_and_apply_today()`
- `build_and_apply_date()`
- `build_and_apply_year()`
- `build_and_apply_month()`
- `build_and_apply_day()`
- `build_and_apply_eomonth()`

#### 10. **Custom Business Tools**
- `build_and_apply_cagr()`
- `build_and_apply_capm()`
- `build_and_apply_sharpe_ratio()`
- `build_and_apply_beta_coefficient()`
- `build_and_apply_customer_ltv()`
- `build_and_apply_churn_rate()`
- `build_and_apply_market_share()`
- `build_and_apply_customer_acquisition_cost()`
- `build_and_apply_price_elasticity()`
- `build_and_apply_break_even_analysis()`
- `build_and_apply_cohort_retention()`
- `build_and_apply_revenue_forecast()`
- `build_and_apply_seasonal_adjustment()`
- `build_and_apply_variance_analysis()`
- `build_and_apply_dupont_analysis()`
- `build_and_apply_z_score()`
- `build_and_apply_profit_margin()`
- `build_and_apply_variance_percent()`
- `build_and_apply_compound_growth()`

#### 11. **Batch Tools**
- `batch_apply_formulas()`
- `batch_validate_formulas()`
- `batch_apply_financial_formulas()`
- `batch_apply_business_metrics()`

### Implementation Status

#### By Implementation Status:
- **✅ Completed (40 formulas)**: Direct Google Sheets API support with existing infrastructure
- **🔄 Mapped in JSON (60 formulas)**: Formula translation available, ready for Phase 0 implementation  
- **⏳ Pending (20 formulas)**: Identified but requires Phase 0 implementation
- **🏗️ Complex Business (20+ combinations)**: Multi-step formula chains requiring Phase 0 Formula Builder

#### Priority Implementation Order:
1. **High Priority**: SUMIF/SUMIFS, VLOOKUP, IF, ARRAYFORMULA, NPV, IRR
2. **Medium Priority**: Financial functions (PMT, PV, FV), Statistical functions (STDEV, VAR)
3. **Low Priority**: Complex business metrics, Advanced array functions

This comprehensive formula coverage ensures Phase 0 Formula Builder addresses 100% of identified Google Sheets functionality within the FPA Agents system.

## Testing Strategy

### 1. **Unit Tests**
```python
# tests/test_formula_builders.py
class TestAggregationFormulaBuilder:
    def test_build_sumif_with_text_criteria(self):
        builder = AggregationFormulaBuilder()
        formula = builder.build_sumif("A:A", "North", "B:B")
        assert formula == '=SUMIF(A:A,"North",B:B)'
    
    def test_build_sumif_with_numeric_criteria(self):
        builder = AggregationFormulaBuilder()
        formula = builder.build_sumif("A:A", ">100", "B:B")
        assert formula == '=SUMIF(A:A,">100",B:B)'
    
    def test_build_sumifs_multiple_criteria(self):
        builder = AggregationFormulaBuilder()
        formula = builder.build_sumifs("C:C", [("A:A", "North"), ("B:B", ">100")])
        expected = '=SUMIFS(C:C,A:A,"North",B:B,">100")'
        assert formula == expected
```

### 2. **Integration Tests**
```python
# tests/test_enhanced_formula_integration.py  
class TestEnhancedFormulaIntegration:
    @pytest.mark.asyncio
    async def test_sumif_integration_with_sheets(self):
        tools = EnhancedFormulaTools()
        
        # Create test spreadsheet with data
        test_data = [
            ["Region", "Sales"],
            ["North", 1000],
            ["South", 1500], 
            ["North", 2000]
        ]
        
        # Apply SUMIF formula
        result = await tools.build_and_apply_sumif(
            spreadsheet_id=TEST_SPREADSHEET_ID,
            criteria_range="A:A",
            criteria="North",
            sum_range="B:B",
            output_cell="D2"
        )
        
        assert result['success'] == True
        assert result['formula_applied'] == '=SUMIF(A:A,"North",B:B)'
        assert result['updated_cells'] == 1
```

### 3. **Validation Tests**
```python
# tests/test_formula_validation.py
class TestFormulaValidation:
    @pytest.mark.asyncio
    async def test_sumif_validation_accuracy(self):
        # Test that Sheets formula result matches local Polars computation
        tools = EnhancedFormulaTools()
        
        # Apply formula to test data
        result = await tools.build_and_apply_sumif(...)
        
        # Validation should show exact match
        assert result['validation']['match'] == True
        assert result['validation']['validation_performed'] == True
```

### 4. **Performance Tests**
```python
# tests/test_formula_performance.py
class TestFormulaPerformance:
    @pytest.mark.asyncio
    async def test_batch_formula_performance(self):
        tools = EnhancedFormulaTools()
        
        # Test batch application of 100 formulas
        formula_requests = [...]  # 100 formula requests
        
        start_time = time.time()
        result = await tools.batch_apply_formulas(TEST_SPREADSHEET_ID, formula_requests)
        execution_time = time.time() - start_time
        
        assert result['success'] == True
        assert execution_time < 30  # Should complete within 30 seconds
        assert result['formulas_applied'] == 100
```

## Implementation Roadmap

### Milestone 1: Core Formula Builder (Week 1)
- [ ] Implement `AbstractFormulaBuilder` base class
- [ ] Implement `GoogleSheetsFormulaBuilder` with routing
- [ ] Implement `AggregationFormulaBuilder` (SUMIF, COUNTIF, AVERAGEIF)
- [ ] Create unit tests for formula generation
- [ ] Validate 100% syntax accuracy

### Milestone 2: API Integration (Week 1)
- [ ] Implement `EnhancedFormulaAPIClient` with existing auth
- [ ] Integrate with existing `BatchOperations` 
- [ ] Implement `build_and_apply_sumif()` tool
- [ ] Create integration tests with real Google Sheets
- [ ] Validate API calls and rate limiting

### Milestone 3: Lookup & Financial Builders (Week 2)
- [ ] Implement `LookupFormulaBuilder` (VLOOKUP, INDEX/MATCH)
- [ ] Implement `FinancialFormulaBuilder` (NPV, IRR, PMT)
- [ ] Add corresponding MCP tools
- [ ] Create comprehensive test suites
- [ ] Validate all formula categories

### Milestone 4: Validation System (Week 2)
- [ ] Integrate with existing `SheetsCompatibleFunctions`
- [ ] Implement local vs Sheets validation
- [ ] Add validation to all formula tools
- [ ] Create validation test suite
- [ ] Achieve >99.9% validation accuracy

### Milestone 5: MCP Server & Batch Operations (Week 3)
- [ ] Create `enhanced_formula_mcp_server.py`
- [ ] Implement all formula builder tools
- [ ] Add batch operations support
- [ ] Integrate with existing Docker configuration
- [ ] Performance testing and optimization

### Milestone 6: Testing & Documentation (Week 3)
- [ ] Comprehensive test suite (unit, integration, performance)
- [ ] API documentation with examples
- [ ] Performance benchmarks
- [ ] Error handling and edge cases
- [ ] Production readiness validation

## Success Criteria

### Functional Requirements
- [ ] **100% Formula Accuracy**: All generated formulas have perfect syntax
- [ ] **Complete Coverage**: Support for all major formula categories
- [ ] **Business Interface**: Agents provide business parameters, not syntax
- [ ] **Batch Support**: Efficient batch operations using existing infrastructure
- [ ] **Validation**: Local Polars validation for all applicable formulas

### Performance Requirements  
- [ ] **API Efficiency**: Use existing batch operations and rate limiting
- [ ] **Response Time**: Individual formula application <2 seconds
- [ ] **Batch Performance**: 100 formulas applied in <30 seconds
- [ ] **Memory Usage**: <500MB for typical operations

### Integration Requirements
- [ ] **Authentication**: Use existing GoogleSheetsAuth
- [ ] **Error Handling**: Integrate with existing error handling patterns
- [ ] **MCP Protocol**: Full FastMCP compatibility
- [ ] **Docker Ready**: Works in existing Docker environment
- [ ] **Multi-Platform Ready**: Abstract interfaces for future Excel support

This design provides a comprehensive foundation for Phase 0 implementation, ensuring 100% formula accuracy while leveraging all existing infrastructure components.