## 11. Excel Style Misc Utils

### FORMULATEXT

**Purpose:** Returns the formula in a referenced cell as text, which can help in auditing or documentation.

**Parameters:**
- `run_context`: RunContext object for file operations
- `reference`: Cell reference or identifier

**Returns:** str - Text string representing formula (simulated)

**Use Cases:**
- Audit financial model formulas for accuracy
- Document complex calculation logic in spreadsheets
- Extract formula references for model validation
- When you need to "extract formula", "audit calculations", "document logic", or "validate formulas"

**Example:**
```python
FORMULATEXT(ctx, "A1")  # Returns "=SUM(B1:B10)"
FORMULATEXT(ctx, {"cell": "B2", "formula": "=AVERAGE(C1:C5)"})  # Returns "=AVERAGE(C1:C5)"
```

### TRANSPOSE

**Purpose:** Converts rows to columns or vice versa, useful for rearranging data.

**Parameters:**
- `run_context`: RunContext object for file operations
- `array`: 2D array, DataFrame, or file path to transpose
- `output_filename`: Filename to save transposed results as parquet file

**Returns:** Path - Path to saved transposed data file

**Use Cases:**
- Restructure financial data for different analysis perspectives
- Convert row-based transaction data to column format
- Prepare data for matrix operations in financial modeling
- When you need to "transpose", "flip data", "restructure", or "convert orientation"

**Example:**
```python
data = [[1, 2, 3], [4, 5, 6]]
TRANSPOSE(ctx, data, output_filename="transposed.parquet")  # Returns transposed data
```

### CELL

**Purpose:** Return information about cell formatting, location, or contents.

**Parameters:**
- `run_context`: RunContext object for file operations
- `info_type`: Type of information to return
- `reference`: Cell reference (optional)

**Returns:** Any - Various types depending on info_type

**Use Cases:**
- Extract cell metadata for financial report generation
- Validate cell references in complex models
- Get system information for environment documentation
- When you need "cell info", "reference details", "metadata", or "environment info"

**Example:**
```python
CELL(ctx, "address", "A1")  # Returns "$A$1"
CELL(ctx, "row", "B5")  # Returns 5
CELL(ctx, "col", "C3")  # Returns 3
CELL(ctx, "type", 123.45)  # Returns "v"
```

### INFO

**Purpose:** Return information about operating environment.

**Parameters:**
- `run_context`: RunContext object for file operations
- `type_text`: Type of information to return

**Returns:** str - Text string with system info

**Use Cases:**
- Document system environment for financial model audits
- Check compatibility for financial analysis tools
- Record system specifications for performance tracking
- When you need "system info", "environment details", "compatibility check", or "audit documentation"

**Example:**
```python
INFO(ctx, "version")  # Returns "Python 3.11.0"
INFO(ctx, "system")  # Returns "Darwin"
INFO(ctx, "release")  # Returns "22.1.0"
```

### N

**Purpose:** Convert value to number using Decimal precision.

**Parameters:**
- `run_context`: RunContext object for file operations
- `value`: Value to convert

**Returns:** Decimal - Numeric value or 0

**Use Cases:**
- Convert mixed data types to consistent numeric format
- Clean financial data with text and numeric values
- Prepare data for mathematical calculations
- When you need to "convert to number", "clean data", "standardize format", or "prepare for calculations"

**Example:**
```python
N(ctx, True)  # Returns Decimal('1')
N(ctx, False)  # Returns Decimal('0')
N(ctx, "123.45")  # Returns Decimal('123.45')
N(ctx, "text")  # Returns Decimal('0')
```

### T

**Purpose:** Convert value to text.

**Parameters:**
- `run_context`: RunContext object for file operations
- `value`: Value to convert

**Returns:** str - Text string or empty string

**Use Cases:**
- Extract text content from mixed data types
- Clean financial data for text-based processing
- Prepare data for reporting and documentation
- When you need to "convert to text", "extract strings", "clean text data", or "prepare for reporting"

**Example:**
```python
T(ctx, 123)  # Returns ""
T(ctx, "Hello")  # Returns "Hello"
T(ctx, True)  # Returns ""
T(ctx, None)  # Returns ""


## 12. Financial Calendar Operations

### FISCAL_YEAR

**Purpose:** Convert calendar date to fiscal year for financial reporting and analysis.

**Parameters:**
- `run_context`: RunContext object for file operations
- `date`: Date to convert (datetime.date or string)
- `fiscal_year_start_month`: Fiscal year start month (1-12, where 1=January)

**Returns:** int - Fiscal year

**Use Cases:**
- Convert transaction dates to appropriate fiscal years for reporting
- Align financial data with company's fiscal calendar
- Calculate fiscal year for budgeting and planning
- When you need to "convert to fiscal year", "align with fiscal calendar", "financial year mapping", or "reporting period conversion"

**Example:**
```python
FISCAL_YEAR(ctx, '2024-03-15', fiscal_year_start_month=4)  # Returns 2023
```

### FISCAL_QUARTER

**Purpose:** Convert date to fiscal quarter for period-based financial analysis.

**Parameters:**
- `run_context`: RunContext object for file operations
- `date`: Date to convert (datetime.date or string)
- `fiscal_year_start_month`: Fiscal year start month (1-12, where 1=January)

**Returns:** str - Fiscal quarter ('Q1', 'Q2', 'Q3', 'Q4')

**Use Cases:**
- Group financial data by fiscal quarters for reporting
- Analyze seasonal business patterns within fiscal periods
- Calculate quarterly financial metrics and KPIs
- When you need "fiscal quarter", "quarter grouping", "period analysis", or "quarterly reporting"

**Example:**
```python
FISCAL_QUARTER(ctx, '2024-03-15', fiscal_year_start_month=4)  # Returns 'Q4'
```

### BUSINESS_DAYS_BETWEEN

**Purpose:** Calculate business days between dates excluding weekends and holidays for financial timeline calculations.

**Parameters:**
- `run_context`: RunContext object for file operations
- `start_date`: Start date (datetime.date or string)
- `end_date`: End date (datetime.date or string)
- `holidays_list`: Optional list of holiday dates to exclude (datetime.date or strings)

**Returns:** int - Number of business days between dates

**Use Cases:**
- Calculate settlement periods for financial transactions
- Determine working day durations for project timelines
- Compute business day intervals for SLA tracking
- When you need "business days", "working days", "settlement calculation", or "exclude weekends"

**Example:**
```python
BUSINESS_DAYS_BETWEEN(ctx, '2024-01-01', '2024-01-31', holidays_list=['2024-01-15'])  # Returns 22
```

### END_OF_PERIOD

**Purpose:** Get end date of period (month, quarter, year) for financial period analysis and reporting.

**Parameters:**
- `run_context`: RunContext object for file operations
- `date`: Date to convert (datetime.date or string)
- `period_type`: Type of period ('month', 'quarter', 'year')

**Returns:** datetime.date - End date of the period

**Use Cases:**
- Calculate month-end, quarter-end, or year-end dates for reporting
- Determine accrual periods for financial calculations
- Set period boundaries for financial analysis
- When you need "period end", "month end", "quarter end", or "reporting periods"

**Example:**
```python
END_OF_PERIOD(ctx, '2024-03-15', period_type='quarter')  # Returns datetime.date(2024, 3, 31)
```

### PERIOD_OVERLAP

**Purpose:** Calculate overlap between two periods in days for contract and revenue analysis.

**Parameters:**
- `run_context`: RunContext object for file operations
- `start1`: First period start date (datetime.date or string)
- `end1`: First period end date (datetime.date or string)
- `start2`: Second period start date (datetime.date or string)
- `end2`: Second period end date (datetime.date or string)

**Returns:** int - Number of overlapping days (0 if no overlap)

**Use Cases:**
- Analyze contract period overlaps for legal and financial review
- Calculate revenue recognition periods for overlapping services
- Determine timeline intersections for project management
- When you need "period overlap", "date intersection", "contract analysis", or "timeline overlap"

**Example:**
```python
PERIOD_OVERLAP(ctx, '2024-01-01', '2024-06-30', '2024-04-01', '2024-09-30')  # Returns 91


## 13. Forecasting and Projection

### LINEAR_FORECAST

**Purpose:** Simple linear trend forecasting using least squares regression for financial trend analysis and projections.

**Parameters:**
- `run_context`: RunContext object for file operations
- `historical_data`: Historical data series (DataFrame, Series, list, or file path)
- `forecast_periods`: Number of periods to forecast
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - DataFrame with forecasted values or path to saved file

**Use Cases:**
- Project revenue trends based on historical performance
- Forecast expense patterns for budget planning
- Predict future cash flows using linear trends
- Simple financial forecasting for short-term planning
- When you need "linear trend", "simple forecast", "trend projection", or "least squares forecasting"

**Example:**
```python
LINEAR_FORECAST(ctx, [100, 105, 110, 115, 120], forecast_periods=3)
LINEAR_FORECAST(ctx, "sales_data.parquet", forecast_periods=12, output_filename="forecast.parquet")
```

### MOVING_AVERAGE

**Purpose:** Calculate moving averages for smoothing and forecasting using Polars rolling operations for trend analysis.

**Parameters:**
- `run_context`: RunContext object for file operations
- `data_series`: Data series (DataFrame, Series, list, or file path)
- `window_size`: Window size for moving average
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - DataFrame with moving averages or path to saved file

**Use Cases:**
- Smooth volatile financial data for trend identification
- Create baseline forecasts using historical averages
- Analyze revenue patterns with rolling windows
- Reduce noise in financial time series data
- When you need "moving average", "rolling average", "trend smoothing", or "data smoothing"

**Example:**
```python
MOVING_AVERAGE(ctx, [10, 12, 14, 16, 18, 20], window_size=3)
MOVING_AVERAGE(ctx, "revenue_data.parquet", window_size=12, output_filename="ma_results.parquet")
```

### EXPONENTIAL_SMOOTHING

**Purpose:** Exponentially weighted forecasting for trend analysis with adaptive weighting of recent observations.

**Parameters:**
- `run_context`: RunContext object for file operations
- `data_series`: Data series (DataFrame, Series, list, or file path)
- `smoothing_alpha`: Smoothing parameter (0 < alpha <= 1)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - DataFrame with smoothed values or path to saved file

**Use Cases:**
- Forecast financial metrics with emphasis on recent data
- Smooth volatile time series with exponential weighting
- Create adaptive forecasts that respond to recent changes
- Weight recent observations more heavily than older data
- When you need "exponential smoothing", "weighted forecast", "adaptive smoothing", or "recent emphasis"

**Example:**
```python
EXPONENTIAL_SMOOTHING(ctx, [100, 105, 102, 108, 110], smoothing_alpha=0.3)
EXPONENTIAL_SMOOTHING(ctx, "sales_data.parquet", smoothing_alpha=0.2, output_filename="smoothed.parquet")
```

### SEASONAL_DECOMPOSE

**Purpose:** Decompose time series into trend, seasonal, residual components using moving averages for pattern analysis.

**Parameters:**
- `run_context`: RunContext object for file operations
- `time_series_data`: Time series data (DataFrame, Series, list, or file path)
- `seasonal_periods`: Number of periods in a season (e.g., 12 for monthly data)
- `model`: Decomposition model ("additive" or "multiplicative")
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - DataFrame with decomposed components or path to saved file

**Use Cases:**
- Identify seasonal patterns in revenue or expense data
- Separate trend from seasonal effects for analysis
- Understand cyclical components in financial metrics
- Decompose complex time series for detailed examination
- When you need "seasonal decomposition", "trend analysis", "pattern identification", or "component analysis"

**Example:**
```python
SEASONAL_DECOMPOSE(ctx, quarterly_sales, seasonal_periods=4)
SEASONAL_DECOMPOSE(ctx, "monthly_data.parquet", seasonal_periods=12, output_filename="decomposed.parquet")
```

### SEASONAL_ADJUST

**Purpose:** Remove seasonal patterns from time series for underlying trend analysis and comparison.

**Parameters:**
- `run_context`: RunContext object for file operations
- `time_series`: Time series data (DataFrame, Series, list, or file path)
- `seasonal_periods`: Number of seasonal periods
- `model`: Adjustment model ("additive" or "multiplicative")
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - DataFrame with seasonally adjusted series or path to saved file

**Use Cases:**
- Remove seasonal effects for true trend comparison
- Analyze underlying performance without seasonal distortion
- Create seasonally adjusted financial metrics
- Compare performance across different time periods fairly
- When you need "seasonal adjustment", "trend isolation", "seasonal removal", or "fair comparison"

**Example:**
```python
SEASONAL_ADJUST(ctx, monthly_sales, seasonal_periods=12)
SEASONAL_ADJUST(ctx, "monthly_data.parquet", seasonal_periods=12, output_filename="adjusted.parquet")
```

### TREND_COEFFICIENT

**Purpose:** Calculate trend coefficient (slope per period) using linear regression for growth rate analysis.

**Parameters:**
- `run_context`: RunContext object for file operations
- `time_series_data`: Time series data (DataFrame, Series, list, or file path)

**Returns:** Decimal - Trend coefficient (slope per period)

**Use Cases:**
- Measure growth rates in revenue or expense trends
- Quantify the rate of change in financial metrics
- Calculate slope coefficients for performance analysis
- Determine directional momentum in financial data
- When you need "trend slope", "growth rate", "rate of change", or "momentum measurement"

**Example:**
```python
TREND_COEFFICIENT(ctx, [100, 105, 110, 115, 120])  # Returns Decimal('5.0')
TREND_COEFFICIENT(ctx, "quarterly_revenue.parquet")  # Returns Decimal('2.5')
```

### CYCLICAL_PATTERN

**Purpose:** Identify cyclical patterns in data using autocorrelation analysis for business cycle detection.

**Parameters:**
- `run_context`: RunContext object for file operations
- `time_series`: Time series data (DataFrame, Series, list, or file path)
- `cycle_length`: Expected cycle length to analyze
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - DataFrame with cyclical indicators or path to saved file

**Use Cases:**
- Detect business cycles in economic or financial data
- Identify recurring patterns in performance metrics
- Analyze cyclical behavior in market data
- Find periodic patterns in time series data
- When you need "cycle detection", "pattern recognition", "business cycles", or "periodic analysis"

**Example:**
```python
CYCLICAL_PATTERN(ctx, economic_data, cycle_length=60)
CYCLICAL_PATTERN(ctx, "economic_data.parquet", cycle_length=60, output_filename="cycles.parquet")
```

### AUTO_CORRELATION

**Purpose:** Calculate autocorrelation of time series using SciPy correlation functions for pattern identification.

**Parameters:**
- `run_context`: RunContext object for file operations
- `time_series`: Time series data (DataFrame, Series, list, or file path)
- `lags`: Number of lags to calculate
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - DataFrame with correlation coefficients or path to saved file

**Use Cases:**
- Identify persistence patterns in financial data
- Detect seasonality through autocorrelation peaks
- Analyze time series dependencies and patterns
- Measure how current values relate to past values
- When you need "autocorrelation", "pattern persistence", "time dependency", or "lag analysis"

**Example:**
```python
AUTO_CORRELATION(ctx, monthly_data, lags=12)
AUTO_CORRELATION(ctx, "monthly_data.parquet", lags=12, output_filename="autocorr.parquet")
```

### HOLT_WINTERS

**Purpose:** Holt-Winters exponential smoothing (Triple exponential smoothing) for complex forecasting with trend and seasonality.

**Parameters:**
- `run_context`: RunContext object for file operations
- `time_series`: Time series data (DataFrame, Series, list, or file path)
- `seasonal_periods`: Number of seasonal periods
- `trend_type`: Trend type ("add" for additive, "mul" for multiplicative, None for no trend)
- `seasonal_type`: Seasonal type ("add" for additive, "mul" for multiplicative)
- `forecast_periods`: Number of periods to forecast (default: 0)
- `alpha`: Level smoothing parameter (auto-optimized if None)
- `beta`: Trend smoothing parameter (auto-optimized if None)
- `gamma`: Seasonal smoothing parameter (auto-optimized if None)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - DataFrame with forecast and components or path to saved file

**Use Cases:**
- Complex forecasting with both trend and seasonal components
- Multi-period financial projections with seasonality
- Advanced time series forecasting for business planning
- Comprehensive forecasting with multiple components
- When you need "triple exponential smoothing", "advanced forecasting", "trend and seasonality", or "Holt-Winters"

**Example:**
```python
HOLT_WINTERS(ctx, quarterly_sales, seasonal_periods=4, trend_type="add", seasonal_type="add")
HOLT_WINTERS(ctx, "quarterly_data.parquet", seasonal_periods=4, forecast_periods=8, output_filename="hw.parquet")


## 14. Logical and Error Handling Functions

### IF

**Purpose:** Return different values depending on whether a condition is met for financial decision-making and conditional calculations.

**Parameters:**
- `run_context`: RunContext object for file operations
- `logical_test`: Logical test to evaluate
- `value_if_true`: Value to return if test is True
- `value_if_false`: Value to return if test is False

**Returns:** Any - Value based on condition result

**Use Cases:**
- Budget variance analysis and financial decision-making
- Credit approval logic and risk assessment
- Performance bonus calculations and incentive modeling
- Conditional financial reporting and analysis
- When you need "conditional logic", "if-then", "decision making", or "conditional evaluation"

**Example:**
```python
actual = 105000
budget = 100000
variance_status = IF(actual > budget, "Over Budget", "Within Budget")
# Returns "Over Budget"

credit_score = 750
annual_income = 85000
approval = IF(credit_score >= 700 and annual_income >= 50000, "Approved", "Denied")
# Returns "Approved"
```

### IFS

**Purpose:** Test multiple conditions without nesting several IF statements for complex financial decision trees and business rule evaluation.

**Parameters:**
- `run_context`: RunContext object for file operations
- `conditions_and_values`: Alternating logical tests and values (condition1, value1, condition2, value2, ...)

**Returns:** Any - Value from first true condition

**Use Cases:**
- Credit rating assignment and risk classification
- Commission tier calculation and performance incentives
- Budget approval workflow and authorization levels
- Multi-tier pricing models and business rule implementation
- When you need "multiple conditions", "complex logic", "tiered evaluation", or "decision trees"

**Example:**
```python
credit_score = 780
rating = IFS(
    credit_score >= 800, "AAA",
    credit_score >= 750, "AA",
    credit_score >= 700, "A",
    credit_score >= 650, "BBB",
    True, "Below Investment Grade"
)
# Returns "AA"

sales_amount = 125000
commission_rate = IFS(
    sales_amount >= 200000, 0.08,
    sales_amount >= 150000, 0.06,
    sales_amount >= 100000, 0.04,
    sales_amount >= 50000, 0.02,
    True, 0.01
)
# Returns 0.04
```

### AND

**Purpose:** Test if all conditions are true for financial compliance checks and comprehensive risk assessment.

**Parameters:**
- `run_context`: RunContext object for file operations
- `logical_tests`: Multiple logical tests to evaluate

**Returns:** bool or pl.Series - True if all conditions are true, False otherwise

**Use Cases:**
- Investment criteria validation and portfolio screening
- Loan approval requirements and credit underwriting
- Budget compliance check and financial control
- Multi-criteria validation and quality assurance
- When you need "all conditions", "compliance check", "multi-criteria", or "comprehensive validation"

**Example:**
```python
pe_ratio = 15.2
debt_to_equity = 0.3
roe = 0.18
meets_criteria = AND(pe_ratio < 20, debt_to_equity < 0.5, roe > 0.15)
# Returns True

credit_score = 720
debt_to_income = 0.35
employment_years = 3
down_payment_pct = 0.20
loan_approved = AND(
    credit_score >= 700,
    debt_to_income <= 0.40,
    employment_years >= 2,
    down_payment_pct >= 0.15
)
# Returns True
```

### OR

**Purpose:** Test if any condition is true for financial risk flagging and alternative criteria evaluation.

**Parameters:**
- `run_context`: RunContext object for file operations
- `logical_tests`: Multiple logical tests to evaluate

**Returns:** bool or pl.Series - True if any condition is true, False otherwise

**Use Cases:**
- Risk flag detection and early warning systems
- Alternative payment methods and flexible business rules
- Audit trigger conditions and compliance monitoring
- Exception handling and contingency planning
- When you need "any condition", "risk detection", "alternative criteria", or "flexible logic"

**Example:**
```python
debt_ratio = 0.85
liquidity_ratio = 0.8
profit_margin = -0.05
risk_flag = OR(debt_ratio > 0.8, liquidity_ratio < 1.0, profit_margin < 0)
# Returns True

revenue_variance = 0.12
expense_variance = 0.08
margin_change = 0.15
audit_required = OR(
    revenue_variance > 0.10,
    expense_variance > 0.10,
    margin_change > 0.10
)
# Returns True
```

### NOT

**Purpose:** Reverse the logical value of a condition for financial logic inversion and exception handling.

**Parameters:**
- `run_context`: RunContext object for file operations
- `logical`: Logical value to reverse

**Returns:** bool or pl.Series - Opposite boolean value

**Use Cases:**
- Investment exclusion criteria and ethical screening
- Non-compliance detection and regulatory reporting
- Opposite condition checking and inverse logic
- Exception handling and negative criteria implementation
- When you need "reverse logic", "inversion", "exclusion", or "negative conditions"

**Example:**
```python
is_tobacco_company = False
is_ethical_investment = NOT(is_tobacco_company)
# Returns True

meets_regulations = False
requires_action = NOT(meets_regulations)
# Returns True
```

### XOR

**Purpose:** Exclusive OR - returns True if odd number of arguments are True for mutually exclusive conditions.

**Parameters:**
- `run_context`: RunContext object for file operations
- `logical_tests`: Multiple logical tests to evaluate

**Returns:** bool or pl.Series - True if odd number of conditions are true

**Use Cases:**
- Mutually exclusive investment options and portfolio allocation
- Alternative approval paths and decision frameworks
- Exclusive market conditions and strategic positioning
- Either-or business decisions and resource allocation
- When you need "exclusive conditions", "mutual exclusion", "either-or", or "single selection"

**Example:**
```python
invest_in_stocks = True
invest_in_bonds = False
invest_in_real_estate = False
single_investment = XOR(invest_in_stocks, invest_in_bonds, invest_in_real_estate)
# Returns True

ceo_approval = False
board_approval = True
committee_approval = False
has_approval = XOR(ceo_approval, board_approval, committee_approval)
# Returns True
```

### IFERROR

**Purpose:** Return a specified value if a formula results in an error for robust financial calculations.

**Parameters:**
- `run_context`: RunContext object for file operations
- `value`: Value or calculation to test
- `value_if_error`: Value to return if error occurs

**Returns:** Any - Original value or error replacement

**Use Cases:**
- Safe division for financial ratios and performance metrics
- Lookup with fallback for missing data and reference tables
- Safe percentage calculation and variance analysis
- Error handling in complex financial models
- When you need "error handling", "safe calculations", "fallback values", or "graceful errors"

**Example:**
```python
revenue = 1000000
shares_outstanding = 0  # Could cause division by zero
eps = IFERROR(revenue / shares_outstanding, "N/A")
# Returns "N/A"

actual = 105000
budget = 0  # Could cause division by zero
variance_pct = IFERROR((actual - budget) / budget * 100, 0)
# Returns 0
```

### IFNA

**Purpose:** Return a specified value if a formula results in #N/A error for handling missing data scenarios.

**Parameters:**
- `run_context`: RunContext object for file operations
- `value`: Value to test for #N/A error
- `value_if_na`: Value to return if #N/A error

**Returns:** Any - Original value or #N/A replacement

**Use Cases:**
- Product lookup with fallback and inventory management
- Customer credit rating lookup and risk assessment
- Exchange rate lookup and currency conversion
- Missing data handling in financial reports
- When you need "missing data", "lookup fallback", "data completion", or "N/A handling"

**Example:**
```python
product_code = "PROD999"
product_price = "#N/A"  # Not found in lookup
price = IFNA(product_price, 0)
# Returns 0

customer_id = "NEW001"
credit_rating = "N/A"  # New customer, no rating yet
rating = IFNA(credit_rating, "Unrated")
# Returns "Unrated"
```

### ISERROR

**Purpose:** Test if value is an error for financial data validation and error detection.

**Parameters:**
- `run_context`: RunContext object for file operations
- `value`: Value to test for error condition

**Returns:** bool - True if value is an error, False otherwise

**Use Cases:**
- Validate calculation results and model outputs
- Check lookup results and data integrity
- Error detection in financial calculations and reports
- Robust error handling workflows and quality control
- When you need "error detection", "validation", "quality control", or "data integrity"

**Example:**
```python
division_result = "#DIV/0!"
has_error = ISERROR(division_result)
# Returns True

customer_data = "#N/A"
lookup_failed = ISERROR(customer_data)
# Returns True
```

### ISBLANK

**Purpose:** Test if cell is blank for financial data completeness validation and missing data detection.

**Parameters:**
- `run_context`: RunContext object for file operations
- `value`: Value to test for blank condition

**Returns:** bool - True if value is blank/null, False otherwise

**Use Cases:**
- Check for missing financial data and completeness assessment
- Validate required fields and data entry quality
- Data quality assurance and reporting completeness
- Missing data detection in financial datasets
- When you need "blank detection", "completeness check", "missing data", or "data quality"

**Example:**
```python
quarterly_revenue = None
data_missing = ISBLANK(quarterly_revenue)
# Returns True

customer_name = ""
name_blank = ISBLANK(customer_name)
# Returns True
```

### ISNUMBER

**Purpose:** Test if value is a number for financial data validation and numeric type checking.

**Parameters:**
- `run_context`: RunContext object for file operations
- `value`: Value to test for numeric type

**Returns:** bool - True if value is numeric, False otherwise

**Use Cases:**
- Validate financial inputs and data entry validation
- Check calculation results and numeric integrity
- Ensure numeric calculations on valid data types
- Data type validation in financial models
- When you need "numeric validation", "type checking", "data validation", or "numeric integrity"

**Example:**
```python
revenue_input = "1000000"
is_valid_revenue = ISNUMBER(float(revenue_input)) if revenue_input.replace('.','').isdigit() else False
# Returns True

profit_margin = 0.15
is_valid_margin = ISNUMBER(profit_margin)
# Returns True
```

### ISTEXT

**Purpose:** Test if value is text for financial data categorization and text field validation.

**Parameters:**
- `run_context`: RunContext object for file operations
- `value`: Value to test for text type

**Returns:** bool - True if value is text, False otherwise

**Use Cases:**
- Validate text fields and categorical data
- Check account codes and identifier validation
- Text field validation in financial reports
- Data type handling in reporting systems
- When you need "text validation", "field validation", "categorical data", or "text checking"

**Example:**
```python
department_name = "Finance"
is_text_field = ISTEXT(department_name)
# Returns True

account_code = "ACC-001"
is_text_code = ISTEXT(account_code)
# Returns True
```

### SWITCH

**Purpose:** Compare expression against list of values and return corresponding result for financial categorization and business rule implementation.

**Parameters:**
- `run_context`: RunContext object for file operations
- `expression`: Expression to compare
- `values_and_results`: Alternating value and result pairs (value1, result1, value2, result2, ...)
- `default`: Default value if no matches found (optional)

**Returns:** Any - Matched result or default value

**Use Cases:**
- Department budget allocation and resource planning
- Credit rating to interest rate mapping and pricing models
- Performance tier commission rates and incentive structures
- Multi-tier business rules and categorization systems
- When you need "value mapping", "categorization", "tiered logic", or "business rules"

**Example:**
```python
department = "Marketing"
budget_multiplier = SWITCH(
    department,
    "Sales", 1.2,
    "Marketing", 1.1,
    "Operations", 1.0,
    "HR", 0.8,
    default=0.9
)
# Returns 1.1

credit_rating = "AA"
interest_rate = SWITCH(
    credit_rating,
    "AAA", 0.025,
    "AA", 0.030,
    "A", 0.035,
    "BBB", 0.045,
    default=0.060
)
# Returns 0.030

## 15. Lookup and Reference Functions

### VLOOKUP

**Purpose:** Search for a value in the first column of a table and return a value in the same row from a specified column.

**Parameters:**
- `run_context`: RunContext object for file operations
- `lookup_value`: Value to search for
- `table_array`: Table to search in (list, DataFrame, NumPy array, or file path)
- `col_index`: Column index to return (1-based)
- `range_lookup`: Whether to find approximate match (default: False for exact match)

**Returns:** Any - Value from the specified column in the matching row

**Use Cases:**
- Look up employee salaries by ID from a personnel table
- Find product prices by SKU from a product catalog
- Retrieve customer information by account number
- When you need to "look up", "find", "search", or "retrieve" data from tables

**Example:**
```python
VLOOKUP(ctx, "EMP001", employee_data, col_index=3)  # Returns salary for employee EMP001
VLOOKUP(ctx, 1001, "product_catalog.parquet", col_index=2, range_lookup=False)  # Returns price for product 1001
```

### HLOOKUP

**Purpose:** Search for a value in the first row of a table and return a value in the same column from a specified row.

**Parameters:**
- `run_context`: RunContext object for file operations
- `lookup_value`: Value to search for
- `table_array`: Table to search in (list, DataFrame, NumPy array, or file path)
- `row_index`: Row index to return (1-based)
- `range_lookup`: Whether to find approximate match (default: False for exact match)

**Returns:** Any - Value from the specified row in the matching column

**Use Cases:**
- Look up quarterly results from time-series data arranged horizontally
- Find performance metrics across different categories in row format
- Retrieve data from pivot tables with horizontal orientation
- When you need "horizontal lookup", "row-based search", or "transposed lookup"

**Example:**
```python
HLOOKUP(ctx, "Q1", quarterly_data, row_index=2)  # Returns Q1 value from second row
HLOOKUP(ctx, 2024, "annual_metrics.parquet", row_index=3, range_lookup=False)  # Returns 2024 metric from row 3
```

### INDEX

**Purpose:** Return a value at a given position in an array or table.

**Parameters:**
- `run_context`: RunContext object for file operations
- `array`: Array or table to index (list, DataFrame, Series, NumPy array, or file path)
- `row_num`: Row number (1-based)
- `column_num`: Column number (1-based, optional for 1D arrays)

**Returns:** Any - Value at the specified position

**Use Cases:**
- Extract specific values from financial matrices and tables
- Retrieve data from calculated arrays and results
- Access elements in multi-dimensional financial datasets
- When you need "array indexing", "position-based retrieval", or "direct access"

**Example:**
```python
INDEX(ctx, [[1, 2], [3, 4]], row_num=2, column_num=1)  # Returns 3
INDEX(ctx, [10, 20, 30], row_num=2)  # Returns 20
INDEX(ctx, "financial_matrix.parquet", row_num=5, column_num=3)  # Returns value at row 5, column 3
```

### MATCH

**Purpose:** Find the relative position of an item in an array.

**Parameters:**
- `run_context`: RunContext object for file operations
- `lookup_value`: Value to find
- `lookup_array`: Array to search in (list, Series, NumPy array, or file path)
- `match_type`: Match type (-1=smallest >=, 0=exact, 1=largest <=, default: 0)

**Returns:** int - Position (1-based) of the matching item

**Use Cases:**
- Find position of values for use with INDEX function
- Locate items in sorted lists for ranking analysis
- Identify positions in reference arrays for data alignment
- When you need "position finding", "array searching", or "relative location"

**Example:**
```python
MATCH(ctx, "Apple", ["Apple", "Banana", "Cherry"])  # Returns 1
MATCH(ctx, 150, [100, 200, 300], match_type=1)  # Returns 1 (largest <= 150)
MATCH(ctx, "target_value", "lookup_data.parquet", match_type=0)  # Returns exact match position
```

### XLOOKUP

**Purpose:** Modern, flexible lookup function replacing VLOOKUP/HLOOKUP with enhanced capabilities.

**Parameters:**
- `run_context`: RunContext object for file operations
- `lookup_value`: Value to search for
- `lookup_array`: Array to search in (list, Series, NumPy array, or file path)
- `return_array`: Array to return values from (list, Series, NumPy array, or file path)
- `if_not_found`: Value to return if no match found (optional)
- `match_mode`: Match mode (-1=next smaller, 0=exact, 1=next larger, 2=wildcard, default: 0)
- `search_mode`: Search mode (-2=reverse binary, -1=reverse, 1=forward, 2=binary, default: 1)

**Returns:** Any - Value from return_array corresponding to the match

**Use Cases:**
- Advanced lookups with flexible matching options
- Wildcard searches for partial matches
- Bidirectional searching for performance optimization
- Robust lookups with custom fallback values
- When you need "advanced lookup", "flexible matching", "wildcard search", or "enhanced VLOOKUP"

**Example:**
```python
XLOOKUP(ctx, "Product*", product_names, prices, match_mode=2)  # Wildcard search
XLOOKUP(ctx, 100, lookup_values, return_values, if_not_found="Not Found")  # Custom fallback
XLOOKUP(ctx, "EMP001", "employee_ids.parquet", "salaries.parquet")  # File-based lookup
```

### LOOKUP

**Purpose:** Simple lookup function (vector form) that finds the largest value less than or equal to lookup_value.

**Parameters:**
- `run_context`: RunContext object for file operations
- `lookup_value`: Value to search for
- `lookup_vector`: Vector to search in (list, Series, NumPy array, or file path)
- `result_vector`: Vector to return values from (optional, defaults to lookup_vector)

**Returns:** Any - Value from result_vector corresponding to the match

**Use Cases:**
- Grade lookup based on score ranges
- Tax bracket calculations for progressive taxation
- Commission rate determination based on sales thresholds
- When you need "simple lookup", "range matching", or "threshold-based retrieval"

**Example:**
```python
LOOKUP(ctx, 85, [0, 60, 70, 80, 90], ["F", "D", "C", "B", "A"])  # Returns "B"
LOOKUP(ctx, 50000, [0, 25000, 50000, 75000], [0.1, 0.15, 0.2, 0.25])  # Returns 0.15
LOOKUP(ctx, 100, "score_ranges.parquet", "grades.parquet")  # File-based grade lookup
```

### CHOOSE

**Purpose:** Return a value from a list based on index number.

**Parameters:**
- `index_num`: Index number (1-based) of value to return
- `*values`: Variable number of values to choose from

**Returns:** Any - Value at the specified index position

**Use Cases:**
- Dynamic selection from predefined options
- Conditional value selection based on calculated indices
- Menu-driven financial calculations
- When you need "value selection", "indexed choice", or "dynamic picking"

**Example:**
```python
CHOOSE(2, "Low", "Medium", "High")  # Returns "Medium"
CHOOSE(1, 0.05, 0.10, 0.15, 0.20)  # Returns 0.05
CHOOSE(risk_level, "Conservative", "Moderate", "Aggressive")  # Dynamic selection
```

### OFFSET

**Purpose:** Create dynamic ranges based on reference point with offset.

**Parameters:**
- `run_context`: RunContext object for file operations
- `reference`: Reference point (list, DataFrame, or file path)
- `rows`: Number of rows to offset
- `cols`: Number of columns to offset
- `height`: Height of returned range (optional, default: 1)
- `width`: Width of returned range (optional, default: 1)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** Any or Path - Single value, DataFrame, or path to saved file

**Use Cases:**
- Create dynamic ranges for financial modeling
- Build flexible references for complex calculations
- Generate offset data ranges for analysis
- When you need "dynamic ranges", "offset references", or "flexible positioning"

**Example:**
```python
OFFSET(ctx, data_table, rows=1, cols=2)  # Returns single cell offset by 1 row, 2 columns
OFFSET(ctx, financial_data, rows=0, cols=1, height=5, width=3)  # Returns 5x3 range
OFFSET(ctx, "reference_data.parquet", rows=2, cols=1, output_filename="offset_result.parquet")  # File-based offset
```

### INDIRECT

**Purpose:** Create references based on text strings (simplified implementation).

**Parameters:**
- `ref_text`: Text string representing a reference
- `a1_style`: Whether to use A1-style references (default: True)

**Returns:** str - Reference text string

**Use Cases:**
- Dynamic reference creation from calculated strings
- Building flexible formulas with variable references
- Creating indirect links for complex financial models
- When you need "text-based references", "dynamic linking", or "indirect addressing"

**Example:**
```python
INDIRECT("A1")  # Returns "A1"
INDIRECT("Sheet2!B5")  # Returns "Sheet2!B5"
INDIRECT(f"A{row_number}")  # Dynamic cell reference
```

### ADDRESS

**Purpose:** Create cell address as text.

**Parameters:**
- `row_num`: Row number
- `column_num`: Column number
- `abs_num`: Absolute/relative reference type (1=$A$1, 2=A$1, 3=$A1, 4=A1, default: 1)
- `a1`: Whether to use A1-style references (default: True)
- `sheet_text`: Sheet name (optional)

**Returns:** str - Cell address as text

**Use Cases:**
- Generate cell references for dynamic formulas
- Create address strings for documentation and reporting
- Build reference strings for complex financial models
- When you need "address generation", "cell reference creation", or "reference building"

**Example:**
```python
ADDRESS(1, 1)  # Returns "$A$1"
ADDRESS(5, 3, abs_num=4)  # Returns "C5"
ADDRESS(10, 2, sheet_text="Data")  # Returns "Data!$B$10"
```

### ROW

**Purpose:** Return row number of reference.

**Parameters:**
- `run_context`: RunContext object for file operations
- `reference`: Reference (list, DataFrame, or file path, optional)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** int, list[int], or Path - Row number(s) or path to saved file

**Use Cases:**
- Get row numbers for data processing and indexing
- Create row-based references for financial calculations
- Generate sequence numbers for financial records
- When you need "row numbers", "position tracking", or "index generation"

**Example:**
```python
ROW(ctx)  # Returns 1
ROW(ctx, [[1, 2], [3, 4], [5, 6]])  # Returns [1, 2, 3]
ROW(ctx, "financial_data.parquet", output_filename="row_numbers.parquet")  # File-based row extraction
```

### COLUMN

**Purpose:** Return column number of reference.

**Parameters:**
- `run_context`: RunContext object for file operations
- `reference`: Reference (list, DataFrame, or file path, optional)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** int, list[int], or Path - Column number(s) or path to saved file

**Use Cases:**
- Get column numbers for data processing and indexing
- Create column-based references for financial calculations
- Generate sequence numbers for financial categories
- When you need "column numbers", "position tracking", or "index generation"

**Example:**
```python
COLUMN(ctx)  # Returns 1
COLUMN(ctx, [[1, 2, 3], [4, 5, 6]])  # Returns [1, 2, 3]
COLUMN(ctx, "financial_data.parquet", output_filename="column_numbers.parquet")  # File-based column extraction
```

### ROWS

**Purpose:** Return number of rows in reference.

**Parameters:**
- `run_context`: RunContext object for file operations
- `array`: Array or table (list, DataFrame, NumPy array, or file path)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** int or Path - Number of rows or path to saved file

**Use Cases:**
- Count records in financial datasets
- Determine data size for processing and analysis
- Validate data completeness and integrity
- When you need "row counting", "data size", or "record counting"

**Example:**
```python
ROWS(ctx, [[1, 2], [3, 4], [5, 6]])  # Returns 3
ROWS(ctx, "transaction_data.parquet")  # Returns number of transaction records
ROWS(ctx, financial_table, output_filename="row_count.parquet")  # File-based row counting
```

### COLUMNS

**Purpose:** Return number of columns in reference.

**Parameters:**
- `run_context`: RunContext object for file operations
- `array`: Array or table (list, DataFrame, NumPy array, or file path)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** int or Path - Number of columns or path to saved file

**Use Cases:**
- Count fields in financial datasets
- Determine data structure for processing and analysis
- Validate data format and schema compliance
- When you need "column counting", "field counting", or "structure analysis"

**Example:**
```python
COLUMNS(ctx, [[1, 2, 3], [4, 5, 6]])  # Returns 3
COLUMNS(ctx, "financial_data.parquet")  # Returns number of financial metrics
COLUMNS(ctx, data_table, output_filename="column_count.parquet")  # File-based column counting

## 16. Statistical and Trend Analysis Functions

### STDEV_P

**Purpose:** Calculate the standard deviation for a full population using Decimal precision.

**Parameters:**
- `run_context`: RunContext object for file operations
- `values`: Array or range of numeric values (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal - Population standard deviation

**Use Cases:**
- Calculate population risk metrics for investment portfolios
- Measure volatility in financial returns for risk assessment
- Analyze population-level variance in performance metrics
- When you need "population standard deviation", "risk measurement", "volatility analysis", or "population variance"

**Example:**
```python
STDEV_P(ctx, [2, 4, 4, 4, 5, 5, 7, 9])  # Returns Decimal('2.0')
STDEV_P(ctx, "data_file.parquet")  # Returns Decimal('1.58113883008419')
```

### STDEV_S

**Purpose:** Calculate the standard deviation for a sample using Decimal precision.

**Parameters:**
- `run_context`: RunContext object for file operations
- `values`: Array or range of numeric values (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal - Sample standard deviation

**Use Cases:**
- Analyze sample-based risk metrics for financial planning
- Measure sample volatility in market data analysis
- Calculate sample standard deviation for statistical inference
- When you need "sample standard deviation", "sample risk", "statistical analysis", or "sample variance"

**Example:**
```python
STDEV_S(ctx, [2, 4, 4, 4, 5, 5, 7, 9])  # Returns Decimal('2.1380899352993')
STDEV_S(ctx, "data_file.parquet")  # Returns Decimal('1.58113883008419')
```

### VAR_P

**Purpose:** Calculate variance for a population using Decimal precision.

**Parameters:**
- `run_context`: RunContext object for file operations
- `values`: Array or range of numeric values (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal - Population variance

**Use Cases:**
- Calculate population variance for risk modeling
- Measure squared deviations in financial metrics
- Analyze population-level dispersion in data
- When you need "population variance", "risk variance", "dispersion measurement", or "squared deviation"

**Example:**
```python
VAR_P(ctx, [2, 4, 4, 4, 5, 5, 7, 9])  # Returns Decimal('4.0')
VAR_P(ctx, "data_file.parquet")  # Returns Decimal('2.5')
```

### VAR_S

**Purpose:** Calculate variance for a sample using Decimal precision.

**Parameters:**
- `run_context`: RunContext object for file operations
- `values`: Array or range of numeric values (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal - Sample variance

**Use Cases:**
- Analyze sample variance for statistical inference
- Measure sample-based risk in financial data
- Calculate variance for hypothesis testing
- When you need "sample variance", "statistical variance", "sample dispersion", or "risk analysis"

**Example:**
```python
VAR_S(ctx, [2, 4, 4, 4, 5, 5, 7, 9])  # Returns Decimal('4.571428571428571')
VAR_S(ctx, "data_file.parquet")  # Returns Decimal('2.5')
```

### MEDIAN

**Purpose:** Determine the middle value in a dataset using Decimal precision.

**Parameters:**
- `run_context`: RunContext object for file operations
- `values`: Array or range of numeric values (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal - Median value

**Use Cases:**
- Find central tendency in skewed financial data
- Calculate median salary or compensation figures
- Determine typical values when outliers exist
- When you need "median", "middle value", "central tendency", or "robust average"

**Example:**
```python
MEDIAN(ctx, [1, 2, 3, 4, 5])  # Returns Decimal('3')
MEDIAN(ctx, [1, 2, 3, 4])  # Returns Decimal('2.5')
MEDIAN(ctx, "data_file.parquet")  # Returns Decimal('3')
```

### MODE

**Purpose:** Find the most frequently occurring value in a dataset using Decimal precision.

**Parameters:**
- `run_context`: RunContext object for file operations
- `values`: Array or range of numeric values (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal or list[Decimal] - Most frequent value(s)

**Use Cases:**
- Identify most common transaction amounts
- Find typical order sizes or quantities
- Analyze recurring expense patterns
- When you need "most common", "frequent value", "typical value", or "recurring pattern"

**Example:**
```python
MODE(ctx, [1, 2, 2, 3, 3, 3])  # Returns Decimal('3')
MODE(ctx, [1, 1, 2, 2, 3])  # Returns [Decimal('1'), Decimal('2')]
MODE(ctx, "data_file.parquet")  # Returns Decimal('3')
```

### CORREL

**Purpose:** Measure the correlation between two datasets using Decimal precision.

**Parameters:**
- `run_context`: RunContext object for file operations
- `range1`: First range of values (float, Decimal, Polars Series, NumPy array, or file path)
- `range2`: Second range of values (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal - Correlation coefficient (-1 to 1)

**Use Cases:**
- Analyze relationship between stock prices and market indices
- Measure correlation between revenue and marketing spend
- Identify correlated financial metrics for portfolio analysis
- When you need "correlation", "relationship analysis", "linear association", or "dependency measurement"

**Example:**
```python
CORREL(ctx, [1, 2, 3, 4, 5], range2=[2, 4, 6, 8, 10])  # Returns Decimal('1.0')
CORREL(ctx, [1, 2, 3, 4, 5], range2=[5, 4, 3, 2, 1])  # Returns Decimal('-1.0')
CORREL(ctx, "data1.parquet", range2="data2.parquet")  # Returns Decimal('0.8')
```

### COVARIANCE_P

**Purpose:** Calculate covariance for a population using Decimal precision.

**Parameters:**
- `run_context`: RunContext object for file operations
- `range1`: First range of values (float, Decimal, Polars Series, NumPy array, or file path)
- `range2`: Second range of values (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal - Population covariance

**Use Cases:**
- Measure population-level relationship between financial variables
- Calculate covariance for portfolio risk analysis
- Analyze joint variability in economic indicators
- When you need "population covariance", "joint variability", "risk covariance", or "variable relationship"

**Example:**
```python
COVARIANCE_P(ctx, [1, 2, 3, 4, 5], range2=[2, 4, 6, 8, 10])  # Returns Decimal('4.0')
COVARIANCE_P(ctx, "data1.parquet", range2="data2.parquet")  # Returns Decimal('2.5')
```

### COVARIANCE_S

**Purpose:** Calculate covariance for a sample using Decimal precision.

**Parameters:**
- `run_context`: RunContext object for file operations
- `range1`: First range of values (float, Decimal, Polars Series, NumPy array, or file path)
- `range2`: Second range of values (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal - Sample covariance

**Use Cases:**
- Analyze sample-based relationships in financial data
- Calculate covariance for statistical inference
- Measure sample covariance for risk modeling
- When you need "sample covariance", "statistical covariance", "sample relationship", or "data association"

**Example:**
```python
COVARIANCE_S(ctx, [1, 2, 3, 4, 5], range2=[2, 4, 6, 8, 10])  # Returns Decimal('5.0')
COVARIANCE_S(ctx, "data1.parquet", range2="data2.parquet")  # Returns Decimal('3.333333333333333333333333333')
```

### TREND

**Purpose:** Predict future values based on linear trends using Decimal precision.

**Parameters:**
- `run_context`: RunContext object for file operations
- `known_y`: Known y values (float, Decimal, Polars Series, NumPy array, or file path)
- `known_x`: Known x values (optional, defaults to 1, 2, 3, ...)
- `new_x`: New x values for prediction (optional, defaults to continuation of known_x)
- `const`: Whether to force intercept through zero (default True)
- `output_filename`: Filename to save prediction results

**Returns:** Path - Path to saved prediction results

**Use Cases:**
- Forecast revenue trends based on historical data
- Predict expense patterns for budget planning
- Project financial metrics using linear trends
- When you need "linear trend", "trend forecasting", "linear projection", or "trend analysis"

**Example:**
```python
TREND(ctx, [1, 2, 3, 4, 5], known_x=[1, 2, 3, 4, 5], new_x=[6, 7, 8], output_filename="trend_results.parquet")
```

### FORECAST

**Purpose:** Predict a future value based on linear regression using Decimal precision.

**Parameters:**
- `run_context`: RunContext object for file operations
- `new_x`: New x value for prediction
- `known_y`: Known y values (float, Decimal, Polars Series, NumPy array, or file path)
- `known_x`: Known x values (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal - Single predicted value

**Use Cases:**
- Forecast single future values for financial planning
- Predict individual data points based on historical trends
- Calculate projected values for specific time periods
- When you need "single forecast", "point prediction", "linear projection", or "future value"

**Example:**
```python
FORECAST(ctx, 6, known_y=[1, 2, 3, 4, 5], known_x=[1, 2, 3, 4, 5])  # Returns Decimal('6.0')
FORECAST(ctx, 10, known_y="y_data.parquet", known_x="x_data.parquet")  # Returns Decimal('12.5')
```

### FORECAST_LINEAR

**Purpose:** Predict a future value based on linear regression (newer version).

**Parameters:**
- `run_context`: RunContext object for file operations
- `new_x`: New x value for prediction
- `known_y`: Known y values (float, Decimal, Polars Series, NumPy array, or file path)
- `known_x`: Known x values (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal - Single predicted value

**Use Cases:**
- Modern linear forecasting for financial projections
- Updated forecasting method with improved accuracy
- Linear prediction for business planning and analysis
- When you need "modern forecast", "linear prediction", "updated forecasting", or "improved projection"

**Example:**
```python
FORECAST_LINEAR(ctx, 6, known_y=[1, 2, 3, 4, 5], known_x=[1, 2, 3, 4, 5])  # Returns Decimal('6.0')
```

### GROWTH

**Purpose:** Forecast exponential growth trends using Decimal precision.

**Parameters:**
- `run_context`: RunContext object for file operations
- `known_y`: Known y values (float, Decimal, Polars Series, NumPy array, or file path)
- `known_x`: Known x values (optional, defaults to 1, 2, 3, ...)
- `new_x`: New x values for prediction (optional, defaults to continuation of known_x)
- `const`: Whether to include constant term (default True)
- `output_filename`: Filename to save prediction results

**Returns:** Path - Path to saved prediction results

**Use Cases:**
- Forecast exponential revenue growth patterns
- Predict compound growth in financial metrics
- Model exponential trends in business data
- When you need "exponential growth", "compound forecasting", "exponential projection", or "growth modeling"

**Example:**
```python
GROWTH(ctx, [1, 2, 4, 8, 16], known_x=[1, 2, 3, 4, 5], new_x=[6, 7, 8], output_filename="growth_results.parquet")
```

### SLOPE

**Purpose:** Calculate slope of linear regression line using Decimal precision.

**Parameters:**
- `run_context`: RunContext object for file operations
- `known_ys`: Known y values (float, Decimal, Polars Series, NumPy array, or file path)
- `known_xs`: Known x values (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal - Slope of regression line

**Use Cases:**
- Measure rate of change in financial trends
- Calculate growth rates for business metrics
- Determine directional momentum in data
- When you need "slope", "rate of change", "growth rate", or "trend direction"

**Example:**
```python
SLOPE(ctx, [1, 2, 3, 4, 5], known_xs=[1, 2, 3, 4, 5])  # Returns Decimal('1.0')
SLOPE(ctx, "y_data.parquet", known_xs="x_data.parquet")  # Returns Decimal('2.5')
```

### INTERCEPT

**Purpose:** Calculate y-intercept of linear regression line using Decimal precision.

**Parameters:**
- `run_context`: RunContext object for file operations
- `known_ys`: Known y values (float, Decimal, Polars Series, NumPy array, or file path)
- `known_xs`: Known x values (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal - Y-intercept of regression line

**Use Cases:**
- Determine baseline values in financial models
- Calculate starting points for trend analysis
- Find initial conditions in regression models
- When you need "y-intercept", "baseline value", "starting point", or "regression intercept"

**Example:**
```python
INTERCEPT(ctx, [2, 4, 6, 8, 10], known_xs=[1, 2, 3, 4, 5])  # Returns Decimal('0.0')
INTERCEPT(ctx, "y_data.parquet", known_xs="x_data.parquet")  # Returns Decimal('1.5')
```

### RSQ

**Purpose:** Calculate R-squared of linear regression using Decimal precision.

**Parameters:**
- `run_context`: RunContext object for file operations
- `known_ys`: Known y values (float, Decimal, Polars Series, NumPy array, or file path)
- `known_xs`: Known x values (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal - R-squared value (coefficient of determination)

**Use Cases:**
- Measure goodness of fit for financial models
- Evaluate predictive power of regression models
- Assess model accuracy in financial forecasting
- When you need "R-squared", "coefficient of determination", "model fit", or "prediction accuracy"

**Example:**
```python
RSQ(ctx, [1, 2, 3, 4, 5], known_xs=[1, 2, 3, 4, 5])  # Returns Decimal('1.0')
RSQ(ctx, "y_data.parquet", known_xs="x_data.parquet")  # Returns Decimal('0.95')
```

### LINEST

**Purpose:** Calculate linear regression statistics using Decimal precision.

**Parameters:**
- `run_context`: RunContext object for file operations
- `known_ys`: Known y values (float, Decimal, Polars Series, NumPy array, or file path)
- `known_xs`: Known x values (optional, defaults to 1, 2, 3, ...)
- `const`: Whether to include constant term (default True)
- `stats_flag`: Whether to include additional statistics (default False)
- `output_filename`: Filename to save regression statistics

**Returns:** Path - Path to saved regression statistics

**Use Cases:**
- Comprehensive linear regression analysis for financial modeling
- Detailed statistical analysis of business relationships
- Advanced regression diagnostics and model evaluation
- When you need "linear regression statistics", "regression analysis", "model diagnostics", or "statistical summary"

**Example:**
```python
LINEST(ctx, [1, 2, 3, 4, 5], known_xs=[1, 2, 3, 4, 5], output_filename="linest_results.parquet")
```

### LOGEST

**Purpose:** Calculate exponential regression statistics using Decimal precision.

**Parameters:**
- `run_context`: RunContext object for file operations
- `known_ys`: Known y values (float, Decimal, Polars Series, NumPy array, or file path)
- `known_xs`: Known x values (optional, defaults to 1, 2, 3, ...)
- `const`: Whether to include constant term (default True)
- `stats_flag`: Whether to include additional statistics (default False)
- `output_filename`: Filename to save regression statistics

**Returns:** Path - Path to saved regression statistics

**Use Cases:**
- Exponential growth modeling for business forecasting
- Compound interest and growth rate analysis
- Non-linear relationship modeling in finance
- When you need "exponential regression", "growth modeling", "non-linear analysis", or "compound modeling"

**Example:**
```python
LOGEST(ctx, [1, 2, 4, 8, 16], known_xs=[1, 2, 3, 4, 5], output_filename="logest_results.parquet")
```

### RANK

**Purpose:** Calculate rank of number in array using 1-based ranking.

**Parameters:**
- `run_context`: RunContext object for file operations
- `number`: Number to rank
- `ref`: Reference array (float, Decimal, Polars Series, NumPy array, or file path)
- `order`: Sort order (0 = descending, 1 = ascending, default 0)

**Returns:** int - Rank of number (1-based)

**Use Cases:**
- Rank investment performance for portfolio analysis
- Order customer segments by revenue or profitability
- Sort financial metrics for competitive analysis
- When you need "ranking", "ordering", "performance ranking", or "competitive positioning"

**Example:**
```python
RANK(ctx, 85, ref=[100, 85, 90, 75, 95], order=0)  # Returns 4
RANK(ctx, 85, ref=[100, 85, 90, 75, 95], order=1)  # Returns 2
RANK(ctx, 85, ref="data_file.parquet", order=0)  # Returns 3
```

### PERCENTRANK

**Purpose:** Calculate percentile rank using Decimal precision.

**Parameters:**
- `run_context`: RunContext object for file operations
- `array`: Array of values (float, Decimal, Polars Series, NumPy array, or file path)
- `x`: Value to rank
- `significance`: Number of significant digits (default 3)

**Returns:** Decimal - Percentile rank (0 to 1)

**Use Cases:**
- Analyze relative position in performance distributions
- Compare individual metrics against benchmarks
- Percentile analysis for risk assessment
- When you need "percentile rank", "relative position", "distribution analysis", or "benchmark comparison"

**Example:**
```python
PERCENTRANK(ctx, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], x=7)  # Returns Decimal('0.667')
PERCENTRANK(ctx, "data_file.parquet", x=85, significance=4)  # Returns Decimal('0.7500')



## 17. Text and Data Management Functions

### CONCAT

**Purpose:** Merge text strings together (modern version).

**Parameters:**
- `run_context`: RunContext object for file operations
- `texts`: Text strings to concatenate (supports file paths)

**Returns:** str - Combined text string

**Use Cases:**
- Combine financial labels and values for reporting
- Create composite identifiers for financial instruments
- Merge text data from multiple sources
- When you need to "concatenate", "combine text", "merge strings", or "join text"

**Example:**
```python
CONCAT(ctx, "Hello", " ", "World")  # Returns 'Hello World'
CONCAT(ctx, "Revenue: $", 1000, " Million")  # Returns 'Revenue: $1000 Million'
CONCAT(ctx, "data.csv")  # File input - Returns 'concatenated content from file'
```

### CONCATENATE

**Purpose:** Merge text strings together (legacy version).

**Parameters:**
- `run_context`: RunContext object for file operations
- `texts`: Text strings to concatenate (supports file paths)

**Returns:** str - Combined text string

**Use Cases:**
- Legacy text concatenation for compatibility
- Combine period labels with years for financial reporting
- Merge budget categories with amounts
- When you need "legacy concatenate", "text merging", "string combination", or "backward compatibility"

**Example:**
```python
CONCATENATE(ctx, "Q1", " ", "2024")  # Returns 'Q1 2024'
CONCATENATE(ctx, "Budget: ", 50000)  # Returns 'Budget: 50000'
CONCATENATE(ctx, "data.csv")  # File input - Returns 'concatenated content from file'
```

### TEXT

**Purpose:** Format numbers or dates as text with a specified format.

**Parameters:**
- `run_context`: RunContext object for file operations
- `value`: Value to format (number, date, or file path)
- `format_text`: Format text (Excel-style format codes)

**Returns:** str - Formatted text string

**Use Cases:**
- Format financial figures for presentation reports
- Convert percentages and currency for display
- Format dates in standardized reporting formats
- When you need "format numbers", "format dates", "text formatting", or "financial presentation"

**Example:**
```python
TEXT(ctx, 0.125, format_text="0.00%")  # Returns '12.50%'
TEXT(ctx, 1234567.89, format_text="$#,##0.00")  # Returns '$1,234,567.89'
TEXT(ctx, Decimal('0.0825'), format_text="0.000%")  # Returns '8.250%'
TEXT(ctx, datetime(2024, 3, 15), format_text="yyyy-mm-dd")  # Returns '2024-03-15'
```

### LEFT

**Purpose:** Extract characters from the left side of a text string.

**Parameters:**
- `run_context`: RunContext object for file operations
- `text`: Text string or file path
- `num_chars`: Number of characters to extract

**Returns:** str - Text substring

**Use Cases:**
- Extract stock ticker symbols from composite identifiers
- Get prefixes from financial account codes
- Extract date components from formatted strings
- When you need "left substring", "prefix extraction", "character extraction", or "text slicing"

**Example:**
```python
LEFT(ctx, "Financial Planning", num_chars=9)  # Returns 'Financial'
LEFT(ctx, "AAPL-2024-Q1", num_chars=4)  # Returns 'AAPL'
LEFT(ctx, "data.csv", num_chars=5)  # File input - Returns 'first'
```

### RIGHT

**Purpose:** Extract characters from the right side of a text string.

**Parameters:**
- `run_context`: RunContext object for file operations
- `text`: Text string or file path
- `num_chars`: Number of characters to extract

**Returns:** str - Text substring

**Use Cases:**
- Extract quarter indicators from period codes
- Get file extensions from filenames
- Extract suffixes from financial identifiers
- When you need "right substring", "suffix extraction", "end characters", or "text trimming"

**Example:**
```python
RIGHT(ctx, "Financial Planning", num_chars=8)  # Returns 'Planning'
RIGHT(ctx, "AAPL-2024-Q1", num_chars=2)  # Returns 'Q1'
RIGHT(ctx, "data.csv", num_chars=3)  # File input - Returns 'ast'
```

### MID

**Purpose:** Extract characters from the middle of a text string.

**Parameters:**
- `run_context`: RunContext object for file operations
- `text`: Text string or file path
- `start_num`: Starting position (1-based)
- `num_chars`: Number of characters to extract

**Returns:** str - Text substring

**Use Cases:**
- Extract year components from financial period codes
- Get middle segments from structured identifiers
- Parse specific sections from formatted data
- When you need "middle substring", "text extraction", "character range", or "string parsing"

**Example:**
```python
MID(ctx, "Financial Planning", start_num=11, num_chars=8)  # Returns 'Planning'
MID(ctx, "AAPL-2024-Q1", start_num=6, num_chars=4)  # Returns '2024'
MID(ctx, "data.csv", start_num=2, num_chars=3)  # File input - Returns 'ata'
```

### LEN

**Purpose:** Count the number of characters in a text string.

**Parameters:**
- `run_context`: RunContext object for file operations
- `text`: Text string or file path

**Returns:** int - Character count

**Use Cases:**
- Validate length of financial identifiers
- Check data entry completeness
- Measure text field sizes for reporting
- When you need "text length", "character count", "string size", or "length validation"

**Example:**
```python
LEN(ctx, "Financial Planning")  # Returns 18
LEN(ctx, "AAPL")  # Returns 4
LEN(ctx, "data.csv")  # File input - Returns 10
```

### FIND

**Purpose:** Locate one text string within another (case-sensitive).

**Parameters:**
- `run_context`: RunContext object for file operations
- `find_text`: Text to find or file path
- `within_text`: Text to search within or file path
- `start_num`: Starting position (1-based, optional)

**Returns:** int - Position (1-based) or -1 if not found

**Use Cases:**
- Locate specific terms in financial documents
- Find position of delimiters in structured data
- Search for keywords in report text
- When you need "case-sensitive search", "text position", "string location", or "exact matching"

**Example:**
```python
FIND(ctx, "Plan", "Financial Planning")  # Returns 11
FIND(ctx, "plan", "Financial Planning")  # Returns -1 (case-sensitive)
FIND(ctx, "2024", "AAPL-2024-Q1", start_num=1)  # Returns 6
```

### SEARCH

**Purpose:** Locate one text string within another (not case-sensitive).

**Parameters:**
- `run_context`: RunContext object for file operations
- `find_text`: Text to find or file path
- `within_text`: Text to search within or file path
- `start_num`: Starting position (1-based, optional)

**Returns:** int - Position (1-based) or -1 if not found

**Use Cases:**
- Search for financial terms regardless of case
- Find keywords in mixed-case data
- Locate text patterns in reports
- When you need "case-insensitive search", "text finding", "pattern matching", or "flexible search"

**Example:**
```python
SEARCH(ctx, "plan", "Financial Planning")  # Returns 11
SEARCH(ctx, "PLAN", "Financial Planning")  # Returns 11 (case-insensitive)
SEARCH(ctx, "q1", "AAPL-2024-Q1", start_num=1)  # Returns 11
```

### REPLACE

**Purpose:** Replace a portion of a text string with another text string.

**Parameters:**
- `run_context`: RunContext object for file operations
- `old_text`: Original text or file path
- `start_num`: Starting position (1-based)
- `num_chars`: Number of characters to replace
- `new_text`: New text

**Returns:** str - Modified text string

**Use Cases:**
- Update year values in financial period codes
- Correct typos in financial labels
- Modify structured identifiers
- When you need "text replacement", "string modification", "character substitution", or "text editing"

**Example:**
```python
REPLACE(ctx, "Financial Planning", start_num=11, num_chars=8, new_text="Analysis")  # Returns 'Financial Analysis'
REPLACE(ctx, "AAPL-2023-Q1", start_num=6, num_chars=4, new_text="2024")  # Returns 'AAPL-2024-Q1'
```

### SUBSTITUTE

**Purpose:** Replace occurrences of old text with new text.

**Parameters:**
- `run_context`: RunContext object for file operations
- `text`: Original text or file path
- `old_text`: Text to replace
- `new_text`: New text
- `instance_num`: Instance number to replace (optional, replaces all if None)

**Returns:** str - Modified text string

**Use Cases:**
- Update company names in financial reports
- Replace currency codes in multi-currency data
- Correct recurring terms in documentation
- When you need "text substitution", "string replacement", "bulk editing", or "selective replacement"

**Example:**
```python
SUBSTITUTE(ctx, "Financial Planning and Financial Analysis", old_text="Financial", new_text="Business")  # Returns 'Business Planning and Business Analysis'
SUBSTITUTE(ctx, "Q1-Q1-Q1", old_text="Q1", new_text="Q2", instance_num=2)  # Returns 'Q1-Q2-Q1'
```

### TRIM

**Purpose:** Remove extra spaces from text.

**Parameters:**
- `run_context`: RunContext object for file operations
- `text`: Text string or file path

**Returns:** str - Cleaned text string

**Use Cases:**
- Clean imported financial data with inconsistent spacing
- Prepare text for accurate comparisons
- Remove formatting artifacts from reports
- When you need "space removal", "text cleaning", "whitespace cleanup", or "data sanitization"

**Example:**
```python
TRIM(ctx, "  Extra   Spaces  ")  # Returns 'Extra Spaces'
TRIM(ctx, "  Financial Planning  ")  # Returns 'Financial Planning'
```

### CLEAN

**Purpose:** Remove non-printable characters.

**Parameters:**
- `run_context`: RunContext object for file operations
- `text`: Text string or file path

**Returns:** str - Cleaned text string

**Use Cases:**
- Remove hidden characters from imported financial data
- Clean text files with encoding issues
- Prepare data for accurate text processing
- When you need "character cleaning", "non-printable removal", "data purification", or "text sanitization"

**Example:**
```python
CLEAN(ctx, "Financial\x00Planning\x01")  # Returns 'FinancialPlanning'
CLEAN(ctx, "Clean\tText\n")  # Returns 'Clean\tText\n' (keeps printable whitespace)
```

### UPPER

**Purpose:** Convert text to uppercase.

**Parameters:**
- `run_context`: RunContext object for file operations
- `text`: Text string or file path

**Returns:** str - Uppercase text string

**Use Cases:**
- Standardize financial identifiers for consistency
- Prepare text for case-insensitive comparisons
- Format headers in financial reports
- When you need "uppercase conversion", "text standardization", "case conversion", or "formatting"

**Example:**
```python
UPPER(ctx, "hello world")  # Returns 'HELLO WORLD'
UPPER(ctx, "Financial Planning")  # Returns 'FINANCIAL PLANNING'
```

### LOWER

**Purpose:** Convert text to lowercase.

**Parameters:**
- `run_context`: RunContext object for file operations
- `text`: Text string or file path

**Returns:** str - Lowercase text string

**Use Cases:**
- Standardize text for database lookups
- Prepare text for case-insensitive processing
- Format user input for consistency
- When you need "lowercase conversion", "text normalization", "case standardization", or "data preparation"

**Example:**
```python
LOWER(ctx, "HELLO WORLD")  # Returns 'hello world'
LOWER(ctx, "Financial Planning")  # Returns 'financial planning'
```

### PROPER

**Purpose:** Convert text to proper case.

**Parameters:**
- `run_context`: RunContext object for file operations
- `text`: Text string or file path

**Returns:** str - Proper case text string

**Use Cases:**
- Format names and titles in financial reports
- Standardize customer names for presentation
- Prepare text for professional documentation
- When you need "proper case", "title case", "capitalization", or "text formatting"

**Example:**
```python
PROPER(ctx, "hello world")  # Returns 'Hello World'
PROPER(ctx, "financial planning")  # Returns 'Financial Planning'
```

### VALUE

**Purpose:** Convert text to number using Decimal precision.

**Parameters:**
- `run_context`: RunContext object for file operations
- `text`: Text string or file path

**Returns:** Decimal - Numeric value

**Use Cases:**
- Convert formatted financial text to numeric values
- Parse currency and percentage data from reports
- Extract numbers from mixed text formats
- When you need "text to number", "numeric conversion", "financial parsing", or "data extraction"

**Example:**
```python
VALUE(ctx, "123.45")  # Returns Decimal('123.45')
VALUE(ctx, "$1,234.56")  # Returns Decimal('1234.56')
VALUE(ctx, "12.5%")  # Returns Decimal('0.125')
VALUE(ctx, "(500)")  # Returns Decimal('-500') (negative in parentheses)
```

### TEXTJOIN

**Purpose:** Join text strings with delimiter.

**Parameters:**
- `run_context`: RunContext object for file operations
- `delimiter`: Delimiter string
- `ignore_empty`: Ignore empty values
- `texts`: Text strings to join (supports file paths)

**Returns:** str - Combined text string

**Use Cases:**
- Create comma-separated lists for financial reports
- Combine multiple data points into single fields
- Generate structured identifiers from components
- When you need "text joining", "delimiter separation", "list creation", or "string aggregation"

**Example:**
```python
TEXTJOIN(ctx, ", ", True, "Apple", "", "Banana", "Cherry")  # Returns 'Apple, Banana, Cherry'
TEXTJOIN(ctx, " | ", False, "Q1", "Q2", "Q3", "Q4")  # Returns 'Q1 | Q2 | Q3 | Q4'
TEXTJOIN(ctx, ",", True, "data.csv")  # File input - Returns 'value1,value2,value3'
