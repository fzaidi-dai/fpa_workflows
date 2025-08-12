# Financial Planning & Analysis (FP&A) AI Agent Tools Documentation

This documentation provides comprehensive information about available tools for FP&A AI agents, organized by category to enable optimal tool selection for financial analysis tasks.

## 1. Basic Math and Aggregation

### SUM

**Purpose:** Add up a range of numbers using Decimal precision for financial accuracy.

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Array or range of numeric values (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal - Sum of all values

**Use Cases:**
- Calculate total revenue, expenses, or cash flows
- Sum financial metrics across periods or categories
- Aggregate budget line items
- When you need to "add up", "total", "sum", or "aggregate" financial data

**Example:**
```python
SUM(ctx, [1, 2, 3, 4, 5])  # Returns Decimal('15')
SUM(ctx, "revenue_data.parquet")  # Sum from file
```

### AVERAGE

**Purpose:** Calculate the mean of a dataset using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Array or range of numeric values (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal - Mean of all values

**Use Cases:**
- Calculate average revenue per customer
- Find mean expense amounts
- Determine typical performance metrics
- When you need to find "average", "mean", or "typical" values

**Example:**
```python
AVERAGE(ctx, [10, 20, 30])  # Returns Decimal('20')
```

### MIN

**Purpose:** Identify the smallest number in a dataset using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Array or range of numeric values (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal - Minimum value

**Use Cases:**
- Find lowest cost, price, or expense
- Identify minimum performance thresholds
- Determine floor values for budgeting
- When you need "minimum", "lowest", "smallest", or "floor" values

**Example:**
```python
MIN(ctx, [10, 5, 20, 3])  # Returns Decimal('3')
```

### MAX

**Purpose:** Identify the largest number in a dataset using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Array or range of numeric values (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal - Maximum value

**Use Cases:**
- Find highest revenue, profit, or performance
- Identify peak values for capacity planning
- Determine ceiling values for budgets
- When you need "maximum", "highest", "largest", or "peak" values

**Example:**
```python
MAX(ctx, [10, 5, 20, 3])  # Returns Decimal('20')
```

### PRODUCT

**Purpose:** Multiply values together using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Array or range of numeric values (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal - Product of all values

**Use Cases:**
- Calculate compound growth rates
- Multiply price by quantity calculations
- Compute cumulative factors
- When you need to "multiply", "compound", or calculate "products"

**Example:**
```python
PRODUCT(ctx, [2, 3, 4])  # Returns Decimal('24')
```

### MEDIAN

**Purpose:** Calculate the middle value of a dataset using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Series/array of numbers (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal - Median value

**Use Cases:**
- Find typical values when outliers exist
- Analyze salary or compensation distributions
- Determine middle performance metrics
- When you need "median", "middle", or "50th percentile" values

**Example:**
```python
MEDIAN(ctx, [1, 2, 3, 4, 5])  # Returns Decimal('3')
```

### MODE

**Purpose:** Find the most frequently occurring value using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Series/array of numbers (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal or list of Decimals - Most frequent value(s)

**Use Cases:**
- Identify most common transaction amounts
- Find typical order sizes or quantities
- Analyze recurring expense patterns
- When you need "most common", "frequent", or "typical" values

**Example:**
```python
MODE(ctx, [1, 2, 2, 3, 3, 3])  # Returns Decimal('3')
```

### PERCENTILE

**Purpose:** Calculate specific percentiles using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Series/array of numbers (float, Decimal, Polars Series, NumPy array, or file path)
- `percentile_value`: Percentile value (0-1)

**Returns:** Decimal - Percentile value

**Use Cases:**
- Risk analysis (95th percentile for VaR)
- Performance benchmarking
- Outlier detection and analysis
- When you need "percentile", "quartile", or "quantile" analysis

**Example:**
```python
PERCENTILE(ctx, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], percentile_value=0.75)  # Returns Decimal('7.75')
```

### POWER

**Purpose:** Raise numbers to a power using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Series/array of base numbers (float, Decimal, Polars Series, NumPy array, or file path)
- `power`: Exponent (single value applied to all numbers)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** list[Decimal] - Results of values^power for each value

**Use Cases:**
- Calculate compound interest (1.05^years)
- Growth rate calculations
- Risk modeling with exponential functions
- When you need "power", "exponent", or "compound" calculations

**Example:**
```python
POWER(ctx, [2, 3, 4], power=2)  # Returns [Decimal('4'), Decimal('9'), Decimal('16')]
```

### SQRT

**Purpose:** Calculate square root using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Series/array of numbers (float, Decimal, Polars Series, NumPy array, or file path)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** list[Decimal] - Square roots of all values

**Use Cases:**
- Volatility calculations (standard deviation)
- Risk metrics calculations
- Geometric calculations for financial modeling
- When you need "square root", "volatility", or "standard deviation" calculations

**Example:**
```python
SQRT(ctx, [25, 16, 9])  # Returns [Decimal('5'), Decimal('4'), Decimal('3')]
```

### EXP

**Purpose:** Calculate e^x using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Series/array of exponents (float, Decimal, Polars Series, NumPy array, or file path)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** list[Decimal] - e^values for each value

**Use Cases:**
- Continuous compounding calculations
- Exponential growth modeling
- Option pricing models (Black-Scholes)
- When you need "exponential", "continuous compounding", or "e^x" calculations

**Example:**
```python
EXP(ctx, [1, 2, 3])  # Returns exponential values
```

### LN

**Purpose:** Calculate natural logarithm using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Series/array of numbers (float, Decimal, Polars Series, NumPy array, or file path)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** list[Decimal] - Natural logarithms of all values

**Use Cases:**
- Calculate continuously compounded returns
- Log-normal distribution modeling
- Growth rate transformations
- When you need "natural log", "ln", or "logarithmic" transformations

**Example:**
```python
LN(ctx, [2.718281828459045, 1, 10])  # Returns natural log values
```

### LOG

**Purpose:** Calculate logarithm with specified base using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Series/array of numbers (float, Decimal, Polars Series, NumPy array, or file path)
- `base`: Base of logarithm (optional, defaults to 10)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** list[Decimal] - Logarithms of all values

**Use Cases:**
- Base-10 logarithmic transformations
- Custom base calculations for specific models
- Data normalization and scaling
- When you need "log", "logarithm", or specific base calculations

**Example:**
```python
LOG(ctx, [100, 1000, 10000], base=10)  # Returns [Decimal('2'), Decimal('3'), Decimal('4')]
```

### ABS

**Purpose:** Calculate absolute value using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Series/array of numbers (float, Decimal, Polars Series, NumPy array, or file path)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** list[Decimal] - Absolute values of all input values

**Use Cases:**
- Calculate variance or deviation magnitudes
- Remove negative signs from differences
- Risk calculations requiring positive values
- When you need "absolute value", "magnitude", or "positive" values

**Example:**
```python
ABS(ctx, [-10, -5, 10, 15])  # Returns [Decimal('10'), Decimal('5'), Decimal('10'), Decimal('15')]
```

### SIGN

**Purpose:** Return sign of numbers (-1, 0, or 1).

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Series/array of numbers (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** list[int] - Signs of all input values (-1 for negative, 0 for zero, 1 for positive)

**Use Cases:**
- Determine direction of changes (positive/negative)
- Classify gains vs losses
- Conditional logic based on value signs
- When you need to identify "positive", "negative", or "direction" of values

**Example:**
```python
SIGN(ctx, [-15, 15, 0, -10, 20])  # Returns [-1, 1, 0, -1, 1]
```

### MOD

**Purpose:** Calculate remainder after division using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `dividends`: Series/array of dividend values (float, Decimal, Polars Series, NumPy array, or file path)
- `divisors`: Series/array of divisor values (same length as dividends, or single value, or file path)

**Returns:** list[Decimal] - Remainders of dividend % divisor for each pair

**Use Cases:**
- Calculate periodic patterns (monthly, quarterly cycles)
- Determine remainder amounts in allocations
- Modular arithmetic for financial calculations
- When you need "remainder", "modulo", or "cyclical" calculations

**Example:**
```python
MOD(ctx, [23, 10, 17], divisors=[5, 3, 4])  # Returns [Decimal('3'), Decimal('1'), Decimal('1')]
```

### ROUND

**Purpose:** Round numbers to specified digits using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Series/array of numbers to round (float, Decimal, Polars Series, NumPy array, or file path)
- `num_digits`: Number of decimal places

**Returns:** list[Decimal] - Rounded numbers

**Use Cases:**
- Format financial reports to specific precision
- Round currency amounts to cents
- Standardize decimal places for presentation
- When you need to "round", "format", or "precision" control

**Example:**
```python
ROUND(ctx, [3.14159, 2.71828, 1.41421], num_digits=2)  # Returns [Decimal('3.14'), Decimal('2.72'), Decimal('1.41')]
```

### ROUNDUP

**Purpose:** Round numbers up using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Series/array of numbers to round up (float, Decimal, Polars Series, NumPy array, or file path)
- `num_digits`: Number of decimal places

**Returns:** list[Decimal] - Rounded up numbers

**Use Cases:**
- Conservative estimates and budgeting
- Ceiling calculations for capacity planning
- Ensure minimum thresholds are met
- When you need "round up", "ceiling", or "conservative" estimates

**Example:**
```python
ROUNDUP(ctx, [3.14159, 2.71828, 1.41421], num_digits=2)  # Returns [Decimal('3.15'), Decimal('2.72'), Decimal('1.42')]
```

### ROUNDDOWN

**Purpose:** Round numbers down using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Series/array of numbers to round down (float, Decimal, Polars Series, NumPy array, or file path)
- `num_digits`: Number of decimal places

**Returns:** list[Decimal] - Rounded down numbers

**Use Cases:**
- Conservative revenue projections
- Floor calculations for minimum values
- Ensure maximum limits are not exceeded
- When you need "round down", "floor", or "conservative" calculations

**Example:**
```python
ROUNDDOWN(ctx, [3.14159, 2.71828, 1.41421], num_digits=2)  # Returns [Decimal('3.14'), Decimal('2.71'), Decimal('1.41')]
```

### WEIGHTED_AVERAGE

**Purpose:** Calculate weighted average of values using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Array of values (float, Decimal, Polars Series, NumPy array, or file path)
- `weights`: Array of weights (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal - Weighted average

**Use Cases:**
- Portfolio performance calculations
- Weighted cost of capital (WACC)
- Average prices weighted by volume
- When you need "weighted average", "portfolio", or "importance-weighted" calculations

**Example:**
```python
WEIGHTED_AVERAGE(ctx, [100, 200, 300], weights=[0.2, 0.3, 0.5])  # Returns Decimal('230')
```

### GEOMETRIC_MEAN

**Purpose:** Calculate geometric mean using Decimal precision (useful for growth rates).

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Series/array of positive numbers (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal - Geometric mean

**Use Cases:**
- Calculate average growth rates (CAGR)
- Portfolio return calculations
- Compound annual growth rate analysis
- When you need "geometric mean", "CAGR", "compound growth", or "average growth rate"

**Example:**
```python
GEOMETRIC_MEAN(ctx, [1.05, 1.08, 1.12, 1.03])  # Returns average growth multiplier
```

### HARMONIC_MEAN

**Purpose:** Calculate harmonic mean using Decimal precision (useful for rates/ratios).

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Series/array of positive numbers (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal - Harmonic mean

**Use Cases:**
- Average P/E ratios for portfolios
- Average interest rates
- Cost per unit calculations
- When you need "harmonic mean", "average rates", "P/E ratios", or "rate averaging"

**Example:**
```python
HARMONIC_MEAN(ctx, [15.2, 22.8, 18.5, 12.3])  # Returns average P/E ratio
```

### CUMSUM

**Purpose:** Calculate cumulative sum using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Series/array of numbers (float, Decimal, Polars Series, NumPy array, or file path)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** list[Decimal] - Array of cumulative sums

**Use Cases:**
- Running totals for cash flow analysis
- Cumulative revenue or expense tracking
- Year-to-date calculations
- When you need "cumulative", "running total", "YTD", or "progressive sum"

**Example:**
```python
CUMSUM(ctx, [10, 20, 30, 40])  # Returns [Decimal('10'), Decimal('30'), Decimal('60'), Decimal('100')]
```

### CUMPROD

**Purpose:** Calculate cumulative product using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Series/array of numbers (float, Decimal, Polars Series, NumPy array, or file path)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** list[Decimal] - Array of cumulative products

**Use Cases:**
- Compound growth calculations
- Cumulative return analysis
- Progressive multiplication factors
- When you need "cumulative product", "compound", "progressive growth", or "cumulative returns"

**Example:**
```python
CUMPROD(ctx, [1.05, 1.08, 1.12])  # Returns [Decimal('1.05'), Decimal('1.134'), Decimal('1.269')]
```

### VARIANCE_WEIGHTED

**Purpose:** Calculate weighted variance using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `values`: Array of values (returns, prices, or other financial metrics) (float, Decimal, Polars Series, NumPy array, or file path)
- `weights`: Array of weights (portfolio weights, importance factors, etc.) (float, Decimal, Polars Series, NumPy array, or file path)

**Returns:** Decimal - Weighted variance

**Use Cases:**
- Portfolio risk analysis
- Weighted dispersion calculations
- Risk-weighted variance measurements
- When you need "weighted variance", "portfolio risk", "weighted dispersion", or "risk analysis"

**Example:**
```python
VARIANCE_WEIGHTED(ctx, [0.12, 0.08, 0.15, 0.06], weights=[0.4, 0.3, 0.2, 0.1])  # Returns portfolio variance
```

## 2. Conditional Aggregation and Counting

### COUNTIF

**Purpose:** Count cells that meet one condition.

**Parameters:**
- `ctx`: RunContext object for file operations
- `range_to_evaluate`: Range of values to evaluate (list, Polars Series, NumPy array, or file path)
- `criteria`: Criteria to match (supports operators: >, <, >=, <=, =, <>, wildcards *, ?)

**Returns:** int - Count of cells meeting the criteria

**Use Cases:**
- Count high-value transactions for risk analysis
- Identify outliers or specific segments in data
- Count transactions above/below thresholds
- When you need to "count", "tally", or "enumerate" based on conditions

**Example:**
```python
COUNTIF(ctx, [100, 200, 150, 300, 50], criteria=">150")  # Returns 2
COUNTIF(ctx, ["Sales", "Marketing", "Sales", "IT"], criteria="Sales")  # Returns 2
```

### COUNTIFS

**Purpose:** Count cells that meet multiple conditions across different ranges.

**Parameters:**
- `ctx`: RunContext object for file operations
- `criteria_ranges`: List of ranges to evaluate (list, Polars Series, NumPy array, or file paths)
- `criteria_values`: List of criteria corresponding to each range

**Returns:** int - Count of rows where all criteria are met

**Use Cases:**
- Count high-value sales in specific regions for territory analysis
- Multi-dimensional data analysis and segmentation
- Complex filtering with multiple conditions
- When you need to count based on "multiple criteria", "AND conditions", or "intersecting filters"

**Example:**
```python
amounts = [100, 200, 150, 300, 50]
categories = ["A", "B", "A", "A", "B"]
COUNTIFS(ctx, [amounts, categories], criteria_values=[">100", "A"])  # Returns 2
```

### SUMIF

**Purpose:** Sum numbers that meet one condition using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `range_to_evaluate`: Range to evaluate against criteria (list, Polars Series, NumPy array, or file path)
- `criteria`: Criteria to match (supports operators: >, <, >=, <=, =, <>, wildcards *, ?)
- `sum_range`: Range to sum (optional, defaults to range_to_evaluate, can be file path)

**Returns:** Decimal - Sum of values meeting the criteria

**Use Cases:**
- Sum revenue for specific regions or categories
- Calculate conditional totals for financial analysis
- Aggregate expenses above budget thresholds
- When you need "conditional sum", "filtered total", or "sum where" calculations

**Example:**
```python
categories = ["A", "B", "A", "C", "A"]
values = [100, 200, 150, 300, 50]
SUMIF(ctx, categories, criteria="A", sum_range=values)  # Returns Decimal('300')
```

### SUMIFS

**Purpose:** Sum numbers that meet multiple conditions using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `sum_range`: Range of values to sum (list, Polars Series, NumPy array, or file path)
- `criteria_ranges`: List of ranges to evaluate (must all be same length as sum_range)
- `criteria_values`: List of criteria corresponding to each range

**Returns:** Decimal - Sum of values where all criteria are met

**Use Cases:**
- Multi-dimensional revenue analysis
- Complex cost allocation and summation
- Sum values based on multiple filtering criteria
- When you need "conditional sum with multiple criteria", "multi-filter aggregation", or "complex summation"

**Example:**
```python
amounts = [100, 200, 150, 300, 50]
categories = ["A", "B", "A", "A", "B"]
regions = ["North", "South", "North", "West", "South"]
SUMIFS(ctx, amounts, criteria_ranges=[categories, regions], criteria_values=["A", "North"])  # Returns Decimal('250')
```

### AVERAGEIF

**Purpose:** Calculate average of cells that meet one condition using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `range_to_evaluate`: Range to evaluate against criteria (list, Polars Series, NumPy array, or file path)
- `criteria`: Criteria to match (supports operators: >, <, >=, <=, =, <>, wildcards *, ?)
- `average_range`: Range to average (optional, defaults to range_to_evaluate, can be file path)

**Returns:** Decimal - Average of values meeting the criteria

**Use Cases:**
- Average transaction size by customer segment
- Calculate conditional averages for performance analysis
- Average performance metrics for specific groups
- When you need "conditional average", "filtered mean", or "average where" calculations

**Example:**
```python
categories = ["A", "B", "A", "C", "A"]
values = [100, 200, 150, 300, 50]
AVERAGEIF(ctx, categories, criteria="A", average_range=values)  # Returns Decimal('100')
```

### AVERAGEIFS

**Purpose:** Calculate average of cells that meet multiple conditions using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `average_range`: Range of values to average (list, Polars Series, NumPy array, or file path)
- `criteria_ranges`: List of ranges to evaluate (must all be same length as average_range)
- `criteria_values`: List of criteria corresponding to each range

**Returns:** Decimal - Average of values where all criteria are met

**Use Cases:**
- Multi-dimensional performance analysis
- Average returns across multiple factors
- Complex conditional averaging for financial metrics
- When you need "conditional average with multiple criteria", "multi-filter averaging", or "sophisticated mean calculations"

**Example:**
```python
amounts = [100, 200, 150, 300, 50]
categories = ["A", "B", "A", "A", "B"]
regions = ["North", "South", "North", "West", "South"]
AVERAGEIFS(ctx, amounts, criteria_ranges=[categories, regions], criteria_values=["A", "North"])  # Returns Decimal('125')
```

### MAXIFS

**Purpose:** Find maximum value based on multiple criteria using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `max_range`: Range of values to find maximum from (list, Polars Series, NumPy array, or file path)
- `criteria_ranges`: List of ranges to evaluate (must all be same length as max_range)
- `criteria_values`: List of criteria corresponding to each range

**Returns:** Decimal - Maximum value where all criteria are met

**Use Cases:**
- Find peak performance metrics in specific segments
- Identify highest values with multiple conditions
- Maximum revenue analysis by region and product
- When you need "conditional maximum", "filtered peak", or "max where multiple conditions"

**Example:**
```python
amounts = [100, 200, 150, 300, 50]
categories = ["A", "B", "A", "A", "B"]
regions = ["North", "South", "North", "West", "South"]
MAXIFS(ctx, amounts, criteria_ranges=[categories, regions], criteria_values=["A", "North"])  # Returns Decimal('150')
```

### MINIFS

**Purpose:** Find minimum value based on multiple criteria using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `min_range`: Range of values to find minimum from (list, Polars Series, NumPy array, or file path)
- `criteria_ranges`: List of ranges to evaluate (must all be same length as min_range)
- `criteria_values`: List of criteria corresponding to each range

**Returns:** Decimal - Minimum value where all criteria are met

**Use Cases:**
- Find lowest costs in specific segments
- Identify minimum performance thresholds
- Minimum value analysis with multiple conditions
- When you need "conditional minimum", "filtered floor", or "min where multiple conditions"

**Example:**
```python
amounts = [100, 200, 150, 300, 50]
categories = ["A", "B", "A", "A", "B"]
regions = ["North", "South", "North", "West", "South"]
MINIFS(ctx, amounts, criteria_ranges=[categories, regions], criteria_values=["A", "North"])  # Returns Decimal('100')
```

### SUMPRODUCT

**Purpose:** Sum the products of corresponding ranges using Decimal precision.

**Parameters:**
- `ctx`: RunContext object for file operations
- `*ranges`: Two or more ranges of equal length to multiply and sum

**Returns:** Decimal - Sum of products of corresponding elements

**Use Cases:**
- Portfolio value calculations (quantities × prices)
- Revenue calculations (units × unit prices)
- Weighted sum calculations
- When you need "multiply and sum", "portfolio calculations", or "weighted aggregations"

**Example:**
```python
SUMPRODUCT(ctx, [1, 2, 3], [4, 5, 6])  # Returns Decimal('32') = (1×4) + (2×5) + (3×6)
SUMPRODUCT(ctx, [10, 20], [5, 3], [2, 1])  # Returns Decimal('160') = (10×5×2) + (20×3×1)
```

### COUNTBLANK

**Purpose:** Count blank/empty cells in a range.

**Parameters:**
- `ctx`: RunContext object for file operations
- `range_to_evaluate`: Range to evaluate for blank/null values (list, Polars Series, NumPy array, or file path)

**Returns:** int - Count of blank/null cells

**Use Cases:**
- Data quality assessment and completeness analysis
- Identify missing data in financial datasets
- Count incomplete records for reporting
- When you need to assess "data completeness", "missing values", or "data quality"

**Example:**
```python
COUNTBLANK(ctx, [1, None, 3, "", 5, None])  # Returns 3
COUNTBLANK(ctx, ["A", "B", "", None, "C"])  # Returns 2
```

### COUNTA

**Purpose:** Count non-empty cells in a range.

**Parameters:**
- `ctx`: RunContext object for file operations
- `range_to_evaluate`: Range to evaluate for non-empty values (list, Polars Series, NumPy array, or file path)

**Returns:** int - Count of non-empty cells

**Use Cases:**
- Determine actual dataset size excluding missing values
- Count valid transaction records
- Data completeness analysis
- When you need to count "valid records", "non-missing data", or "actual data points"

**Example:**
```python
COUNTA(ctx, [1, None, 3, "", 5, 0])  # Returns 4 (counts 1, 3, 5, 0)
COUNTA(ctx, ["A", "B", "", None, "C"])  # Returns 3 (counts "A", "B", "C")
```

### AGGREGATE

**Purpose:** Perform various aggregations with error handling and filtering using Decimal precision.

**Parameters:**
- `function_num`: Function number (1=AVERAGE, 2=COUNT, 3=COUNTA, 4=MAX, 5=MIN, 6=PRODUCT, 9=SUM, 12=MEDIAN, 14=LARGE, 15=SMALL)
- `options`: Options for handling errors and filtering (0=default, 2=ignore errors, 3=ignore hidden and errors)
- `array`: Array to aggregate
- `k`: Additional parameter for functions like LARGE, SMALL, PERCENTILE

**Returns:** Decimal - Aggregated result

**Use Cases:**
- Robust financial calculations ignoring errors
- Clean aggregation of messy financial data
- Error-resistant analysis for data with inconsistencies
- When you need "error-resistant calculations", "robust aggregation", or "clean financial analysis"

**Example:**
```python
AGGREGATE(9, options=2, array=[10, "Error", 20, 30])  # Returns Decimal('60') (SUM ignoring errors)
AGGREGATE(4, options=0, array=[10, 20, 30, 40])  # Returns Decimal('40') (MAX)
```

### SUBTOTAL

**Purpose:** Calculate subtotals with filtering capability using Decimal precision.

**Parameters:**
- `function_num`: Function number (101-111 ignore hidden values, 1-11 include all)
- `ref1`: Reference range to calculate subtotal for

**Returns:** Decimal - Subtotal result

**Use Cases:**
- Financial reporting with filtered data
- Subtotals that ignore hidden rows
- Hierarchical financial analysis
- When you need "filtered subtotals", "visible data aggregation", or "reporting subtotals"

**Example:**
```python
SUBTOTAL(109, ref1=[10, 20, 30, 40])  # Returns Decimal('100') (SUM)
SUBTOTAL(101, ref1=[10, 20, 30, 40])  # Returns Decimal('25') (AVERAGE)


## 3. Conditional Logic

### MULTI_CONDITION_LOGIC

**Purpose:** Apply complex multi-condition logic with nested if/elif/else structures for hierarchical business rule implementation.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame to apply logic to, or file path to load data from
- `condition_tree`: Nested dictionary defining conditional logic structure
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - DataFrame with conditional results column, or path if output_filename provided

**Use Cases:**
- Credit risk assessment with multiple criteria
- Investment allocation strategies based on age and risk tolerance
- Revenue categorization for financial reporting
- Complex business rule implementation
- When you need "hierarchical conditions", "nested business rules", "multi-tier classification", or "complex decision trees"

**Example:**
```python
risk_tree = {
    'if': 'credit_score >= 750',
    'then': 'Low Risk',
    'elif': [
        {'condition': 'credit_score >= 650', 'then': 'Medium Risk'},
        {'condition': 'debt_ratio <= 0.3', 'then': 'Medium Risk'}
    ],
    'else': 'High Risk'
}
MULTI_CONDITION_LOGIC(ctx, customer_df, condition_tree=risk_tree)
```

### NESTED_IF_LOGIC

**Purpose:** Handle nested conditional statements with cascading if-then-else logic for complex financial decision trees.

**Parameters:**
- `run_context`: RunContext object for file operations
- `conditions_list`: List of condition strings or Polars expressions
- `results_list`: List of corresponding results for each condition
- `default_value`: Default value if no conditions are met
- `df_context`: Optional DataFrame context for string condition evaluation
- `output_filename`: Optional filename to save results as parquet file

**Returns:** List, DataFrame, or Path - Conditional results or path if output_filename provided

**Use Cases:**
- Bond rating classification systems
- Commission tier calculations based on sales performance
- Investment risk categorization
- Customer segmentation with multiple criteria
- Performance bonus calculations
- When you need "cascading conditions", "tiered classification", "sequential evaluation", or "priority-based logic"

**Example:**
```python
conditions = ['credit_score >= 800', 'credit_score >= 700', 'credit_score >= 600']
ratings = ['AAA', 'AA', 'A']
NESTED_IF_LOGIC(ctx, conditions, results_list=ratings, default_value='BBB', df_context=bond_df)
```

### CASE_WHEN

**Purpose:** SQL-style CASE WHEN logic for multiple conditional branches with clean and efficient conditional processing.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame to apply case logic to, or file path to load data from
- `case_conditions`: List of dictionaries with 'when' and 'then' keys
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - DataFrame with case results column, or path if output_filename provided

**Use Cases:**
- Customer segment classification based on revenue
- Performance rating systems
- Investment allocation based on multiple factors
- Product categorization for financial analysis
- When you need "SQL-style conditions", "multiple branches", "case statements", or "conditional classification"

**Example:**
```python
segments = [
    {'when': 'annual_revenue >= 1000000', 'then': 'Enterprise'},
    {'when': 'annual_revenue >= 100000', 'then': 'Corporate'},
    {'when': 'annual_revenue >= 10000', 'then': 'SMB'},
    {'else': 'Startup'}
]
CASE_WHEN(ctx, customer_df, case_conditions=segments)
```

### CONDITIONAL_AGGREGATION

**Purpose:** Aggregate data based on conditions, similar to SQL HAVING clause, for sophisticated financial analysis.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame to aggregate, or file path to load data from
- `group_columns`: List of columns to group by
- `condition`: Condition string to filter data before aggregation
- `aggregation_func`: Aggregation function ('sum', 'count', 'mean', 'max', 'min', 'std', 'var')
- `target_column`: Column to aggregate (required for most functions except 'count')
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - DataFrame with conditional aggregations, or path if output_filename provided

**Use Cases:**
- Sum high-value transactions by region
- Count profitable customers by segment
- Average revenue for enterprise clients by quarter
- Maximum deal size analysis for large deals
- Conditional financial metrics calculation
- When you need "conditional aggregation", "filtered summation", "conditional metrics", or "business rule aggregation"

**Example:**
```python
CONDITIONAL_AGGREGATION(
    ctx, sales_df,
    group_columns=['region'],
    condition='amount > 1000',
    aggregation_func='sum',
    target_column='amount'
)
```

## 4. Data Cleaning Operations

### STANDARDIZE_CURRENCY

**Purpose:** Standardize currency formats for financial data consistency across different systems and reporting requirements.

**Parameters:**
- `run_context`: RunContext object for file operations
- `currency_series`: Series of currency values in various formats (List, Polars Series, or file path)
- `target_format`: Target currency format (e.g., 'USD', 'EUR', 'GBP', 'JPY')
- `output_filename`: Filename to save standardized currency results

**Returns:** Path - Path to the saved parquet file containing standardized currency values

**Use Cases:**
- Multi-currency financial reporting consolidation
- Regulatory compliance for international transactions
- Data preparation for financial analysis and modeling
- Standardization for accounting system integration
- When you need to "standardize currency", "normalize money formats", "consolidate currencies", or "clean financial data"

**Example:**
```python
STANDARDIZE_CURRENCY(ctx, ["$1,234.56", "USD 1234.56", "1234.56 USD"], target_format="USD", output_filename="std_currency.parquet")
```

### CLEAN_NUMERIC

**Purpose:** Clean numeric data by removing non-numeric characters and converting to proper numeric format for mathematical operations.

**Parameters:**
- `run_context`: RunContext object for file operations
- `mixed_series`: Series with mixed data containing numbers and non-numeric characters (List, Polars Series, or file path)
- `output_filename`: Filename to save cleaned numeric results

**Returns:** Path - Path to the saved parquet file containing cleaned numeric values

**Use Cases:**
- Preparing imported financial data for analysis
- Cleaning data from various accounting systems
- Standardizing numeric formats across data sources
- Converting text-based financial reports to numeric format
- When you need to "clean numbers", "extract numeric values", "remove formatting", or "prepare data for calculations"

**Example:**
```python
CLEAN_NUMERIC(ctx, ["$1,234.56", "€987.65", "(500.00)", "2,345.67%"], output_filename="clean_numbers.parquet")
```

### NORMALIZE_NAMES

**Purpose:** Normalize company/customer names for consistent identification and reporting across financial systems.

**Parameters:**
- `run_context`: RunContext object for file operations
- `name_series`: Series of names to normalize (List, Polars Series, or file path)
- `normalization_rules`: Dictionary mapping variations to standard forms
- `output_filename`: Filename to save normalized names

**Returns:** Path - Path to the saved parquet file containing normalized names

**Use Cases:**
- Customer data deduplication and master data management
- Regulatory reporting with standardized entity names
- Financial consolidation across subsidiaries
- Vendor management and procurement standardization
- When you need to "standardize names", "normalize entities", "clean company names", or "deduplicate customers"

**Example:**
```python
rules = {"incorporated": "Inc.", "corporation": "Corp.", "company": "Co."}
NORMALIZE_NAMES(ctx, ["Apple Inc.", "Apple Incorporated", "APPLE INC"], normalization_rules=rules, output_filename="std_companies.parquet")
```

### REMOVE_DUPLICATES

**Purpose:** Remove duplicate records with configurable options for financial data integrity and accurate reporting.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame to process (Polars DataFrame or file path)
- `subset_columns`: List of column names to check for duplicates
- `keep_method`: Method to keep records ('first', 'last', 'none')
- `output_filename`: Filename to save deduplicated results

**Returns:** Path - Path to the saved parquet file containing deduplicated data

**Use Cases:**
- Transaction deduplication in payment processing
- Customer master data management
- Financial report accuracy and compliance
- Data warehouse ETL processes
- When you need to "remove duplicates", "deduplicate data", "clean records", or "ensure data integrity"

**Example:**
```python
REMOVE_DUPLICATES(ctx, transactions_df, subset_columns=["transaction_id"], keep_method="first", output_filename="unique_transactions.parquet")
```

### STANDARDIZE_DATES

**Purpose:** Convert various date formats to a standardized format for consistent financial reporting and time-series analysis.

**Parameters:**
- `run_context`: RunContext object for file operations
- `date_series`: Series of dates in various formats (List, Polars Series, or file path)
- `target_format`: Target date format string (e.g., '%Y-%m-%d', '%m/%d/%Y')
- `output_filename`: Filename to save standardized dates

**Returns:** Path - Path to the saved parquet file containing standardized dates

**Use Cases:**
- Financial report standardization across systems
- Regulatory compliance for date formatting requirements
- Data warehouse ETL date normalization
- Cross-border financial data integration
- When you need to "standardize dates", "normalize date formats", "clean date data", or "prepare time series data"

**Example:**
```python
STANDARDIZE_DATES(ctx, ["01/15/2023", "2023-01-15", "15-Jan-2023"], target_format="%Y-%m-%d", output_filename="std_dates.parquet")
```

## 5. Data Comparison and Ranking

### RANK_BY_COLUMN

**Purpose:** Rank records by column values with financial precision for performance analysis and competitive benchmarking.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame to rank (Polars DataFrame or file path)
- `column`: Column to rank by
- `ascending`: Sort order (default False for descending - highest values get rank 1)
- `method`: Ranking method ('average', 'min', 'max', 'dense', 'ordinal')
- `output_filename`: Filename to save results as parquet file

**Returns:** Path - Path to saved results file with ranking column added

**Use Cases:**
- Ranking investment portfolios by annual return for performance evaluation
- Customer ranking by total revenue for account prioritization
- Product profitability rankings for strategic decisions
- Employee performance rankings for compensation planning
- When you need to "rank", "order", "prioritize", or "compare performance"

### PERCENTILE_RANK

**Purpose:** Calculate percentile rank for each value to understand relative performance and position within datasets.

**Parameters:**
- `run_context`: RunContext object for file operations
- `series`: Series to rank (Polars Series, list, NumPy array, or file path)
- `method`: Ranking method for ties ('average', 'min', 'max', 'dense', 'ordinal')
- `output_filename`: Filename to save results as parquet file

**Returns:** Path - Path to saved results file containing percentile ranks (0-100)

**Use Cases:**
- Risk assessment showing what percentage of returns fall below each value
- Credit score analysis relative to portfolio distribution
- Performance benchmarking against peer groups
- Outlier detection and threshold analysis
- When you need "percentile analysis", "relative position", "benchmarking", or "distribution analysis"

### COMPARE_PERIODS

**Purpose:** Compare financial values between specified periods with comprehensive variance analysis for period-over-period reporting.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame containing period data (Polars DataFrame or file path)
- `value_column`: Column containing values to compare
- `period_column`: Column containing period identifiers
- `periods_to_compare`: List of exactly 2 period identifiers to compare
- `output_filename`: Filename to save results as parquet file

**Returns:** Path - Path to saved results file containing period comparison with variance metrics

**Use Cases:**
- Year-over-year revenue analysis for growth tracking
- Quarterly expense comparisons for budget management
- Monthly performance variance reporting
- Period-over-period profitability analysis
- When you need "period comparison", "variance analysis", "growth tracking", or "trend analysis"

### VARIANCE_FROM_TARGET

**Purpose:** Calculate variance from target values with comprehensive analysis for budget control and performance management.

**Parameters:**
- `run_context`: RunContext object for file operations
- `actual_values`: Actual values achieved (Series, list, NumPy array, or file path)
- `target_values`: Target/budget values (Series, list, NumPy array, or file path)
- `output_filename`: Filename to save results as parquet file

**Returns:** Path - Path to saved results file containing variance analysis with absolute and percentage variances

**Use Cases:**
- Budget variance analysis for financial control
- Sales performance vs targets for commission calculations
- Expense variance reporting for cost management
- KPI tracking against strategic objectives
- When you need "budget variance", "target analysis", "performance tracking", or "variance reporting"

### RANK_CORRELATION

**Purpose:** Calculate Spearman rank correlation coefficient to measure monotonic relationships between financial variables.

**Parameters:**
- `run_context`: RunContext object for file operations
- `series1`: First series for correlation (Series, list, NumPy array, or file path)
- `series2`: Second series for correlation (Series, list, NumPy array, or file path)

**Returns:** float - Spearman rank correlation coefficient (-1 to 1)

**Use Cases:**
- Analyzing relationship between credit scores and default rates
- Investment performance correlation across different periods
- Company size vs profitability relationship analysis
- Risk factor correlation for portfolio management
- When you need "correlation analysis", "relationship measurement", "monotonic relationships", or "rank correlation"

## 6. Data Filtering and Selection

### FILTER_BY_DATE_RANGE

**Purpose:** Filter DataFrame rows by date range using optimized date operations for time-based financial analysis.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame to filter (Polars DataFrame or file path)
- `date_column`: Name of the date column to filter on
- `start_date`: Start date in ISO format (YYYY-MM-DD)
- `end_date`: End date in ISO format (YYYY-MM-DD)
- `output_filename`: Filename to save filtered results as parquet file

**Returns:** Path - Path to saved filtered DataFrame

**Use Cases:**
- Filter transactions for quarterly or annual reporting periods
- Extract data for specific fiscal years or quarters
- Analyze performance within date ranges for trend analysis
- Regulatory reporting requiring specific time periods
- When you need to "filter by date", "time period analysis", "date range selection", or "period-specific data"

**Example:**
```python
FILTER_BY_DATE_RANGE(ctx, transactions_df, date_column='transaction_date', start_date='2024-01-01', end_date='2024-12-31', output_filename='yearly_transactions.parquet')
```

### FILTER_BY_VALUE

**Purpose:** Filter DataFrame rows based on column values using comparison operators for targeted data analysis.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame to filter (Polars DataFrame or file path)
- `column`: Column name to filter on
- `operator`: Comparison operator ('>', '<', '>=', '<=', '==', '!=')
- `value`: Value to compare against
- `output_filename`: Filename to save filtered results as parquet file

**Returns:** Path - Path to saved filtered DataFrame

**Use Cases:**
- Filter high-value transactions for risk analysis
- Extract customers above revenue thresholds
- Identify outliers or exceptional performance
- Filter data for specific value ranges or criteria
- When you need to "filter by value", "threshold analysis", "conditional selection", or "value-based filtering"

**Example:**
```python
FILTER_BY_VALUE(ctx, sales_df, column='amount', operator='>', value=1000, output_filename='high_value_sales.parquet')
```

### FILTER_BY_MULTIPLE_CONDITIONS

**Purpose:** Filter DataFrame using multiple conditions with AND logic for complex data selection criteria.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame to filter (Polars DataFrame or file path)
- `conditions_dict`: Dictionary of conditions {column: value} or {column: 'operator:value'}
- `output_filename`: Filename to save filtered results as parquet file

**Returns:** Path - Path to saved filtered DataFrame

**Use Cases:**
- Multi-criteria customer segmentation analysis
- Complex risk assessment with multiple factors
- Advanced filtering for regulatory compliance
- Sophisticated data selection for modeling
- When you need "multiple conditions", "complex filtering", "AND logic", or "multi-criteria selection"

**Example:**
```python
FILTER_BY_MULTIPLE_CONDITIONS(ctx, df, conditions_dict={'region': 'North', 'sales': '>:1000', 'status': 'active'}, output_filename='filtered_data.parquet')
```

### TOP_N

**Purpose:** Select top N records by value using optimized ranking operations for performance analysis.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame to select from (Polars DataFrame or file path)
- `column`: Column to sort by for ranking
- `n`: Number of top records to select
- `ascending`: Sort order (False for top values, True for bottom values)
- `output_filename`: Filename to save selected results as parquet file

**Returns:** Path - Path to saved DataFrame with top N records

**Use Cases:**
- Identify top-performing customers, products, or regions
- Select highest revenue transactions for analysis
- Find best-performing investments or portfolios
- Extract top performers for benchmarking
- When you need "top performers", "highest values", "best results", or "ranking analysis"

**Example:**
```python
TOP_N(ctx, customers_df, column='revenue', n=10, ascending=False, output_filename='top_customers.parquet')
```

### BOTTOM_N

**Purpose:** Select bottom N records by value using optimized ranking for identifying underperformers.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame to select from (Polars DataFrame or file path)
- `column`: Column to sort by for ranking
- `n`: Number of bottom records to select
- `output_filename`: Filename to save selected results as parquet file

**Returns:** Path - Path to saved DataFrame with bottom N records

**Use Cases:**
- Identify underperforming customers or products
- Find lowest margin transactions for cost analysis
- Extract worst-performing investments for review
- Analyze bottom performers for improvement strategies
- When you need "underperformers", "lowest values", "worst results", or "bottom ranking"

**Example:**
```python
BOTTOM_N(ctx, products_df, column='profit_margin', n=5, output_filename='lowest_margin_products.parquet')
```

### SAMPLE_DATA

**Purpose:** Extract random samples from large datasets using optimized sampling for statistical analysis.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame to sample from (Polars DataFrame or file path)
- `n_samples`: Number of random samples to extract
- `random_state`: Random seed for reproducible sampling (optional)
- `output_filename`: Filename to save sampled results as parquet file

**Returns:** Path - Path to saved DataFrame with sampled records

**Use Cases:**
- Create representative samples for statistical analysis
- Reduce large datasets for faster processing
- Generate test datasets for model validation
- Random sampling for audit or quality control
- When you need "random sampling", "statistical sampling", "data reduction", or "representative subset"

**Example:**
```python
SAMPLE_DATA(ctx, large_dataset_df, n_samples=1000, random_state=42, output_filename='sample_data.parquet')
```

## 7. Data Transformation and Pivoting

### PIVOT_TABLE

**Purpose:** Create pivot tables with aggregations by groups for financial data analysis and reporting.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame to pivot or file path
- `index_cols`: Index columns for grouping
- `value_cols`: Value columns to aggregate
- `agg_func`: Aggregation function ('sum', 'mean', 'count', etc.)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - Pivoted DataFrame or path to saved file

**Use Cases:**
- Create summary tables for financial reporting
- Aggregate revenue by region and product categories
- Generate cross-tabulated financial metrics
- Transform transactional data into summary format
- When you need "pivot", "cross-tabulate", "summarize", or "aggregate by groups"

**Example:**
```python
PIVOT_TABLE(ctx, sales_df, index_cols=['region'], value_cols=['revenue'], agg_func='sum')
```

### UNPIVOT

**Purpose:** Transform wide data to long format using melt operation for data normalization and analysis.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame to unpivot or file path
- `identifier_cols`: Identifier columns to keep
- `value_cols`: Value columns to melt
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - Unpivoted DataFrame or path to saved file

**Use Cases:**
- Convert quarterly financial data from wide to long format
- Normalize time-series data for analysis
- Prepare data for time-series forecasting models
- Transform spreadsheet-style data for database loading
- When you need "melt", "unpivot", "normalize", or "transform wide to long"

**Example:**
```python
UNPIVOT(ctx, df, identifier_cols=['customer_id'], value_cols=['Q1', 'Q2', 'Q3', 'Q4'])
```

### GROUP_BY

**Purpose:** Group data and apply aggregation functions for financial analysis and reporting.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame to group or file path
- `grouping_cols`: Grouping columns
- `agg_func`: Aggregation function ('sum', 'mean', 'count', etc.)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - Grouped DataFrame or path to saved file

**Use Cases:**
- Calculate total revenue by customer segments
- Average performance metrics by time periods
- Count transactions by category for analysis
- Group financial data for comparative analysis
- When you need "group by", "aggregate", "summarize by groups", or "categorize data"

**Example:**
```python
GROUP_BY(ctx, sales_df, grouping_cols=['category'], agg_func='sum')
```

### CROSS_TAB

**Purpose:** Create cross-tabulation tables for multi-dimensional financial analysis and reporting.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame to cross-tabulate or file path
- `row_vars`: Row variables
- `col_vars`: Column variables
- `values`: Values to aggregate
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - Cross-tabulated DataFrame or path to saved file

**Use Cases:**
- Analyze sales performance across regions and products
- Create contingency tables for risk analysis
- Generate multi-dimensional financial reports
- Compare performance metrics across multiple dimensions
- When you need "cross-tabulate", "pivot with multiple dimensions", "contingency table", or "multi-dimensional analysis"

**Example:**
```python
CROSS_TAB(ctx, df, row_vars=['region'], col_vars=['product'], values=['sales'])
```

### GROUP_BY_AGG

**Purpose:** Group a DataFrame by one or more columns and apply multiple aggregation functions for comprehensive financial analysis.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame to group or file path
- `group_by_cols`: List of columns to group by
- `agg_dict`: Dictionary of column-aggregation function pairs
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - Grouped DataFrame with multiple aggregations or path to saved file

**Use Cases:**
- Calculate multiple metrics (sum, count, average) by customer segments
- Generate comprehensive financial summaries with various aggregations
- Create detailed performance reports with multiple KPIs
- Analyze data with different aggregation requirements per column
- When you need "multiple aggregations", "complex grouping", "multi-metric analysis", or "detailed summarization"

**Example:**
```python
GROUP_BY_AGG(ctx, df, group_by_cols=['region'], agg_dict={'revenue': 'sum', 'users': 'count'})
```

### STACK

**Purpose:** Stack multiple columns into single column for data normalization and time-series analysis.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame to stack or file path
- `columns_to_stack`: Columns to stack
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - Stacked DataFrame or path to saved file

**Use Cases:**
- Convert quarterly financial data into time-series format
- Normalize multi-column data for analysis
- Prepare data for longitudinal studies
- Transform wide-format financial reports to long format
- When you need "stack", "melt multiple columns", "normalize columns", or "convert to time-series"

**Example:**
```python
STACK(ctx, df, columns_to_stack=['Q1', 'Q2', 'Q3', 'Q4'])
```

### UNSTACK

**Purpose:** Unstack index level to columns for data transformation and reporting.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame to unstack or file path
- `level_to_unstack`: Level to unstack
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - Unstacked DataFrame or path to saved file

**Use Cases:**
- Convert time-series data back to wide format for reporting
- Transform normalized data for spreadsheet-style presentation
- Create pivot-style reports from stacked data
- Prepare data for visualization tools requiring wide format
- When you need "unstack", "pivot from long to wide", "denormalize", or "spread data"

**Example:**
```python
UNSTACK(ctx, stacked_df, level_to_unstack='quarter')
```

### MERGE

**Purpose:** Merge/join two DataFrames for data integration and comprehensive financial analysis.

**Parameters:**
- `run_context`: RunContext object for file operations
- `left_df`: Left DataFrame or file path
- `right_df`: Right DataFrame or file path
- `join_keys`: Join keys (single string or list of strings)
- `join_type`: Join type ('inner', 'left', 'right', 'full', 'cross', 'semi', 'anti')
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - Merged DataFrame or path to saved file

**Use Cases:**
- Combine customer data with transaction data for analysis
- Integrate financial statements from different sources
- Join master data with transactional data
- Merge lookup tables with main datasets
- When you need "join", "merge", "combine datasets", or "data integration"

**Example:**
```python
MERGE(ctx, sales_df, customer_df, join_keys='customer_id', join_type='left')
```

### CONCAT

**Purpose:** Concatenate DataFrames for combining similar datasets and data aggregation.

**Parameters:**
- `run_context`: RunContext object for file operations
- `dataframes`: List of DataFrames or file paths
- `axis`: Axis to concatenate on (0 for vertical, 1 for horizontal)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - Concatenated DataFrame or path to saved file

**Use Cases:**
- Combine monthly sales data into annual reports
- Merge similar datasets from different periods
- Aggregate data from multiple sources with same structure
- Stack financial reports from different departments
- When you need "concatenate", "combine vertically", "merge similar data", or "stack datasets"

**Example:**
```python
CONCAT(ctx, [df1, df2, df3], axis=0)
```

### FILL_FORWARD

**Purpose:** Forward fill missing values for time-series data completion and financial data cleaning.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame or Series to fill or file path
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - DataFrame with filled values or path to saved file

**Use Cases:**
- Fill missing stock prices in time-series data
- Complete financial reports with missing values
- Handle gaps in transaction data
- Clean time-series financial datasets
- When you need "forward fill", "fill missing values", "complete time series", or "data imputation"

**Example:**
```python
FILL_FORWARD(ctx, revenue_series)
```

### INTERPOLATE

**Purpose:** Interpolate missing values for smooth financial data analysis and modeling.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame or Series to interpolate or file path
- `method`: Interpolation method ('linear')
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - DataFrame with interpolated values or path to saved file

**Use Cases:**
- Estimate missing values in financial time-series
- Smooth price data for technical analysis
- Fill gaps in economic indicators
- Prepare data for financial modeling and forecasting
- When you need "interpolate", "estimate missing values", "smooth data", or "fill gaps"

**Example:**
```python
INTERPOLATE(ctx, data_series, method='linear')


## 8. Data Validation and Quality

### CHECK_DUPLICATES

**Purpose:** Identify duplicate records in dataset using Polars efficient duplicate detection.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame to check for duplicates (DataFrame or file path)
- `columns_to_check`: List of column names to check for duplicates
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - DataFrame with duplicate flags or path to saved file

**Use Cases:**
- Detect duplicate transaction IDs in financial data for fraud prevention
- Identify duplicate customer records for master data management
- Find repeated entries in accounting ledgers
- Data quality assessment for financial reporting
- When you need to "check duplicates", "find repeated records", "data deduplication", or "identify redundant entries"

**Example:**
```python
CHECK_DUPLICATES(ctx, transactions_df, columns_to_check=['transaction_id'])
CHECK_DUPLICATES(ctx, "transactions.parquet", columns_to_check=['transaction_id', 'amount'])
```

### VALIDATE_DATES

**Purpose:** Validate date formats and ranges using Polars date parsing capabilities.

**Parameters:**
- `run_context`: RunContext object for file operations
- `date_series`: Date series to validate (Series, list, or file path)
- `min_date`: Minimum acceptable date (ISO format: YYYY-MM-DD)
- `max_date`: Maximum acceptable date (ISO format: YYYY-MM-DD)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.Series or Path - Series with validation flags or path to saved file

**Use Cases:**
- Validate transaction dates are within fiscal year for compliance
- Check employee hire dates for reasonableness
- Verify financial statement dates for reporting periods
- Data quality control for time-series financial analysis
- When you need to "validate dates", "check date ranges", "date format validation", or "temporal data quality"

**Example:**
```python
VALIDATE_DATES(ctx, date_column, min_date='2020-01-01', max_date='2025-12-31')
VALIDATE_DATES(ctx, "dates.parquet", min_date='2020-01-01', max_date='2025-12-31')
```

### CHECK_NUMERIC_RANGE

**Purpose:** Validate numeric values within expected ranges using Polars efficient range checking.

**Parameters:**
- `run_context`: RunContext object for file operations
- `numeric_series`: Numeric series to validate (Series, list, or file path)
- `min_value`: Minimum acceptable value (inclusive)
- `max_value`: Maximum acceptable value (inclusive)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.Series or Path - Series with validation flags or path to saved file

**Use Cases:**
- Validate revenue amounts are positive and reasonable for financial controls
- Check interest rates are within expected bounds for risk management
- Verify employee ages for HR data quality
- Data validation for financial modeling inputs
- When you need to "validate ranges", "check numeric bounds", "data validation", or "outlier detection"

**Example:**
```python
CHECK_NUMERIC_RANGE(ctx, revenue_column, min_value=0, max_value=1000000)
CHECK_NUMERIC_RANGE(ctx, "financial_data.parquet", min_value=0, max_value=1000000)
```

### OUTLIER_DETECTION

**Purpose:** Detect statistical outliers using IQR or z-score methods with SciPy integration.

**Parameters:**
- `run_context`: RunContext object for file operations
- `numeric_series`: Numeric series to analyze (Series, list, or file path)
- `method`: Detection method ('iqr' or 'z-score')
- `threshold`: Detection threshold (1.5 for IQR, 2-3 for z-score typically)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.Series or Path - Series with outlier flags or path to saved file

**Use Cases:**
- Detect outlier transactions for fraud detection and risk assessment
- Identify unusual stock returns for investment analysis
- Find anomalous revenue figures for financial review
- Statistical quality control for financial data
- When you need to "detect outliers", "find anomalies", "statistical analysis", or "risk detection"

**Example:**
```python
OUTLIER_DETECTION(ctx, sales_data, method='iqr', threshold=1.5)
OUTLIER_DETECTION(ctx, "returns.parquet", method='z-score', threshold=2.5)
```

### COMPLETENESS_CHECK

**Purpose:** Check data completeness by column using Polars efficient null counting.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame to check for completeness (DataFrame or file path)

**Returns:** Dict[str, float] - Dictionary with column names and completeness percentages (0-100)

**Use Cases:**
- Assess customer data completeness for CRM systems
- Evaluate financial statement completeness for reporting
- Data quality assessment for regulatory compliance
- Master data management and data governance
- When you need to "check completeness", "assess data quality", "null analysis", or "data coverage"

**Example:**
```python
COMPLETENESS_CHECK(ctx, financial_data_df)
COMPLETENESS_CHECK(ctx, "customer_data.parquet")
```

### CONSISTENCY_CHECK

**Purpose:** Check data consistency across related fields using configurable business rules.

**Parameters:**
- `run_context`: RunContext object for file operations
- `df`: DataFrame to check for consistency (DataFrame or file path)
- `consistency_rules`: Rules for consistency checking (e.g., {'total': ['subtotal', 'tax']})
- `output_filename`: Optional filename to save results as parquet file

**Returns:** pl.DataFrame or Path - DataFrame with consistency flags or path to saved file

**Use Cases:**
- Verify invoice totals equal sum of components for accounting accuracy
- Check budget vs actual consistency for financial planning
- Validate balance sheet equation: Assets = Liabilities + Equity
- Data integrity checks for financial calculations
- When you need to "check consistency", "validate calculations", "business rule validation", or "financial integrity"

**Example:**
```python
CONSISTENCY_CHECK(ctx, df, consistency_rules={'total': ['subtotal', 'tax']})
CONSISTENCY_CHECK(ctx, "invoices.parquet", consistency_rules={'total': ['subtotal', 'tax']})


## 9. Date and Time Functions

### TODAY

**Purpose:** Return the current date.

**Parameters:**
- `run_context`: RunContext object for file operations

**Returns:** datetime.date - Current date

**Use Cases:**
- Get current date for financial reporting timestamps
- Set today's date as reference point for calculations
- Date stamp financial transactions
- When you need "current date", "today's date", or "reference date"

**Example:**
```python
TODAY(ctx)  # Returns datetime.date(2025, 1, 8)
```

### NOW

**Purpose:** Return the current date and time.

**Parameters:**
- `run_context`: RunContext object for file operations

**Returns:** datetime.datetime - Current date and time

**Use Cases:**
- Timestamp financial transactions with precise time
- Audit trail creation for financial operations
- Real-time financial data processing
- When you need "current timestamp", "exact time", or "datetime now"

**Example:**
```python
NOW(ctx)  # Returns datetime.datetime(2025, 1, 8, 14, 30, 45, 123456)
```

### DATE

**Purpose:** Construct a date from year, month, and day components.

**Parameters:**
- `run_context`: RunContext object for file operations
- `year`: Year (e.g., 2025)
- `month`: Month (1-12)
- `day`: Day (1-31)

**Returns:** datetime.date - Constructed date

**Use Cases:**
- Create specific dates for financial modeling
- Build date references for reporting periods
- Construct historical dates for analysis
- When you need to "create date", "build date", or "construct date from components"

**Example:**
```python
DATE(ctx, 2025, 4, 15)  # Returns datetime.date(2025, 4, 15)
```

### YEAR

**Purpose:** Extract the year from a date.

**Parameters:**
- `run_context`: RunContext object for file operations
- `date`: Date to extract year from (date object, string, or file path)

**Returns:** int - Year component

**Use Cases:**
- Extract year for annual financial analysis
- Group data by year for reporting
- Calculate age-based financial metrics
- When you need "extract year", "get year", or "year from date"

**Example:**
```python
YEAR(ctx, datetime.date(2025, 4, 15))  # Returns 2025
YEAR(ctx, "2025-04-15")  # Returns 2025
```

### MONTH

**Purpose:** Extract the month from a date.

**Parameters:**
- `run_context`: RunContext object for file operations
- `date`: Date to extract month from (date object, string, or file path)

**Returns:** int - Month component (1-12)

**Use Cases:**
- Extract month for monthly financial reporting
- Seasonal analysis and forecasting
- Monthly budget variance calculations
- When you need "extract month", "get month", or "month from date"

**Example:**
```python
MONTH(ctx, datetime.date(2025, 4, 15))  # Returns 4
MONTH(ctx, "2025-04-15")  # Returns 4
```

### DAY

**Purpose:** Extract the day from a date.

**Parameters:**
- `run_context`: RunContext object for file operations
- `date`: Date to extract day from (date object, string, or file path)

**Returns:** int - Day component (1-31)

**Use Cases:**
- Extract day for daily financial tracking
- Payment due date calculations
- Daily cash flow analysis
- When you need "extract day", "get day", or "day from date"

**Example:**
```python
DAY(ctx, datetime.date(2025, 4, 15))  # Returns 15
DAY(ctx, "2025-04-15")  # Returns 15
```

### EDATE

**Purpose:** Calculate a date a given number of months before or after a specified date.

**Parameters:**
- `run_context`: RunContext object for file operations
- `start_date`: Start date (date object, string, or file path)
- `months`: Number of months to add (positive) or subtract (negative)

**Returns:** datetime.date - Calculated date

**Use Cases:**
- Calculate maturity dates for financial instruments
- Project future cash flow dates
- Determine anniversary dates for contracts
- When you need "add months", "subtract months", or "date arithmetic"

**Example:**
```python
EDATE(ctx, datetime.date(2025, 1, 15), 3)  # Returns datetime.date(2025, 4, 15)
EDATE(ctx, "2025-01-15", -2)  # Returns datetime.date(2024, 11, 15)
```

### EOMONTH

**Purpose:** Find the end of the month for a given date.

**Parameters:**
- `run_context`: RunContext object for file operations
- `start_date`: Start date (date object, string, or file path)
- `months`: Number of months to add (positive) or subtract (negative)

**Returns:** datetime.date - End of month date

**Use Cases:**
- Calculate reporting period end dates
- Determine month-end closing dates
- Set deadline dates for monthly processes
- When you need "end of month", "month end", or "last day of month"

**Example:**
```python
EOMONTH(ctx, datetime.date(2025, 1, 15), 0)  # Returns datetime.date(2025, 1, 31)
EOMONTH(ctx, "2025-01-15", 2)  # Returns datetime.date(2025, 3, 31)
```

### DATEDIF

**Purpose:** Calculate the difference between two dates.

**Parameters:**
- `run_context`: RunContext object for file operations
- `start_date`: Start date (date object, string, or file path)
- `end_date`: End date (date object, string, or file path)
- `unit`: Unit of difference ("Y" for years, "M" for months, "D" for days)

**Returns:** int - Difference in specified unit

**Use Cases:**
- Calculate investment holding periods
- Determine loan terms and durations
- Age calculation for financial products
- When you need "date difference", "time between dates", or "period calculation"

**Example:**
```python
DATEDIF(ctx, datetime.date(2024, 1, 1), datetime.date(2025, 1, 1), "Y")  # Returns 1
DATEDIF(ctx, "2024-01-01", "2024-04-01", "M")  # Returns 3
```

### YEARFRAC

**Purpose:** Calculate the fraction of a year between two dates.

**Parameters:**
- `run_context`: RunContext object for file operations
- `start_date`: Start date (date object, string, or file path)
- `end_date`: End date (date object, string, or file path)
- `basis`: Day count basis (0=30/360 US, 1=Actual/Actual, 2=Actual/360, 3=Actual/365, 4=30/360 European)

**Returns:** Decimal - Fraction of year between dates

**Use Cases:**
- Calculate interest accrual periods
- Determine partial year financial metrics
- Prorate calculations for time-based allocations
- When you need "year fraction", "time period in years", or "day count fraction"

**Example:**
```python
YEARFRAC(ctx, datetime.date(2024, 1, 1), datetime.date(2024, 7, 1), 1)  # Returns Decimal('0.4972677595628415')
```

### WORKDAY

**Purpose:** Return a future or past date excluding weekends and holidays.

**Parameters:**
- `run_context`: RunContext object for file operations
- `start_date`: Start date (date object, string, or file path)
- `days`: Number of working days to add (positive) or subtract (negative)
- `holidays`: Optional list of holiday dates to exclude

**Returns:** datetime.date - Calculated working day

**Use Cases:**
- Calculate payment due dates excluding weekends
- Determine business day delivery schedules
- Set realistic project timelines
- When you need "working days", "business days", or "exclude weekends"

**Example:**
```python
WORKDAY(ctx, datetime.date(2025, 1, 1), 5)  # Returns datetime.date(2025, 1, 8)
```

### NETWORKDAYS

**Purpose:** Count working days between two dates.

**Parameters:**
- `run_context`: RunContext object for file operations
- `start_date`: Start date (date object, string, or file path)
- `end_date`: End date (date object, string, or file path)
- `holidays`: Optional list of holiday dates to exclude

**Returns:** int - Number of working days

**Use Cases:**
- Calculate business days for SLA tracking
- Determine project duration in working days
- Count trading days for financial analysis
- When you need "working day count", "business days between", or "network days"

**Example:**
```python
NETWORKDAYS(ctx, datetime.date(2025, 1, 1), datetime.date(2025, 1, 10))  # Returns 8
```

### DATE_RANGE

**Purpose:** Generate a series of dates between a start and end date with a specified frequency, essential for creating financial model timelines.

**Parameters:**
- `run_context`: RunContext object for file operations
- `start_date`: Start date (date object, string, or file path)
- `end_date`: End date (date object, string, or file path)
- `frequency`: Frequency ('D' for daily, 'W' for weekly, 'M' for month-end, 'Q' for quarter-end, 'Y' for year-end)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** list[datetime.date] - Series of dates

**Use Cases:**
- Create financial model timelines and projections
- Generate reporting schedules for regular intervals
- Build cash flow forecasting periods
- When you need "date series", "time periods", or "financial timeline"

**Example:**
```python
DATE_RANGE(ctx, "2025-01-01", "2025-03-31", "M")  # Returns [datetime.date(2025, 1, 31), datetime.date(2025, 2, 28), datetime.date(2025, 3, 31)]
```

### WEEKDAY

**Purpose:** Return day of week as number.

**Parameters:**
- `run_context`: RunContext object for file operations
- `serial_number`: Date (date object, string, or file path)
- `return_type`: Return type (1=Sunday=1 to Saturday=7, 2=Monday=1 to Sunday=7, 3=Monday=0 to Sunday=6)

**Returns:** int - Day of week number

**Use Cases:**
- Analyze weekly patterns in financial data
- Determine day-of-week effects on trading
- Schedule recurring financial processes
- When you need "day of week", "weekday number", or "week analysis"

**Example:**
```python
WEEKDAY(ctx, datetime.date(2025, 1, 8))  # Returns 4 (Wednesday)
```

### QUARTER

**Purpose:** Extract quarter from date.

**Parameters:**
- `run_context`: RunContext object for file operations
- `date`: Date to extract quarter from (date object, string, or file path)

**Returns:** int - Quarter (1-4)

**Use Cases:**
- Group financial data by quarters for reporting
- Calculate quarterly financial metrics
- Analyze seasonal business patterns
- When you need "quarter from date", "extract quarter", or "quarterly analysis"

**Example:**
```python
QUARTER(ctx, datetime.date(2024, 7, 15))  # Returns 3
```

### TIME

**Purpose:** Create time value from hours, minutes, seconds.

**Parameters:**
- `run_context`: RunContext object for file operations
- `hour`: Hour (0-23)
- `minute`: Minute (0-59)
- `second`: Second (0-59)

**Returns:** datetime.time - Time value

**Use Cases:**
- Create specific time values for scheduling
- Build timestamp references for transactions
- Set time-based triggers for processes
- When you need "create time", "build time", or "time from components"

**Example:**
```python
TIME(ctx, 14, 30, 0)  # Returns datetime.time(14, 30)
```

### HOUR

**Purpose:** Extract hour from time.

**Parameters:**
- `run_context`: RunContext object for file operations
- `serial_number`: Time value (time object, datetime object, string, or file path)

**Returns:** int - Hour (0-23)

**Use Cases:**
- Extract hour for time-based financial analysis
- Analyze hourly trading patterns
- Process time-stamped transaction data
- When you need "extract hour", "get hour", or "hour from time"

**Example:**
```python
HOUR(ctx, datetime.time(14, 30, 0))  # Returns 14
```

### MINUTE

**Purpose:** Extract minute from time.

**Parameters:**
- `run_context`: RunContext object for file operations
- `serial_number`: Time value (time object, datetime object, string, or file path)

**Returns:** int - Minute (0-59)

**Use Cases:**
- Extract minute for precise time analysis
- Process high-frequency financial data
- Analyze time-stamped transaction details
- When you need "extract minute", "get minute", or "minute from time"

**Example:**
```python
MINUTE(ctx, datetime.time(14, 30, 45))  # Returns 30
```

### SECOND

**Purpose:** Extract second from time.

**Parameters:**
- `run_context`: RunContext object for file operations
- `serial_number`: Time value (time object, datetime object, string, or file path)

**Returns:** int - Second (0-59)

**Use Cases:**
- Extract second for high-precision time analysis
- Process timestamped financial transactions
- Analyze real-time market data
- When you need "extract second", "get second", or "second from time"

**Example:**
```python
SECOND(ctx, datetime.time(14, 30, 45))  # Returns 45


## 10. Excel Style Array and Dynamic Spill Functions

## UNIQUE

**Purpose:** Extract a list of unique values from a range.

**Parameters:**
- `run_context`: RunContext object for file operations
- `array`: Array to process (list, Series, DataFrame, or file path)
- `by_col`: Process by column (optional, not implemented for basic arrays)
- `exactly_once`: Return only values that appear exactly once (optional)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** list[Any] - Array of unique values

**Use Cases:**
- Remove duplicate entries from financial transaction lists
- Extract unique customer IDs for analysis
- Identify distinct product categories or regions
- Data cleaning and deduplication for financial datasets
- When you need to "remove duplicates", "extract unique values", "deduplicate data", or "find distinct entries"

**Example:**
```python
UNIQUE(ctx, [1, 2, 2, 3, 3, 3])  # Returns [1, 2, 3]
UNIQUE(ctx, [1, 2, 2, 3, 3, 3], exactly_once=True)  # Returns [1]
UNIQUE(ctx, "data.parquet", output_filename="unique_results.parquet")  # Returns [1, 2, 3, 4, 5]
```

### SORT

**Purpose:** Sort data or arrays dynamically.

**Parameters:**
- `run_context`: RunContext object for file operations
- `array`: Array to sort (list, Series, DataFrame, or file path)
- `sort_index`: Sort index (optional, for multi-column sorting)
- `sort_order`: Sort order (1 for ascending, -1 for descending, optional)
- `by_col`: Sort by column (optional, not implemented for basic arrays)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** list[Any] - Sorted array

**Use Cases:**
- Sort financial transactions by amount or date
- Rank investment portfolios by performance
- Order customer data by revenue or priority
- Prepare data for time-series analysis
- When you need to "sort", "order", "rank", or "arrange data"

**Example:**
```python
SORT(ctx, [3, 1, 4, 1, 5])  # Returns [1, 1, 3, 4, 5]
SORT(ctx, [3, 1, 4, 1, 5], sort_order=-1)  # Returns [5, 4, 3, 1, 1]
SORT(ctx, "data.parquet", output_filename="sorted_results.parquet")  # Returns [1, 2, 3, 4, 5]
```

### SORTBY

**Purpose:** Sort an array by values in another array.

**Parameters:**
- `run_context`: RunContext object for file operations
- `array`: Array to sort (list, Series, DataFrame, or file path)
- `by_arrays_and_orders`: Array to sort by (list, Series, DataFrame, or file path)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** list[Any] - Sorted array

**Use Cases:**
- Sort customer names by corresponding revenue amounts
- Order products by their profit margins
- Rank employees by performance scores
- Sort financial data by external criteria
- When you need to "sort by another array", "order by external values", "rank by criteria", or "sort with custom keys"

**Example:**
```python
SORTBY(ctx, ['apple', 'banana', 'cherry'], by_arrays_and_orders=[3, 1, 2])  # Returns ['banana', 'cherry', 'apple']
SORTBY(ctx, [100, 200, 300], by_arrays_and_orders=[3, 1, 2])  # Returns [200, 300, 100]
SORTBY(ctx, "values.parquet", by_arrays_and_orders="sort_keys.parquet", output_filename="sortby_results.parquet")  # Returns ['banana', 'cherry', 'apple']
```

### FILTER

**Purpose:** Return only those records that meet specified conditions.

**Parameters:**
- `run_context`: RunContext object for file operations
- `array`: Array to filter (list, Series, DataFrame, or file path)
- `include`: Boolean array indicating which elements to include
- `if_empty`: Value to return if no elements match (optional)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** list[Any] - Filtered array

**Use Cases:**
- Filter transactions above certain thresholds for risk analysis
- Extract profitable customers from customer lists
- Select high-performing investments for portfolio review
- Data segmentation for targeted financial analysis
- When you need to "filter", "select", "subset", or "conditional selection"

**Example:**
```python
FILTER(ctx, [1, 2, 3, 4, 5], include=[True, False, True, False, True])  # Returns [1, 3, 5]
FILTER(ctx, ['a', 'b', 'c'], include=[False, False, False], if_empty='none')  # Returns 'none'
FILTER(ctx, "data.parquet", include=[True, False, True], output_filename="filtered_results.parquet")  # Returns [1, 3]
```

### SEQUENCE

**Purpose:** Generate a list of sequential numbers in an array format.

**Parameters:**
- `run_context`: RunContext object for file operations
- `rows`: Number of rows
- `columns`: Number of columns (optional, defaults to 1)
- `start`: Starting number (optional, defaults to 1)
- `step`: Step size (optional, defaults to 1)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** list[list[int]] - Array of sequential numbers

**Use Cases:**
- Generate sequential IDs for financial records
- Create index arrays for data processing
- Build time series indices for forecasting
- Generate row numbers for financial reports
- When you need to "generate sequence", "create index", "number series", or "sequential numbering"

**Example:**
```python
SEQUENCE(ctx, 3)  # Returns [[1], [2], [3]]
SEQUENCE(ctx, 2, columns=3, start=5, step=2)  # Returns [[5, 7, 9], [11, 13, 15]]
SEQUENCE(ctx, 3, columns=2, output_filename="sequence_results.parquet")  # Returns [[1, 2], [3, 4], [5, 6]]
```

### RAND

**Purpose:** Generate random numbers between 0 and 1.

**Parameters:**
- `run_context`: RunContext object for file operations

**Returns:** float - Random decimal between 0 and 1

**Use Cases:**
- Generate random samples for financial simulations
- Create random weights for portfolio allocation
- Monte Carlo simulations for risk analysis
- Random sampling for audit procedures
- When you need "random numbers", "stochastic generation", "simulation", or "random sampling"

**Example:**
```python
result = RAND(ctx)  # Returns random float between 0 and 1
0 <= result < 1  # True
```

### RANDBETWEEN

**Purpose:** Generate random integers between two values.

**Parameters:**
- `run_context`: RunContext object for file operations
- `bottom`: Lower bound (inclusive)
- `top`: Upper bound (inclusive)

**Returns:** int - Random integer within range

**Use Cases:**
- Generate random transaction IDs for testing
- Create random samples within specific ranges
- Simulate discrete financial scenarios
- Random selection for audit sampling
- When you need "random integers", "bounded random", "discrete random", or "range sampling"

**Example:**
```python
result = RANDBETWEEN(ctx, 1, 10)  # Returns random integer between 1 and 10
1 <= result <= 10  # True
result = RANDBETWEEN(ctx, -5, 5)  # Returns random integer between -5 and 5
-5 <= result <= 5  # True
```

### FREQUENCY

**Purpose:** Calculate frequency distribution.

**Parameters:**
- `run_context`: RunContext object for file operations
- `data_array`: Data array (list, Series, DataFrame, or file path)
- `bins_array`: Bins array defining the intervals (list, Series, DataFrame, or file path)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** list[int] - Array of frequencies

**Use Cases:**
- Analyze transaction amount distributions
- Create histograms for financial data visualization
- Risk analysis using frequency distributions
- Performance analysis with binning
- When you need "frequency distribution", "histogram", "binning", or "distribution analysis"

**Example:**
```python
FREQUENCY(ctx, [1, 2, 3, 4, 5, 6], bins_array=[2, 4, 6])  # Returns [2, 2, 2, 0]
FREQUENCY(ctx, [1.5, 2.5, 3.5, 4.5], bins_array=[2, 3, 4])  # Returns [1, 1, 1, 1]
FREQUENCY(ctx, "data.parquet", bins_array="bins.parquet", output_filename="frequency_results.parquet")  # Returns [2, 2, 2, 0]
```

### TRANSPOSE

**Purpose:** Transpose array orientation.

**Parameters:**
- `run_context`: RunContext object for file operations
- `array`: Array to transpose (2D list, DataFrame, or file path)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** list[list[Any]] - Transposed array

**Use Cases:**
- Convert row-based financial data to column format
- Prepare data for different analysis tools
- Transform matrix data for mathematical operations
- Reorganize financial reports for presentation
- When you need to "transpose", "flip matrix", "convert orientation", or "matrix transformation"

**Example:**
```python
TRANSPOSE(ctx, [[1, 2, 3], [4, 5, 6]])  # Returns [[1, 4], [2, 5], [3, 6]]
TRANSPOSE(ctx, [[1, 2], [3, 4], [5, 6]])  # Returns [[1, 3, 5], [2, 4, 6]]
TRANSPOSE(ctx, "matrix.parquet", output_filename="transposed_results.parquet")  # Returns [[1, 4], [2, 5], [3, 6]]
```

### MMULT

**Purpose:** Matrix multiplication.

**Parameters:**
- `run_context`: RunContext object for file operations
- `array1`: First matrix (2D list, DataFrame, or file path)
- `array2`: Second matrix (2D list, DataFrame, or file path)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** list[list[float]] - Matrix product

**Use Cases:**
- Portfolio optimization calculations
- Financial modeling with matrix operations
- Risk analysis using covariance matrices
- Linear algebra operations for quantitative finance
- When you need "matrix multiplication", "linear algebra", "portfolio math", or "quantitative operations"

**Example:**
```python
MMULT(ctx, [[1, 2], [3, 4]], array2=[[5, 6], [7, 8]])  # Returns [[19.0, 22.0], [43.0, 50.0]]
MMULT(ctx, [[1, 2, 3]], array2=[[4], [5], [6]])  # Returns [[32.0]]
MMULT(ctx, "matrix1.parquet", array2="matrix2.parquet", output_filename="mmult_results.parquet")  # Returns [[19.0, 22.0], [43.0, 50.0]]
```

### MINVERSE

**Purpose:** Matrix inverse.

**Parameters:**
- `run_context`: RunContext object for file operations
- `array`: Square matrix to invert (2D list, DataFrame, or file path)
- `output_filename`: Optional filename to save results as parquet file

**Returns:** list[list[float]] - Inverse matrix

**Use Cases:**
- Solve systems of linear equations in financial models
- Portfolio optimization and risk analysis
- Regression analysis for financial forecasting
- Matrix operations in quantitative finance
- When you need "matrix inverse", "linear system solving", "regression math", or "quantitative analysis"

**Example:**
```python
MINVERSE(ctx, [[1, 2], [3, 4]])  # Returns [[-2.0, 1.0], [1.5, -0.5]]
MINVERSE(ctx, [[2, 0], [0, 2]])  # Returns [[0.5, 0.0], [0.0, 0.5]]
MINVERSE(ctx, "matrix.parquet", output_filename="inverse_results.parquet")  # Returns [[-2.0, 1.0], [1.5, -0.5]]
```

### MDETERM

**Purpose:** Matrix determinant.

**Parameters:**
- `run_context`: RunContext object for file operations
- `array`: Square matrix to calculate determinant of (2D list, DataFrame, or file path)

**Returns:** float - Determinant value

**Use Cases:**
- Check if matrix is invertible for financial calculations
- Analyze system stability in financial models
- Determine linear independence in quantitative analysis
- Matrix validation for financial computations
- When you need "matrix determinant", "invertibility check", "linear algebra", or "matrix validation"

**Example:**
```python
MDETERM(ctx, [[1, 2], [3, 4]])  # Returns -2.0
MDETERM(ctx, [[2, 0], [0, 2]])  # Returns 4.0
MDETERM(ctx, "matrix.parquet")

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
