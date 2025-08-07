Basic Math and Aggregation

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
