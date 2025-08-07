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
