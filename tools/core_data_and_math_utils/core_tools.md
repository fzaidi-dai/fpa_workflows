# Financial Planning Tools

## Overview
This document defines the tool categories and functions available to our financial planning AI agent. Each tool is designed to help users with specific financial planning tasks and analysis by composing these functions in smart ways e.g. by chaining them.

## Tool Categories

### 1. Basic Arithmetic & Aggregation
These functions are the building blocks for financial summaries and aggregations.

- **SUM**
  - Purpose: Add up a range of numbers
  - Input: Array or range of numeric values
  - Output: Single numeric value (sum)
  - Example: =SUM(A1:A10)
- **AVERAGE**
  - Purpose: Calculate the mean of a dataset
  - Input: Array or range of numeric values
  - Output: Single numeric value极 (mean)
  - Example: =AVERAGE(B1:B10)
- **MIN**
  - Purpose: Identify the smallest number in a dataset
  - Input: Array or range of numeric values
  - Output: Single numeric value (minimum)
  - Example: =MIN(C1:C10)
- **MAX**
  - Purpose: Identify the largest number in a dataset
  - Input: Array or range of numeric values
  - Output: Single numeric value (maximum)
  - Example: =MAX(C1:C10)
- **PRODUCT**
  - Purpose: Multiply values together
  - Input: Array or range of numeric values
  - Output: Single numeric value (product)
  - Example: =PRODUCT(D1:D4)
- **MEDIAN**
  - Purpose: Calculate the middle value of a dataset
  - Input: Series/array of numbers
  - Output: Float
  - Example: =MEDIAN(A1:A10)
- **MODE**
  - Purpose: Find the most frequently occurring value
  - Input: Series/array of numbers
  - Output: Float or list of floats
  - Example: =MODE(A1:A10)
- **PERCENTILE/QUANTILE**
  - Purpose: Calculate specific percentiles (e.g., 25th, 75th percentile)
  - Input: Series/array of numbers, percentile value (0-1)
  - Output: Float
  - Example: =PERCENTILE(A1:A10, 0.75)

- **POWER**
  - Purpose: Raise number to a power
  - Input: number, power
  - Output: Float
  - Example: POWER(1.05, 10)

- **SQRT**
  - Purpose: Calculate square root
  - Input: number
  - Output: Float
  - Example: SQRT(25)

- **EXP**
  - Purpose: Calculate e^x
  - Input: number
  - Output: Float
  - Example: EXP(1)

- **LN**
  - Purpose: Calculate natural logarithm
  - Input: number
  - Output: Float
  - Example: LN(2.718)

- **LOG**
  - Purpose: Calculate logarithm with specified base
  - Input: number, [base]
  - Output: Float
  - Example: LOG(100, 10)

- **ABS**
  - Purpose: Calculate absolute value
  - Input: number
  - Output: Float
  - Example: ABS(-10)

- **SIGN**
  - Purpose: Return sign of number (-1, 0, or 1)
  - Input: number
  - Output: Integer
  - Example: SIGN(-15)

- **MOD**
  - Purpose: Calculate remainder after division
  - Input: number, divisor
  - Output: Float
  - Example: MOD(23, 5)

- **ROUND**
  - Purpose: Round number to specified digits
  - Input: number, num_digits
  - Output: Float
  - Example: ROUND(3.14159, 2)

- **ROUNDUP**
  - Purpose: Round number up
  - Input: number, num_digits
  - Output: Float
  - Example: ROUNDUP(3.14159, 2)

- **ROUNDDOWN**
  - Purpose: Round number down
  - Input: number, num_digits
  - Output: Float
  - Example: ROUNDDOWN(3.14159, 2)

- **WEIGHTED_AVERAGE**
  - Purpose: Calculate weighted average of values
  - Input: values (array), weights (array)
  - Output: Float
  - Example: WEIGHTED_AVERAGE([100, 200, 300], [0.2, 0.3, 0.5])

- **GEOMETRIC_MEAN**
  - Purpose: Calculate geometric mean (useful for growth rates)
  - Input: Series/array of positive numbers
  - Output: Float
  - Example: GEOMETRIC_MEAN([1.05, 1.08, 1.12, 1.03])

- **HARMONIC_MEAN**
  - Purpose: Calculate harmonic mean (useful for rates/ratios)
  - Input: Series/array of positive numbers
  - Output: Float
  - Example: HARMONIC_MEAN([2, 4, 8])

- **CUMSUM**
  - Purpose: Calculate cumulative sum
  - Input: Series/array of numbers
  - Output: Array of cumulative sums
  - Example: CUMSUM([10, 20, 30, 40])

- **CUMPROD**
  - Purpose: Calculate cumulative product
  - Input: Series/array of numbers
  - Output: Array of cumulative products
  - Example: CUMPROD([1.极05, 1.08, 1.12])

- **VARIANCE_WEIGHTED**
  - Purpose: Calculate weighted variance
  - Input: values (array), weights (array)
  - Output: Float
  - Example: VARIANCE_WEIGHTED([100, 200, 300], [0.2, 0.3, 0.5])


### 2. Conditional Aggregation & Counting
These functions allow you to work with data subsets based on specific criteria.

- **SUMIF**
  - Purpose: Sum numbers that meet one condition
  - Input: Range to evaluate, criteria, sum range (optional)
  - Output: Single numeric value
  - Example: =SUMIF(A1:A10, ">100", B1:B10)
- **SUMIFS**
  - Purpose: Sum numbers that meet multiple conditions
  - Input: Sum range, criteria ranges, criteria values
  - Output: Single numeric value
  - Example: =SUMIFS(C1:C10, A1:A10, ">100", B1:B10, "Sales")
- **COUNTIF**
  - Purpose: Count cells that meet one condition
  - Input: Range to evaluate, criteria
  - Output: Integer count
  - Example: =COUNTIF(A1:A10, ">100")
- **COUNTIFS**
  - Purpose: Count cells that meet multiple conditions
  - Input: Criteria ranges and criteria values (pairs)
  - Output: Integer count
  - Example: =COUNTIFS(A1:A10, ">100", B1:B10, "Sales")
- **AVERAGEIF**
  - Purpose: Calculate average of cells that meet one condition
  - Input: Range to evaluate, criteria, average range (optional)
  - Output: Single numeric value
  - Example: =AVERAGEIF(A1:A10, ">100", B1:B10)
- **AVERAGEIFS**
  - Purpose: Calculate average of cells that meet multiple conditions
  - Input: Average range, criteria ranges, criteria values
  - Output: Single numeric value
  - Example: =AVERAGEIFS(C1:C10, A1:A10, ">100", B1:B10, "Sales")
- **MAXIFS**
  - Purpose: Find maximum value based on multiple criteria
  - Input: Max range, criteria ranges, criteria values
  - Output: Single numeric value
  - Example: =MAXIFS(C1:C10, A1:A10, ">100", B1:B10, "Sales")
- **MINIFS**
  - Purpose: Find minimum value based on multiple criteria
  - Input: Min range, criteria ranges, criteria values
  - Output: Single numeric value
  - Example: =MINIFS(C1:C10, A1:A10, ">100", B1:B10, "Sales")

- **SUMPRODUCT**
  - Purpose: Sum the products of corresponding ranges
  - Input: range1, range2, [range3, ...]
  - Output: Single numeric value
  - Example: SUMPRODUCT(A1:A10, B1:B10)

- **AGGREGATE**
  - Purpose: Perform various aggregations with error handling and filtering
  - Input: function_num (int), options (int), array, [k]
  - Output: Single numeric value
  - Example: AGGREGATE(1, 5, A1:A10) # Sum ignoring errors

- **SUBTOTAL**
  - Purpose: Calculate subtotals with filtering capability
  - Input: function_num (int), ref1, [ref2, ...]
  - Output: Single numeric value
  - Example: SUBTOTAL(109, A1:A10) # Sum of visible cells

- **COUNTBLANK**
  - Purpose: Count blank/empty cells in a range
  - Input: Range to evaluate
  - Output: Integer count
  - Example: COUNTBLANK(A1:A10)

- **COUNTA**
  - Purpose: Count non-empty cells in a range
  - Input: Range to evaluate
  - Output: Integer count
  - Example: COUNTA(A1:A10)

### 3. Lookup & Reference Functions
These are invaluable when you need to retrieve data from a table or array dynamically.

- **VLOOKUP**
  - Purpose: Search for a value in a vertical range
  - Input: lookup_value, table_array, col_index, range_lookup (optional)
  - Output: Single value from specified column
  - Example: =VLOOKUP(lookup_value, table_array, col_index, [range_lookup])
- **HLOOKUP**
  - Purpose: Search for a value in a horizontal range
  - Input: lookup_value, table_array, row_index, range_lookup (optional)
  - Output: Single value from specified row
  - Example: =HLOOKUP(lookup_value, table_array, row_index, [range_lookup])
- **INDEX**
  - Purpose: Return a value at a given position in an array
  - Input: array, row_num, column_num (optional)
  - Output: Single value at specified position
  - Example: =INDEX(return_range, row_num, [column_num])
- **MATCH**
  - Purpose: Find the relative position of an item in an array
  - Input: lookup_value, lookup_array, match_type
  - Output: Integer position
  - Example: =MATCH(lookup_value, lookup_range, 0)
- **XLOOKUP**
  - Purpose: Modern, flexible lookup function replacing VLOOKUP/HLOOKUP
  - Input: lookup_value, lookup_array, return_array, if_not_found (optional)
  - Output: Value from return array or if_not_found value
  - Example: =XLOOKUP(lookup_value, lookup_array, return_array, [if_not_found])
- **OFFSET**
  - Purpose: Create dynamic ranges based on reference point
  - Input: reference, rows, cols, height (optional), width (optional)
  - Output: Range reference
  - Example: =OFFSET(reference, rows, cols, [height], [width])
- **INDIRECT**
  - Purpose: Create references based on text strings
  - Input: ref_text, a1_style (optional)
  - Output: Range reference
  - Example: =INDIRECT(ref_text)
- **CHOOSE**
  - Purpose: Return a value from a list based on index number
  - Input: index_num, value1, value2, ...
  - Output: Selected value
  - Example: =CHOOSE(index_num, value1, value2, …)

- **LOOKUP**
  - Purpose: Simple lookup function (vector or array form)
  - Input: lookup_value, lookup_vector, result_vector
  - Output: Single value
  - Example: LOOKUP(lookup_value, lookup_vector, result_vector)

- **ADDRESS**
  - Purpose: Create cell address as text
  - Input: row_num, column_num, [abs_num], [a1], [极 sheet_text]
  - Output: Text string (cell address)
  - Example: ADDRESS(1, 1, 1, TRUE, "Sheet1")

- **ROW**
  - Purpose: Return row number of reference
  - Input: [reference]
  - Output: Integer or array of integers
  - Example: ROW(A5)

- **COLUMN**
  - Purpose: Return column number of reference
  - Input: [reference]
  - Output: Integer or array of integers
  - Example: COLUMN(B1)

- **ROWS**
  - Purpose: Return number of rows in reference
  - Input: array
  - Output: Integer
  - Example: ROWS(A1:A10)

- **COLUMNS**
  - Purpose: Return number of columns in reference
  - Input: array
  - Output: Integer
  - Example: COLUMNS(A1:E1)

### 4. Logical & Error-Handling Functions
These functions help structure decision-making processes and manage errors gracefully.

- **IF**
  - Purpose: Return different values depending on whether a condition is met
  - Input: logical_test, value_if_true, value_if_false
  - Output: Value based on condition result
  - Example: =IF(A1 > 100, "Above Budget", "Within Budget")
- **IFERROR**
  - Purpose: Return a specified value if a formula results in an error
  - Input: value, value_if_error
  - Output: Original value or error replacement
  - Example: =IFERROR(formula, alternative_value)
- **IFNA**
  - Purpose: Return a specified value if a formula results in #N/A error
  - Input: value, value_if_na
  - Output: Original value or #N/A replacement
  - Example: =IFNA(formula, alternative_value)
- **IFS**
  - Purpose: Test multiple conditions without nesting several IF statements
  - Input: logical_test1, value_if_true1, logical_test2, value_if_true2, ...
  - Output: Value from first true condition
  - Example: =IFS(A1>100, "High", A1>50, "Medium", TRUE, "Low")
- **AND**
  - Purpose: Test if all conditions are true
  - Input: logical1, logical2, ...
  - Output: TRUE if all conditions are true, FALSE otherwise
  - Example: =AND(condition1, condition2)
- **OR**
  - Purpose: Test if any condition is true
  - Input: logical1, logical2, ...
  - Output: TRUE if any condition is true, FALSE otherwise
  - Example: =OR(condition1, condition2)
- **NOT**
  - Purpose: Reverse the logical value of a condition
  - Input: logical
  - Output: Opposite boolean value
  - Example: =NOT(condition)

- **SWITCH**
  - Purpose: Compare expression against list of values and return corresponding result
  - Input: expression, value1, result1, [value2, result2], ..., [default]
  - Output: Matched result or default
  - Example: SWITCH(A1, 1, "One", 2, "Two", "Other")

- **XOR**
  - Purpose: Exclusive OR - returns TRUE if odd number of arguments are TRUE
  - Input: logical1, logical2, ...
  - Output: Boolean
  - Example: XOR(TRUE, FALSE, TRUE)

- **ISBLANK**
  - Purpose: Test if cell is blank
  - Input: value
  - Output: Boolean
  - Example: ISBLANK(A1)

- **ISNUMBER**
  - Purpose: Test if value is a number
  - Input: value
  - Output: Boolean
  - Example: ISNUMBER(A1)

- **ISTEXT**
  - Purpose: Test if value is text
  - Input: value
  - Output: Boolean
  - Example: ISTEXT(A1)

- **ISERROR**
  - Purpose: Test if value is an error
  - Input: value
  - Output: Boolean
  - Example: ISERROR(A1/B1)

### 5. Financial Functions
These functions are specifically tailored to perform time value of money calculations and asset valuations.

- **NPV**
  - Purpose: Calculate the present value of a series of cash flows at a constant discount rate
  - Input: discount_rate, cash_flow1, cash_flow2, ...
  - Output: Single numeric value (net present value)
  - Example: =NPV(discount_rate, cash_flow1, cash_flow2, …)
- **IRR**
  - Purpose: Determine the discount rate that makes the net present value of cash flows zero
  - Input: cash_flow_range, guess (optional)
  - Output: Single numeric value (internal rate of return)
  - Example: =IRR(cash_flow_range)
- **XNPV**
  - Purpose: Calculate NPV for cash flows that occur at irregular intervals
  - Input: discount_rate, cash_flow_range, date_range
  - Output: Single numeric value (net present value)
  - Example: =XNPV(discount_rate, cash_flow_range, date_range)
- **XIR极R**
  - Purpose: Calculate IRR for cash flows that occur at irregular intervals
  - Input: cash_flow_range, date_range, guess (optional)
  - Output: Single numeric value (internal rate of return)
  - Example: =XIRR(cash_flow_range, date_range)
- **PMT**
  - Purpose: Calculate the payment for a loan based on constant payments and interest rate
  - Input: interest_rate, number_of_periods, present_value, future_value (optional), type (optional)
  - Output: Single numeric value (payment amount)
  - Example: =PMT(interest_rate, number_of_periods, present_value)
- **IPMT**
  - Purpose: Determine the interest portion for a specific period of a loan payment
  - Input: interest_rate, period, number_of_period极s, present_value, future_value (optional), type (optional)
  - Output: Single numeric value (interest payment)
  - Example: =IPMT(interest_rate, period, number_of_periods, present_value)
- **PPMT**
  - Purpose: Determine the principal portion for a specific period of a loan payment
  - Input: interest_rate, period, number_of_periods, present_value, future_value (optional), type (optional)
  - Output: Single numeric value (principal payment)
  - Example: =PPMT(interest_rate, period, number_of_periods, present_value)
- **PV**
  - Purpose: Compute the present value of an investment given a constant interest rate
  - Input: interest_rate, number_of_periods, payment, future_value (optional), type (optional)
  - Output: Single numeric value (present value)
  - Example: =PV(interest_rate, number_of_periods, payment, [future_value])
- **FV**
  - Purpose: Compute the future value of an investment given a constant interest rate
  - Input: interest_rate, number_of_periods, payment, present_value (optional), type (optional)
  - Output: Single numeric value (future value)
  - Example: =FV(interest_rate, number_of_periods, payment, [present_value])
- **NPER**
  - Purpose: Determine the duration of an investment in periods
  - Input: interest_rate, payment, present_value, future_value (optional), type (optional)
  - Output: Single numeric value (number of periods)
  - Example: =NPER(interest_rate, payment, present_value)
- **RATE**
  - Purpose: Determine the interest rate per period for an annuity
  - Input: number_of_periods, payment, present_value, future_value (optional), type (optional), guess (optional)
  - Output: Single numeric value (interest rate)
  - Example: =RATE(number_of_periods, payment, present_value)
- **CUMIPMT**
  - Purpose: Calculate cumulative interest payments over a range of periods
  - Input: interest_rate, number_of_periods, present_value, start_period, end_period, type
  - Output: Single numeric value (cumulative interest)
  - Example: =CUMIPMT(interest_rate, number_of_periods, present_value, start_period, end_period, type)
- **CUMPRINC**
  - Purpose: Calculate cumulative principal payments over a range of periods
  - Input: interest_rate, number_of_periods, present_value, start_period, end_period, type
  - Output: Single numeric value (cumulative principal)
  - Example: =CUMPRINC(interest_rate, number_of_periods, present_value, start_period, end_period, type)
- **SLN**
  - Purpose: Compute straight-line depreciation
  - Input: cost, salvage_value, useful_life
  - Output: Single numeric value (annual depreciation)
  - Example: =SLN(cost, salvage, life)
- **SYD**
  - Purpose: Compute sum-of-years' digits depreciation
  - Input: cost, salvage_value, useful_life, period
  - Output: Single numeric value (period depreciation)
  - Example: =SYD(cost, salvage, life, period)
- **DDB**
  - Purpose: Compute double-declining balance depreciation
  - Input: cost, salvage_value, useful_life, period, factor (optional)
  - Output: Single numeric value (period depreciation)
  - Example: =DDB(cost, salvage, life, period, [factor])
- **DB**
  - Purpose: Compute fixed-declining balance depreciation
  - Input: cost, salvage_value, useful_life, period, month (optional)
  - Output: Single numeric value (period depreciation)
  - Example: =DB(cost, salvage, life, period, [month])
- **PRICE**
  - Purpose: Calculate bond price
  - Input: settlement, maturity, rate, yield, redemption, frequency, basis (optional)
  - Output: Single numeric value (bond price)
  - Example: =PRICE(settlement, maturity, rate, yield, redemption, frequency, [basis])
- **YIELD**
  - Purpose: Calculate bond yield
  - Input: settlement, maturity, rate, price, redemption, frequency, basis (optional)
  - Output: Single numeric value (bond yield)
  - Example: =YIELD(settlement, maturity, rate, price, redemption, frequency, [basis])
- **DURATION**
  - Purpose: Calculate bond duration
  - Input: settlement, maturity, coupon, yield, frequency, basis (optional)
  - Output: Single numeric value (bond duration)
  - Example: =DURATION(settlement, maturity, coupon, yield, frequency, [basis])

- **ACCRINT**
  - Purpose: Calculate accrued interest for periodic interest securities
  - Input: issue, first_interest, settlement, rate, par, frequency, [basis], [calc_method]
  - Output: Float (accrued interest)
  - Example: ACCRINT("2024-01-01", "2024-07-01", "2024-03-01", 0.08, 1000, 2)

- **ACCRINTM**
  - Purpose: Calculate accrued interest for maturity securities
  - Input: issue, settlement, rate, par, [basis]
  - Output: Float (accrued interest)
  - Example: ACCRINTM("2024-01-01", "2024-12-31", 0.05, 1000)

- **EFFECT**
  - Purpose: Calculate effective annual interest rate
  - Input: nominal_rate, npery
  - Output: Float (effective rate)
  - Example: EFFECT(0.12, 12)

- **NOMINAL**
  - Purpose: Calculate nominal annual interest rate
  - Input: effect_rate, npery
  - Output: Float (nominal rate)
  - Example: NOMINAL(0.1268, 12)

- **MIRR**
  - Purpose: Modified internal rate of return
  - Input: values, finance_rate, reinvest_rate
  - Output: Float (modified IRR)
  - Example: MIRR([-1000, 200, 400, 500], 0.10, 0.12)

### 6. Date & Time Functions
Essential for forecasting, scheduling cash flows, and working with time series data.

- **TODAY**
  - Purpose: Return the current date
  - Input: No parameters
  - Output: Current date
  - Example: =TODAY()
- **NOW**
  - Purpose: Return the current date and time
  - Input: No parameters
  - Output: Current date and time
  - Example: =NOW()
- **DATE**
  - Purpose: Construct a date from year, month, and day components
  - Input: year, month, day
  - Output: Date value
  - Example: =DATE(2025, 4, 15)
- **YEAR**
  - Purpose: Extract the year from a date
  - Input: date
  - Output: Integer (year)
  - Example: =YEAR(A1)
- **MONTH**
  - Purpose: Extract the month from a date
  - Input: date
  - Output: Integer (month 1-12)
  - Example: =MONTH(A1)
- **DAY**
  - Purpose: Extract the day from a date
  - Input: date
  - Output: Integer (day 1-31)
  - Example: =DAY(A1)
- **EDATE**
  - Purpose: Calculate a date a given number of months before or after a specified date
  - Input: start_date, months
  - Output: Date value
  - Example: =EDATE(start_date, months)
- **EOMONTH**
  - Purpose: Find the end of the month for a given date
  - Input: start_date, months
  - Output: Date value (end of month)
  - Example: =EOMONTH(start_date, months)
- **DATEDIF**
  - Purpose: Calculate the difference between two dates
  - Input: start_date, end_date, unit
  - Output: Integer (difference in specified unit)
  - Example: =DATEDIF(start_date, end_date, "unit")
- **YEARFRAC**
  - Purpose: Calculate the fraction of a year between two dates极
  - Input: start_date, end_date, basis (optional)
  - Output: Decimal fraction of year
  - Example: =YEARFRAC(start_date, end_date)
- **WORKDAY**
  - Purpose: Return a future or past date excluding weekends and holidays
  - Input: start_date, days, holidays (optional)
  - Output: Date value
  - Example: =WORKDAY(start_date, days, [holidays])
- **NETWORKDAYS**
  - Purpose: Count working days between two dates
  - Input: start_date, end_date, holidays (optional)
  - Output: Integer (number of working days)
  - Example: =NETWORKDAYS(start_date, end_date, [holidays])
- **DATE_RANGE**
  - Purpose: Generate a series of dates between a start and end date with a specified frequency, essential for creating financial model timelines
  - Input: start_date (str or date), end_date (str or date), frequency (str, e.g., 'M' for month-end, 'D' for day, 'Q' for quarter-end)
  - Output: Series of dates
  - Example: =DATE_RANGE("2025-01-01", "2025-12-31", "M")

- **WEEKDAY**
  - Purpose: Return day of week as number
  - Input: serial_number, [return_type]
  - Output: Integer (1-7)
  - Example: WEEKDAY(DATE(2024,1,1))

- **QUARTER**
  - Purpose: Extract quarter from date
  - Input: date
  - Output: Integer (1-4)
  - Example: QUARTER(DATE(2024,7,15))

- **TIME**
  - Purpose: Create time value from hours, minutes, seconds
  - Input: hour, minute, second
  - Output: Time value
  - Example: TIME(14, 30, 0)

- **HOUR**
  - Purpose: Extract hour from time
  - Input: serial_number
  - Output: Integer (0-23)
  - Example: HOUR(TIME(14,30,0))

- **MINUTE**
  - Purpose: Extract minute from time
  - Input: serial_number
  - Output: Integer (0-59)
  - Example: MINUTE(TIME(14,30,45))

- **SECOND**
  - Purpose: Extract second from time
  - Input: serial_number
  - Output: Integer (0-59)
  - Example: SECOND(TIME(14,30,45))

### 7. Text & Data Management Functions
Useful for generating labels, combining text, and cleaning up data reports.

- **CONCAT**
  - Purpose: Merge text strings together (modern version)
  - Input: text1, text2, ...
  - Output: Combined text string
  - Example: =CONCAT(text1, text2, …)
- **CONCATENATE**
  - Purpose: Merge text strings together (legacy version)
  - Input: text1, text2, ...
  - Output: Combined text string
  - Example: =CONCATENATE(text1, text2, …)
- **TEXT**
  - Purpose: Format numbers or dates as text with a specified format
  - Input: value, format_text
  - Output: Formatted text string
  - Example: =TEXT(A1, "0.00%")
- **LEFT**
  - Purpose: Extract characters from the left side of a text string
  - Input: text, num_chars
  - Output: Text substring
  - Example: =LEFT(text, num_chars)
- **RIGHT**
  - Purpose: Extract characters from the right side of a text string
  - Input: text, num_chars
  - Output: Text substring
  - Example: =RIGHT(text, num_chars)
- **MID**
  - Purpose: Extract characters from the middle of a text string
  - Input: text, start_num, num_chars
  - Output: Text substring
  - Example: =MID(text, start_num, num_chars)
- **LEN**
  - Purpose: Count the number of characters in a text string
  - Input: text
  - Output: Integer (character count)
  - Example: =极LEN(text)
- **FIND**
  - Purpose: Locate one text string within another (case-sensitive)
  - Input: find_text, within_text, start_num (optional)
 极 Output: Integer (position)
  - Example: =FIND(find_text, within_text)
- **SEARCH**
  - Purpose: Locate one text string within another (not case-sensitive)
  - Input: find_text, within_text, start_num (optional)
  - Output: Integer (position)
  - Example: =SEARCH(find_text, within_text)
- **REPLACE**
  - Purpose: Replace a portion of a text string with another text string
  - Input: old_text, start_num, num_chars, new_text
  - Output: Modified text string
  - Example: =REPLACE(old_text, start极_num, num_chars, new_text)
- **SUBSTITUTE**
  - Purpose: Replace occurrences of old text with new text
  - Input: text, old_text, new_text, instance_num (optional)
  - Output: Modified text string
  - Example: =SUBSTITUTE(text, old_text, new_text)

- **TRIM**
  - Purpose: Remove extra spaces from text
  - Input: text
  - Output: Cleaned text string
  - Example: TRIM("  Extra   Spaces  ")

- **CLEAN**
  - Purpose: Remove non-printable characters
  - Input: text
  - Output: Cleaned text string
  - Example: CLEAN(text_with_nonprints)

- **UPPER**
  - Purpose: Convert text to uppercase
  - Input: text
  - Output: Uppercase text string
  - Example: UPPER("hello world")

- **LOWER**
  - Purpose: Convert text to lowercase
  - Input: text
  - Output: Lowercase text string
  - Example: LOWER("HELLO WORLD")

- **PROPER**
  - Purpose: Convert text to proper case
  - Input: text
  - Output: Proper case text string
  - Example: PROPER("hello world")

- **VALUE**
  - Purpose: Convert text to number
  - Input: text
  - Output: Numeric value
  - Example: VALUE("123.45")

- **TEXTJOIN**
  - Purpose: Join text strings with delimiter
  - Input: delimiter, ignore_empty, text1, [text2], ...
  - Output: Combined text string
  - Example: TEXTJOIN(", ", TRUE, A1:A5)

### 8. Statistical & Trend Analysis Functions
These functions support forecasting and risk analysis by uncovering trends and relationships in data.

- **STDEV.P**
  - Purpose: Calculate the standard deviation for a full population
  - Input: Array or range of numeric values
  - Output: Single numeric value (population standard deviation)
  - Example: =STDEV.P(data_range)
- **STDEV.S**
  - Purpose: Calculate the standard deviation for a sample
  - Input: Array or range of numeric values
  - Output: Single numeric value (sample standard deviation)
  - Example: =STDEV.S(data_range)
- **VAR.P**
  - Purpose: Calculate variance for a population
  - Input: Array or range of numeric values
  - Output: Single numeric value (population variance)
  - Example: =VAR.P(data_range)
- **VAR.S**
  - Purpose: Calculate variance for a sample
  - Input: Array or range of numeric values
  - Output: Single numeric value (sample variance)
  - Example: =VAR.S(data_range)
- **MEDIAN**
  - Purpose: Determine the middle value in a dataset
  - Input: Array or range of numeric values
  - Output: Single numeric value (median)
  - Example: =MEDIAN(data_range)
- **MODE**
  - Purpose: Find the most frequently occurring value in a dataset
  - Input: Array or range of numeric values
  - Output: Single numeric value (mode)
  - Example: =MODE(data_range)
- **CORREL**
  - Purpose: Measure the correlation between two datasets
  - Input: range1, range2
  - Output: Single numeric value (-1 to 1)
  - Example: =CORREL(range1, range2)
- **COVARIANCE.P**
  - Purpose: Calculate covariance for a population
  - Input: range1, range2
  - Output: Single numeric value (population covariance)
  - Example: =COVARIANCE.P(range1, range2)
- **COVARIANCE.S**
  - Purpose: Calculate covariance for a sample
  - Input: range1, range2
  - Output: Single numeric value (sample covariance)
  - Example: =COVARIANCE.S(range1, range2)
- **TREND**
  - Purpose: Predict future values based on linear trends
  - Input: known_y's, known_x's (optional), new_x's (optional), const (optional)
  - Output: Array of predicted values
  - Example: =TREND(known_y's, [known_x's], [new_x's])
- **FORECAST**
  - Purpose: Predict a future value based on linear regression
  - Input: new_x, known_y's, known_x's
  - Output: Single predicted value
  - Example: =FORECAST(new_x, known_y's, known_x's)
- **FORECAST.LINEAR**
  - Purpose: Predict a future value based on linear regression (newer version)
  - Input: new_x, known_y's, known_x's
  - Output: Single predicted value
  - Example: =FORECAST.LINEAR(new_x, known_y's, known_x's)
- **GROWTH**
  - Purpose: Forecast exponential growth trends
  - Input: known_y's, known_x's (optional), new极_x's (optional), const (optional)
  - Output: Array of predicted values
  - Example: =GROWTH(known_y's, [known_x's], [new_x's])

- **SLOPE**
  - Purpose: Calculate slope of linear regression line
  - Input: known_ys, known_xs
  - Output: Float (slope)
  - Example: SLOPE(B1:B10, A1:A10)

- **INTERCEPT**
  - Purpose: Calculate y-intercept of linear regression line
  - Input: known_ys, known_xs
  - Output: Float (intercept)
  - Example: INTERCEPT(B1:B10, A1:A10)

- **RSQ**
  - Purpose: Calculate R-squared of linear regression
  - Input: known_ys, known_xs
  - Output: Float (R-squared)
  - Example: RSQ(B1:B10, A1:A10)

- **LINEST**
  - Purpose: Calculate linear regression statistics
  - Input: known_ys, [known_xs], [const], [stats]
  - Output: Array of regression statistics
  - Example: LINEST(B1:B10, A1:A10, TRUE, TRUE)

- **LOGEST**
  - Purpose: Calculate exponential regression statistics
  - Input: known_ys, [known_xs], [const], [stats]
  - Output: Array of regression statistics
  - Example: LOGEST(B1:B10, A1:A10, TRUE, TRUE)

- **RANK**
  - Purpose: Calculate rank of number in array
  - Input: number, ref, [order]
  - Output: Integer (rank)
  - Example: RANK(85, A1:A10, 0)

- **PERCENTRANK**
  - Purpose: Calculate percentile rank
  - Input: array, x, [significance]
  - Output: Float (percentile rank)
  - Example: PERCENTRANK(A1:A10, 85)

### 9. Array and Dynamic Spill Functions (Modern Excel)
These functions help in performing calculations across ranges and enabling dynamic results.

- **UNIQUE**
  - Purpose: Extract a list of unique values from a range
  - Input: array, by_col (optional), exactly_once (optional)
  - Output: Array of unique values
  - Example: =UNIQUE(range)
- **SORT**
  - Purpose: Sort data or arrays dynamically
  - Input: array, sort_index (optional), sort_order (optional), by_col (optional)
  - Output: Sorted array
  - Example: =SORT(range)
- **SORTBY**
  - Purpose: Sort an array by values in another array
  - Input: array, by_array1, sort_order1 (optional), by_array2 (optional), sort_order2 (optional), ...
  - Output: Sorted array
  - Example: =SORTBY(array, by_array)
- **FILTER**
  - Purpose: Return only those records that meet specified conditions
  - Input: array, include, if_empty (optional)
  - Output: Filtered array
  - Example: =FILTER(range, condition)
- **SEQUENCE**
  - Purpose: Generate a list of sequential numbers in an array format
  - Input: rows, columns (optional), start (optional), step (optional)
  - Output: Array of sequential numbers
  - Example: =SEQUENCE(rows, [columns], [start], [step])
- **RAND**
  - Purpose: Generate random numbers between 0 and 1
  - Input: No parameters
  - Output: Random decimal between 0 and 1
  - Example: =RAND()
- **RANDBETWEEN**
  - Purpose: Generate random integers between two values
  - Input: bottom, top
  - Output: Random integer within range
  - Example: =RANDBETWEEN(lower, upper)


- **FREQUENCY**
  - Purpose: Calculate frequency distribution
  - Input: data_array, bins_array
  - Output: Array of frequencies
  - Example: FREQUENCY(A1:A100, C1:C10)

- **TRANSPOSE**
  - Purpose: Transpose array orientation
  - Input: array
  - Output: Transposed array
  - Example: TRANSPOSE(A1:E5)

- **MMULT**
  - Purpose: Matrix multiplication
  - Input: array1, array2
  - Output: Matrix product
  - Example: MMULT(A1:B3, D1:E2)

- **MINVERSE**
  - Purpose: Matrix inverse
  - Input: array
  - Output: Inverse matrix
  - Example: MINVERSE(A1:B2)

- **MDETERM**
  - Purpose: Matrix determinant
  - Input: array
  - Output: Float (determinant)
  - Example: MDETERM(A1:B2)

### 10. Additional Useful Functions
These functions further aid in analysis, documentation, or advanced computations.

- **FORMULATEXT**
  - Purpose: Returns the formula in a referenced cell as text, which can help in auditing or documentation
  - Input: reference
  - Output: Text string (formula)
  - Example: =FORMULATEXT(A1)
- **TRANSPOSE**
  - Purpose: Converts rows to columns or vice versa, useful for rearranging data
  - Input: array
  - Output: Transposed array
  - Example: =TRANSPOSE(A1:B10)


- **CELL**
  - Purpose: Return information about cell formatting, location, or contents
  - Input: info_type, [reference]
  - Output: Various types depending on info_type
  - Example: CELL("address", A1)

- **INFO**
  - Purpose: Return information about operating environment
  - Input: type_text
  - Output: Text string with system info
  - Example: INFO("version")

- **N**
  - Purpose: Convert value to number
  - Input: value
  - Output: Numeric value or 0
  - Example: N(TRUE)

- **T**
  - Purpose: Convert value to text
  - Input: value
  - Output: Text string or empty string
  - Example: T(123)

### 11. Data Transformation & Pivoting
- **PIVOT_TABLE**
  - Purpose: Create pivot tables with aggregations by groups
  - Input: DataFrame, index columns, value columns, aggregation functions
  - Output: DataFrame
  - Example: PIVOT_TABLE(sales_df, ['region'], ['revenue'], 'sum')
- **UNPIVOT/MELT**
  - Purpose: Transform wide data to long format
  - Input: DataFrame, identifier columns, value columns
  - Output: DataFrame
  - Example: UNPIVOT(df, ['customer_id'], ['Q1', 'Q2', 'Q3', 'Q4'])
- **GROUP_BY**
  - Purpose: Group data and apply aggregation functions
  - Input: DataFrame, grouping columns, aggregation functions
  - Output: DataFrame
  - Example: GROUP_BY(sales_df, ['category'], 'sum')
- **CROSS_TAB**
  - Purpose: Create cross-tabulation tables
  - Input: DataFrame, row variables, column variables, values
  - Output: DataFrame
  - Example: CROSS_TAB(df, ['region'], ['product'], ['sales'])
- **GROUP_BY_AGG**
  - Purpose: Group a DataFrame by one or more columns and then apply one or more aggregation functions (like sum, mean, count) to specified columns. This is more versatile than a simple GROUP_BY
  - Input: df (DataFrame), group_by_cols (list of str), agg_dict (dict, e.g., {'revenue': 'sum', 'users': 'count'})
  - Output: DataFrame
  - Example: GROUP_BY_AGG(df, ['region'], {'revenue': 'sum', 'customers': 'count'})

- **STACK**
  - Purpose: Stack multiple columns into single column
  - Input: DataFrame, columns_to_stack
  - Output: DataFrame
  - Example: STACK(df, ['Q1', 'Q2', 'Q3', 'Q4'])

- **UNSTACK**
  - Purpose: Unstack index level to columns
  - Input: DataFrame, level_to_unstack
  - Output: DataFrame
  - Example: UNSTACK(stacked_df, 'quarter')

- **MERGE**
  - Purpose: Merge/join two DataFrames
  - Input: left_df, right_df, join_keys, join_type
  - Output极: DataFrame
  - Example: MERGE(sales_df, customer_df, 'customer_id', 'left')

- **CONCAT**
  - Purpose: Concatenate DataFrames
  - Input: list_of_dataframes, axis
  - Output: DataFrame
  - Example: CONCAT([df1, df2, df3], axis=0)

- **FILL_FORWARD**
  - Purpose: Forward fill missing values
  - Input: DataFrame or Series
  - Output: DataFrame or Series with filled values
  - Example: FILL_FORWARD(revenue_series)

- **INTERPOLATE**
  - Purpose: Interpolate missing values
  - Input: DataFrame or Series, method
  - Output: DataFrame or Series with interpolated values
  - Example: INTERPOLATE(data_series, 'linear')

### 12. Forecasting & Projection
- **LINEAR_FORECAST**
  - Purpose: Simple linear trend forecasting
  - Input: Historical data series, forecast periods
  - Output: Series with forecasted values
  - Example: LINEAR_FORECAST(historical_sales, forecast_periods=12)
- **MOVING_AVERAGE**
  - Purpose: Calculate moving averages for smoothing and forecasting
  - Input: Data series, window size
  - Output: Series with moving averages
  - Example: MOVING_AVERAGE(monthly_revenue_series, window_size=3)
- **EXPONENTIAL_SMOOTHING**
  - Purpose: Exponentially weighted forecasting
  - Input: Data series, smoothing parameter
  - Output: Series with smoothed/forecasted values
  - Example: EXPONENTIAL_SMOOTHING(sales_data, smoothing_alpha=0.3)
- **SEASONAL_DECOMPOSE**
  - Purpose: Decompose time series into trend, seasonal, residual components
  - Input: Time series data with date index
  - Output: DataFrame with decomposed components
  - Example: SEASONAL_DECOMPOSE(quarterly_sales_ts)

- **SEASONAL_ADJUST**
  - Purpose: Remove seasonal patterns from time series
  - Input: time_series, seasonal_periods
  - Output: Series with seasonal adjustment
  - Example: SEASONAL_ADJUST(monthly_sales, 12)

- **TREND_COEFFICIENT**
  - Purpose: Calculate trend coefficient (slope per period)
  - Input: time_series_data
  - Output: Float (trend coefficient)
  - Example: TREND_COEFFICIENT(quarterly_revenue)

- **CYCLICAL_PATTERN**
  - Purpose: Identify cyclical patterns in data
  - Input: time_series, cycle_length
  - Output: Series with cyclical indicators
  - Example: CYCLICAL_PATTERN(economic_data, 60)

- **AUTO_CORRELATION**
  - Purpose: Calculate autocorrelation of time series
  - Input: time_series, lags
  - Output: Array of correlation coefficients
  - Example: AUTO_CORRELATION(monthly_data, 12)

- **HOLT_WINTERS**
  - Purpose: Holt-Winters exponential smoothing
  - Input: time_series, seasonal_periods, trend_type, seasonal_type
  - Output: Dict with forecast and components
  - Example: HOLT_WINTERS(quarterly_sales, 4, 'add', 'add')

### 13. Data Validation & Quality
These functions ensure data integrity and quality for financial analysis.

- **CHECK_DUPLICATES**
  - Purpose: Identify duplicate records in dataset
  - Input: DataFrame, columns_to_check
  - Output: DataFrame with duplicate flags
  - Example: CHECK_DUPLICATES(transactions_df, ['transaction_id'])

- **VALIDATE_DATES**
  - Purpose: Validate date formats and ranges
  - Input: date_series, min_date, max_date
  - Output: Series with validation flags
  - Example: VALIDATE_DATES(date_column, '2020-01-01', '2025-12-31')

- **CHECK_NUMERIC_RANGE**
  - Purpose: Validate numeric values within expected ranges
  - Input: numeric_series, min_value, max_value
  - Output: Series with validation flags
  - Example: CHECK_NUMERIC_RANGE(revenue_column, 0, 1000000)

- **OUTLIER_DETECTION**
  - Purpose: Detect statistical outliers using IQR or z-score methods
  - Input: numeric_series, method, threshold
  - Output: Series with outlier flags
  - Example: OUTLIER_DETECTION(sales_data, 'iqr', 1.5)

- **COMPLETENESS_CHECK**
  - Purpose: Check data completeness by column
  - Input: DataFrame
  - Output: Dict with completeness percentages
  - Example: COMPLETENESS_CHECK(financial_data_df)

- **CONSISTENCY_CHECK**
  - Purpose: Check data consistency across related fields
  - Input: DataFrame, consistency_rules
  - Output: DataFrame with consistency flags
  - Example: CONSISTENCY_CHECK(df, {'total': ['subtotal', 'tax']})

### 14. Data Filtering & Selection
Functions for filtering and selecting data subsets.

- **FILTER_BY_DATE_RANGE**
  - Purpose: Filter DataFrame by date range
  - Input: DataFrame, date_column, start_date, end_date
  - Output: Filtered DataFrame
  - Example: FILTER_BY_DATE_RANGE(df, 'transaction_date', '2024-01-01', '2024-12-31')

- **FILTER_BY_VALUE**
  - Purpose: Filter DataFrame by column values
  - Input: DataFrame, column, operator, value
  - Output: Filtered DataFrame
  - Example: FILTER_BY_VALUE(sales_df, 'amount', '>', 1000)

- **FILTER_BY_MULTIPLE_CONDITIONS**
  - Purpose: Filter DataFrame by multiple conditions
  - Input: DataFrame, conditions_dict
  - Output: Filtered DataFrame
  - Example: FILTER_BY_MULTIPLE_CONDITIONS(df, {'region': 'North', 'sales': '>1000'})

- **TOP_N**
  - Purpose: Select top N records by value
  - Input: DataFrame, column, n, ascending
  - Output: DataFrame with top N records
  - Example: TOP_N(customers_df, 'revenue', 10, False)

- **BOTTOM_N**
  - Purpose: Select bottom N records极 by value
  - Input: DataFrame, column, n
  - Output: DataFrame with bottom N records
  - Example: BOTTOM_N(products_df, 'profit_margin', 5)

- **SAMPLE_DATA**
  - Purpose: Sample random records from DataFrame
  - Input: DataFrame, n_samples, random_state
  - Output: DataFrame with sampled records
  - Example: SAMPLE_DATA(large_dataset_df, 1000, 42)

### 15. Comparison & Ranking
Functions for comparing values and creating rankings.

- **RANK_BY_COLUMN**
  - Purpose: Rank records by column values
  - Input: DataFrame, column, ascending, method
  - Output: DataFrame with rank column
  - Example: RANK_BY_COLUMN(sales_df, 'revenue', False, 'dense')

- **PERCENTILE_RANK**
  - Purpose: Calculate percentile rank for each value
  - Input: Series, method
  - Output: Series with percentile ranks
  - Example: PERCENTILE_RANK(sales_amounts, 'average')

- **COMPARE_PERIODS**
  - Purpose: Compare values between periods
  - Input: DataFrame, value_column, period_column, periods_to_compare
  - Output: DataFrame with period comparisons
  - Example: COMPARE_PERIODS(monthly_data, 'revenue', 'month', ['2024-01', '2023-01'])

- **VARIANCE_FROM_TARGET**
  - Purpose: Calculate variance from target values
  - Input: actual_values, target_values
  - Output: Series with variances and percentages
  - Example: VARIANCE_FROM_TARGET(actual_sales, budget_sales)

- **RANK_CORRELATION**
  - Purpose: Calculate rank correlation between two series
  - Input: series1, series2
  - Output: Float (correlation coefficient)
  - Example: RANK_CORRELATION(performance_scores, salary_ranks)

### 16. Pattern Recognition
Functions for identifying patterns in financial data.

- **DETECT_SEASONALITY**
  - Purpose: Detect seasonal patterns in time series
  - Input: time_series, seasonal_period
  - Output: Dict with seasonality metrics
  - Example: DETECT_SEASONALITY(monthly_sales, 12)

- **IDENTIFY_TREND**
  - Purpose: Identify trend direction and strength
  - Input: time_series
  - Output: Dict with trend characteristics
  - Example: IDENTIFY_TREND(quarterly_revenue)

- **FIND_BREAKPOINTS**
  - Purpose: Identify structural breaks in time series
  - Input: time_series, min_size
  - Output: Array of breakpoint indices
  - Example: FIND_BREAKPOINTS(stock_prices, 30)

- **CYCLE_DETECTION**
  - Purpose: Detect cyclical patterns
  - Input: time_series, min_cycle_length, max_cycle_length
  - Output: Dict with cycle characteristics
  - Example: CYCLE_DETECTION(economic_indicators, 6, 24)

- **ANOMALY_DETECTION**
  - Purpose: Detect anomalies in time series
  - Input: time_series, method, sensitivity
  - Output: Series with anomaly flags
  - Example: ANOMALY_DETECTION(daily_transactions, 'isolation_forest', 0.1)

### 17. Conditional Logic
Advanced conditional operations for complex business rules.

- **MULTI_CONDITION_LOGIC**
  - Purpose: Apply complex multi-condition logic
  - Input: DataFrame, condition_tree
  - Output: Series with results
  - Example: MULTI_CONDITION_LOGIC(df, {'if': 'revenue > 1000', 'then': 'high', 'elif': 'revenue > 500', 'then': 'medium', 'else': 'low'})

- **NESTED_IF_LOGIC**
  - Purpose: Handle nested conditional statements
  - Input: conditions_list, results_list, default_value
  - Output: Series with conditional results
  - Example: NESTED_IF_LOGIC([cond1, cond2, cond3], [result1, result2, result3], default)

- **CASE_WHEN**
  - Purpose: SQL-style CASE WHEN logic
  - Input: DataFrame, case_conditions
  - Output: Series with case results
  - Example: CASE_WHEN(df, [{'when': 'score >= 90', 'then': 'A'}, {'when': 'score >= 80', 'then': 'B'}])

- **CONDITIONAL_AGGREGATION**
  - Purpose: Aggregate based on conditions
  - Input: DataFrame, group_columns, condition, aggregation_func
  - Output: DataFrame with conditional aggregations
  - Example: CONDITIONAL_AGGREGATION(sales_df, ['region'], 'amount > 100', 'sum')

### 18. Financial Calendar Operations
Functions for handling financial calendars and periods.

- **FISCAL_YEAR**
  - Purpose: Convert calendar date to fiscal year
  - Input: date, fiscal_year_start_month
  - Output: Integer (fiscal year)
  - Example: FISCAL_YEAR('2024-03-15', 4)

- **FISCAL_QUARTER**
  - Purpose: Convert date to fiscal quarter
  - Input: date, fiscal_year_start_month
  - Output: String (fiscal quarter)
  - Example: FISCAL_QUARTER('2024-03-15', 4)

- **BUSINESS_DAYS_BETWEEN**
  - Purpose: Calculate business days between dates
  - Input: start_date, end_date, holidays_list
  - Output: Integer (business days)
  - Example: BUSINESS_DAYS_BETWEEN('2024-01-01', '2024-01-31', ['2024-01-15'])

- **END_OF_PERIOD**
  - Purpose: Get end date of period (month, quarter, year)
  - Input: date, period_type
  - Output: Date
  - Example: END_OF_PERIOD('2024-03-15', 'quarter')

- **PERIOD_OVERLAP**
  - Purpose: Calculate overlap between two periods
  - Input: start1, end1, start2, end2
  - Output: Integer (overlap days)
  - Example: PERIOD_OVERLAP('2024-01-01', '2024-06-30', '2024-04-01', '2024-09-30')

### 19. Data Cleaning Operations
Functions for cleaning and standardizing financial data.

- **STANDARDIZE_CURRENCY**
  - Purpose: Standardize currency formats
  - Input: currency_series, target_format
  - Output: Series with standardized currency
  - Example: STANDARDIZE_CURRENCY(mixed_currency_data, 'USD')

- **CLEAN_NUMERIC**
  - Purpose: Clean numeric data removing non-numeric characters
  - Input: mixed_series
  - Output: Series with clean numeric values
  - Example: CLEAN_NUMERIC(['$1,234.56', '€987.65', '¥1000'])

- **NORMALIZE_NAMES**
  - Purpose: Normalize company/customer names
  - Input: name_series, normalization_rules
  - Output: Series with normalized names
  - Example: NORMALIZE_NAMES(company_names, standardization_dict)

- **REMOVE_DUPLICATES**
  - Purpose: Remove duplicate records with options
  - Input: DataFrame, subset_columns, keep_method
  - Output: DataFrame without duplicates
  - Example: REMOVE_DUPLICATES(transactions_df, ['customer_id', 'date'], 'first')

- **STANDARDIZE_DATES**
  - Purpose: Convert various date formats to standard format
  - Input: date_series, target_format
  - Output: Series with standardized dates
  - Example: STANDARDIZE_DATES(mixed_date_formats, '%Y-%m-%d')

