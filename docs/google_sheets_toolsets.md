# Google Sheets Toolsets Documentation

This document provides comprehensive information about the 82 available Google Sheets formulas distributed across 9 specialized MCP servers for the FP&A multi-agent system.

## Overview

The Google Sheets integration provides exactly 82 Excel-like functions distributed across 9 category-focused MCP servers. Each server maintains a focused toolset (4-14 formulas each) optimized for AI agent usability while providing 100% formula accuracy.

## MCP Server Architecture

### Core Infrastructure Servers (5 servers)

#### 1. Data Server (`google_sheets_data_mcp_server.py`)
**Purpose**: Core data operations for reading and writing spreadsheet data
- **Read Operations**: Extract data from cells, ranges, and entire sheets
- **Write Operations**: Update cells, ranges, and bulk data insertion
- **Data Validation**: Ensure data integrity during operations
- **Use Cases**: Loading financial data, updating calculated results, data synchronization

#### 2. Structure Server (`google_sheets_structure_mcp_server.py`)
**Purpose**: Spreadsheet and worksheet management operations
- **Sheet Management**: Create, delete, rename worksheets
- **Workbook Operations**: Manage spreadsheet properties and structure
- **Layout Control**: Organize data across multiple sheets
- **Use Cases**: Setting up analysis workbooks, organizing data by category

#### 3. Format Server (`google_sheets_format_mcp_server.py`)
**Purpose**: Cell and range formatting for professional presentations
- **Cell Formatting**: Number formats, currency, percentages, dates
- **Conditional Formatting**: Highlight trends, exceptions, and key metrics
- **Style Management**: Colors, fonts, borders, alignment
- **Use Cases**: Executive dashboards, financial reports, data visualization

#### 4. Chart Server (`google_sheets_chart_mcp_server.py`)
**Purpose**: Data visualization and chart creation
- **Chart Types**: Line, bar, pie, scatter, combo charts
- **Chart Configuration**: Titles, legends, axes, data series
- **Interactive Elements**: Filters, drill-down capabilities
- **Use Cases**: Trend analysis, performance dashboards, executive summaries

#### 5. Validation Server (`google_sheets_validation_mcp_server.py`)
**Purpose**: Data quality and validation rules
- **Input Validation**: Ensure data meets business rules
- **Range Validation**: Check data consistency across ranges
- **Error Detection**: Identify and flag data quality issues
- **Use Cases**: Financial data integrity, audit trails, compliance checks

## Formula Category Servers (9 servers - 82 total formulas)

### 1. Aggregation Functions (`aggregation_formula_mcp_server.py`) - 13 formulas
**Purpose**: Mathematical aggregation and summary operations

**Value Aggregation Operations:**
- **SUM**: Calculate totals for revenue, expenses, quantities
- **AVERAGE**: Compute mean values for performance metrics  
- **MAX**: Identify highest values in datasets
- **MIN**: Identify lowest values in datasets
- **COUNT**: Count numeric data points
- **COUNTA**: Count non-empty values
- **SUBTOTAL**: Aggregate with hidden row awareness

**Conditional Aggregation Operations:**
- **SUMIF**: Sum values meeting single criteria (e.g., revenue by region)
- **SUMIFS**: Sum values meeting multiple criteria
- **COUNTIF**: Count occurrences meeting single criteria (e.g., customers by segment)
- **COUNTIFS**: Count occurrences meeting multiple criteria
- **AVERAGEIF**: Average calculations with single condition
- **AVERAGEIFS**: Average calculations with multiple conditions

**Business Applications:**
- Revenue aggregation by time period, region, or product
- Expense analysis across departments or cost centers
- Customer segmentation based on purchase behavior
- Performance metric calculations with complex filtering

### 2. Lookup and Reference (`lookup_formula_mcp_server.py`) - 6 formulas
**Purpose**: Data matching and cross-referencing operations

**Data Matching Operations:**
- **VLOOKUP**: Vertical lookup for retrieving related information
- **HLOOKUP**: Horizontal lookup for table-based data retrieval
- **XLOOKUP**: Advanced lookup with multiple criteria and error handling
- **INDEX**: Return value at specific position in array
- **MATCH**: Find position of value in array
- **INDEX_MATCH**: Combined flexible lookup operation

**Business Applications:**
- Customer detail retrieval for transaction analysis
- Product information lookup for sales data
- Employee data matching for payroll analysis
- Supplier information cross-reference

### 3. Financial Functions (`financial_formula_mcp_server.py`) - 14 formulas
**Purpose**: Specialized financial calculations and analysis

**Time Value of Money:**
- **NPV**: Net present value calculations for investment analysis
- **IRR**: Internal rate of return for project evaluation
- **MIRR**: Modified internal rate of return
- **XIRR**: Internal rate of return for irregular cash flows
- **XNPV**: Net present value for irregular cash flows
- **PV**: Present value calculations
- **FV**: Future value calculations
- **PMT**: Payment calculations for financing scenarios

**Financial Analysis:**
- **NPER**: Number of payment periods
- **RATE**: Interest rate calculations
- **IPMT**: Interest portion of payment
- **PPMT**: Principal portion of payment
- **SLN**: Straight-line depreciation
- **DDB**: Double-declining balance depreciation

**Business Applications:**
- Capital investment evaluation
- Loan and financing analysis
- Budget planning and forecasting
- Asset valuation and depreciation

### 4. Business Intelligence (`business_formula_mcp_server.py`) - 14 formulas
**Purpose**: Advanced business analysis and KPI calculations

**Performance Metrics:**
- **PROFIT_MARGIN**: Calculate profit margins for products/services
- **VARIANCE_PERCENT**: Period-over-period variance analysis
- **COMPOUND_GROWTH**: Compound growth rate calculations
- **CAGR**: Compound Annual Growth Rate
- **MARKET_SHARE**: Market share calculations

**Customer Analytics:**
- **CUSTOMER_LTV**: Customer Lifetime Value calculations
- **CHURN_RATE**: Customer churn rate analysis
- **CUSTOMER_ACQUISITION_COST**: CAC calculations

**Financial Analysis:**
- **CAPM**: Capital Asset Pricing Model
- **SHARPE_RATIO**: Risk-adjusted return calculations
- **BETA_COEFFICIENT**: Market risk analysis
- **BREAK_EVEN_ANALYSIS**: Break-even point calculations
- **DUPONT_ANALYSIS**: Return on equity decomposition
- **Z_SCORE**: Financial health assessment

**Business Applications:**
- Monthly/quarterly performance reporting
- Customer analytics and retention analysis
- Investment and risk assessment
- Operational efficiency metrics

### 5. Array Functions (`array_formula_mcp_server.py`) - 7 formulas
**Purpose**: Multi-dimensional data operations and array calculations

**Array Operations:**
- **ARRAYFORMULA**: Apply formulas to entire arrays automatically
- **TRANSPOSE**: Rotate data from rows to columns or vice versa
- **UNIQUE**: Extract unique values from datasets
- **SORT**: Sort data by specified criteria
- **FILTER**: Filter data based on conditions
- **SEQUENCE**: Generate sequential number arrays
- **SUMPRODUCT**: Multiply corresponding array elements and sum

**Business Applications:**
- Large dataset processing
- Multi-dimensional financial analysis
- Bulk data transformations
- Complex scenario modeling

### 6. Text Functions (`text_formula_mcp_server.py`) - 8 formulas
**Purpose**: Text processing and string manipulation

**Text Processing:**
- **CONCATENATE**: Combine text strings
- **LEFT**: Extract characters from left side of text
- **RIGHT**: Extract characters from right side of text
- **MID**: Extract characters from middle of text
- **LEN**: Calculate length of text strings
- **UPPER**: Convert text to uppercase
- **LOWER**: Convert text to lowercase
- **TRIM**: Remove extra spaces from text

**Business Applications:**
- Customer name standardization
- Address parsing and cleaning
- Product code formatting
- Data import preparation

### 7. Logical Functions (`logical_formula_mcp_server.py`) - 4 formulas
**Purpose**: Conditional logic and decision-making operations

**Conditional Operations:**
- **IF**: Simple and nested conditional logic
- **AND**: Test multiple conditions (all must be true)
- **OR**: Test multiple conditions (any can be true)
- **NOT**: Reverse logical values

**Business Applications:**
- Customer segmentation logic
- Risk categorization
- Performance rating systems
- Automated business rule enforcement

### 8. Statistical Functions (`statistical_formula_mcp_server.py`) - 7 formulas
**Purpose**: Statistical analysis and data science operations

**Descriptive Statistics:**
- **MEDIAN**: Find middle value in dataset
- **STDEV**: Calculate standard deviation
- **VAR**: Calculate variance
- **MODE**: Find most frequently occurring value
- **PERCENTILE**: Calculate percentile values
- **PERCENTRANK**: Find rank as percentage
- **RANK**: Determine rank of value in dataset

**Business Applications:**
- Financial performance analysis
- Risk assessment and modeling
- Quality control metrics
- Predictive analytics preparation

### 9. DateTime Functions (`datetime_formula_mcp_server.py`) - 7 formulas
**Purpose**: Date and time calculations for temporal analysis

**Date Operations:**
- **NOW**: Current date and time
- **TODAY**: Current date
- **DATE**: Create date from year, month, day
- **YEAR**: Extract year from date
- **MONTH**: Extract month from date
- **DAY**: Extract day from date
- **EOMONTH**: End of month calculations

**Business Applications:**
- Accounts receivable aging
- Project timeline analysis
- Seasonal trend identification
- Compliance deadline tracking

## Integration Patterns for FP&A Analysis

### Data Flow Architecture
1. **Data Input**: Load financial data through Data Server
2. **Structure Setup**: Organize worksheets using Structure Server
3. **Analysis Execution**: Apply appropriate formula servers for calculations
4. **Visualization**: Create charts and formatting through Chart/Format servers
5. **Validation**: Ensure data quality through Validation Server

### Formula Selection Guidelines

#### For Value Aggregation (Revenue, Costs, Totals)
**Use Aggregation Server**: SUM, AVERAGE, SUMIF, SUMIFS
- "Calculate total revenue by customer segment"
- "Sum monthly expenses across departments"
- "Average performance metrics by region"

#### For Occurrence Counting (Customers, Transactions, Events)
**Use Aggregation Server**: COUNT, COUNTIF, COUNTIFS
- "Count unique customers per region"
- "Identify frequency of support tickets by category"
- "Count transactions meeting specific criteria"

#### For Data Matching (Cross-referencing, Lookups)
**Use Lookup Server**: VLOOKUP, INDEX/MATCH, XLOOKUP
- "Retrieve customer details for each transaction"
- "Cross-reference product information with sales data"
- "Match employee data across systems"

#### For Conditional Analysis (Filtering, Categorization)
**Use Logical Server**: IF, AND, OR combined with other servers
- "Categorize customers based on purchase behavior"
- "Apply risk ratings based on multiple criteria"
- "Implement automated business rules"

#### For Financial Calculations (NPV, IRR, Depreciation)
**Use Financial Server**: Specialized financial analysis
- "Evaluate capital investment opportunities"
- "Calculate loan payment schedules"
- "Assess asset depreciation strategies"

#### For Advanced Analytics (Statistics, Trends)
**Use Statistical + Business Servers**: Performance analysis
- "Analyze customer behavior patterns"
- "Calculate risk-adjusted returns"
- "Identify market trends and outliers"

## Analytical Workflow Examples

### Customer Lifetime Value Analysis
1. **Data Matching**: Use VLOOKUP to combine customer and transaction data
2. **Value Aggregation**: Apply SUMIF to calculate total customer revenue
3. **Occurrence Counting**: Use COUNTIF to determine purchase frequency
4. **Business Calculation**: Calculate LTV using CUSTOMER_LTV formula
5. **Statistical Analysis**: Use PERCENTILE to segment customers

### Revenue Trend Analysis
1. **Value Aggregation**: Sum revenue by periods using SUM with date criteria
2. **Business Intelligence**: Calculate growth rates using COMPOUND_GROWTH
3. **Conditional Analysis**: Filter data by product/region using IF statements
4. **Statistical Analysis**: Identify trends using moving averages
5. **Visualization**: Create trend charts through Chart Server

### Financial Performance Dashboard
1. **Data Consolidation**: Use array functions to combine multiple data sources
2. **KPI Calculations**: Apply business formulas for key metrics
3. **Variance Analysis**: Calculate actual vs. budget using VARIANCE_PERCENT
4. **Risk Assessment**: Use statistical functions for performance distributions
5. **Presentation**: Format results using Format Server

## Error Handling and Data Integrity

### Financial Data Principles
- **No Imputation**: Never estimate or fill missing financial data
- **Audit Trails**: Maintain clear record of all calculations
- **Validation Checks**: Verify data consistency across operations
- **Error Transparency**: Clearly flag and report data quality issues

### Best Practices
- **Explicit Operation Selection**: Choose specific servers based on analytical needs
- **Data Source Validation**: Verify data quality before analysis
- **Formula Documentation**: Maintain clear record of calculation logic
- **Result Verification**: Cross-check critical calculations

## Complete Formula Reference

### All 82 Formulas by Server:

**Aggregation (13)**: SUM, AVERAGE, COUNT, COUNTA, MAX, MIN, SUMIF, SUMIFS, COUNTIF, COUNTIFS, AVERAGEIF, AVERAGEIFS, SUBTOTAL

**Lookup (6)**: VLOOKUP, HLOOKUP, XLOOKUP, INDEX, MATCH, INDEX_MATCH

**Financial (14)**: NPV, IRR, MIRR, XIRR, XNPV, PMT, PV, FV, NPER, RATE, IPMT, PPMT, SLN, DDB

**Business (14)**: PROFIT_MARGIN, VARIANCE_PERCENT, COMPOUND_GROWTH, CAGR, CUSTOMER_LTV, CHURN_RATE, CAPM, SHARPE_RATIO, BETA_COEFFICIENT, MARKET_SHARE, CUSTOMER_ACQUISITION_COST, BREAK_EVEN_ANALYSIS, DUPONT_ANALYSIS, Z_SCORE

**Array (7)**: ARRAYFORMULA, TRANSPOSE, UNIQUE, SORT, FILTER, SEQUENCE, SUMPRODUCT

**Text (8)**: CONCATENATE, LEFT, RIGHT, MID, LEN, UPPER, LOWER, TRIM

**Logical (4)**: IF, AND, OR, NOT

**Statistical (7)**: MEDIAN, STDEV, VAR, MODE, PERCENTILE, PERCENTRANK, RANK

**DateTime (7)**: NOW, TODAY, DATE, YEAR, MONTH, DAY, EOMONTH

**Total: 82 formulas across 9 focused servers**

This comprehensive toolset enables sophisticated financial analysis while maintaining data integrity, providing clear business insights, and supporting the High-Level Planner in making informed decisions about which analytical operations to perform on specific data sources.