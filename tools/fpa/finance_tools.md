# Financial Planning Tools

## Overview
This document defines the higher level and specialized finiancial planning and analysis tool categories and functions available to our financial planning AI agent. Each tool is designed to help users with specific financial planning tasks and analysis.

## Tool Categories

### 20. Cohort & Customer Analytics
- **COHORT_ANALYSIS**
  - Purpose: Create cohort retention/revenue tables
  - Input: DataFrame with customer_id, order_date, cohort_period, value
  - Output: DataFrame with cohort metrics
  - Example: COHORT_ANALYSIS(transactions_df, 'customer_id', 'order_date', 'revenue')
- **RETENTION_RATE**
  - Purpose: Calculate customer retention rates by period
  - Input: DataFrame with customer activity by period
  - Output: Series/DataFrame with retention percentages
  - Example: RETENTION_RATE(customer_activity_df, 'month', 'active')
- **CHURN_RATE**
  - Purpose: Calculate customer churn rates
  - Input: DataFrame with customer status by period
  - Output: Float or Series
  - Example: CHURN_RATE(customer_status_df, 'churned_flag')
- **LTV_CALCULATION**
  - Purpose: Calculate Customer Lifetime Value
  - Input: Average revenue per user, churn rate, profit margin
  - Output: Float
  - Example: LTV_CALCULATION(arpu=50, churn_rate=0.05, profit_margin=0.3)
- **PAYBACK_PERIOD**
  - Purpose: Calculate when cumulative profit equals initial investment
  - Input: Series of cash flows, initial investment
  - Output: Integer (period number) or None
  - Example: PAYBACK_PERIOD([-1000, 300, 400, 500], 1000)
- **ASSIGN_COHORT**
  - Purpose: Assign a cohort identifier (e.g., acquisition month/year) to each customer based on their first transaction or sign-up date
  - Input: df (DataFrame), customer_id_col (str), date_col (str), cohort_period (str, e.g., 'M' for month, 'Q' for quarter)
  - Output: DataFrame (with a new 'cohort' column)
  - Example: ASSIGN_COHORT(df, 'customer_id', 'signup_date', 'M')
- **ARPU**
  - Purpose: Calculate the average revenue generated per user or account over a specific period
  - Input: total_revenue (float), total_users (int)
  - Output: Float
  - Example: ARPU(total_revenue=100000, total_users=500)

### 21. Revenue & Cost Analysis
- **UNIT_ECONOMICS**
  - Purpose: Calculate key unit economics metrics
  - Input: Revenue per unit, cost per unit, quantity
  - Output: Dict with profit margins, contribution margins
  - Example: UNIT_ECONOMICS(revenue_per_unit=100, cost_per_unit=60, quantity=1000)
- **BREAK_EVEN_ANALYSIS**
  - Purpose: Calculate break-even point in units and revenue
  - Input: Fixed costs, variable cost per unit, price per unit
  - Output: Dict with break-even units and revenue
  - Example: BREAK_EVEN_ANALYSIS(fixed_costs=50000, variable_cost=30, price=50)
- **MARGIN_ANALYSIS**
  - Purpose: Calculate various margin metrics
  - Input: Revenue, costs (COGS, operating, etc.)
  - Output: Dict with gross, operating, net margins
  - Example: MARGIN_ANALYSIS(revenue=1000000, cogs=600000, operating_costs=200000)
- **COST_ALLOCATION**
  - Purpose: Allocate indirect costs across departments/products
  - Input: DataFrame with costs, allocation bases
  - Output: DataFrame with allocated costs
  - Example: COST_ALLOCATION(cost_df, allocation_basis='headcount')

### 22. Cash Flow & Working Capital
- **CASH_FLOW_STATEMENT**
  - Purpose: Generate cash flow statement from financial data
  - Input: Income statement, balance sheet data
  - Output: DataFrame with operating, investing, financing cash flows
  - Example: CASH_FLOW_STATEMENT(income_stmt_df, balance_sheet_df)
- **WORKING_CAPITAL_ANALYSIS**
  - Purpose: Calculate working capital metrics
  - Input: Current assets, current liabilities, sales data
  - Output: Dict with working capital ratios and days metrics
  - Example: WORKING_CAPITAL_ANALYSIS(current_assets=500000, current_liabilities=300000, annual_sales=2000000)
- **DSO_CALCULATION**
  - Purpose: Calculate Days Sales Outstanding
  - Input: Accounts receivable, revenue, period days
  - Output: Float
  - Example: DSO_CALCULATION(accounts_receivable=100000, revenue=1200000, period_days=365)
- **DPO_CALCULATION**
  - Purpose: Calculate Days Payable Outstanding
  - Input: Accounts payable, COGS, period days
  - Output: Float
  - Example: DPO_CALCULATION(accounts_payable=75000, cogs=800000, period_days=365)

### 23. Risk & Sensitivity Analysis
- **SCENARIO_ANALYSIS**
  - Purpose: Run multiple scenarios with different assumptions
  - Input: Base case assumptions, scenario variations
  - Output: DataFrame with results for each scenario
  - Example: SCENARIO_ANALYSIS(base_assumptions, scenarios=['optimistic', 'pessimistic'])
- **MONTE_CARLO_SIMULATION**
  - Purpose: Run probabilistic simulations
  - Input: Variables with probability distributions, model function
  - Output: Array of simulation results
  - Example: MONTE_CARLO_SIMULATION(revenue_dist, cost_dist, profit_model, n_simulations=1000)
- **SENSITIVITY_ANALYSIS**
  - Purpose: Analyze sensitivity to key variables
  - Input: Base model, variable ranges
  - Output: DataFrame showing impact of variable changes
  - Example: SENSITIVITY_ANALYSIS(dcf_model, {'discount_rate': [0.08, 0.12], 'growth_rate': [0.02, 0.05]})
- **VAR_CALCULATION**
  - Purpose: Calculate Value at Risk
  - Input: Returns data, confidence level
  - Output: Float representing VaR
  - Example: VAR_CALCULATION(portfolio_returns, confidence_level=0.95)

### 24. Valuation & Investment
- **DCF_VALUATION**
  - Purpose: Perform discounted cash flow valuation
  - Input: Cash flow projections, discount rate, terminal value
  - Output: Dict with NPV, terminal value, total valuation
  - Example: DCF_VALUATION(cash_flows=[100, 110, 121], discount_rate=0.10, terminal_value=1000)
- **COMPARABLE_ANALYSIS**
  - Purpose: Create comparable company analysis
  - Input: DataFrame with company metrics, multiples
  - Output: DataFrame with valuation ranges
  - Example: COMPARABLE_ANALYSIS(company_metrics_df, multiples=['P/E', 'EV/EBITDA'])
- **WACC_CALCULATION**
  - Purpose: Calculate Weighted Average Cost of Capital
  - Input: Cost of equity, cost of debt, debt/equity weights, tax rate
  - Output: Float
  - Example: WACC_CALCULATION(cost_equity=0.12, cost_debt=0.06, weight_equity=0.6, weight_debt=0.4, tax_rate=0.25)
- **BETA_CALCULATION**
  - Purpose: Calculate stock beta vs market
  - Input: Stock returns, market returns
  - Output: Float
  - Example: BETA_CALCULATION(stock_returns_series, market_returns_series)

### 25. Budget & Planning
- **BUDGET_VARIANCE**
  - Purpose: Calculate budget vs actual variances
  - Input: Budget DataFrame, actual DataFrame
  - Output: DataFrame with variances and percentages
  - Example: BUDGET_VARIANCE(budget_df, actual_df)
- **ROLLING_FORECAST**
  - Purpose: Create rolling forecasts
  - Input: Historical data, forecast periods, update frequency
  - Output: DataFrame with rolling forecast values
  - Example: ROLLING_FORECAST(historical_data, forecast_periods=12, update_frequency='monthly')
- **ZERO_BASED_BUDGET**
  - Purpose: Build budgets from zero base
  - Input: Activity drivers, cost per activity
  - Output: DataFrame with justified budget amounts
  - Example: ZERO_BASED_BUDGET(activity_drivers_df, cost_per_activity_dict)
- **FLEX_BUDGET**
  - Purpose: Create flexible budgets based on activity levels
  - Input: Fixed costs, variable costs, activity levels
  - Output: DataFrame with flexible budget
  - Example: FLEX_BUDGET(fixed_costs=100000, variable_cost_per_unit=25, activity_levels=[1000, 1500, 2000])

### 26. Performance Metrics
- **KPI_DASHBOARD**
  - Purpose: Calculate and format key performance indicators
  - Input: Raw financial data, KPI definitions
  - Output: Dict with calculated KPIs
  - Example: KPI_DASHBOARD(financial_data_df, kpi_definitions_dict)
- **PERFORMANCE_RATIOS**
  - Purpose: Calculate financial ratios
  - Input: Financial statement data
  - Output: Dict with liquidity, efficiency, profitability ratios
  - Example: PERFORMANCE_RATIOS(balance_sheet_df, income_statement_df)
- **BENCHMARK_ANALYSIS**
  - Purpose: Compare metrics against benchmarks
  - Input: Company metrics, industry benchmarks
  - Output: DataFrame with comparisons and rankings
  - Example: BENCHMARK_ANALYSIS(company_metrics_dict, industry_benchmarks_df)
- **TREND_ANALYSIS**
  - Purpose: Analyze trends over multiple periods
  - Input: Multi-period financial data
  - Output: DataFrame with trend percentages and analysis
  - Example: TREND_ANALYSIS(multi_period_data_df, periods=5)

### 27. Tax & Compliance
- **TAX_PROVISION**
  - Purpose: Calculate tax provisions
  - Input: Pre-tax income, tax rates, adjustments
  - Output: Dict with current and deferred tax
  - Example: TAX_PROVISION(pretax_income=1000000, tax_rate=0.25, adjustments_dict={'timing_diff': 50000})
- **TRANSFER_PRICING**
  - Purpose: Calculate transfer pricing adjustments
  - Input: Intercompany transactions, arm's length prices
  - Output: DataFrame with pricing adjustments
  - Example: TRANSFER_PRICING(intercompany_transactions_df, arms_length_prices_dict)
- **DEPRECIATION_SCHEDULE**
  - Purpose: Create asset depreciation schedules
  - Input: Asset details, depreciation methods, useful lives
  - Output: DataFrame with annual depreciation
  - Example: DEPRECIATION_SCHEDULE(asset_cost=100000, useful_life=5, method='straight_line')
- **TAX_OPTIMIZATION**
  - Purpose: Identify tax optimization opportunities
  - Input: Entity structure, income allocation
  - Output: Dict with optimization strategies
  - Example: TAX_OPTIMIZATION(entity_structure_dict, income_allocation_df)

### 28. Treasury & Capital Management
- **CAPITAL_STRUCTURE**
  - Purpose: Analyze optimal capital structure
  - Input: Debt levels, equity, cost of capital
  - Output: Dict with optimal debt/equity ratios
  - Example: CAPITAL_STRUCTURE(debt_levels=[0, 25, 50, 75], equity=1000000, cost_equity=0.12, cost_debt=0.06)
- **LIQUIDITY_MANAGEMENT**
  - Purpose: Manage cash and liquidity needs
  - Input: Cash flows, credit facilities, investment options
  - Output: Dict with liquidity recommendations
  - Example: LIQUIDITY_MANAGEMENT(cash_flows_df, credit_facilities_dict, investment_options_list)
- **HEDGE_EFFECTIVENESS**
  - Purpose: Test hedge effectiveness for accounting
  - Input: Hedge instrument data, hedged item data
  - Output: Dict with effectiveness metrics
  - Example: HEDGE_EFFECTIVENESS(hedge_instrument_df, hedged_item_df)
- **INTEREST_RATE_ANALYSIS**
  - Purpose: Analyze interest rate exposure and strategies
  - Input: Interest-bearing assets/liabilities, rate scenarios
  - Output: DataFrame with rate impact analysis
  - Example: INTEREST_RATE_ANALYSIS(rate_sensitive_positions_df, rate_scenarios=[0.02, 0.04, 0.06])

### 29. M&A & Corporate Development
- **MERGER_MODEL**
  - Purpose: Model merger financials
  - Input: Acquirer data, target data, deal terms
  - Output: Dict with pro forma results
  - Example: MERGER_MODEL(acquirer_financials_df, target_financials_df, deal_terms_dict)
- **SYNERGY_ANALYSIS**
  - Purpose: Quantify merger synergies
  - Input: Revenue synergies, cost synergies, implementation costs
  - Output: Dict with synergy NPV
  - Example: SYNERGY_ANALYSIS(revenue_synergies=10000000, cost_synergies=5000000, implementation_costs=2000000)
- **ACCRETION_DILUTION**
  - Purpose: Calculate EPS accretion/dilution
  - Input: Deal metrics, financing structure
  - Output: Dict with EPS imp

### 30. Financial Statement Analysis
- **COMMON_SIZE_ANALYSIS**
  - Purpose: Create common size financial statements
  - Input: Financial statement data
  - Output: DataFrame with percentages
  - Example: COMMON_SIZE_ANALYSIS(income_statement_df, base_item='revenue')
- **DUPONT_ANALYSIS**
  - Purpose: Decompose ROE into components
  - Input: Financial statement data
  - Output: Dict with DuPont components
  - Example: DUPONT_ANALYSIS(net_income=100000, revenue=1000000, total_assets=500000, equity=300000)
- **ALTMAN_Z_SCORE**
  - Purpose: Calculate bankruptcy prediction score
  - Input: Financial ratios
  - Output: Float representing Z-score
  - Example: ALTMAN_Z_SCORE(working_capital_ratio=0.2, retained_earnings_ratio=0.15, ebit_ratio=0.1)
- **FINANCIAL_STRENGTH**
  - Purpose: Assess overall financial health
  - Input: Multiple financial metrics
  - Output: Dict with strength indicators
  - Example: FINANCIAL_STRENGTH(financial_metrics_dict)

### 31. International Finance
- **CURRENCY_CONVERSION**
  - Purpose: Convert currencies using rates
  - Input: Amount, from currency, to currency, rate
  - Output: Float
  - Example: CURRENCY_CONVERSION(amount=1000, from_currency='EUR', to_currency='USD', rate=1.10)
- **FX_HEDGE_ANALYSIS**
  - Purpose: Analyze foreign exchange hedging strategies
  - Input: FX exposure, hedge instruments, scenarios
  - Output: DataFrame with hedge outcomes
  - Example: FX_HEDGE_ANALYSIS(fx_exposure_df, hedge_instruments_list, fx_scenarios=[0.9, 1.0, 1.1])
- **CONSOLIDATION**
  - Purpose: Consolidate multi-currency financials
  - Input: Subsidiary financials, FX rates, consolidation rules
  - Output: DataFrame with consolidated results
  - Example: CONSOLIDATION(subsidiary_financials_dict, fx_rates_dict, consolidation_rules_dict)
- **HYPERINFLATION_ADJUSTMENT**
  - Purpose: Adjust for hyperinflation accounting
  - Input: Historical cost data, inflation indices
  - Output: DataFrame with adjusted amounts
  - Example: HYPERINFLATION_ADJUSTMENT(historical_costs_df, inflation_indices_series)

### 32. ESG & Sustainability
- **CARBON_FOOTPRINT**
  - Purpose: Calculate carbon footprint metrics
  - Input: Activity data, emission factors
  - Output: Dict with CO2 equivalent emissions
  - Example: CARBON_FOOTPRINT(activity_data_df, emission_factors_dict)
- **ESG_SCORING**
  - Purpose: Calculate ESG performance scores
  - Input: ESG metrics, weights, benchmarks
  - Output: Dict with composite scores
  - Example: ESG_SCORING(esg_metrics_dict, weights={'E': 0.4, 'S': 0.3, 'G': 0.3}, benchmarks_dict)
- **SUSTAINABILITY_ROI**
  - Purpose: Calculate ROI of sustainability investments
  - Input: Investment costs, savings, timeline
  - Output: Dict with ROI metrics
  - Example: SUSTAINABILITY_ROI(investment_costs=1000000, annual_savings=200000, timeline=5)
- **GREEN_FINANCE**
  - Purpose: Analyze green financing options
  - Input: Project costs, green incentives, rates
  - Output: Dict with financing alternatives
  - Example: GREEN_FINANCE(project_costs=5000000, green_incentives_dict, market_rates_dict)

### 33. Real Estate & Asset Management
- **PROPERTY_VALUATION**
  - Purpose: Value real estate investments
  - Input: Cash flows, cap rates, comparable sales
  - Output: Dict with valuation estimates
  - Example: PROPERTY_VALUATION(annual_noi=100000, cap_rate=0.06, comparable_sales_list)
- **LEASE_ANALYSIS**
  - Purpose: Analyze lease vs buy decisions
  - Input: Lease terms, purchase costs, assumptions
  - Output: Dict with NPV comparison
  - Example: LEASE_ANALYSIS(lease_payment=5000, lease_term=60, purchase_price=250000, discount_rate=0.08)
- **ASSET_ALLOCATION**
  - Purpose: Optimize asset portfolio allocation
  - Input: Asset classes, returns, risk metrics
  - Output: Dict with optimal weights
  - Example: ASSET_ALLOCATION(asset_returns_df, risk_metrics_dict, target_return=0.08)
- **DEPRECIATION_REAL_ESTATE**
  - Purpose: Calculate real estate depreciation
  - Input: Property cost, useful life, method
  - Output: DataFrame with depreciation schedule
  - Example: DEPRECIATION_REAL_ESTATE(property_cost=1000000, useful_life=27.5, method='straight_line')

### 34. Technology & Automation
- **AUTOMATION_ROI**
  - Purpose: Calculate ROI of automation projects
  - Input: Implementation costs, labor savings, timeline
  - Output: Dict with ROI and payback metrics
  - Example: AUTOMATION_ROI(implementation_costs=500000, annual_labor_savings=150000, timeline=5)
- **DIGITAL_TRANSFORMATION**
  - Purpose: Assess digital transformation value
  - Input: Investment costs, efficiency gains, revenue impact
  - Output: Dict with business case metrics
  - Example: DIGITAL_TRANSFORMATION(investment_costs=2000000, efficiency_gains=300000, revenue_impact=500000)
- **DATA_QUALITY_SCORE**
  - Purpose: Score data quality for financial reporting
  - Input: Data completeness, accuracy, consistency metrics
  - Output: Float representing quality score
  - Example: DATA_QUALITY_SCORE(completeness=0.95, accuracy=0.98, consistency=0.92)
- **PROCESS_EFFICIENCY**
  - Purpose: Measure financial process efficiency
  - Input: Process times, error rates, costs
  - Output: Dict with efficiency metrics
  - Example: PROCESS_EFFICIENCY(process_times_dict, error_rates_dict, process_costs_dict)

### 35. Insurance & Actuarial
- **RESERVE_CALCULATION**
  - Purpose: Calculate insurance reserves
  - Input: Claim data, development patterns, assumptions
  - Output: Dict with reserve estimates
  - Example: RESERVE_CALCULATION(claim_triangles_df, development_patterns_dict, assumptions_dict)
- **LOSS_RATIO**
  - Purpose: Calculate insurance loss ratios
  - Input: Claims paid, premiums earned
  - Output: Float
  - Example: LOSS_RATIO(claims_paid=8000000, premiums_earned=10000000)
- **ACTUARIAL_PRESENT_VALUE**
  - Purpose: Calculate present value of future benefits
  - Input: Benefit payments, mortality tables, discount rates
  - Output: Float
  - Example: ACTUARIAL_PRESENT_VALUE(benefit_payments_series, mortality_table_df, discount_rate=0.04)
- **RISK_PREMIUM**
  - Purpose: Calculate risk premiums for insurance
  - Input: Expected losses, expenses, profit margin
  - Output: Float
  - Example: RISK_PREMIUM(expected_losses=7500000, expenses=1500000, profit_margin=0.15)

### 36. Commodities & Trading
- **COMMODITY_PRICING**
  - Purpose: Price commodity derivatives
  - Input: Spot prices, volatility, risk-free rate, time to expiration
  - Output: Dict with option/futures prices
  - Example: COMMODITY_PRICING(spot_price=100, volatility=0.25, risk_free_rate=0.05, time_to_expiration=0.25)
- **TRADING_PNL**
  - Purpose: Calculate trading profit and loss
  - Input: Positions, market data, transaction costs
  - Output: DataFrame with PnL attribution
  - Example: TRADING_PNL(positions_df, market_data_df, transaction_costs=0.001)
- **RISK_METRICS**
  - Purpose: Calculate trading risk metrics
  - Input: Portfolio positions, correlation matrix, volatilities
  - Output: Dict with VaR, Expected Shortfall
  - Example: RISK_METRICS(portfolio_positions_df, correlation_matrix, volatilities_dict)
- **MARK_TO_MARKET**
  - Purpose: Mark trading positions to market
  - Input: Positions, current market prices
  - Output: DataFrame with market values
  - Example: MARK_TO_MARKET(positions_df, current_prices_dict)

### 37. Reporting & Visualization
- **FINANCIAL_DASHBOARD**
  - Purpose: Create executive financial dashboards
  - Input: Financial data, KPI definitions, formatting rules
  - Output: Dict with dashboard components
  - Example: FINANCIAL_DASHBOARD(financial_data_df, kpi_definitions_dict, formatting_rules_dict)
- **VARIANCE_REPORT**
  - Purpose: Generate variance analysis reports
  - Input: Actual vs budget/forecast data
  - Output: Formatted DataFrame with variance analysis
  - Example: VARIANCE_REPORT(actual_data_df, budget_data_df, variance_thresholds={'significant': 0.05})
- **EXECUTIVE_SUMMARY**
  - Purpose: Generate executive summary of financial performance
  - Input: Financial metrics, commentary templates
  - Output: Dict with summary components
  - Example: EXECUTIVE_SUMMARY(financial_metrics_dict, commentary_templates_dict)
- **CHART_GENERATION**
  - Purpose: Generate financial charts and graphs
  - Input: Data series, chart specifications
  - Output: Chart object
  - Example: CHART_GENERATION(revenue_series, chart_type='line', title='Revenue Trend')

### 38. Validation & Quality Assurance
- **MODEL_VALIDATION**
  - Purpose: Validate financial model calculations and logic
  - Input: model_df: DataFrame, validation_rules: dict
  - Output: Dict with validation results
  - Example: MODEL_VALIDATION(financial_model_df, validation_rules_dict)
- **AUDIT_TRAIL**
  - Purpose: Create audit trails showing calculation dependencies
  - Input: worksheet: object, cell_range: str
  - Output: DataFrame with formula dependencies
  - Example: AUDIT_TRAIL(excel_worksheet, cell_range='A1:Z100')
- **SENSITIVITY_VALIDATION**
  - Purpose: Validate that sensitivity analyses produce reasonable results
  - Input: base_case: dict, sensitivity_results: DataFrame
  - Output: Dict with validation metrics
  - Example: SENSITIVITY_VALIDATION(base_case_dict, sensitivity_results_df)
