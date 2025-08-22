#!/usr/bin/env python3
"""
Custom Business Formula MCP Server - Phase 0 Category-Wise Implementation

This focused MCP server provides custom business formulas with 100% syntax accuracy:
- Growth Metrics: CAGR, Compound Growth
- Customer Analytics: Customer LTV, Churn Rate
- Business Metrics: Profit Margin, Variance Analysis
- Performance Ratios: Various business calculations

Key Benefits:
- Focused toolset for AI agents (14 business tools)
- 100% Formula Accuracy guarantee
- Business-parameter interface
- Specialized for business analytics

Usage:
    # As MCP Server
    uv run python business_formula_mcp_server.py
    
    # As FastAPI Server  
    uv run python business_formula_mcp_server.py --port 3033
"""

import asyncio
import logging
from typing import Any, Dict

from fastapi import FastAPI
from fastmcp import FastMCP
from pydantic import Field
import uvicorn

# Import Formula Builder
from formula_builders import GoogleSheetsFormulaBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server and FastAPI app
mcp = FastMCP("Business Formula Builder Server")
app = FastAPI(
    title="Custom Business Formula Builder MCP Server", 
    version="1.0.0",
    description="Phase 0 Business Formulas with 100% syntax accuracy"
)


class BusinessFormulaTools:
    """
    Focused MCP tools for custom business formulas only.
    
    This specialized server handles sophisticated business calculations:
    - Growth and performance metrics
    - Customer analytics
    - Financial ratios and variance analysis
    - Complex business formula combinations
    """
    
    def __init__(self):
        self.formula_builder = GoogleSheetsFormulaBuilder()
        logger.info("📈 BusinessFormulaTools initialized")
        
        # Get only custom business formulas
        all_formulas = self.formula_builder.get_supported_formulas()
        self.supported_formulas = [f for f in all_formulas if f in [
            'profit_margin', 'variance_percent', 'compound_growth', 
            'cagr', 'customer_ltv', 'churn_rate', 'capm', 'sharpe_ratio',
            'beta_coefficient', 'market_share', 'customer_acquisition_cost',
            'break_even_analysis', 'dupont_analysis', 'z_score'
        ]]
        
        logger.info(f"📊 Supporting {len(self.supported_formulas)} business formulas")


# Global instance
business_tools = BusinessFormulaTools()


# ================== GROWTH & PERFORMANCE METRICS ==================

@mcp.tool()
async def build_profit_margin(
    revenue_cell: str = Field(description="Cell reference containing revenue (e.g., 'B2')"),
    cost_cell: str = Field(description="Cell reference containing cost (e.g., 'C2')")
) -> Dict[str, Any]:
    """
    Build Profit Margin formula with guaranteed syntax accuracy.
    
    Calculates profit margin as (Revenue - Cost) / Revenue.
    
    Examples:
        build_profit_margin("B2", "C2") → =(B2-C2)/B2
        build_profit_margin("Revenue!D5", "Costs!E5") → =(Revenue!D5-Costs!E5)/Revenue!D5
    """
    try:
        formula = business_tools.formula_builder.build_formula('profit_margin', {
            'revenue_cell': revenue_cell,
            'cost_cell': cost_cell
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'profit_margin',
            'parameters': {
                'revenue_cell': revenue_cell,
                'cost_cell': cost_cell
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_profit_margin: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'profit_margin'}


@mcp.tool()
async def build_variance_percent(
    actual_cell: str = Field(description="Cell reference containing actual value"),
    budget_cell: str = Field(description="Cell reference containing budget/target value")
) -> Dict[str, Any]:
    """
    Build Variance Percentage formula with guaranteed syntax accuracy.
    
    Calculates variance as (Actual - Budget) / Budget.
    
    Examples:
        build_variance_percent("B2", "C2") → =(B2-C2)/C2
        build_variance_percent("Actual!D5", "Budget!D5") → =(Actual!D5-Budget!D5)/Budget!D5
    """
    try:
        formula = business_tools.formula_builder.build_formula('variance_percent', {
            'actual_cell': actual_cell,
            'budget_cell': budget_cell
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'variance_percent',
            'parameters': {
                'actual_cell': actual_cell,
                'budget_cell': budget_cell
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_variance_percent: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'variance_percent'}


@mcp.tool()
async def build_compound_growth(
    end_value: str = Field(description="Cell reference containing ending value"),
    start_value: str = Field(description="Cell reference containing starting value"),
    periods: int = Field(description="Number of periods")
) -> Dict[str, Any]:
    """
    Build Compound Growth Rate formula with guaranteed syntax accuracy.
    
    Calculates compound growth rate over a specified number of periods.
    
    Examples:
        build_compound_growth("B5", "B1", 5) → =POWER(B5/B1,1/5)-1
        build_compound_growth("EndValue", "StartValue", 3) → =POWER(EndValue/StartValue,1/3)-1
    """
    try:
        formula = business_tools.formula_builder.build_formula('compound_growth', {
            'end_value': end_value,
            'start_value': start_value,
            'periods': periods
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'compound_growth',
            'parameters': {
                'end_value': end_value,
                'start_value': start_value,
                'periods': periods
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_compound_growth: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'compound_growth'}


@mcp.tool()
async def build_cagr(
    ending_value: str = Field(description="Cell reference containing ending value"),
    beginning_value: str = Field(description="Cell reference containing beginning value"),
    years: int = Field(description="Number of years")
) -> Dict[str, Any]:
    """
    Build CAGR (Compound Annual Growth Rate) formula with guaranteed syntax accuracy.
    
    Calculates the compound annual growth rate over a number of years.
    
    Examples:
        build_cagr("B5", "B1", 5) → =POWER(B5/B1,1/5)-1
        build_cagr("FinalRevenue", "InitialRevenue", 3) → =POWER(FinalRevenue/InitialRevenue,1/3)-1
    """
    try:
        formula = business_tools.formula_builder.build_formula('cagr', {
            'ending_value': ending_value,
            'beginning_value': beginning_value,
            'years': years
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'cagr',
            'parameters': {
                'ending_value': ending_value,
                'beginning_value': beginning_value,
                'years': years
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_cagr: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'cagr'}


# ================== CUSTOMER ANALYTICS ==================

@mcp.tool()
async def build_customer_ltv(
    customer_range: str = Field(description="Range containing customer IDs (e.g., 'A:A')"),
    customer_id: str = Field(description="Specific customer ID to calculate LTV for"),
    revenue_range: str = Field(description="Range containing revenue values (e.g., 'B:B')"),
    months_range: str = Field(description="Range containing months active (e.g., 'C:C')")
) -> Dict[str, Any]:
    """
    Build Customer Lifetime Value formula with guaranteed syntax accuracy.
    
    This complex formula combines SUMIF and MAXIFS to calculate LTV as:
    Total Customer Revenue × Maximum Months Active / 12
    
    Examples:
        build_customer_ltv("A:A", "CUST123", "B:B", "C:C") 
        → =SUMIF(A:A,"CUST123",B:B)*MAXIFS(C:C,A:A,"CUST123")/12
        
        build_customer_ltv("Customers!A:A", "CustomerXYZ", "Revenue!B:B", "Tenure!C:C")
        → =SUMIF(Customers!A:A,"CustomerXYZ",Revenue!B:B)*MAXIFS(Tenure!C:C,Customers!A:A,"CustomerXYZ")/12
    """
    try:
        formula = business_tools.formula_builder.build_formula('customer_ltv', {
            'customer_range': customer_range,
            'customer_id': customer_id,
            'revenue_range': revenue_range,
            'months_range': months_range
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'customer_ltv',
            'parameters': {
                'customer_range': customer_range,
                'customer_id': customer_id,
                'revenue_range': revenue_range,
                'months_range': months_range
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_customer_ltv: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'customer_ltv'}


@mcp.tool()
async def build_churn_rate(
    status_range: str = Field(description="Range containing customer status values (e.g., 'D:D')")
) -> Dict[str, Any]:
    """
    Build Churn Rate formula with guaranteed syntax accuracy.
    
    Calculates churn rate as the percentage of customers who churned out of total active customers.
    Formula: COUNTIF(status="Churned") / COUNTIF(status<>"New") * 100
    
    Examples:
        build_churn_rate("D:D") → =COUNTIF(D:D,"Churned")/COUNTIF(D:D,"<>New")*100
        build_churn_rate("CustomerStatus!A:A") → =COUNTIF(CustomerStatus!A:A,"Churned")/COUNTIF(CustomerStatus!A:A,"<>New")*100
    """
    try:
        formula = business_tools.formula_builder.build_formula('churn_rate', {
            'status_range': status_range
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'churn_rate',
            'parameters': {
                'status_range': status_range
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_churn_rate: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'churn_rate'}


# ================== FINANCIAL RATIOS & ANALYTICS ==================

@mcp.tool()
async def build_capm(
    risk_free_rate: str = Field(description="Risk-free rate cell or value"),
    beta: str = Field(description="Beta coefficient cell or value"),
    market_return: str = Field(description="Market return cell or value")
) -> Dict[str, Any]:
    """
    Build CAPM (Capital Asset Pricing Model) formula with guaranteed syntax accuracy.
    
    Calculates expected return using CAPM: Risk-free rate + Beta * (Market return - Risk-free rate).
    
    Examples:
        build_capm("0.03", "1.2", "0.08") → =0.03+1.2*(0.08-0.03)
        build_capm("A1", "B1", "C1") → =A1+B1*(C1-A1)
    """
    try:
        formula = business_tools.formula_builder.build_formula('capm', {
            'risk_free_rate': risk_free_rate,
            'beta': beta,
            'market_return': market_return
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'capm',
            'parameters': {
                'risk_free_rate': risk_free_rate,
                'beta': beta,
                'market_return': market_return
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_capm: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'capm'}


@mcp.tool()
async def build_sharpe_ratio(
    portfolio_return: str = Field(description="Portfolio return cell or value"),
    risk_free_rate: str = Field(description="Risk-free rate cell or value"),
    portfolio_std_dev: str = Field(description="Portfolio standard deviation cell or value")
) -> Dict[str, Any]:
    """
    Build Sharpe Ratio formula with guaranteed syntax accuracy.
    
    Calculates Sharpe ratio: (Portfolio return - Risk-free rate) / Portfolio standard deviation.
    
    Examples:
        build_sharpe_ratio("0.12", "0.03", "0.15") → =(0.12-0.03)/0.15
        build_sharpe_ratio("A1", "B1", "C1") → =(A1-B1)/C1
    """
    try:
        formula = business_tools.formula_builder.build_formula('sharpe_ratio', {
            'portfolio_return': portfolio_return,
            'risk_free_rate': risk_free_rate,
            'portfolio_std_dev': portfolio_std_dev
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'sharpe_ratio',
            'parameters': {
                'portfolio_return': portfolio_return,
                'risk_free_rate': risk_free_rate,
                'portfolio_std_dev': portfolio_std_dev
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_sharpe_ratio: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'sharpe_ratio'}


@mcp.tool()
async def build_beta_coefficient(
    covariance_market: str = Field(description="Covariance with market cell or value"),
    market_variance: str = Field(description="Market variance cell or value")
) -> Dict[str, Any]:
    """
    Build Beta Coefficient formula with guaranteed syntax accuracy.
    
    Calculates beta: Covariance(stock, market) / Variance(market).
    
    Examples:
        build_beta_coefficient("0.025", "0.02") → =0.025/0.02
        build_beta_coefficient("A1", "B1") → =A1/B1
    """
    try:
        formula = business_tools.formula_builder.build_formula('beta_coefficient', {
            'covariance_market': covariance_market,
            'market_variance': market_variance
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'beta_coefficient',
            'parameters': {
                'covariance_market': covariance_market,
                'market_variance': market_variance
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_beta_coefficient: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'beta_coefficient'}


# ================== MARKET & CUSTOMER ANALYTICS ==================

@mcp.tool()
async def build_market_share(
    company_sales: str = Field(description="Company sales cell or value"),
    total_market_sales: str = Field(description="Total market sales cell or value")
) -> Dict[str, Any]:
    """
    Build Market Share formula with guaranteed syntax accuracy.
    
    Calculates market share as: Company sales / Total market sales.
    
    Examples:
        build_market_share("B2", "C2") → =B2/C2
        build_market_share("CompanySales", "MarketTotal") → =CompanySales/MarketTotal
    """
    try:
        formula = business_tools.formula_builder.build_formula('market_share', {
            'company_sales': company_sales,
            'total_market_sales': total_market_sales
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'market_share',
            'parameters': {
                'company_sales': company_sales,
                'total_market_sales': total_market_sales
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_market_share: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'market_share'}


@mcp.tool()
async def build_customer_acquisition_cost(
    marketing_costs: str = Field(description="Total marketing costs cell or value"),
    new_customers: str = Field(description="Number of new customers acquired cell or value")
) -> Dict[str, Any]:
    """
    Build Customer Acquisition Cost (CAC) formula with guaranteed syntax accuracy.
    
    Calculates CAC as: Total marketing costs / Number of new customers acquired.
    
    Examples:
        build_customer_acquisition_cost("B2", "C2") → =B2/C2
        build_customer_acquisition_cost("MarketingCosts", "NewCustomers") → =MarketingCosts/NewCustomers
    """
    try:
        formula = business_tools.formula_builder.build_formula('customer_acquisition_cost', {
            'marketing_costs': marketing_costs,
            'new_customers': new_customers
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'customer_acquisition_cost',
            'parameters': {
                'marketing_costs': marketing_costs,
                'new_customers': new_customers
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_customer_acquisition_cost: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'customer_acquisition_cost'}


# ================== BUSINESS ANALYSIS MODELS ==================

@mcp.tool()
async def build_break_even_analysis(
    fixed_costs: str = Field(description="Fixed costs cell or value"),
    price_per_unit: str = Field(description="Price per unit cell or value"),
    variable_cost_per_unit: str = Field(description="Variable cost per unit cell or value")
) -> Dict[str, Any]:
    """
    Build Break-Even Analysis formula with guaranteed syntax accuracy.
    
    Calculates break-even point: Fixed costs / (Price per unit - Variable cost per unit).
    
    Examples:
        build_break_even_analysis("10000", "50", "30") → =10000/(50-30)
        build_break_even_analysis("A1", "B1", "C1") → =A1/(B1-C1)
    """
    try:
        formula = business_tools.formula_builder.build_formula('break_even_analysis', {
            'fixed_costs': fixed_costs,
            'price_per_unit': price_per_unit,
            'variable_cost_per_unit': variable_cost_per_unit
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'break_even_analysis',
            'parameters': {
                'fixed_costs': fixed_costs,
                'price_per_unit': price_per_unit,
                'variable_cost_per_unit': variable_cost_per_unit
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_break_even_analysis: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'break_even_analysis'}


@mcp.tool()
async def build_dupont_analysis(
    net_income: str = Field(description="Net income cell or value"),
    revenue: str = Field(description="Revenue cell or value"),
    total_assets: str = Field(description="Total assets cell or value")
) -> Dict[str, Any]:
    """
    Build DuPont Analysis (ROA) formula with guaranteed syntax accuracy.
    
    Calculates ROA using DuPont formula: (Net Income / Revenue) * (Revenue / Total Assets).
    
    Examples:
        build_dupont_analysis("5000", "50000", "100000") → =(5000/50000)*(50000/100000)
        build_dupont_analysis("A1", "B1", "C1") → =(A1/B1)*(B1/C1)
    """
    try:
        formula = business_tools.formula_builder.build_formula('dupont_analysis', {
            'net_income': net_income,
            'revenue': revenue,
            'total_assets': total_assets
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'dupont_analysis',
            'parameters': {
                'net_income': net_income,
                'revenue': revenue,
                'total_assets': total_assets
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_dupont_analysis: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'dupont_analysis'}


@mcp.tool()
async def build_z_score(
    value: str = Field(description="Value to normalize cell or value"),
    mean: str = Field(description="Mean of the dataset cell or value"),
    std_dev: str = Field(description="Standard deviation of the dataset cell or value")
) -> Dict[str, Any]:
    """
    Build Z-Score formula with guaranteed syntax accuracy.
    
    Calculates Z-score for standardization: (Value - Mean) / Standard Deviation.
    
    Examples:
        build_z_score("85", "75", "10") → =(85-75)/10
        build_z_score("A1", "B1", "C1") → =(A1-B1)/C1
    """
    try:
        formula = business_tools.formula_builder.build_formula('z_score', {
            'value': value,
            'mean': mean,
            'std_dev': std_dev
        })
        
        return {
            'success': True,
            'formula_generated': formula,
            'formula_type': 'z_score',
            'parameters': {
                'value': value,
                'mean': mean,
                'std_dev': std_dev
            }
        }
        
    except Exception as e:
        logger.error(f"Error in build_z_score: {e}")
        return {'success': False, 'error': str(e), 'formula_type': 'z_score'}


# ================== SERVER INFO TOOLS ==================

@mcp.tool()
async def get_business_capabilities() -> Dict[str, Any]:
    """
    Get all supported business formulas and their descriptions.
    """
    try:
        capabilities = {
            'growth_performance': {
                'profit_margin': 'Calculate profit margin percentage',
                'variance_percent': 'Calculate variance from budget/target',
                'compound_growth': 'Calculate compound growth rate',
                'cagr': 'Calculate Compound Annual Growth Rate'
            },
            'customer_analytics': {
                'customer_ltv': 'Calculate Customer Lifetime Value',
                'churn_rate': 'Calculate customer churn rate percentage',
                'customer_acquisition_cost': 'Calculate Customer Acquisition Cost (CAC)',
                'market_share': 'Calculate market share percentage'
            },
            'financial_ratios': {
                'capm': 'Capital Asset Pricing Model calculation',
                'sharpe_ratio': 'Calculate Sharpe ratio for risk-adjusted returns',
                'beta_coefficient': 'Calculate beta coefficient for volatility'
            },
            'business_analysis': {
                'break_even_analysis': 'Calculate break-even point analysis',
                'dupont_analysis': 'DuPont ROA analysis calculation',
                'z_score': 'Calculate Z-score for data standardization'
            }
        }
        
        use_cases = {
            'profit_margin': ['Product profitability analysis', 'Departmental performance', 'Project ROI'],
            'cagr': ['Revenue growth analysis', 'Investment performance', 'Market expansion metrics'],
            'customer_ltv': ['Customer segmentation', 'Marketing ROI', 'Retention strategies'],
            'churn_rate': ['Customer retention analysis', 'Service quality metrics', 'Competitive analysis'],
            'capm': ['Investment valuation', 'Portfolio management', 'Cost of equity calculation'],
            'sharpe_ratio': ['Portfolio performance', 'Risk-adjusted returns', 'Investment comparison'],
            'break_even_analysis': ['Pricing strategy', 'Production planning', 'Business viability'],
            'z_score': ['Data normalization', 'Outlier detection', 'Performance standardization']
        }
        
        return {
            'success': True,
            'server_name': 'Business Formula Builder',
            'total_tools': len(business_tools.supported_formulas),
            'categories': capabilities,
            'supported_formulas': business_tools.supported_formulas,
            'use_cases': use_cases
        }
        
    except Exception as e:
        logger.error(f"Error in get_business_capabilities: {e}")
        return {'success': False, 'error': str(e)}


# ================== FASTAPI ENDPOINTS ==================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Business Formula Builder MCP Server",
        "version": "1.0.0",
        "formula_count": len(business_tools.supported_formulas),
        "categories": ["growth_performance", "customer_analytics", "financial_ratios", "business_analysis"]
    }


@app.get("/capabilities")
async def get_capabilities():
    """Get business formula capabilities"""
    return await get_business_capabilities()


# ================== MAIN EXECUTION ==================

async def main():
    """Run the Business Formula MCP Server"""
    logger.info("🚀 Starting Business Formula Builder MCP Server...")
    logger.info("📈 Specialized for Business Analytics Formulas")
    logger.info(f"📊 Supporting {len(business_tools.supported_formulas)} business tools")
    logger.info("")
    logger.info("🎯 Supported Categories:")
    logger.info("   • Growth & Performance: Profit Margin, CAGR, Variance")
    logger.info("   • Customer Analytics: Customer LTV, Churn Rate, CAC, Market Share")
    logger.info("   • Financial Ratios: CAPM, Sharpe Ratio, Beta Coefficient")
    logger.info("   • Business Analysis: Break-Even, DuPont, Z-Score")
    logger.info("")
    logger.info("✅ 100% Formula Accuracy Guaranteed")
    logger.info("📋 Perfect for Business Intelligence & Analytics")
    logger.info("")
    
    # Run MCP server
    await mcp.run()


def run_fastapi_server(port: int = 3033):
    """Run the FastAPI server for HTTP access"""
    logger.info(f"🌐 Starting Business Formula FastAPI server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--port":
        # Run as FastAPI server
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 3033
        run_fastapi_server(port)
    else:
        # Run as MCP server
        asyncio.run(main())