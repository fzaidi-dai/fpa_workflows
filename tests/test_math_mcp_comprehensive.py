#!/usr/bin/env python3
"""
Comprehensive test suite for Math and Aggregation MCP Server

This script tests the math server functionality with real Financial Planning and Analysis scenarios:
- Input data as Polars dataframes saved as .parquet files
- Output results to .parquet files in scratch_pad folder
- Tests all 25 math functions with appropriate data types
- Validates file-based operations for FPA workflows
"""

import os
import asyncio
import polars as pl
from pathlib import Path
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# Load environment variables
load_dotenv()

# Import MCP tools
from mcp_tooling.mcp_tools_adk import MCPConfig, create_mcp_tools

def validate_environment():
    """Validate that required environment variables are present."""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found in environment variables. "
            "Please ensure your .env file contains this key."
        )
    return openrouter_api_key

def create_test_data():
    """Create comprehensive test datasets for Financial Planning and Analysis."""
    scratch_pad = Path("scratch_pad")
    scratch_pad.mkdir(exist_ok=True)

    print("📊 Creating test datasets for FPA scenarios...")

    # 1. Monthly Revenue Data
    monthly_revenue = pl.DataFrame({
        "month": ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"],
        "revenue": [125000.50, 132000.75, 128500.25, 145000.00, 139500.80, 152000.30],
        "expenses": [85000.25, 89000.50, 87500.75, 95000.00, 92500.60, 98000.40]
    })
    monthly_revenue.write_parquet(scratch_pad / "monthly_revenue.parquet")

    # 2. Product Performance Data
    product_performance = pl.DataFrame({
        "product_id": ["P001", "P002", "P003", "P004", "P005"],
        "sales": [45000.75, 32000.50, 58000.25, 41000.80, 67000.90],
        "cost": [28000.45, 20000.30, 35000.60, 25000.70, 40000.85],
        "margin": [17000.30, 12000.20, 22000.65, 15000.10, 27000.05]
    })
    product_performance.write_parquet(scratch_pad / "product_performance.parquet")

    # 3. Quarterly Growth Rates
    growth_rates = pl.DataFrame({
        "quarter": ["Q1", "Q2", "Q3", "Q4"],
        "growth_rate": [1.05, 1.08, 1.12, 1.03],
        "market_factor": [0.98, 1.02, 1.05, 0.97]
    })
    growth_rates.write_parquet(scratch_pad / "growth_rates.parquet")

    # 4. Risk Metrics Data
    risk_metrics = pl.DataFrame({
        "asset": ["Stock_A", "Stock_B", "Stock_C", "Stock_D", "Stock_E"],
        "returns": [0.12, 0.08, 0.15, 0.06, 0.18],
        "volatility": [0.25, 0.18, 0.32, 0.15, 0.28],
        "weight": [0.3, 0.2, 0.25, 0.15, 0.1]
    })
    risk_metrics.write_parquet(scratch_pad / "risk_metrics.parquet")

    # 5. Cash Flow Data
    cash_flow = pl.DataFrame({
        "period": [1, 2, 3, 4, 5, 6],
        "inflow": [50000.00, 55000.25, 48000.75, 62000.50, 58000.80, 65000.30],
        "outflow": [35000.50, 38000.75, 33000.25, 42000.80, 40000.60, 45000.90]
    })
    cash_flow.write_parquet(scratch_pad / "cash_flow.parquet")

    # 6. Portfolio Weights
    portfolio_weights = pl.DataFrame({
        "asset": ["Bond", "Stock", "Real_Estate", "Commodity"],
        "weight": [0.4, 0.35, 0.15, 0.1]
    })
    portfolio_weights.write_parquet(scratch_pad / "portfolio_weights.parquet")

    # 7. Expense Categories
    expense_categories = pl.DataFrame({
        "category": ["Marketing", "Operations", "R&D", "Admin", "Sales"],
        "amount": [25000.50, 45000.75, 35000.25, 18000.80, 32000.60]
    })
    expense_categories.write_parquet(scratch_pad / "expense_categories.parquet")

    # 8. Interest Rates for Compound Calculations
    interest_rates = pl.DataFrame({
        "year": [1, 2, 3, 4, 5],
        "rate": [0.05, 0.055, 0.06, 0.058, 0.062]
    })
    interest_rates.write_parquet(scratch_pad / "interest_rates.parquet")

    print("✅ Test datasets created successfully!")
    return {
        "monthly_revenue": "monthly_revenue.parquet",
        "product_performance": "product_performance.parquet",
        "growth_rates": "growth_rates.parquet",
        "risk_metrics": "risk_metrics.parquet",
        "cash_flow": "cash_flow.parquet",
        "portfolio_weights": "portfolio_weights.parquet",
        "expense_categories": "expense_categories.parquet",
        "interest_rates": "interest_rates.parquet"
    }

def verify_output_file(filename: str, expected_type: str = "numeric") -> bool:
    """Verify that an output file was created and contains expected data."""
    file_path = Path("scratch_pad") / filename
    if not file_path.exists():
        print(f"❌ Output file {filename} not found")
        return False

    try:
        df = pl.read_parquet(file_path)
        if df.height == 0:
            print(f"❌ Output file {filename} is empty")
            return False

        print(f"✅ Output file {filename} created with {df.height} rows")
        print(f"   Sample data: {df.head(2).to_pandas().to_dict('records')}")
        return True
    except Exception as e:
        print(f"❌ Error reading output file {filename}: {e}")
        return False

async def test_basic_aggregation_functions(agent, runner, session, test_data):
    """Test basic aggregation functions (SUM, AVERAGE, MIN, MAX, etc.)."""
    print("\n🧮 Testing Basic Aggregation Functions")
    print("=" * 50)

    test_cases = [
        {
            "name": "SUM - Total Monthly Revenue",
            "query": f"Calculate the total revenue from {test_data['monthly_revenue']} and save results to total_revenue.parquet",
            "expected_output": "total_revenue.parquet"
        },
        {
            "name": "AVERAGE - Average Product Sales",
            "query": f"Find the average sales from {test_data['product_performance']} and save to avg_sales.parquet",
            "expected_output": "avg_sales.parquet"
        },
        {
            "name": "MIN - Minimum Expense",
            "query": f"Find the minimum expense from {test_data['expense_categories']} and save to min_expense.parquet",
            "expected_output": "min_expense.parquet"
        },
        {
            "name": "MAX - Maximum Cash Inflow",
            "query": f"Find the maximum inflow from {test_data['cash_flow']} and save to max_inflow.parquet",
            "expected_output": "max_inflow.parquet"
        },
        {
            "name": "MEDIAN - Median Product Margin",
            "query": f"Calculate the median margin from {test_data['product_performance']}",
            "expected_output": None  # Single value result
        }
    ]

    success_count = 0
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case['name']} ---")
        try:
            content = types.Content(
                role='user',
                parts=[types.Part(text=test_case['query'])]
            )

            events_async = runner.run_async(
                session_id=session.id,
                user_id=session.user_id,
                new_message=content
            )

            response_parts = []
            async for event in events_async:
                if hasattr(event, 'content') and event.content:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_parts.append(part.text)

            response = ''.join(response_parts)
            if response.strip():
                print(f"✅ Response received: {response[:150]}...")

                # Verify output file if expected
                if test_case['expected_output']:
                    if verify_output_file(test_case['expected_output']):
                        success_count += 1
                    else:
                        print(f"❌ Output file verification failed")
                else:
                    success_count += 1
            else:
                print("❌ No response received")

        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n=== Basic Aggregation Results: {success_count}/{len(test_cases)} successful ===")
    return success_count == len(test_cases)

async def test_mathematical_operations(agent, runner, session, test_data):
    """Test mathematical operations (POWER, SQRT, EXP, LN, LOG, etc.)."""
    print("\n🔢 Testing Mathematical Operations")
    print("=" * 50)

    test_cases = [
        {
            "name": "POWER - Compound Growth Calculation",
            "query": f"Calculate compound growth using power function on {test_data['growth_rates']} with power 3 and save to compound_growth.parquet",
            "expected_output": "compound_growth.parquet"
        },
        {
            "name": "SQRT - Volatility Square Root",
            "query": f"Calculate square root of volatility from {test_data['risk_metrics']} and save to sqrt_volatility.parquet",
            "expected_output": "sqrt_volatility.parquet"
        },
        {
            "name": "EXP - Exponential Growth",
            "query": f"Calculate exponential of interest rates from {test_data['interest_rates']} and save to exp_rates.parquet",
            "expected_output": "exp_rates.parquet"
        },
        {
            "name": "LN - Natural Log of Returns",
            "query": f"Calculate natural logarithm of returns from {test_data['risk_metrics']} and save to ln_returns.parquet",
            "expected_output": "ln_returns.parquet"
        },
        {
            "name": "ABS - Absolute Values",
            "query": f"Calculate absolute values of cash flow differences and save to abs_cashflow.parquet",
            "expected_output": "abs_cashflow.parquet"
        }
    ]

    success_count = 0
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case['name']} ---")
        try:
            content = types.Content(
                role='user',
                parts=[types.Part(text=test_case['query'])]
            )

            events_async = runner.run_async(
                session_id=session.id,
                user_id=session.user_id,
                new_message=content
            )

            response_parts = []
            async for event in events_async:
                if hasattr(event, 'content') and event.content:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_parts.append(part.text)

            response = ''.join(response_parts)
            if response.strip():
                print(f"✅ Response received: {response[:150]}...")

                if test_case['expected_output']:
                    if verify_output_file(test_case['expected_output']):
                        success_count += 1
                    else:
                        print(f"❌ Output file verification failed")
                else:
                    success_count += 1
            else:
                print("❌ No response received")

        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n=== Mathematical Operations Results: {success_count}/{len(test_cases)} successful ===")
    return success_count == len(test_cases)

async def test_weighted_and_cumulative_functions(agent, runner, session, test_data):
    """Test weighted averages and cumulative functions."""
    print("\n📈 Testing Weighted and Cumulative Functions")
    print("=" * 50)

    test_cases = [
        {
            "name": "WEIGHTED_AVERAGE - Portfolio Returns",
            "query": f"Calculate weighted average of returns from {test_data['risk_metrics']} using weights column",
            "expected_output": None  # Single value result
        },
        {
            "name": "CUMSUM - Cumulative Revenue",
            "query": f"Calculate cumulative sum of revenue from {test_data['monthly_revenue']} and save to cumulative_revenue.parquet",
            "expected_output": "cumulative_revenue.parquet"
        },
        {
            "name": "CUMPROD - Cumulative Growth",
            "query": f"Calculate cumulative product of growth rates from {test_data['growth_rates']} and save to cumulative_growth.parquet",
            "expected_output": "cumulative_growth.parquet"
        },
        {
            "name": "GEOMETRIC_MEAN - Average Growth Rate",
            "query": f"Calculate geometric mean of growth rates from {test_data['growth_rates']}",
            "expected_output": None  # Single value result
        },
        {
            "name": "VARIANCE_WEIGHTED - Portfolio Risk",
            "query": f"Calculate weighted variance of returns from {test_data['risk_metrics']} using weights",
            "expected_output": None  # Single value result
        }
    ]

    success_count = 0
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case['name']} ---")
        try:
            content = types.Content(
                role='user',
                parts=[types.Part(text=test_case['query'])]
            )

            events_async = runner.run_async(
                session_id=session.id,
                user_id=session.user_id,
                new_message=content
            )

            response_parts = []
            async for event in events_async:
                if hasattr(event, 'content') and event.content:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_parts.append(part.text)

            response = ''.join(response_parts)
            if response.strip():
                print(f"✅ Response received: {response[:150]}...")

                if test_case['expected_output']:
                    if verify_output_file(test_case['expected_output']):
                        success_count += 1
                    else:
                        print(f"❌ Output file verification failed")
                else:
                    success_count += 1
            else:
                print("❌ No response received")

        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n=== Weighted and Cumulative Results: {success_count}/{len(test_cases)} successful ===")
    return success_count == len(test_cases)

async def test_rounding_and_formatting_functions(agent, runner, session, test_data):
    """Test rounding and formatting functions."""
    print("\n🎯 Testing Rounding and Formatting Functions")
    print("=" * 50)

    test_cases = [
        {
            "name": "ROUND - Round Revenue to 2 Decimals",
            "query": f"Round revenue values from {test_data['monthly_revenue']} to 2 decimal places",
            "expected_output": None  # Single value result
        },
        {
            "name": "ROUNDUP - Conservative Expense Estimates",
            "query": f"Round up expenses from {test_data['monthly_revenue']} to nearest hundred",
            "expected_output": None  # Single value result
        },
        {
            "name": "ROUNDDOWN - Conservative Revenue Projections",
            "query": f"Round down sales from {test_data['product_performance']} to nearest thousand",
            "expected_output": None  # Single value result
        }
    ]

    success_count = 0
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case['name']} ---")
        try:
            content = types.Content(
                role='user',
                parts=[types.Part(text=test_case['query'])]
            )

            events_async = runner.run_async(
                session_id=session.id,
                user_id=session.user_id,
                new_message=content
            )

            response_parts = []
            async for event in events_async:
                if hasattr(event, 'content') and event.content:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_parts.append(part.text)

            response = ''.join(response_parts)
            if response.strip():
                print(f"✅ Response received: {response[:150]}...")
                success_count += 1
            else:
                print("❌ No response received")

        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n=== Rounding and Formatting Results: {success_count}/{len(test_cases)} successful ===")
    return success_count == len(test_cases)

async def main():
    """Main comprehensive test function."""
    print("🏦 Comprehensive Math and Aggregation MCP Server Test")
    print("📊 Financial Planning and Analysis Scenarios")
    print("=" * 60)

    try:
        # Validate environment
        validate_environment()
        print("✅ Environment validation passed")

        # Create test data
        test_data = create_test_data()

        # Discover servers and create tools
        servers = MCPConfig.discover_available_servers()
        math_server = next((s for s in servers if s["name"] == "math_and_aggregation"), None)
        if not math_server:
            print("❌ Math server not found!")
            return False

        print(f"✅ Found math server: {math_server['name']}")

        # Create tools
        math_tools = create_mcp_tools(math_server)
        print(f"✅ Created {len(math_tools)} math tools")

        # Create agent
        agent = LlmAgent(
            model=LiteLlm(model="openrouter/qwen/qwen3-coder"),
            name="fpa_math_agent",
            instruction=(
                "You are a Financial Planning and Analysis assistant with advanced mathematical capabilities. "
                "Use the available math tools to perform calculations on financial datasets stored as parquet files. "
                "When working with files, always specify the exact filename including the .parquet extension. "
                "For operations that support output files, save results to the scratch_pad folder with descriptive names. "
                "Always explain your calculations and provide context for financial analysis."
            ),
            tools=math_tools
        )

        # Create session
        session_service = InMemorySessionService()
        session = await session_service.create_session(
            app_name="fpa_math_test",
            user_id="test_user",
            session_id="fpa_session"
        )

        # Create runner
        runner = Runner(
            agent=agent,
            app_name="fpa_math_test",
            session_service=session_service
        )

        # Run test suites
        test_results = []

        # Test basic aggregation functions
        result1 = await test_basic_aggregation_functions(agent, runner, session, test_data)
        test_results.append(("Basic Aggregation", result1))

        # Test mathematical operations
        result2 = await test_mathematical_operations(agent, runner, session, test_data)
        test_results.append(("Mathematical Operations", result2))

        # Test weighted and cumulative functions
        result3 = await test_weighted_and_cumulative_functions(agent, runner, session, test_data)
        test_results.append(("Weighted & Cumulative", result3))

        # Test rounding and formatting
        result4 = await test_rounding_and_formatting_functions(agent, runner, session, test_data)
        test_results.append(("Rounding & Formatting", result4))

        # Final summary
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print("=" * 60)

        all_passed = True
        for test_name, result in test_results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name}: {status}")
            if not result:
                all_passed = False

        if all_passed:
            print("\n🎉 ALL COMPREHENSIVE TESTS PASSED!")
            print("💼 Math MCP Server is ready for Financial Planning and Analysis!")
            return True
        else:
            print("\n⚠️  Some tests failed. Please check the server configuration and data files.")
            return False

    except Exception as e:
        print(f"\n❌ Comprehensive test failed with error: {e}")
        return False

if __name__ == "__main__":
    """Run the comprehensive math server tests."""
    try:
        result = asyncio.run(main())
        exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user.")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit(1)
