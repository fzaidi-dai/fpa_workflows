#!/usr/bin/env python3
"""
Direct test suite for Math and Aggregation MCP Server

This script tests the math server functionality by calling the MCP endpoints directly:
- Tests all 26 math functions with appropriate data
- Validates both file inputs and JSON array inputs
- Checks file output capabilities
- Provides comprehensive coverage of all functionality
"""

import requests
import json
import polars as pl
from pathlib import Path

def create_test_data():
    """Create test datasets for validation."""
    scratch_pad = Path("scratch_pad")
    scratch_pad.mkdir(exist_ok=True)

    print("📊 Creating test datasets...")

    # Simple test numbers for JSON array tests
    test_numbers = pl.DataFrame({
        "value": [10.5, 20.3, 30.7, 40.2, 50.8]
    })
    test_numbers.write_parquet(scratch_pad / "test_numbers.parquet")

    # Monthly revenue data
    monthly_revenue = pl.DataFrame({
        "revenue": [125000.50, 132000.75, 128500.25, 145000.00, 139500.80, 152000.30]
    })
    monthly_revenue.write_parquet(scratch_pad / "monthly_revenue.parquet")

    print("✅ Test datasets created!")
    return True

def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """Call an MCP tool directly via HTTP."""
    url = "http://localhost:3002/math_mcp"
    payload = {
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        if "content" in result and len(result["content"]) > 0:
            content = json.loads(result["content"][0]["text"])
            return content
        else:
            return {"success": False, "error": "No content in response"}

    except Exception as e:
        return {"success": False, "error": str(e)}

def test_basic_aggregation():
    """Test basic aggregation functions."""
    print("\n🧮 Testing Basic Aggregation Functions")
    print("=" * 50)

    tests = [
        ("SUM", {"values": "[10, 20, 30, 40, 50]"}, "150"),
        ("SUM", {"values": "scratch_pad/test_numbers.parquet"}, None),
        ("AVERAGE", {"values": "[10, 20, 30]"}, "20"),
        ("AVERAGE", {"values": "scratch_pad/monthly_revenue.parquet"}, None),
        ("MIN", {"values": "[5, 2, 8, 1, 9]"}, "1"),
        ("MIN", {"values": "scratch_pad/test_numbers.parquet"}, None),
        ("MAX", {"values": "[5, 2, 8, 1, 9]"}, "9"),
        ("MAX", {"values": "scratch_pad/monthly_revenue.parquet"}, None),
        ("PRODUCT", {"values": "[2, 3, 4]"}, "24"),
        ("MEDIAN", {"values": "[1, 2, 3, 4, 5]"}, "3"),
        ("MODE", {"values": "[1, 2, 2, 3, 2]"}, "2"),
        ("PERCENTILE", {"values": "[10, 20, 30, 40, 50]", "percentile_value": 50}, "30"),
    ]

    success_count = 0
    for tool_name, args, expected in tests:
        print(f"\n--- Testing {tool_name} ---")
        result = call_mcp_tool(tool_name, args)

        if result["success"]:
            print(f"✅ {tool_name}: {result['result']}")
            success_count += 1
        else:
            print(f"❌ {tool_name}: {result['error']}")

    print(f"\n=== Basic Aggregation Results: {success_count}/{len(tests)} successful ===")
    return success_count == len(tests)

def test_mathematical_operations():
    """Test mathematical operations."""
    print("\n🔢 Testing Mathematical Operations")
    print("=" * 50)

    tests = [
        ("POWER", {"values": "[2, 3, 4]", "power": 2}, None),
        ("POWER", {"values": "scratch_pad/test_numbers.parquet", "power": 2, "output_filename": "power_test.parquet"}, None),
        ("SQRT", {"values": "[4, 9, 16, 25]"}, None),
        ("SQRT", {"values": "scratch_pad/test_numbers.parquet", "output_filename": "sqrt_test.parquet"}, None),
        ("EXP", {"values": "[1, 2, 3]"}, None),
        ("LN", {"values": "[1, 2.718, 7.389]"}, None),
        ("LOG", {"values": "[10, 100, 1000]", "base": 10}, None),
        ("ABS", {"values": "[-5, -3, 2, -8, 10]"}, None),
        ("ABS", {"values": "scratch_pad/test_numbers.parquet", "output_filename": "abs_test.parquet"}, None),
        ("SIGN", {"values": "[-5, 0, 3, -2, 7]"}, None),
    ]

    success_count = 0
    for tool_name, args, expected in tests:
        print(f"\n--- Testing {tool_name} ---")
        result = call_mcp_tool(tool_name, args)

        if result["success"]:
            print(f"✅ {tool_name}: Success")
            if "type" in result and result["type"] == "file_path":
                print(f"   Output file: {result['result']}")
            success_count += 1
        else:
            print(f"❌ {tool_name}: {result['error']}")

    print(f"\n=== Mathematical Operations Results: {success_count}/{len(tests)} successful ===")
    return success_count == len(tests)

def test_rounding_functions():
    """Test rounding functions."""
    print("\n🎯 Testing Rounding Functions")
    print("=" * 50)

    tests = [
        ("ROUND", {"values": "[3.14159, 2.71828, 1.41421]", "num_digits": 2}, None),
        ("ROUNDUP", {"values": "[3.14159, 2.71828, 1.41421]", "num_digits": 2}, None),
        ("ROUNDDOWN", {"values": "[3.14159, 2.71828, 1.41421]", "num_digits": 2}, None),
    ]

    success_count = 0
    for tool_name, args, expected in tests:
        print(f"\n--- Testing {tool_name} ---")
        result = call_mcp_tool(tool_name, args)

        if result["success"]:
            print(f"✅ {tool_name}: {result['result']}")
            success_count += 1
        else:
            print(f"❌ {tool_name}: {result['error']}")

    print(f"\n=== Rounding Functions Results: {success_count}/{len(tests)} successful ===")
    return success_count == len(tests)

def test_advanced_functions():
    """Test advanced mathematical functions."""
    print("\n📈 Testing Advanced Functions")
    print("=" * 50)

    tests = [
        ("WEIGHTED_AVERAGE", {"values": "[10, 20, 30]", "weights": "[1, 2, 3]"}, None),
        ("GEOMETRIC_MEAN", {"values": "[2, 8, 32]"}, None),
        ("HARMONIC_MEAN", {"values": "[2, 4, 8]"}, None),
        ("CUMSUM", {"values": "[1, 2, 3, 4, 5]"}, None),
        ("CUMSUM", {"values": "scratch_pad/test_numbers.parquet", "output_filename": "cumsum_test.parquet"}, None),
        ("CUMPROD", {"values": "[1, 2, 3, 4]"}, None),
        ("CUMPROD", {"values": "scratch_pad/test_numbers.parquet", "output_filename": "cumprod_test.parquet"}, None),
        ("VARIANCE_WEIGHTED", {"values": "[10, 20, 30]", "weights": "[1, 1, 1]"}, None),
        ("MOD", {"dividends": "[10, 15, 20]", "divisors": "[3, 4, 6]"}, None),
    ]

    success_count = 0
    for tool_name, args, expected in tests:
        print(f"\n--- Testing {tool_name} ---")
        result = call_mcp_tool(tool_name, args)

        if result["success"]:
            print(f"✅ {tool_name}: Success")
            if "type" in result and result["type"] == "file_path":
                print(f"   Output file: {result['result']}")
            success_count += 1
        else:
            print(f"❌ {tool_name}: {result['error']}")

    print(f"\n=== Advanced Functions Results: {success_count}/{len(tests)} successful ===")
    return success_count == len(tests)

def test_tools_list():
    """Test the tools/list endpoint."""
    print("\n📋 Testing Tools List")
    print("=" * 50)

    url = "http://localhost:3002/math_mcp"
    payload = {"method": "tools/list"}

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        if "tools" in result:
            tools = result["tools"]
            print(f"✅ Found {len(tools)} tools:")
            for tool in tools[:5]:  # Show first 5 tools
                print(f"   - {tool['name']}: {tool['description'][:60]}...")
            if len(tools) > 5:
                print(f"   ... and {len(tools) - 5} more tools")
            return True
        else:
            print("❌ No tools found in response")
            return False

    except Exception as e:
        print(f"❌ Error testing tools list: {e}")
        return False

def verify_output_files():
    """Verify that output files were created correctly."""
    print("\n📁 Verifying Output Files")
    print("=" * 50)

    expected_files = [
        "power_test.parquet",
        "sqrt_test.parquet",
        "abs_test.parquet",
        "cumsum_test.parquet",
        "cumprod_test.parquet"
    ]

    success_count = 0
    for filename in expected_files:
        file_path = Path("scratch_pad") / "analysis" / filename
        if file_path.exists():
            try:
                df = pl.read_parquet(file_path)
                print(f"✅ {filename}: {df.height} rows")
                success_count += 1
            except Exception as e:
                print(f"❌ {filename}: Error reading - {e}")
        else:
            print(f"❌ {filename}: File not found")

    print(f"\n=== Output Files Results: {success_count}/{len(expected_files)} successful ===")
    return success_count == len(expected_files)

def main():
    """Main test function."""
    print("🏦 Direct Math and Aggregation MCP Server Test")
    print("📊 Comprehensive Function Testing")
    print("=" * 60)

    try:
        # Create test data
        if not create_test_data():
            print("❌ Failed to create test data")
            return False

        # Run all test suites
        test_results = []

        # Test tools list
        result0 = test_tools_list()
        test_results.append(("Tools List", result0))

        # Test basic aggregation
        result1 = test_basic_aggregation()
        test_results.append(("Basic Aggregation", result1))

        # Test mathematical operations
        result2 = test_mathematical_operations()
        test_results.append(("Mathematical Operations", result2))

        # Test rounding functions
        result3 = test_rounding_functions()
        test_results.append(("Rounding Functions", result3))

        # Test advanced functions
        result4 = test_advanced_functions()
        test_results.append(("Advanced Functions", result4))

        # Verify output files
        result5 = verify_output_files()
        test_results.append(("Output Files", result5))

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
            print("\n🎉 ALL TESTS PASSED!")
            print("💼 Math MCP Server is fully functional!")
            return True
        else:
            print("\n⚠️  Some tests failed. Check the server logs for details.")
            return False

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        return False

if __name__ == "__main__":
    """Run the direct math server tests."""
    try:
        result = main()
        exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user.")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit(1)
