# Connected Google Sheets Workflow Plan

## Overview

This document outlines a comprehensive three-phase architecture for creating connected Google Sheets workflows based on user scenarios and data metadata. The system transforms natural language business requirements into fully automated, validated Google Sheets with formulas, charts, and live data connections.

## Architecture Rationale

### The Problem
Current FPA agents face several critical challenges:
1. **Formula Accuracy**: LLMs generate formula strings with ~70% accuracy (unacceptable for financial systems)
2. **Path Selection**: No intelligent routing between Fast Path (local), Sheets Path (formulas), or Dual Path
3. **API Efficiency**: Manual formula application leads to inefficient API usage
4. **Integration Complexity**: Disconnected tools without workflow orchestration

### The Solution
A **three-phase separation of concerns** system:
1. **Phase 1**: Business logic planning (LLM strength)
2. **Phase 2**: Tool discovery and mapping (LLM intelligence)
3. **Phase 3**: Deterministic execution (code reliability)

This eliminates formula string generation by agents while leveraging LLM intelligence for complex business-to-technical mapping.

## Phase 1: High-Level Planner Agent

### Purpose
Transform user scenarios into structured business logic plans using data metadata analysis.

### Components

#### 1. ScenarioAnalyzer
```python
class ScenarioAnalyzer:
    """Parses user scenarios and extracts business objectives"""
    
    def analyze_scenario(self, user_scenario: str) -> Dict[str, Any]:
        """
        Extract key business requirements from natural language scenario
        
        Returns:
        - business_objectives: List of high-level goals
        - data_requirements: What data is needed
        - visualization_needs: Chart/dashboard requirements
        - success_metrics: How to measure completion
        """
```

#### 2. MetadataIntegrator
```python
class MetadataIntegrator:
    """Uses existing metadata tools to understand data structure"""
    
    def __init__(self):
        # Leverage existing metadata functions
        self.get_metadata = get_metadata  # From tools/metadata.py
        self.get_metadata_one_file = get_metadata_one_file
    
    def analyze_data_context(self, data_files: List[str]) -> Dict[str, Any]:
        """
        Create comprehensive data context using existing metadata tools
        
        Returns:
        - schemas: Column types and structures
        - relationships: Potential join keys
        - data_quality: Null counts, unique values
        - suggestions: Recommended operations
        """
```

#### 3. BusinessPlanGenerator
```python
class BusinessPlanGenerator:
    """Creates structured high-level plans with dependencies"""
    
    def generate_plan(self, 
                     scenario: Dict[str, Any], 
                     data_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate business logic plan with step dependencies
        
        Output Format:
        {
            "plan_id": "uuid",
            "business_objective": "Calculate monthly profitability by region",
            "steps": [
                {
                    "step_id": "upload_data",
                    "description": "Upload sales and cost data to sheets",
                    "business_logic": "Make raw data available for analysis",
                    "dependencies": [],
                    "data_inputs": ["sales.csv", "costs.csv"],
                    "expected_outputs": ["SalesData sheet", "CostData sheet"]
                },
                {
                    "step_id": "calculate_revenue",
                    "description": "Calculate total revenue by region",
                    "business_logic": "Sum all sales amounts grouped by region",
                    "dependencies": ["upload_data"],
                    "requires_charts": false,
                    "validation_required": true
                },
                {
                    "step_id": "create_dashboard",
                    "description": "Create profit margin visualization",
                    "business_logic": "Show profit margins as column chart",
                    "dependencies": ["calculate_revenue", "calculate_costs"],
                    "chart_requirements": {
                        "type": "comparison",
                        "data_type": "financial_metrics"
                    }
                }
            ]
        }
        """
```

### Input/Output Example

**Input:**
```python
user_scenario = "I need to analyze Q4 sales performance by region and create a dashboard showing profit margins and top performing products"
data_files = ["data/q4_sales.csv", "data/product_costs.csv", "data/regions.csv"]
```

**Output:**
```json
{
    "plan_id": "plan_001",
    "business_objective": "Q4 sales performance analysis with profit dashboard",
    "steps": [
        {
            "step_id": "data_upload",
            "description": "Upload Q4 sales, costs, and region data",
            "business_logic": "Prepare data sources for analysis",
            "dependencies": [],
            "chart_requirements": null
        },
        {
            "step_id": "regional_analysis", 
            "description": "Calculate sales totals by region",
            "business_logic": "Aggregate sales performance by geographical region",
            "dependencies": ["data_upload"],
            "validation_required": true
        },
        {
            "step_id": "profit_calculation",
            "description": "Determine profit margins per region",
            "business_logic": "Calculate (revenue - costs) / revenue by region",
            "dependencies": ["regional_analysis"]
        },
        {
            "step_id": "dashboard_creation",
            "description": "Create profit margin visualization",
            "business_logic": "Display regional profit margins as interactive chart",
            "dependencies": ["profit_calculation"],
            "chart_requirements": {
                "type": "column_chart",
                "title": "Q4 Profit Margins by Region"
            }
        }
    ]
}
```

## Phase 2: LLM-Based Tool Discovery Agent

### Purpose
Convert high-level business plans into detailed executable plans using comprehensive tool discovery across ALL MCP servers.

### Key Innovation: Comprehensive Tool Discovery

Unlike simple formula discovery, this agent discovers **all types of tools**:

1. **Formula Builder Tools** (not formula strings!)
2. **Data Upload Tools**
3. **Chart Creation Tools** 
4. **Validation Tools**
5. **Structure Management Tools**

### Components

#### 1. ToolCatalogManager
```python
class ToolCatalogManager:
    """Maintains comprehensive catalog of all available tools"""
    
    def __init__(self):
        self.load_all_mcp_servers()
        
    def get_complete_catalog(self) -> Dict[str, Any]:
        """
        Returns comprehensive tool catalog structure:
        {
            "formula_tools": {
                "build_and_apply_sumif": {
                    "server": "sheets_formula_mcp",
                    "parameters": {...},
                    "capabilities": [...],
                    "use_cases": [...]
                }
            },
            "upload_tools": {
                "batch_upload_csvs_to_sheets": {
                    "server": "sheets_data_mcp",
                    "parameters": {...}
                }
            },
            "chart_tools": {
                "create_chart": {
                    "server": "sheets_chart_mcp", 
                    "chart_types": ["LINE", "COLUMN", "PIE", ...],
                    "parameters": {...}
                }
            },
            "validation_tools": {...},
            "structure_tools": {...}
        }
        """
```

#### 2. IntelligentMatcher  
```python
class IntelligentMatcher:
    """Uses LLM intelligence for business-to-tool mapping"""
    
    def match_business_logic_to_tools(self, 
                                    business_step: Dict[str, Any],
                                    available_tools: Dict[str, Any],
                                    data_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        LLM-based intelligent matching (NOT pattern matching)
        
        Example Input:
        business_step = {
            "business_logic": "Sum all sales where customer tier is 'High Value' by region",
            "data_inputs": ["SalesData sheet columns A-D"],
            "expected_output": "Regional high-value sales totals"
        }
        
        Example Output:
        [
            {
                "tool": "build_and_apply_sumifs",
                "confidence": 0.95,
                "reasoning": "SUMIFS perfect for multi-criteria aggregation",
                "parameters": {
                    "criteria_range1": "SalesData!C:C",  # region column
                    "criteria1": "{region_value}",
                    "criteria_range2": "SalesData!D:D",  # tier column
                    "criteria2": "High Value",
                    "sum_range": "SalesData!B:B",        # amount column
                    "output_cell": "Summary!B{row}"
                },
                "validation_tool": "sumif_tool",  # For local validation
                "chart_potential": true
            }
        ]
        """
```

#### 3. ExecutablePlanBuilder
```python
class ExecutablePlanBuilder:
    """Converts business plans to detailed executable plans"""
    
    def build_executable_plan(self, 
                            high_level_plan: Dict[str, Any],
                            tool_mappings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create detailed execution plan with resolved tools and parameters
        
        Output includes:
        - Exact tool calls with resolved parameters  
        - Dependency ordering for execution
        - Validation requirements for each step
        - Chart specifications with positioning
        - Error handling and rollback instructions
        """
```

### Formula Builder Architecture (Critical Innovation)

**Problem**: Agent formula string generation = 70% accuracy
**Solution**: Tools generate formulas, agents provide business parameters

#### Safe Formula Builder Tools
```python
# WRONG WAY (Current Sheets Path - Error Prone)
await apply_formula(
    spreadsheet_id="abc123",
    range_spec="B5", 
    formula="=SUMIFS(SalesData!B:B,SalesData!A:A,\"North\",SalesData!D:D,\">1000\")"
    #        ↑ AGENT GENERATED STRING - ERROR PRONE!
)

# RIGHT WAY (New Formula Builder Tools - Safe)
await build_and_apply_sumifs(
    spreadsheet_id="abc123",
    sum_range="SalesData!B:B",      # Business parameter
    criteria_range1="SalesData!A:A", # Business parameter  
    criteria1="North",               # Business parameter
    criteria_range2="SalesData!D:D", # Business parameter
    criteria2=">1000",              # Business parameter
    output_cell="Summary!B5"        # Business parameter
)
# Tool generates formula internally with 100% accuracy
```

#### Formula Builder Tool Implementation
```python
@mcp.tool()
async def build_and_apply_sumifs(
    spreadsheet_id: str,
    sum_range: str,
    criteria_ranges: List[str],
    criteria_values: List[str], 
    output_cell: str
) -> Dict[str, Any]:
    """
    Build and apply SUMIFS formula with guaranteed syntax accuracy.
    
    Agent provides business parameters, tool handles formula generation.
    """
    # Tool generates formula string with perfect syntax
    formula = FormulaBuilder.build_sumifs(
        sum_range=sum_range,
        criteria_pairs=list(zip(criteria_ranges, criteria_values))
    )
    
    # Apply to sheets using existing infrastructure
    result = await apply_formula_to_sheets(spreadsheet_id, output_cell, formula)
    
    return {
        'success': True,
        'formula_generated': formula,  # Perfect syntax
        'business_parameters': {
            'sum_range': sum_range,
            'criteria': dict(zip(criteria_ranges, criteria_values))
        },
        'output_location': output_cell
    }
```

### Tool Discovery Examples

#### Example 1: Revenue Analysis
**Business Logic**: "Calculate total revenue by product category"

**LLM Discovery Output**:
```json
{
    "primary_tool": {
        "name": "build_and_apply_sumif",
        "parameters": {
            "sum_range": "SalesData!C:C",
            "criteria_range": "SalesData!B:B", 
            "criteria": "Electronics",
            "output_cell": "Summary!B2"
        }
    },
    "validation_tool": {
        "name": "sumif_tool",
        "parameters": {
            "range_path": "data/sales.csv",
            "criteria": "Electronics"
        }
    },
    "chart_tool": {
        "name": "create_chart",
        "parameters": {
            "chart_type": "PIE",
            "data_range": "Summary!A2:B6",
            "title": "Revenue by Product Category"
        }
    }
}
```

#### Example 2: Trend Analysis  
**Business Logic**: "Show monthly sales trend with forecast"

**LLM Discovery Output**:
```json
{
    "data_preparation": {
        "name": "create_pivot_summary",
        "parameters": {
            "source_range": "SalesData!A:D",
            "row_fields": ["Month"],
            "value_fields": ["Sales_Amount"]
        }
    },
    "trend_visualization": {
        "name": "create_chart", 
        "parameters": {
            "chart_type": "LINE",
            "data_range": "PivotSummary!A:B",
            "title": "Monthly Sales Trend",
            "position": {"row": 2, "col": 5}
        }
    },
    "forecast_formula": {
        "name": "build_and_apply_trend",
        "parameters": {
            "historical_range": "PivotSummary!B2:B13",
            "forecast_periods": 3,
            "output_range": "Forecast!B2:B5"
        }
    }
}
```

## Phase 3: Deterministic Execution Engine

### Purpose
Execute detailed plans using existing infrastructure with guaranteed reliability, batching, and validation.

### Components

#### 1. SheetsFirstOrchestrator
```python
class SheetsFirstOrchestrator:
    """Manages entire workflow execution with Sheets-first strategy"""
    
    def __init__(self):
        self.batch_ops = BatchOperations(service)  # Existing infrastructure
        self.rate_limiter = RateLimiter()          # Existing infrastructure
        self.validator = ValidationEngine()
        
    async def execute_workflow(self, executable_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute complete workflow with:
        - Dependency resolution
        - Batch optimization  
        - Rate limiting
        - Step-by-step validation
        - Error recovery
        """
```

#### 2. BatchingEngine
```python
class BatchingEngine:
    """Leverages existing BatchOperations for API efficiency"""
    
    def __init__(self):
        self.batch_ops = BatchOperations(service)
        
    async def batch_execute_operations(self, operations: List[Dict]) -> Dict[str, Any]:
        """
        Group operations by spreadsheet and execute in optimized batches
        
        Uses existing:
        - batch_update() with rate limiting
        - optimize_requests() for logical ordering  
        - execute_with_retry() with exponential backoff
        """
        
        # Group by spreadsheet for efficiency
        grouped_ops = self._group_by_spreadsheet(operations)
        
        results = []
        for spreadsheet_id, ops in grouped_ops.items():
            # Convert to batch request format
            batch_requests = self._convert_to_batch_requests(ops)
            
            # Use existing batch operations with rate limiting
            result = self.batch_ops.execute_with_retry(
                spreadsheet_id, 
                batch_requests,
                max_retries=3
            )
            results.append(result)
            
        return self._consolidate_results(results)
```

#### 3. ValidationEngine
```python
class ValidationEngine:
    """Validates each step using local Polars computation"""
    
    def __init__(self):
        self.sheets_funcs = SheetsCompatibleFunctions()  # Existing
        
    async def validate_step(self, 
                          step_result: Dict[str, Any],
                          validation_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate Sheets results against local Polars computation
        
        Process:
        1. Execute same operation locally with Polars
        2. Read result from Google Sheets  
        3. Compare with tolerance for floating point
        4. Report validation success/failure
        """
        
        # Local computation using existing tools
        if validation_spec['type'] == 'sumif':
            local_result = self.sheets_funcs.SUMIF(
                data=validation_spec['data_path'],
                criteria=validation_spec['criteria']
            )
            
        # Read from Sheets
        sheets_result = await self._read_sheets_value(
            step_result['spreadsheet_id'],
            step_result['output_cell']
        )
        
        # Validate with tolerance
        return self._compare_results(local_result, sheets_result)
```

#### 4. ChartIntegrator
```python
class ChartIntegrator:
    """Integrates with existing chart MCP server"""
    
    def __init__(self):
        self.chart_service = SheetsChartService()  # Existing chart server
        
    async def create_workflow_charts(self, chart_specs: List[Dict]) -> List[Dict]:
        """
        Create charts using existing chart MCP server tools
        
        Uses existing:
        - create_chart() with all chart types
        - get_chart_types() for validation
        - Batch operations for multiple charts
        """
        
        results = []
        for chart_spec in chart_specs:
            # Use existing chart creation tool
            result = await self.chart_service.create_chart(
                spreadsheet_id=chart_spec['spreadsheet_id'],
                sheet_id=chart_spec['sheet_id'], 
                chart_type=chart_spec['chart_type'],  # LINE, COLUMN, PIE, etc.
                data_range=chart_spec['data_range'],
                title=chart_spec['title'],
                position=chart_spec.get('position')
            )
            results.append(result)
            
        return results
```

### Execution Flow Example

```python
# Input: Executable plan from Phase 2
executable_plan = {
    "steps": [
        {
            "step_id": "upload_data",
            "tools": [
                {
                    "name": "batch_upload_csvs_to_sheets",
                    "parameters": {...}
                }
            ]
        },
        {
            "step_id": "calculate_revenue",
            "tools": [
                {
                    "name": "build_and_apply_sumif", 
                    "parameters": {
                        "sum_range": "SalesData!B:B",
                        "criteria_range": "SalesData!A:A",
                        "criteria": "Electronics",
                        "output_cell": "Summary!B2"
                    }
                }
            ],
            "validation": {
                "type": "sumif",
                "data_path": "data/sales.csv", 
                "criteria": "Electronics"
            }
        },
        {
            "step_id": "create_chart",
            "tools": [
                {
                    "name": "create_chart",
                    "parameters": {
                        "chart_type": "COLUMN",
                        "data_range": "Summary!A2:B6",
                        "title": "Revenue by Category"
                    }
                }
            ]
        }
    ]
}

# Execution with existing infrastructure
orchestrator = SheetsFirstOrchestrator()
results = await orchestrator.execute_workflow(executable_plan)

# Output: Complete connected Google Sheets with formulas and charts
{
    "success": True,
    "spreadsheet_url": "https://docs.google.com/spreadsheets/d/...",
    "steps_completed": 3,
    "formulas_applied": 5,
    "charts_created": 2,
    "validations_passed": 5,
    "api_calls_used": 12,  # Optimized through batching
    "execution_time": "45 seconds"
}
```

## Integration with Existing Infrastructure

### 1. Metadata Tools Integration
```python
# Leverage existing metadata analysis
from tools.metadata import get_metadata, get_metadata_one_file

class MetadataIntegrator:
    def analyze_data_files(self, file_paths: List[str]) -> Dict[str, Any]:
        metadata = {}
        for file_path in file_paths:
            # Use existing metadata tool
            file_metadata = get_metadata_one_file(file_path)
            if file_metadata['success']:
                metadata[file_path] = file_metadata['metadata']
        return metadata
```

### 2. Chart Server Integration  
```python
# Use existing chart creation capabilities
from mcp_tooling.google_sheets.chart_server.sheets_chart_mcp import create_chart, get_chart_types

class ChartIntegrator:
    async def create_charts(self, chart_specs: List[Dict]) -> List[Dict]:
        # Get available chart types from existing server
        available_types = await get_chart_types()
        
        results = []
        for spec in chart_specs:
            if spec['chart_type'] in available_types['chart_types']:
                # Use existing chart creation tool
                result = await create_chart(
                    spreadsheet_id=spec['spreadsheet_id'],
                    sheet_id=spec['sheet_id'],
                    chart_type=spec['chart_type'],
                    data_range=spec['data_range'], 
                    title=spec['title'],
                    position=spec.get('position')
                )
                results.append(result)
        return results
```

### 3. Batch Operations Integration
```python
# Leverage existing batch operations and rate limiting
from mcp_tooling.google_sheets.api.batch_ops import BatchOperations
from mcp_tooling.google_sheets.config.rate_limits import USAGE_PATTERNS

class BatchingEngine:
    def __init__(self, usage_pattern='heavy'):
        self.batch_ops = BatchOperations(service)
        self.config = USAGE_PATTERNS[usage_pattern]
        
    async def execute_batched_operations(self, operations: List[Dict]) -> Dict:
        # Group operations by spreadsheet 
        grouped = self._group_operations(operations)
        
        results = []
        for spreadsheet_id, ops in grouped.items():
            # Convert to existing batch format
            batch_requests = self.batch_ops.optimize_requests(ops)
            
            # Execute with existing rate limiting and retry logic
            result = self.batch_ops.execute_with_retry(
                spreadsheet_id=spreadsheet_id,
                requests=batch_requests,
                max_retries=3
            )
            results.append(result)
            
        return self._consolidate_results(results)
```

### 4. Formula Mappings Integration
```python
# Extend existing formula mapping files
from mcp_tooling.formula_mappings import simple_formulas, complex_chains

class FormulaBuilder:
    def __init__(self):
        # Load existing formula mappings
        self.simple_mappings = self._load_json('formula_mappings/simple_formulas.json')
        self.complex_mappings = self._load_json('formula_mappings/complex_chains.json')
        
    def build_formula(self, formula_type: str, parameters: Dict) -> str:
        # Use existing mappings to ensure consistency
        if formula_type in self.simple_mappings:
            template = self.simple_mappings[formula_type]['sheets']
            return self._apply_parameters(template, parameters)
        # ... handle complex formulas
```

## Implementation Structure

```
mcp_tooling/
├── workflow_planning/
│   ├── __init__.py
│   ├── high_level_planner.py        # Phase 1: Business logic planning
│   │   ├── ScenarioAnalyzer
│   │   ├── MetadataIntegrator  
│   │   └── BusinessPlanGenerator
│   ├── tool_discovery_agent.py      # Phase 2: LLM-based tool discovery
│   │   ├── ToolCatalogManager
│   │   ├── IntelligentMatcher
│   │   └── ExecutablePlanBuilder
│   ├── plan_schemas.py              # Data structures and validation
│   └── tool_catalog.py              # Comprehensive tool catalog
├── workflow_execution/
│   ├── __init__.py
│   ├── execution_engine.py          # Phase 3: Deterministic execution  
│   │   ├── SheetsFirstOrchestrator
│   │   └── WorkflowExecutor
│   ├── batching_engine.py           # Batch operations optimization
│   ├── validation_engine.py         # Step-by-step validation
│   ├── chart_integrator.py          # Chart creation integration
│   └── formula_builder.py           # Safe formula generation
├── integration/
│   ├── __init__.py
│   ├── main_workflow_agent.py       # Orchestrates all three phases
│   └── workflow_api.py              # MCP tools for complete system
└── enhanced_formula_tools/          # New safe formula builder tools
    ├── __init__.py
    ├── aggregation_tools.py         # SUMIF, AVERAGEIF, etc. builders
    ├── lookup_tools.py              # VLOOKUP, INDEX/MATCH builders  
    ├── financial_tools.py           # NPV, IRR, etc. builders
    └── array_tools.py               # ARRAYFORMULA builders
```

## Data Flow Examples

### Complete Workflow Example

**Step 1: User Input**
```python
user_scenario = """
I need to analyze Q4 2024 sales data to understand:
1. Which regions performed best
2. What products drove the most revenue  
3. Profit margins by customer segment
4. Create executive dashboard with key charts
"""

data_files = [
    "data/q4_sales.csv",      # Contains: date, region, customer_id, product, amount
    "data/customers.csv",     # Contains: customer_id, segment, tier
    "data/product_costs.csv"  # Contains: product, unit_cost
]
```

**Step 2: Phase 1 - High-Level Planning**
```python
# Planner uses existing metadata tools
metadata = {}
for file in data_files:
    metadata[file] = get_metadata_one_file(file)

# LLM creates business plan
high_level_plan = {
    "business_objective": "Q4 2024 sales performance analysis with executive dashboard",
    "steps": [
        {
            "step_id": "data_preparation",
            "description": "Upload and organize Q4 sales data",
            "business_logic": "Make raw data available for analysis in structured sheets",
            "data_inputs": data_files,
            "dependencies": []
        },
        {
            "step_id": "regional_analysis", 
            "description": "Calculate sales totals by region",
            "business_logic": "Sum sales amounts grouped by region to identify top performers",
            "dependencies": ["data_preparation"],
            "validation_required": True
        },
        {
            "step_id": "product_revenue",
            "description": "Identify top revenue-driving products", 
            "business_logic": "Rank products by total revenue contribution",
            "dependencies": ["data_preparation"]
        },
        {
            "step_id": "profit_analysis",
            "description": "Calculate profit margins by customer segment",
            "business_logic": "Determine (revenue - costs) / revenue for each customer segment",
            "dependencies": ["data_preparation", "regional_analysis"],
            "requires_lookup": True
        },
        {
            "step_id": "executive_dashboard",
            "description": "Create visual dashboard with key metrics",
            "business_logic": "Present insights through charts for executive consumption",
            "dependencies": ["regional_analysis", "product_revenue", "profit_analysis"],
            "chart_requirements": [
                {"type": "column", "data": "regional_performance"},
                {"type": "pie", "data": "product_revenue_share"}, 
                {"type": "bar", "data": "profit_by_segment"}
            ]
        }
    ]
}
```

**Step 3: Phase 2 - Tool Discovery**
```python
# LLM agent discovers specific tools for each step
executable_plan = {
    "steps": [
        {
            "step_id": "data_preparation",
            "tools": [
                {
                    "name": "batch_upload_csvs_to_sheets",
                    "server": "sheets_data_mcp",
                    "parameters": {
                        "uploads": [
                            {
                                "csv_path": "data/q4_sales.csv",
                                "sheet_name": "SalesData",
                                "start_range": "A1"
                            },
                            {
                                "csv_path": "data/customers.csv", 
                                "sheet_name": "CustomerData",
                                "start_range": "A1"
                            },
                            {
                                "csv_path": "data/product_costs.csv",
                                "sheet_name": "CostData", 
                                "start_range": "A1"
                            }
                        ]
                    }
                }
            ]
        },
        {
            "step_id": "regional_analysis",
            "tools": [
                {
                    "name": "build_and_apply_sumif",
                    "server": "enhanced_formula_mcp",
                    "parameters": {
                        "sum_range": "SalesData!E:E",      # amount column
                        "criteria_range": "SalesData!B:B", # region column
                        "criteria": "North",
                        "output_cell": "RegionalSummary!B2"
                    },
                    "repeat_for_criteria": ["North", "South", "East", "West"]
                }
            ],
            "validation": {
                "tool": "sumif_tool", 
                "server": "conditional_mcp_server",
                "parameters": {
                    "range_path": "data/q4_sales.csv",
                    "criteria": "North"  # Validate each region
                }
            }
        },
        {
            "step_id": "product_revenue",
            "tools": [
                {
                    "name": "build_and_apply_sumif",
                    "server": "enhanced_formula_mcp", 
                    "parameters": {
                        "sum_range": "SalesData!E:E",      # amount
                        "criteria_range": "SalesData!D:D", # product  
                        "criteria": "{product_name}",      # Dynamic
                        "output_cell": "ProductSummary!B{row}"
                    },
                    "dynamic_execution": True
                }
            ]
        },
        {
            "step_id": "profit_analysis",
            "tools": [
                {
                    "name": "build_and_apply_vlookup",
                    "server": "enhanced_formula_mcp",
                    "parameters": {
                        "lookup_value": "RegionalSummary!A2",  # region
                        "table_array": "CostData!A:B",        # cost lookup
                        "col_index_num": 2,
                        "range_lookup": False,
                        "output_cell": "ProfitAnalysis!C2"
                    }
                },
                {
                    "name": "build_and_apply_formula",
                    "server": "enhanced_formula_mcp",
                    "parameters": {
                        "formula_type": "profit_margin",
                        "revenue_cell": "RegionalSummary!B2",
                        "cost_cell": "ProfitAnalysis!C2", 
                        "output_cell": "ProfitAnalysis!D2"
                    }
                }
            ]
        },
        {
            "step_id": "executive_dashboard",
            "tools": [
                {
                    "name": "create_chart",
                    "server": "sheets_chart_mcp",
                    "parameters": {
                        "chart_type": "COLUMN",
                        "data_range": "RegionalSummary!A2:B5",
                        "title": "Q4 Sales by Region",
                        "position": {"row": 2, "col": 6}
                    }
                },
                {
                    "name": "create_chart", 
                    "server": "sheets_chart_mcp",
                    "parameters": {
                        "chart_type": "PIE",
                        "data_range": "ProductSummary!A2:B11", 
                        "title": "Revenue Share by Product",
                        "position": {"row": 15, "col": 6}
                    }
                },
                {
                    "name": "create_chart",
                    "server": "sheets_chart_mcp", 
                    "parameters": {
                        "chart_type": "BAR",
                        "data_range": "ProfitAnalysis!A2:D5",
                        "title": "Profit Margins by Segment", 
                        "position": {"row": 28, "col": 6}
                    }
                }
            ]
        }
    ]
}
```

**Step 4: Phase 3 - Deterministic Execution**
```python
# Execute with existing infrastructure
orchestrator = SheetsFirstOrchestrator()
execution_result = await orchestrator.execute_workflow(executable_plan)

# Result: Complete connected Google Sheets
{
    "success": True,
    "execution_time": "2 minutes 15 seconds",
    "spreadsheet_id": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
    "spreadsheet_url": "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
    
    "sheets_created": [
        "SalesData", "CustomerData", "CostData",
        "RegionalSummary", "ProductSummary", "ProfitAnalysis", "Dashboard"
    ],
    
    "formulas_applied": [
        {"cell": "RegionalSummary!B2", "formula": "=SUMIF(SalesData!B:B,\"North\",SalesData!E:E)", "validated": True},
        {"cell": "RegionalSummary!B3", "formula": "=SUMIF(SalesData!B:B,\"South\",SalesData!E:E)", "validated": True},
        {"cell": "ProfitAnalysis!C2", "formula": "=VLOOKUP(A2,CostData!A:B,2,FALSE)", "validated": True},
        {"cell": "ProfitAnalysis!D2", "formula": "=(B2-C2)/B2", "validated": True}
    ],
    
    "charts_created": [
        {"title": "Q4 Sales by Region", "type": "COLUMN", "sheet": "Dashboard", "position": "F2"},
        {"title": "Revenue Share by Product", "type": "PIE", "sheet": "Dashboard", "position": "F15"},
        {"title": "Profit Margins by Segment", "type": "BAR", "sheet": "Dashboard", "position": "F28"}
    ],
    
    "validations_completed": [
        {"step": "regional_analysis", "status": "passed", "tolerance": 0.01},
        {"step": "product_revenue", "status": "passed", "tolerance": 0.01},
        {"step": "profit_analysis", "status": "passed", "tolerance": 0.01}
    ],
    
    "api_efficiency": {
        "total_operations": 47,
        "api_calls_made": 8,      # Optimized through batching
        "rate_limit_hits": 0,
        "average_batch_size": 5.9
    }
}
```

## Benefits and Success Criteria

### Key Benefits

1. **Financial-Grade Accuracy**
   - Formula strings generated by tools (100% accuracy)
   - No agent formula string generation (eliminates 70% error rate)
   - Step-by-step validation using local Polars computation
   - Error detection and rollback capabilities

2. **API Efficiency** 
   - Leverages existing BatchOperations infrastructure
   - Intelligent operation grouping by spreadsheet
   - Rate limiting prevents API quota exhaustion
   - Optimized request ordering for performance

3. **Separation of Concerns**
   - Phase 1: Pure business logic (LLM strength)
   - Phase 2: Intelligent tool mapping (LLM intelligence) 
   - Phase 3: Reliable execution (deterministic code)
   - Clear interfaces between phases

4. **Existing Infrastructure Leverage**
   - Uses all current metadata, chart, and batch operation tools
   - Extends formula mapping files
   - Integrates with all MCP servers
   - No duplication of existing functionality

5. **Scalability**
   - Handles complex multi-step workflows
   - Supports dependency resolution
   - Batch processing for large datasets
   - Chart integration for visualization requirements

### Success Criteria

#### Functional Requirements
- [ ] Agent accepts natural language scenarios
- [ ] System automatically analyzes provided data files
- [ ] Creates complete executable plans without human intervention
- [ ] Executes plans with existing batch operations and rate limiting
- [ ] Produces connected Google Sheets with live formula references
- [ ] Creates charts automatically based on scenario requirements
- [ ] Validates all computations against local Polars results

#### Performance Requirements  
- [ ] API calls optimized through batching (>80% reduction vs individual calls)
- [ ] Rate limiting prevents quota exhaustion (0 HTTP 429 errors)
- [ ] Formula accuracy: 100% (vs 70% with agent-generated strings)
- [ ] Execution time: <5 minutes for typical workflows
- [ ] Validation accuracy: 99.9% match between local and Sheets results

#### Integration Requirements
- [ ] Uses existing `get_metadata()` and `get_metadata_one_file()` tools
- [ ] Integrates with all existing MCP servers
- [ ] Leverages existing `BatchOperations` and rate limiting
- [ ] Uses existing chart creation tools from chart MCP server  
- [ ] Extends existing formula mapping JSON files
- [ ] Compatible with current ADK agent framework

#### Quality Requirements
- [ ] Deterministic execution (same input = same output)
- [ ] Error recovery and rollback capabilities
- [ ] Comprehensive logging for debugging
- [ ] Step-by-step progress tracking
- [ ] Clear error messages for troubleshooting

This comprehensive three-phase system transforms natural language business requirements into fully functional, connected Google Sheets with guaranteed accuracy and efficiency suitable for financial analysis workflows.

## Implementation Sequence

### Phase 0: Formula Builder Implementation (FIRST PRIORITY)

**Rationale**: Implement and test the formula builder tools first to ensure they're stable and available when the planning agents need them.

#### Implementation Steps:
1. **Create Formula Builder Infrastructure**
   - Implement `FormulaBuilder` class with safe formula generation
   - Create base classes for different formula types
   - Add comprehensive testing suite

2. **Implement Enhanced Formula Tools**
   - `build_and_apply_sumif()`, `build_and_apply_sumifs()`
   - `build_and_apply_vlookup()`, `build_and_apply_xlookup()` 
   - `build_and_apply_averageif()`, `build_and_apply_countif()`
   - Array formula builders (`build_and_apply_arrayformula()`)
   - Financial formula builders (`build_and_apply_npv()`, `build_and_apply_irr()`)

3. **Integration with Existing Infrastructure**
   - Integrate with existing `BatchOperations` and rate limiting
   - Use existing Google Sheets API authentication
   - Leverage existing error handling and retry mechanisms

4. **Comprehensive Testing**
   - Unit tests for each formula builder tool
   - Integration tests with real Google Sheets
   - Validation against existing local computation tools
   - Performance testing with batch operations

5. **MCP Server Creation**
   - Create new `enhanced_formula_mcp_server.py`
   - Expose all formula builder tools via MCP protocol
   - Add FastAPI endpoints for HTTP access
   - Update MCP server configuration

#### File Structure for Phase 0:
```
mcp_tooling/
├── enhanced_formula_tools/
│   ├── __init__.py
│   ├── formula_builder.py           # Core FormulaBuilder class
│   ├── aggregation_builders.py      # SUMIF, AVERAGEIF, COUNTIF builders
│   ├── lookup_builders.py           # VLOOKUP, INDEX/MATCH builders  
│   ├── financial_builders.py        # NPV, IRR, PMT builders
│   ├── array_builders.py            # ARRAYFORMULA builders
│   └── custom_builders.py           # Business-specific formula builders
├── enhanced_formula_mcp_server.py   # MCP server exposing formula builders
└── tests/
    ├── test_formula_builders.py     # Unit tests
    ├── test_enhanced_formula_mcp.py # Integration tests
    └── test_formula_validation.py   # Validation tests
```

#### Success Criteria for Phase 0:
- [ ] All formula builder tools implemented and tested
- [ ] 100% formula syntax accuracy verified
- [ ] Integration with existing batch operations confirmed
- [ ] MCP server created and accessible via HTTP
- [ ] Validation against local Polars computation passes
- [ ] Performance benchmarks meet requirements

### Revised Implementation Phases:

#### Phase 1: High-Level Planner Agent (SECOND)
- Implement after formula builders are stable
- Can reference and plan for available formula builder tools
- Reduced risk since formula infrastructure is proven

#### Phase 2: LLM-Based Tool Discovery Agent (THIRD)  
- Implement with full knowledge of available formula builder tools
- Tool catalog includes all tested formula builders
- Can confidently map business logic to proven formula tools

#### Phase 3: Deterministic Execution Engine (FOURTH)
- Implement using proven formula builders
- Execution engine has stable tools to work with
- Integration testing with end-to-end workflows

This revised sequence ensures that the critical formula builder foundation is solid before building the planning and discovery layers on top of it.

## Multi-Platform Architecture Considerations

### Overview

This architecture is designed with multi-platform support in mind. The three-phase separation of concerns provides excellent foundation for extending to Excel, Notion, Airtable, or other spreadsheet platforms with minimal changes to the core business logic.

### Impact Analysis for Excel Adaptation

#### **Phase 0: Formula Builder Implementation** 
**Impact: MAJOR CHANGES**
- **API Changes**: Replace Google Sheets API with Microsoft Graph API
- **Authentication**: Switch from Google OAuth to Microsoft OAuth/Azure AD  
- **Formula Syntax**: Minor differences in Excel vs Google Sheets formulas
- **Batch Operations**: Different batching mechanisms in Microsoft Graph API
- **Rate Limiting**: Different quota limits and patterns

#### **Phase 1: High-Level Planner Agent**
**Impact: MINIMAL TO NONE**
- **Business Logic**: Completely unchanged - still analyzes scenarios and creates business plans
- **Metadata Integration**: Still uses same Polars-based metadata analysis
- **Plan Structure**: Identical JSON structure for business objectives and dependencies
- **LLM Planning**: Same LLM capabilities for business requirement analysis

#### **Phase 2: LLM-Based Tool Discovery Agent** 
**Impact: MODERATE**
- **Tool Catalog**: References Excel-specific formula builders instead of Google Sheets
- **LLM Intelligence**: Same mapping logic from business requirements to tools
- **Plan Structure**: Same executable plan format
- **Chart Discovery**: Maps to Excel chart types instead of Google Sheets chart types

#### **Phase 3: Deterministic Execution Engine**
**Impact: MODERATE** 
- **Execution Logic**: Same workflow orchestration and dependency resolution
- **Validation Engine**: Same Polars validation logic (unchanged)
- **Error Handling**: Same patterns but different API error codes
- **Chart Integration**: Different chart creation API calls

### Platform-Agnostic Architecture Design

#### **Abstract Base Classes**
```python
# Abstract interfaces for platform independence
class AbstractFormulaBuilder:
    """Base class for platform-specific formula builders"""
    @abstractmethod
    def build_sumif(self, criteria_range: str, criteria: str, sum_range: str) -> str:
        pass
    
    @abstractmethod 
    def build_vlookup(self, lookup_value: str, table_array: str, col_index: int) -> str:
        pass

class AbstractAPIClient:
    """Base class for platform-specific API clients"""
    @abstractmethod
    async def batch_update(self, file_id: str, requests: List[Dict]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def authenticate(self) -> Any:
        pass

class AbstractChartCreator:
    """Base class for platform-specific chart creation"""
    @abstractmethod 
    async def create_chart(self, file_id: str, chart_type: str, data_range: str, title: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def get_supported_chart_types(self) -> List[str]:
        pass
```

#### **Platform-Specific Implementations**
```python
# Google Sheets Implementation (Phase 0 Initial Target)
class GoogleSheetsFormulaBuilder(AbstractFormulaBuilder):
    def build_sumif(self, criteria_range: str, criteria: str, sum_range: str) -> str:
        return f"=SUMIF({criteria_range},\"{criteria}\",{sum_range})"
    
    def build_vlookup(self, lookup_value: str, table_array: str, col_index: int) -> str:
        return f"=VLOOKUP({lookup_value},{table_array},{col_index},FALSE)"

class GoogleSheetsAPIClient(AbstractAPIClient):
    def __init__(self):
        self.service = GoogleSheetsAuth().authenticate()
        self.batch_ops = BatchOperations(self.service)
    
    async def batch_update(self, spreadsheet_id: str, requests: List[Dict]) -> Dict[str, Any]:
        return self.batch_ops.execute_with_retry(spreadsheet_id, requests)

# Excel Implementation (Future Extension)
class ExcelFormulaBuilder(AbstractFormulaBuilder):
    def build_sumif(self, criteria_range: str, criteria: str, sum_range: str) -> str:
        # Excel syntax is nearly identical to Google Sheets
        return f"=SUMIF({criteria_range},\"{criteria}\",{sum_range})"
    
    def build_vlookup(self, lookup_value: str, table_array: str, col_index: int) -> str:
        # Excel uses same VLOOKUP syntax
        return f"=VLOOKUP({lookup_value},{table_array},{col_index},FALSE)"

class ExcelAPIClient(AbstractAPIClient):
    def __init__(self):
        self.graph_client = self._create_graph_client()
    
    async def batch_update(self, workbook_id: str, requests: List[Dict]) -> Dict[str, Any]:
        # Microsoft Graph API batch operations
        return await self.graph_client.workbooks[workbook_id].batch_patch(requests)
```

#### **Configuration-Driven Platform Selection**
```python
class WorkflowOrchestrator:
    """Platform-agnostic workflow orchestrator"""
    
    def __init__(self, platform: str = "google_sheets"):
        self.platform = platform
        
        # Factory pattern for platform-specific components
        if platform == "google_sheets":
            self.formula_builder = GoogleSheetsFormulaBuilder()
            self.api_client = GoogleSheetsAPIClient()
            self.chart_creator = GoogleSheetsChartCreator()
        elif platform == "excel":
            self.formula_builder = ExcelFormulaBuilder()
            self.api_client = ExcelAPIClient()  
            self.chart_creator = ExcelChartCreator()
        else:
            raise ValueError(f"Unsupported platform: {platform}")
    
    async def execute_workflow(self, executable_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow using platform-specific implementations"""
        # Same workflow logic works for all platforms
        results = []
        
        for step in executable_plan['steps']:
            for tool_spec in step['tools']:
                if tool_spec['name'].startswith('build_and_apply_'):
                    # Use platform-specific formula builder
                    formula = self.formula_builder.build_formula(
                        tool_spec['formula_type'],
                        tool_spec['parameters']
                    )
                    
                    # Use platform-specific API client
                    result = await self.api_client.apply_formula(
                        tool_spec['file_id'],
                        tool_spec['output_cell'],
                        formula
                    )
                    results.append(result)
        
        return self._consolidate_results(results)
```

### Platform-Specific Differences

#### **API Differences**
```python
# Google Sheets API Structure
google_batch_request = {
    "requests": [
        {
            "updateCells": {
                "range": {"sheetId": 0, "startRowIndex": 1, "startColumnIndex": 1},
                "rows": [{"values": [{"userEnteredValue": {"formulaValue": "=SUM(A1:A10)"}}]}],
                "fields": "userEnteredValue"
            }
        }
    ]
}

# Microsoft Graph API Structure  
excel_batch_request = {
    "requests": [
        {
            "id": "1",
            "method": "PATCH",
            "url": f"/workbooks/{workbook_id}/worksheets/{sheet_id}/range(address='B2')",
            "body": {
                "values": [["=SUM(A1:A10)"]]
            }
        }
    ]
}
```

#### **Formula Syntax Differences**
```python
class FormulaSyntaxMapper:
    """Handle minor syntax differences between platforms"""
    
    GOOGLE_TO_EXCEL_MAPPINGS = {
        "ARRAYFORMULA": "dynamic_array",  # Excel uses dynamic arrays differently
        "QUERY": "FILTER",                # Google QUERY -> Excel FILTER  
        "IMPORTRANGE": "external_reference" # Different external reference syntax
    }
    
    def adapt_formula_for_platform(self, formula: str, target_platform: str) -> str:
        """Adapt formula syntax for target platform"""
        if target_platform == "excel" and "ARRAYFORMULA" in formula:
            # Convert Google Sheets ARRAYFORMULA to Excel dynamic array
            return formula.replace("=ARRAYFORMULA(", "=(").rstrip(")")
        
        return formula
```

#### **Chart Type Mappings**
```python
PLATFORM_CHART_MAPPINGS = {
    "google_sheets": {
        "column": "COLUMN",
        "line": "LINE", 
        "pie": "PIE",
        "bar": "BAR",
        "scatter": "SCATTER"
    },
    "excel": {
        "column": "ColumnClustered",
        "line": "Line",
        "pie": "Pie", 
        "bar": "BarClustered",
        "scatter": "XYScatter"
    }
}

class ChartTypeAdapter:
    def get_platform_chart_type(self, generic_type: str, platform: str) -> str:
        return PLATFORM_CHART_MAPPINGS[platform][generic_type]
```

### Implementation Impact Summary

| Phase | Impact Level | Changes Required |
|-------|-------------|------------------|
| **Phase 0: Formula Builders** | 🔴 **MAJOR** | New API client, authentication, batch operations |
| **Phase 1: High-Level Planner** | 🟢 **NONE** | Business logic is platform-agnostic |
| **Phase 2: Tool Discovery** | 🟡 **MINOR** | Tool catalog references, chart type mapping |
| **Phase 3: Execution Engine** | 🟡 **MODERATE** | API calls, error handling, chart integration |

### Multi-Platform Implementation Strategy

#### **Phase 0 Multi-Platform Implementation**
1. **Create Abstract Interfaces** for all platform-specific operations
2. **Implement Google Sheets First** (as planned in current Phase 0)
3. **Design Platform Factory Pattern** during Google Sheets implementation
4. **Add Excel Implementation** using same interfaces (future phase)
5. **Shared Testing Suite** that validates both platforms

#### **Shared Components Across Platforms**
```python
# These components work identically across all platforms
class MetadataIntegrator:           # ✅ Platform-agnostic
class ScenarioAnalyzer:            # ✅ Platform-agnostic  
class BusinessPlanGenerator:       # ✅ Platform-agnostic
class ValidationEngine:            # ✅ Uses same Polars validation
class IntelligentMatcher:          # ✅ Same LLM logic, different tool catalogs
```

#### **Platform-Specific Components**
```python
# These components need platform implementations
class FormulaBuilder:              # 🔄 Platform-specific API calls
class APIClient:                   # 🔄 Platform-specific authentication & requests  
class ChartCreator:               # 🔄 Platform-specific chart APIs
class BatchOperations:            # 🔄 Platform-specific batching mechanisms
```

### Benefits of Multi-Platform Architecture

#### **Code Reuse**
- **80% Shared Logic**: Business planning, tool discovery, and validation logic
- **20% Platform-Specific**: Only API calls, authentication, and platform syntax
- **Single Codebase**: One system supporting multiple platforms

#### **Consistent User Experience** 
- **Same Workflow**: User provides scenarios and data files regardless of platform
- **Identical Results**: Same business logic produces equivalent outputs
- **Unified Interface**: Single API supports multiple backend platforms

#### **Easy Extension**
- **New Platforms**: Framework ready for Notion, Airtable, LibreOffice Calc
- **Pluggable Architecture**: Add new platforms without changing core logic
- **Configuration-Driven**: Switch platforms via configuration parameter

#### **Shared Validation**
- **Same Polars Logic**: Local validation works identically for all platforms
- **Consistent Accuracy**: Same validation tolerances across platforms
- **Unified Testing**: Test business logic once, validate on all platforms

### Future Platform Extension Examples

#### **Notion Database Integration**
```python
class NotionFormulaBuilder(AbstractFormulaBuilder):
    def build_sumif(self, criteria_range: str, criteria: str, sum_range: str) -> Dict:
        # Notion uses database filters and rollups instead of formulas
        return {
            "type": "rollup",
            "filter": {"property": criteria_range, "text": {"equals": criteria}},
            "aggregation": "sum",
            "relation_property": sum_range
        }
```

#### **Airtable Integration**
```python
class AirtableFormulaBuilder(AbstractFormulaBuilder):
    def build_sumif(self, criteria_range: str, criteria: str, sum_range: str) -> str:
        # Airtable formula syntax similar to Excel/Google
        return f"SUMIF({{{criteria_range}}}, '{criteria}', {{{sum_range}}})"
```

This multi-platform architecture ensures that the significant investment in business logic, LLM intelligence, and workflow orchestration can be leveraged across multiple spreadsheet platforms with minimal additional development effort.