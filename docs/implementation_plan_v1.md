# FPA Agents Implementation Plan v1.0

## Executive Summary

This document outlines the comprehensive implementation plan for the Financial Planning & Analysis (FP&A) AI application. The system combines Google ADK agents with MCP (Model Context Protocol) servers to provide AI-powered financial analysis through Google Sheets, using a dual-layer hybrid architecture that leverages Polars for computation and Google Sheets for formula transparency.

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Phase 1: Google Sheets MCP Server Foundation](#phase-1-google-sheets-mcp-server-foundation)
3. [Phase 2: Polars Integration Enhancement](#phase-2-polars-integration-enhancement)
4. [Phase 3: Multi-Agent Workflow System](#phase-3-multi-agent-workflow-system)
5. [Phase 4: Plan Execution System](#phase-4-plan-execution-system)
6. [Phase 5: Validation & Testing](#phase-5-validation--testing)
7. [Phase 6: Production Readiness](#phase-6-production-readiness)
8. [Technical Specifications](#technical-specifications)
9. [Risk Mitigation](#risk-mitigation)
10. [Success Metrics](#success-metrics)

## System Architecture Overview

### Core Design Principles

1. **Dual-Layer Hybrid Architecture**: Polars for computation, Google Sheets for presentation
2. **Formula Transparency**: All calculations visible as Google Sheets formulas
3. **Tool Reduction Strategy**: Multiple specialized MCP servers to prevent LLM overwhelm
4. **State Management**: Comprehensive checkpointing and recovery mechanisms
5. **Validation-First**: Pre and post-execution validation at every step

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                          │
│                    (Chat-like Interface)                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Google ADK Agents                         │
├─────────────────────────────────────────────────────────────┤
│  • User-Interfacing Agent                                    │
│  • High-Level Planning Agent                                 │
│  • Execution-Level Planning Agent                            │
│  • Execution Agent                                           │
│  • Validation Agent                                          │
│  • Error Handling Sub-Agents                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      MCP Servers Layer                       │
├─────────────────────────────────────────────────────────────┤
│  Google Sheets MCP Servers:                                  │
│  • Structure Server    • Data Server                         │
│  • Formula Server      • Formatting Server                   │
│  • Chart Server        • Validation Server                   │
│                                                               │
│  Existing MCP Servers:                                       │
│  • Filesystem MCP      • Math & Aggregation MCP              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Computation & Storage                      │
├─────────────────────────────────────────────────────────────┤
│  • Polars DataFrames (Local Computation)                     │
│  • Google Sheets (Presentation & Formulas)                   │
│  • File Storage (/data/, /scratch_pad/)                      │
└─────────────────────────────────────────────────────────────┘
```

## Phase 1: Google Sheets MCP Server Foundation

**Timeline: Weeks 1-3**  
**Priority: Critical**

### 1.1 Directory Structure Setup

Create the following directory structure:

```
/mcp_tooling/google_sheets/
├── api/                           # Low-level Google Sheets API wrapper
│   ├── __init__.py
│   ├── auth.py                   # Authentication & service initialization
│   ├── spreadsheet_ops.py        # Core spreadsheet operations
│   ├── value_ops.py              # Cell/range value operations
│   ├── format_ops.py             # Formatting operations
│   ├── formula_ops.py            # Formula application operations
│   ├── batch_ops.py              # Batch operation utilities
│   ├── chart_ops.py              # Google Charts API integration
│   ├── named_range_ops.py        # Named ranges management
│   ├── validation_ops.py         # Data validation rules
│   ├── range_resolver.py         # Unified range handling
│   └── error_handler.py          # Google API error handling
│
├── structure_server/              # MCP Server 1: Spreadsheet structure
│   ├── __init__.py
│   └── sheets_structure_mcp.py
│
├── data_server/                   # MCP Server 2: Data operations
│   ├── __init__.py
│   └── sheets_data_mcp.py
│
├── formatting_server/             # MCP Server 3: Formatting & styling
│   ├── __init__.py
│   └── sheets_formatting_mcp.py
│
├── formula_server/                # MCP Server 4: Formula application
│   ├── __init__.py
│   └── sheets_formula_mcp.py
│
├── chart_server/                  # MCP Server 5: Chart generation
│   ├── __init__.py
│   └── sheets_chart_mcp.py
│
├── validation_server/             # MCP Server 6: Validation rules
│   ├── __init__.py
│   └── sheets_validation_mcp.py
│
├── formula_docs/                  # Formula documentation for agents
│   ├── array_formulas.md
│   ├── date_formulas.md
│   ├── financial_formulas.md
│   ├── lookup_formulas.md
│   ├── math_formulas.md
│   ├── statistical_formulas.md
│   └── text_formulas.md
│
└── formula_mappings/              # Polars to Sheets translations
    ├── simple_formulas.json
    ├── array_formulas.json
    ├── pivot_formulas.json
    ├── financial_formulas.json
    └── complex_chains.json
```

### 1.2 Core API Implementation

#### Authentication Module (`auth.py`)
```python
class GoogleSheetsAuth:
    """Handles Google Sheets API authentication"""
    
    def __init__(self):
        self.service = None
        self.credentials = None
    
    def authenticate_service_account(self, credentials_path: str):
        """Authenticate using service account"""
        
    def authenticate_oauth(self, client_secrets_path: str):
        """Authenticate using OAuth 2.0"""
        
    def get_service(self):
        """Returns authenticated service instance"""
```

#### Range Resolver (`range_resolver.py`)
```python
class RangeResolver:
    """Unified range handling for Polars and Google Sheets"""
    
    @staticmethod
    def sheets_to_polars(range_spec: str) -> tuple:
        """Convert A1:C10 to Polars slice indices"""
        
    @staticmethod
    def polars_to_sheets(df: pl.DataFrame, 
                        row_slice: slice, 
                        col_slice: slice) -> str:
        """Convert Polars slice to A1 notation"""
        
    @staticmethod
    def resolve_range(df: pl.DataFrame, 
                     range_spec: Union[str, dict]) -> pl.DataFrame:
        """Apply range specification to DataFrame"""
```

#### Batch Operations (`batch_ops.py`)
```python
class BatchOptimizer:
    """Optimizes Google Sheets API calls"""
    
    def __init__(self, max_batch_size: int = 100):
        self.queue = []
        self.max_batch_size = max_batch_size
        self.rate_limiter = RateLimiter(
            max_requests=100,
            window_seconds=100
        )
    
    def add_operation(self, operation: dict):
        """Add operation to batch queue"""
        
    def execute_batch(self):
        """Execute all queued operations"""
        
    def optimize_operations(self, operations: list) -> list:
        """Optimize operation order for efficiency"""
```

### 1.3 MCP Server Implementation

Each MCP server will follow this pattern:

```python
# Example: structure_server/sheets_structure_mcp.py
from fastmcp import FastMCP
from ..api import GoogleSheetsAuth, spreadsheet_ops

mcp = FastMCP("Google Sheets Structure Server")

@mcp.tool()
async def create_spreadsheet(
    title: str,
    initial_sheets: list[str] = None
) -> dict:
    """Create a new Google Sheets spreadsheet"""
    
@mcp.tool()
async def add_sheet(
    spreadsheet_id: str,
    sheet_name: str,
    rows: int = 1000,
    columns: int = 26
) -> dict:
    """Add a new sheet to existing spreadsheet"""
    
@mcp.tool()
async def delete_sheet(
    spreadsheet_id: str,
    sheet_id: int
) -> dict:
    """Delete a sheet from spreadsheet"""
```

### 1.4 Formula Translation Layer

```python
class FormulaTranslator:
    """Translates Polars operations to Google Sheets formulas"""
    
    def __init__(self):
        self.mappings = self.load_mappings()
        self.complex_handler = ComplexFormulaHandler()
    
    def translate_operation(self, 
                           polars_op: dict, 
                           sheet_context: dict) -> str:
        """
        Converts Polars operation to Google Sheets formula
        
        Example:
        Input: {"operation": "SUM", "dataframe": "df", "column": "revenue"}
        Output: "=SUM(A2:A100)"
        """
        
    def validate_translation(self, 
                            polars_result: Any, 
                            sheets_formula: str) -> bool:
        """Verify translation produces same result"""
```

## Phase 2: Polars Integration Enhancement

**Timeline: Week 4**  
**Priority: Critical**

### 2.1 Range Parameter Refactoring

Modify all existing Polars MCP tools to accept range parameters:

```python
# Before
def sum_tool(dataframe_path: str) -> FinnOutput:
    df = load_dataframe(dataframe_path)
    # Assumes df is already sliced
    
# After
def sum_tool(
    dataframe_path: str,
    range_spec: Optional[Union[str, dict]] = None
) -> FinnOutput:
    df = load_dataframe(dataframe_path)
    if range_spec:
        df = RangeResolver.resolve_range(df, range_spec)
```

### 2.2 Dual-Layer Execution System

```python
class DualLayerExecutor:
    """Executes computations with Polars, pushes formulas to Sheets"""
    
    def __init__(self):
        self.polars_executor = PolarsExecutor()
        self.sheets_pusher = SheetsPusher()
        self.formula_translator = FormulaTranslator()
        self.validator = DualLayerValidator()
    
    def execute_step(self, step: PlanStep) -> StepResult:
        # 1. Compute with Polars
        polars_result = self.polars_executor.execute(
            step.operation,
            step.input_data
        )
        
        # 2. Translate to Sheets formula
        sheets_formula = self.formula_translator.translate(
            step.operation,
            step.sheet_context
        )
        
        # 3. Push both to Sheets
        self.sheets_pusher.push_with_formula(
            value=polars_result,
            formula=sheets_formula,
            range=step.output_range
        )
        
        # 4. Validate consistency
        validation = self.validator.validate_dual_layer(
            polars_result,
            sheets_formula,
            step.validation_rules
        )
        
        return StepResult(
            success=validation.passed,
            polars_result=polars_result,
            sheets_formula=sheets_formula,
            validation=validation
        )
```

### 2.3 Smart Formula Pushing

```python
class SmartSheetsPusher:
    """Intelligently pushes formulas and values"""
    
    def push_with_context(self, cell_context: CellContext):
        if cell_context.type == "source_data":
            # Push raw values only
            self.push_values(cell_context.range, cell_context.values)
            
        elif cell_context.type == "calculation":
            # Push formula, let Sheets calculate
            self.push_formula(cell_context.range, cell_context.formula)
            
        elif cell_context.type == "complex_calculation":
            # Push formula with validation note
            self.push_formula(cell_context.range, cell_context.formula)
            self.add_note(
                cell_context.range,
                f"Polars computed: {cell_context.computed_value}"
            )
            
        elif cell_context.type == "pivot_table":
            # Create native Sheets pivot
            self.create_pivot_table(cell_context.pivot_config)
```

## Phase 3: Multi-Agent Workflow System

**Timeline: Weeks 5-6**  
**Priority: Critical**

### 3.1 Enhanced Plan Data Structure

```json
{
  "plan_id": "uuid-123456",
  "version": 1,
  "created_at": "2025-01-10T10:00:00Z",
  "user_task": "Create customer payback analysis by cohort",
  "status": "in_progress",
  "checkpoints": [
    {
      "id": "chk_001",
      "step_id": 1,
      "timestamp": "2025-01-10T10:05:00Z",
      "state_snapshot": {}
    }
  ],
  "validation_rules": [
    {
      "type": "range_check",
      "field": "payback_months",
      "min": 0,
      "max": 60
    }
  ],
  "steps": [
    {
      "step_id": 1,
      "description": "Extract customer cohort data",
      "dependencies": [],
      "status": "completed",
      "sub_steps": [
        {
          "sub_id": "1.1",
          "operation": "load_data",
          "source": "customers.parquet",
          "status": "completed",
          "checkpoint_id": "chk_001"
        },
        {
          "sub_id": "1.2",
          "operation": "filter_by_date",
          "range": "2022-01-01:2024-12-31",
          "status": "completed",
          "checkpoint_id": "chk_002"
        }
      ],
      "computation": {
        "polars_operation": "filter_and_group",
        "sheets_formula": "=QUERY(A:D, 'SELECT * WHERE...')",
        "formula_complexity": "moderate"
      },
      "output_artifacts": {
        "local": {
          "path": "/scratch_pad/cohort_data.parquet",
          "schema": {"cohort": "str", "count": "int"}
        },
        "sheets": {
          "spreadsheet_id": "abc123",
          "range": "Cohorts!A1:Z1000"
        }
      },
      "validation": {
        "dual_check": true,
        "tolerance": 0.001,
        "business_rules": ["cohort_count_positive"]
      },
      "rollback_point": "chk_000"
    }
  ]
}
```

### 3.2 High-Level Planning Agent

```python
class HighLevelPlanningAgent:
    """Creates high-level execution plans from user tasks"""
    
    def __init__(self):
        self.llm = LiteLLM(model="openrouter/qwen")
        self.mcp_catalog = self.load_mcp_catalog()
        
    async def create_plan(self, user_task: str) -> Plan:
        # 1. Analyze task
        task_analysis = await self.analyze_task(user_task)
        
        # 2. Generate clarifying questions if needed
        if task_analysis.needs_clarification:
            questions = self.generate_questions(task_analysis)
            answers = await self.get_user_answers(questions)
            task_analysis = self.update_analysis(task_analysis, answers)
        
        # 3. Create step-by-step plan
        steps = self.generate_steps(task_analysis)
        
        # 4. Add dependencies and validation
        steps = self.add_dependencies(steps)
        steps = self.add_validation_rules(steps)
        
        # 5. Get user approval
        plan = Plan(steps=steps, task=user_task)
        approved = await self.get_user_approval(plan)
        
        if approved:
            return plan
        else:
            return await self.revise_plan(plan)
```

### 3.3 Execution-Level Planning Agent

```python
class ExecutionLevelPlanningAgent:
    """Converts high-level plans to executable operations"""
    
    def __init__(self):
        self.mcp_selector = MCPServerSelector()
        self.tool_selector = ToolSelector()
        self.formula_selector = FormulaSelector()
        
    def create_execution_plan(self, high_level_plan: Plan) -> ExecutionPlan:
        execution_steps = []
        
        for step in high_level_plan.steps:
            # 1. Select appropriate MCP server
            mcp_server = self.mcp_selector.select_server(step)
            
            # 2. Select specific tools
            tools = self.tool_selector.select_tools(step, mcp_server)
            
            # 3. Determine if formulas needed
            if step.needs_formula:
                formula_category = self.formula_selector.select_category(step)
                formula = self.formula_selector.select_formula(
                    step, 
                    formula_category
                )
            else:
                formula = None
            
            # 4. Create execution step
            exec_step = ExecutionStep(
                step_id=step.step_id,
                mcp_server=mcp_server,
                tools=tools,
                formula=formula,
                batching_strategy=self.determine_batching(step)
            )
            
            execution_steps.append(exec_step)
        
        # 5. Optimize execution order
        execution_steps = self.optimize_execution_order(execution_steps)
        
        return ExecutionPlan(steps=execution_steps)
```

## Phase 4: Plan Execution System

**Timeline: Week 7**  
**Priority: Critical**

### 4.1 Execution Agent

```python
class ExecutionAgent:
    """Executes plans deterministically"""
    
    def __init__(self):
        self.sub_agent_factory = SubAgentFactory()
        self.checkpoint_manager = CheckpointManager()
        self.state_tracker = StateTracker()
        
    async def execute_plan(self, execution_plan: ExecutionPlan) -> ExecutionResult:
        for step in execution_plan.steps:
            try:
                # 1. Update state
                self.state_tracker.mark_in_progress(step.step_id)
                
                # 2. Create sub-agent for this step
                sub_agent = self.sub_agent_factory.create_agent(
                    step.mcp_server,
                    step.tools
                )
                
                # 3. Execute step
                result = await sub_agent.execute(step)
                
                # 4. Create checkpoint
                checkpoint = self.checkpoint_manager.create_checkpoint(
                    step.step_id,
                    result
                )
                
                # 5. Update state
                self.state_tracker.mark_completed(step.step_id, checkpoint)
                
            except Exception as e:
                # Handle failure
                recovery_result = await self.handle_failure(step, e)
                if not recovery_result.success:
                    return ExecutionResult(
                        success=False,
                        failed_step=step.step_id,
                        error=e
                    )
        
        return ExecutionResult(success=True)
```

### 4.2 Checkpoint & Recovery System

```python
class CheckpointManager:
    """Manages execution checkpoints for recovery"""
    
    def __init__(self):
        self.storage = CheckpointStorage()
        self.sheets_versioning = SheetsVersioning()
        
    def create_checkpoint(self, step_id: str, state: dict) -> Checkpoint:
        checkpoint = Checkpoint(
            id=f"chk_{step_id}_{timestamp()}",
            step_id=step_id,
            timestamp=datetime.now(),
            local_state={
                "dataframes": self.snapshot_dataframes(),
                "files": self.snapshot_files()
            },
            sheets_state={
                "spreadsheet_id": state.get("spreadsheet_id"),
                "revision": self.sheets_versioning.get_current_revision(),
                "ranges": state.get("modified_ranges")
            }
        )
        
        self.storage.save(checkpoint)
        return checkpoint
    
    def rollback_to_checkpoint(self, checkpoint_id: str):
        checkpoint = self.storage.load(checkpoint_id)
        
        # Restore local state
        self.restore_dataframes(checkpoint.local_state["dataframes"])
        self.restore_files(checkpoint.local_state["files"])
        
        # Restore Sheets state
        self.sheets_versioning.revert_to_revision(
            checkpoint.sheets_state["revision"]
        )
        
        return checkpoint
```

### 4.3 Error Handling

```python
class ErrorHandlingSubAgent:
    """Handles and recovers from execution errors"""
    
    async def handle_error(self, step: ExecutionStep, error: Exception) -> RecoveryResult:
        error_type = self.classify_error(error)
        
        if error_type == "rate_limit":
            # Wait and retry
            await self.wait_for_rate_limit()
            return RecoveryResult(action="retry", delay=60)
            
        elif error_type == "formula_error":
            # Try to fix formula
            fixed_formula = await self.fix_formula(step.formula, error)
            return RecoveryResult(
                action="retry_with_fix",
                fixed_formula=fixed_formula
            )
            
        elif error_type == "data_missing":
            # Check if we can recover data
            if self.can_recover_data(step):
                recovered_data = await self.recover_data(step)
                return RecoveryResult(
                    action="retry_with_data",
                    data=recovered_data
                )
            
        elif error_type == "partial_failure":
            # Rollback to last good sub-step
            last_good = self.find_last_good_substep(step)
            return RecoveryResult(
                action="rollback_partial",
                rollback_to=last_good
            )
            
        return RecoveryResult(action="fail", reason=str(error))
```

## Phase 5: Validation & Testing

**Timeline: Week 8**  
**Priority: High**

### 5.1 Validation Agent

```python
class ValidationAgent:
    """Comprehensive validation at all stages"""
    
    def __init__(self):
        self.pre_validator = PreExecutionValidator()
        self.post_validator = PostExecutionValidator()
        self.business_validator = BusinessRuleValidator()
        
    async def validate_pre_execution(self, plan: ExecutionPlan) -> ValidationResult:
        checks = []
        
        # Check data availability
        checks.append(await self.pre_validator.check_data_exists(plan))
        
        # Validate all formulas
        checks.append(await self.pre_validator.validate_formulas(plan))
        
        # Estimate API calls
        api_estimate = self.pre_validator.estimate_api_calls(plan)
        if api_estimate > RATE_LIMIT_THRESHOLD:
            checks.append(ValidationCheck(
                passed=False,
                reason="Exceeds API rate limits",
                suggestion="Enable aggressive batching"
            ))
        
        return ValidationResult(checks=checks)
    
    async def validate_post_execution(self, result: ExecutionResult) -> ValidationResult:
        # Check for formula errors
        formula_check = await self.post_validator.check_formula_errors(
            result.spreadsheet_id
        )
        
        # Validate business rules
        business_check = await self.business_validator.validate(
            result,
            self.get_business_rules(result.task_type)
        )
        
        # End-to-end validation
        e2e_check = await self.validate_against_requirements(
            result,
            result.original_task
        )
        
        return ValidationResult(
            formula_check=formula_check,
            business_check=business_check,
            e2e_check=e2e_check
        )
```

### 5.2 Test Scenarios

```python
class FPATestScenarios:
    """Test scenarios for the three example use cases"""
    
    @pytest.fixture
    def customer_payback_data(self):
        """Generate test data for customer payback analysis"""
        return {
            "customers": generate_customer_data(1000),
            "subscriptions": generate_subscription_data(1000),
            "marketing_spend": generate_marketing_data(12)
        }
    
    async def test_customer_payback_analysis(self, customer_payback_data):
        """Test Example 1: Customer Payback Analysis"""
        task = """
        Create a customer payback analysis by cohort and acquisition month 
        for all cohorts that have at least 12 months of data.
        """
        
        # Execute plan
        result = await execute_fpa_task(task, customer_payback_data)
        
        # Validate results
        assert result.success
        assert "payback_months" in result.sheets
        assert result.sheets["payback_months"].has_conditional_formatting
        assert result.validation.business_rules_passed
    
    async def test_ltv_analysis(self):
        """Test Example 2: Customer Lifetime Value Analysis"""
        
    async def test_product_pl_forecast(self):
        """Test Example 3: New Product P&L Forecast"""
```

## Phase 6: Production Readiness

**Timeline: Week 9**  
**Priority: High**

### 6.1 Observability

```python
class ObservabilityLayer:
    """Comprehensive monitoring and logging"""
    
    def __init__(self):
        self.logger = self.setup_logging()
        self.metrics = self.setup_metrics()
        self.tracer = self.setup_tracing()
        
    def log_api_call(self, api_call: APICall):
        self.logger.info(
            "API Call",
            extra={
                "service": api_call.service,
                "method": api_call.method,
                "duration": api_call.duration,
                "status": api_call.status
            }
        )
        
        self.metrics.increment(
            "api_calls",
            tags={
                "service": api_call.service,
                "status": api_call.status
            }
        )
    
    def track_execution_time(self, step: ExecutionStep):
        with self.tracer.trace("step_execution") as span:
            span.set_tag("step_id", step.step_id)
            span.set_tag("mcp_server", step.mcp_server)
            yield
            span.set_tag("duration", span.duration)
```

### 6.2 Security & Best Practices

```python
class SecurityLayer:
    """Security implementations"""
    
    def __init__(self):
        self.credential_manager = CredentialManager()
        self.access_controller = AccessController()
        
    def validate_spreadsheet_access(self, user_id: str, spreadsheet_id: str):
        """Ensure user has access to spreadsheet"""
        
    def sanitize_formula(self, formula: str) -> str:
        """Prevent formula injection attacks"""
        
    def encrypt_sensitive_data(self, data: dict) -> dict:
        """Encrypt sensitive financial data"""
```

## Technical Specifications

### Formula Translation Mappings

```json
{
  "simple_mappings": {
    "sum": {
      "polars": "df['column'].sum()",
      "sheets": "=SUM({range})",
      "validation": "exact_match"
    },
    "average": {
      "polars": "df['column'].mean()",
      "sheets": "=AVERAGE({range})",
      "validation": "tolerance_0.001"
    }
  },
  "complex_mappings": {
    "moving_average": {
      "polars": "df['column'].rolling(window).mean()",
      "sheets": "=AVERAGE(OFFSET({cell},-{window}+1,0,{window},1))",
      "helper_columns": ["window_start", "window_end"],
      "validation": "tolerance_0.01"
    }
  }
}
```

### API Rate Limiting Strategy

```python
class RateLimitStrategy:
    """Google Sheets API rate limiting"""
    
    # Limits: 100 requests per 100 seconds per user
    MAX_REQUESTS = 100
    WINDOW_SECONDS = 100
    
    # Batching thresholds
    BATCH_SIZE = 50
    BATCH_WAIT_MS = 500
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_BACKOFF = "exponential"
    INITIAL_RETRY_DELAY = 1000  # ms
```

## Risk Mitigation

### Identified Risks and Mitigations

1. **API Rate Limiting**
   - Risk: Hitting Google Sheets API limits
   - Mitigation: Aggressive batching, local computation with Polars

2. **Formula Translation Errors**
   - Risk: Incorrect formula translation
   - Mitigation: Dual validation, extensive testing

3. **Context Window Overflow**
   - Risk: LLM context limits exceeded
   - Mitigation: Context pruning, summarization

4. **Cascading Failures**
   - Risk: One failure affects subsequent steps
   - Mitigation: Checkpointing, rollback capabilities

5. **Data Inconsistency**
   - Risk: Mismatch between Polars and Sheets
   - Mitigation: Continuous validation, dual-check system

## Success Metrics

### Functional Metrics
- Successfully execute all 3 example FP&A scenarios
- Generate Google Sheets with visible, editable formulas
- Maintain <0.01% discrepancy between Polars and Sheets calculations

### Performance Metrics
- Average execution time per step: <5 seconds
- API call efficiency: <50% of rate limit threshold
- Recovery success rate: >95% for transient failures

### Quality Metrics
- Formula accuracy: 99.9%
- Business rule validation pass rate: 100%
- End-to-end validation success: >95%

### User Experience Metrics
- Plan approval on first attempt: >80%
- User modifications required: <10%
- Task completion rate: >90%

## Implementation Timeline

```
Week 1-2: Google Sheets API Foundation
  - Authentication setup
  - Core API operations
  - Range resolver implementation

Week 3: MCP Server Development
  - 6 specialized MCP servers
  - Formula documentation
  - Initial testing

Week 4: Polars Integration
  - Range parameter refactoring
  - Dual-layer execution
  - Formula translation layer

Week 5: Planning Agents
  - High-level planner
  - Execution-level planner
  - Plan data structures

Week 6: Workflow Integration
  - Agent communication
  - Plan approval flow
  - Sub-agent creation

Week 7: Execution System
  - Execution agent
  - Checkpointing
  - Error recovery

Week 8: Validation & Testing
  - Validation agents
  - Test scenarios
  - Performance testing

Week 9: Production Readiness
  - Observability
  - Security
  - Documentation
  - Deployment
```

## Next Steps

1. **Immediate Actions**
   - Set up Google Cloud project and enable Sheets API
   - Create service account credentials
   - Initialize `/mcp_tooling/google_sheets/` directory structure

2. **Week 1 Deliverables**
   - Complete authentication module
   - Implement RangeResolver class
   - Create first MCP server (structure_server)
   - Basic formula translator for simple operations

3. **Success Criteria for Phase 1**
   - Successfully create and modify Google Sheets via API
   - Apply simple formulas (SUM, AVERAGE) with validation
   - Demonstrate batching with 100+ operations

## Appendix

### A. Google Sheets Formula Categories

Based on Google's documentation, formulas are organized into these categories:
- Array formulas
- Database functions
- Date functions
- Engineering functions
- Financial functions
- Google functions
- Info functions
- Logical functions
- Lookup functions
- Math functions
- Operator functions
- Parser functions
- Statistical functions
- Text functions

### B. Example Plan Execution

```python
# Example: Customer Payback Analysis
async def execute_customer_payback_analysis():
    # User task
    task = "Create customer payback analysis by cohort"
    
    # High-level planning
    high_level_plan = await high_level_planner.create_plan(task)
    
    # Execution planning
    execution_plan = await execution_planner.create_execution_plan(
        high_level_plan
    )
    
    # Execute with validation
    result = await executor.execute_plan(execution_plan)
    
    # Validate results
    validation = await validator.validate_post_execution(result)
    
    return {
        "spreadsheet_id": result.spreadsheet_id,
        "success": validation.passed,
        "metrics": result.metrics
    }
```

### C. Formula Translation Examples

```python
# Simple Translation
polars: df["revenue"].sum()
sheets: =SUM(Revenue!B:B)

# Complex Translation with Helper Columns
polars: df.groupby("cohort")["revenue"].cumsum()
sheets: 
  Helper1: =SUMIF($A$2:A2, A2, $B$2:B2)
  Result: =Helper1

# Array Formula Translation
polars: df["price"] * df["quantity"]
sheets: =ARRAYFORMULA(B:B * C:C)
```

---

This implementation plan provides a comprehensive roadmap for building the FP&A application with all critical features, risk mitigations, and success metrics clearly defined.