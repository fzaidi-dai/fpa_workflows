# Multi-Agent FP&A System Implementation Plan

## Table of Contents
1. [System Overview](#system-overview)
2. [Current State Analysis](#current-state-analysis)
3. [Agent Specifications](#agent-specifications)
4. [Implementation Phases](#implementation-phases)
5. [Technical Architecture](#technical-architecture)
6. [Workflow Examples](#workflow-examples)
7. [File Structures](#file-structures)
8. [Integration Points](#integration-points)

---

## System Overview

### Vision
A multi-agent Financial Planning & Analysis (FP&A) system built on Google ADK that provides conversational interaction, automated data processing, and strategic planning capabilities for financial experts.

### Core Components
- **Root Agent**: Conversational FP&A assistant that interacts with users
- **High-Level Planner Agent**: Strategic planning agent that creates actionable plans
- **Data Pipeline**: Polars-based cleaning and metadata generation
- **Google Sheets Integration**: MCP servers for spreadsheet operations
- **Session Management**: ADK-based state management and callbacks

### Key Principles
- **Data Integrity**: No imputation or estimation in financial data cleaning
- **Modular Design**: Clear separation of agent responsibilities
- **Extensibility**: Easy to add specialized agents later
- **Memory Management**: Persistent task context and planning history

---

## Current State Analysis

### ✅ Available Components

#### Tools
- **Filesystem Tool** (`tools/native_filesystem.py`)
  - `read_file`, `write_file`, `list_files`
  - `create_directory`, `delete_file`, `get_file_info`
  - Security: Restricted to `data/`, `scratch_pad/`, `memory/` directories

- **Metadata Tool** (`tools/metadata.py`)
  - `get_metadata()`: Analyzes all CSV files in a folder
  - `get_metadata_one_file()`: Analyzes single CSV file
  - Returns: schema, statistics, null counts, unique counts, sample data
  - Uses Polars for efficient analysis

#### Google Sheets MCP Servers
Following the exact formula server pattern:
- **Format Server** (`google_sheets_format_mcp_server.py`): Cell formatting, conditional formatting
- **Chart Server** (`google_sheets_chart_mcp_server.py`): Chart creation and management
- **Data Server** (`google_sheets_data_mcp_server.py`): Read/write operations
- **Structure Server** (`google_sheets_structure_mcp_server.py`): Spreadsheet/sheet management
- **Validation Server** (`google_sheets_validation_mcp_server.py`): Data validation rules
- **Formula Servers** (9 categories): Aggregation, Lookup, Financial, Business, Array, Text, Logical, Statistical, DateTime

#### Data
- 7 CSV files in `/data` folder:
  - customers.csv
  - marketing.csv
  - operational_costs.csv
  - orders.csv
  - subscriptions.csv
  - support_tickets.csv
  - usage_metrics.csv

### ❌ Components to Build

#### Infrastructure
- `/prompts` folder structure with agent cards
- `docs/google_sheets_toolsets.md` documentation
- `create_task` tool for task folder management
- Context dumping tool for session state persistence
- Prompt provider functions for agents

#### Data Pipeline
- Polars-based financial data cleaning pipeline
- Automated preprocessing validation
- Session management callbacks

#### Agents
- Root Agent implementation
- High-Level Planner Agent implementation
- Agent coordination and communication

---

## Agent Specifications

### Root Agent

#### Purpose
Primary conversational interface for FP&A tasks, managing user interactions and coordinating with other agents.

#### Configuration
- **Model**: LiteLLM with OpenRouter (or Google Vertex AI)
- **Type**: Conversational ADK Agent
- **Persona**: Professional Financial Planning & Analysis Assistant

#### Tools
1. **Filesystem Tool**: Access to data, scratch_pad, memory directories
2. **create_task Tool**: Create and manage task folders
3. **High-Level Planner Tool**: AgentTool for invoking planner

#### Session Variables
- `task`: Current task folder name
- `high_level_plan_status`: "planning" | "processing_feedback" | "approved"
- `high_level_plan_feedback`: User feedback for plan improvement
- `data_files`: Dictionary of file paths (original, cleaned, metadata)

#### Callbacks
- **Pre-run**: Data validation, cleaned file check
- **Post-run**: Context dumping to `current_context.md`

#### Prompt Structure
```
prompts/
└── root_agent_card/
    ├── specs.md          # name: "FPA Assistant", model: "openrouter/...", provider: "litellm"
    ├── description.md    # "Professional FP&A assistant for financial analysis tasks"
    ├── system_prompt.md  # Detailed system instructions
    └── instruction.md    # User query template with placeholders
```

### High-Level Planner Agent

#### Purpose
Generate strategic plans based on available data, toolsets, and user requirements.

#### Configuration
- **Model**: LiteLLM with OpenRouter (or Google Vertex AI)
- **Type**: Planning ADK Agent
- **Focus**: Data-driven strategic planning

#### Tools
1. **Filesystem Tool**: Read/write plan files and access metadata

#### Session Variables (Input)
- `task`: Task folder name from Root Agent
- `available_toolsets`: Content from `google_sheets_toolsets.md`
- `metadata_files`: Dictionary of data file metadata
- `high_level_plan_status`: Current planning state
- `high_level_plan_feedback`: Feedback for plan revision

#### Callbacks
- **Pre-run**: Load toolsets documentation, load metadata, create plan file
- **Post-run**: Context dumping, memory update

#### Outputs
- `high_level_plan.md`: Strategic plan in task folder
- `memory/high_level_plan.md`: Plan history and feedback incorporation

#### Prompt Structure
```
prompts/
└── high_level_planner_card/
    ├── specs.md          # name: "Strategic Planner", model: "openrouter/...", provider: "litellm"
    ├── description.md    # "Creates data-driven strategic plans for FP&A tasks"
    ├── system_prompt.md  # Planning instructions with toolset awareness
    └── instruction.md    # Planning template with placeholders
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Days 1-2)

#### 1.1 Create Prompts Structure
```bash
prompts/
├── root_agent_card/
│   ├── specs.md
│   ├── description.md
│   ├── system_prompt.md
│   └── instruction.md
└── high_level_planner_card/
    ├── specs.md
    ├── description.md
    ├── system_prompt.md
    └── instruction.md
```

#### 1.2 Create Google Sheets Toolsets Documentation
- File: `docs/google_sheets_toolsets.md`
- Content: Categories, tools, use cases for all MCP servers
- Format: Similar to `finn_tool_categories.md`

#### 1.3 Implement Core Tools

**create_task Tool**
```python
def create_task(task_name: str, task_description: str) -> Dict[str, Any]:
    """
    Create a task folder with initial structure.
    
    Creates:
    - /scratch_pad/{task_name}/
    - /scratch_pad/{task_name}/task.md
    - /scratch_pad/{task_name}/memory/
    """
```

**Context Dumping Tool**
```python
def dump_session_context(session_state: Dict, task_folder: str) -> Dict[str, Any]:
    """
    Dump session state to current_context.md in task folder.
    """
```

#### 1.4 Prompt Provider Functions
```python
def root_agent_prompt_provider() -> Dict[str, Any]:
    """Read prompt files and return agent configuration."""
    
def high_level_planner_prompt_provider() -> Dict[str, Any]:
    """Read prompt files and return planner configuration."""
```

### Phase 2: Data Management Pipeline (Days 3-4)

#### 2.1 Financial Data Cleaning Pipeline
```python
def clean_financial_data(file_path: str) -> Dict[str, Any]:
    """
    Clean CSV data for financial analysis.
    
    Rules:
    - NO imputation of missing values
    - Flag missing data for user awareness
    - Standardize date formats
    - Validate numeric columns
    - Preserve exact financial values
    
    Output: {original_name}_cleaned.csv
    """
```

#### 2.2 Metadata Integration
- Use existing `tools/metadata.py` functions
- Generate metadata for each CSV file
- Store in session state as `metadata_files`
- Create unified metadata JSON

#### 2.3 Session Management Callbacks

**Data Validation Callback**
```python
def validate_data_callback(context: SessionContext) -> None:
    """
    Ensure data files are available and cleaned.
    Sets context.data_files with paths.
    """
```

**Metadata Generation Callback**
```python
def generate_metadata_callback(context: SessionContext) -> None:
    """
    Generate metadata for all data files.
    Sets context.metadata_files.
    """
```

**Context Persistence Callback**
```python
def persist_context_callback(context: SessionContext) -> None:
    """
    Save session state to task folder.
    Runs post-agent execution.
    """
```

### Phase 3: Agent Implementation (Days 5-7)

#### 3.1 Root Agent Implementation
```python
class RootAgent:
    def __init__(self):
        # Load prompts using prompt_provider
        # Configure tools
        # Set up callbacks
        
    async def run(self, user_input: str):
        # Conversational flow
        # Task creation
        # Planner invocation
```

#### 3.2 High-Level Planner Implementation
```python
class HighLevelPlanner:
    def __init__(self):
        # Load prompts
        # Configure filesystem access
        # Set up callbacks
        
    async def create_plan(self, context: SessionContext):
        # Access metadata
        # Generate strategic plan
        # Handle feedback
```

#### 3.3 Agent Coordination
- ADK AgentTool for Root → Planner communication
- Session state sharing
- Feedback loop implementation

### Phase 4: Integration & Testing (Days 8-9)

#### 4.1 End-to-End Workflow Testing
- User interaction → Task creation
- Data validation → Metadata generation
- Plan creation → Feedback processing
- Context persistence → Recovery

#### 4.2 Test Scenarios
1. Simple data analysis request
2. Complex multi-step financial analysis
3. Plan revision with feedback
4. Session recovery after interruption

#### 4.3 Documentation
- User guide with examples
- Developer documentation
- Troubleshooting guide

---

## Technical Architecture

### Session State Structure
```python
{
    "task": "customer_churn_analysis",
    "high_level_plan_status": "planning",
    "high_level_plan_feedback": "Include revenue impact analysis",
    "data_files": {
        "customers.csv": {
            "original": "/data/customers.csv",
            "cleaned": "/data/customers_cleaned.csv",
            "metadata": "/data/customers_metadata.json"
        }
    },
    "metadata_files": {
        "customers.csv": {...},  # Full metadata from tool
        "orders.csv": {...}
    },
    "available_toolsets": "...",  # Content from google_sheets_toolsets.md
}
```

### Callback Execution Order

#### Root Agent
1. **Pre-run**:
   - validate_data_callback
   - generate_metadata_callback
2. **Execution**: Agent logic
3. **Post-run**:
   - persist_context_callback

#### High-Level Planner
1. **Pre-run**:
   - load_toolsets_callback
   - load_metadata_callback
   - create_plan_file_callback
2. **Execution**: Planning logic
3. **Post-run**:
   - persist_context_callback
   - update_memory_callback

### Tool Integration Points

#### Google Sheets MCP Servers
- Available for future execution agents
- Documented in `google_sheets_toolsets.md`
- Not directly used by Root or Planner
- Will be used by execution agents (future)

#### Filesystem Operations
- Both agents use filesystem tool
- Security: Restricted to allowed directories
- Task folders in `/scratch_pad`
- Memory in `/scratch_pad/{task}/memory`

---

## Workflow Examples

### Example 1: Simple Analysis Request

1. **User**: "Analyze customer churn for Q4"

2. **Root Agent**:
   - Converses to understand requirements
   - Creates task: "customer_churn_q4_analysis"
   - Sets status: "planning"
   - Invokes High-Level Planner

3. **High-Level Planner**:
   - Reads customers.csv metadata
   - Identifies relevant columns
   - Creates plan with steps:
     1. Load customer data
     2. Calculate churn metrics
     3. Generate trend analysis
     4. Create visualizations
   - Saves to `high_level_plan.md`

4. **Root Agent**:
   - Presents plan to user
   - Gets approval
   - Sets status: "approved"

### Example 2: Plan Revision Flow

1. **User**: Reviews plan, requests changes

2. **Root Agent**:
   - Captures feedback
   - Sets status: "processing_feedback"
   - Sets feedback in session
   - Re-invokes planner

3. **High-Level Planner**:
   - Reads previous plan from memory
   - Incorporates feedback
   - Updates plan
   - Documents changes in memory

4. **Root Agent**:
   - Presents revised plan
   - Continues until approval

---

## File Structures

### Project Structure
```
fpa_agents/
├── agents/
│   ├── root_agent.py
│   └── high_level_planner.py
├── data/
│   ├── *.csv                    # Original data files
│   ├── *_cleaned.csv            # Cleaned versions
│   └── *_metadata.json          # Metadata files
├── scratch_pad/
│   └── {task_name}/
│       ├── task.md              # Task description
│       ├── high_level_plan.md   # Current plan
│       ├── current_context.md   # Session state dump
│       └── memory/
│           └── high_level_plan.md  # Plan history
├── prompts/
│   ├── root_agent_card/
│   └── high_level_planner_card/
├── tools/
│   ├── native_filesystem.py     # ✅ Existing
│   ├── metadata.py              # ✅ Existing
│   ├── create_task.py           # To build
│   └── context_dumper.py        # To build
└── docs/
    ├── google_sheets_toolsets.md  # To create
    └── multi_agent_fpa_system_plan.md  # This file
```

### Task Folder Structure
```
scratch_pad/{task_name}/
├── task.md                      # Task description
├── high_level_plan.md          # Strategic plan
├── current_context.md          # Latest session state
├── memory/
│   ├── high_level_plan.md     # Plan evolution history
│   └── feedback_log.md        # User feedback history
└── data/
    └── intermediate_results/   # Future: execution results
```

---

## Integration Points

### ADK Integration
- **AgentTool**: Root Agent → High-Level Planner
- **SessionContext**: Shared state management
- **Callbacks**: Pre/post execution hooks
- **Prompt Templates**: Jinja-style placeholders

### Google Sheets Integration (Future)
- Execution agents will use MCP servers
- Direct spreadsheet manipulation
- Formula generation with 100% accuracy
- Data visualization and reporting

### Data Pipeline Integration
- Automatic cleaning on file upload
- Metadata generation for all CSV files
- Session state tracking of file paths
- No data loss or estimation

### Memory System
- Task-specific memory folders
- Plan evolution tracking
- Feedback incorporation history
- Context recovery capability

---

## Success Criteria

### Phase 1 Success
- [ ] Prompts structure created
- [ ] Tools implemented and tested
- [ ] Documentation complete

### Phase 2 Success
- [ ] Data cleaning preserves financial integrity
- [ ] Metadata generation automatic
- [ ] Callbacks functioning correctly

### Phase 3 Success
- [ ] Agents communicate effectively
- [ ] Plans generated accurately
- [ ] Feedback loop works

### Phase 4 Success
- [ ] End-to-end workflow smooth
- [ ] Context persistence reliable
- [ ] System ready for production

---

## Future Enhancements

### Additional Agents
- **Execution Agent**: Implements plans using Google Sheets
- **Validation Agent**: Verifies results and data quality
- **Reporting Agent**: Generates executive summaries

### Advanced Features
- Multi-user support with separate workspaces
- Concurrent task execution
- Advanced memory and learning
- Integration with external data sources

### UI Development
- Web interface for file upload
- Task management dashboard
- Plan visualization
- Results presentation

---

## Notes and Considerations

### Security
- Filesystem access restricted to safe directories
- No execution of arbitrary code
- Service account authentication for Google Sheets
- Data privacy and isolation

### Performance
- Polars for efficient data processing
- Async operations where applicable
- Callback optimization
- Session state management efficiency

### Maintainability
- Clear separation of concerns
- Consistent naming conventions
- Comprehensive documentation
- Modular architecture

---

*Last Updated: 2024-12-20*
*Version: 1.0*