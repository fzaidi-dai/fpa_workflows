 We are building an AI based Financial Planning and Analysis application. The user will upload a set of csv files as data. Then the user will give specific tasks/scenarios and the application shall solve a given task by generating and continuously updating a Google Sheet step by step.
 The type of scenarios this application is going to support are given below as examples:

**Example 1: Customer Payback Analysis**
* **User Input:** "Create a customer payback analysis by cohort and acquisition month for all cohorts that have at least 12 months of data. Create 3 tables: monthly, annual, and combined subscriptions. Use the initial subscription type. Allocate marketing spend by % of customers if direct data is unavailable. Highlight payback months (green/red) and create a summary table highlighting cohorts based on payback period (<15 months=green, >15 months=yellow, not paid back=red).

**Example 2: Customer Lifetime Value (LTV) Analysis**
* **User Input:**: Calculate LTV across multiple dimensions using the formula: LTV = (ARPU / Churn Rate) × Profit Margin Key Requirements:
Multi-dimensional analysis (subscription type, plan type, industry, acquisition channel). Use initial subscription/plan type for consistency. Handle zero churn scenarios with configurable assumptions. Generate CAC-to-LTV ratio analysis.

**Example 3: New Product P&L Forecast**
* **User Input:** We are launching a new SaaS product in Jan 2025, priced between our Basic and Pro plans. It has monthly and annual options. Build an Excel model with a 3-year forecasted P&L down to contribution margin. Use a cohort-based retention model based on the average of Basic and Pro plans for the last 3 years. The model needs toggleable assumptions and a monthly forecast view.

- We will be using Google Aghent Development Kit (ADK) for this application.

We have devised the folowing high level design for this application:
- The agentic workflow shall use a set of MCP servers that expose their tools to execute and solve the task.
- The output artifacts shall be intermediate Google sheets and a final Spreadsheet consisting of all the intermediate worksheets with analysis results/reasoning etc., formulas/functions applied to each cell as appropriate in each sheet, and charts and visualizations as the user desired also embedded as individual or part of other sheets.
- We will implement an AI based multi-agent system that will implement a workflow to slove user given tasks/scenarios for this desired output.
- The agentic workflow will have two modes: Plan Mode and Execution Mode in the application.
- In Plan Mode, a user-interfacing agent will ask the user to input a task/scenario through a chat-like interface. The scenario will be analyzed by a high-level planner agent.
- This job of the high-level planner agent will devise a high level plan to solve the given task/scenario step by step.
- The high-level planner agent would send back clarifying questions to the user-interfacing agent, who will in turn ask those questions to the user in a chat-like interface, as and when required during this high level planning phase.
- Once the high-level planner is able to proceed with the plan design after getting answers to all the clarifying questions, it will proceed with designing a step by step plan for solving the task at a high level. The high level here means that it will describe the steps necessary to perform the task in a user-friendly language for each step of the plan.
- The planner would have access to a .md file that contains information about all the MCP servers with a summary of their purpose, exposed tools and meta data etc. This information will help the planner plan better considering what it would have available as tools later when executing the plan.
- Once the high-level plan is finalized, it will be send back to the user for final approval. In case the user approves it, it will be passed to an execution-level planner agent.
- The plan would be a persistent data structure based on JSON and will look something like this:
 {
  "plan": [
    "plan_id": <some random string probably a database id or uuid>
    {
      "step_id": 1,
      "description": "Extract historical retention data for Basic plan customers from the last 3 years.",
      "dependencies": [],
      "output_artifact_name": "basic_plan_retention_data",
      "status": "pending"
    },
    {
      "step_id": 2,
      "description": "Extract historical retention data for Pro plan customers from the last 3 years.",
      "dependencies": [],
      "output_artifact_name": "pro_plan_retention_data",
      "status": "pending"
    },
    {
      "step_id": 3,
      "description": "Calculate monthly retention rates for Basic plan cohorts.",
      "dependencies": [1],
      "output_artifact_name": "basic_monthly_retention_rates",
      "status": "pending"
    },
    {
      "step_id": 4,
      "description": "Calculate monthly retention rates for Pro plan cohorts.",
      "dependencies": [2],
      "output_artifact_name": "pro_monthly_retention_rates",
      "status": "pending"
    },
    {
      "step_id": 5,
      "description": "Average the Basic and Pro monthly retention rates to create the new product retention curve.",
      "dependencies": [3, 4],
      "output_artifact_name": "average_retention_curve",
      "status": "pending"
    },
... and so on.

- As depicted above, the plan at this high-level has an id, and each step as a specific id (number), dependencies (for later extracing parallelism of steps), an output artifact name, a status field to identify if this step is pending execution, completed, in progress, repeated afdter user feedback etc. and a user-friendly description.
- The job of the execution-level planner agent is to create an execution-plan and add it to this high level master plan. This agent shall select the MCP servers relevant to each step, followed by the tool selection for that step. It will most likely be in the form of a nested structure that would specify which MCP server and its selected tools with the tool's arguments in terms of the previous steps output artifacts on which this step is dependent on. This will be used to implement this step in the execution phase.
- In the execution-plan, optimizations like data compression and batching of requests to Google Sheets API will also be incorporated. Note that these features shall be available as tools in the Google Sheets MCP server. This means that execution-plan will encode Google Sheets best practices such as request batching and other optimizations by storing them in a new data structure, let's say something like final-executable plan that contains its own a steps that contain the first three steps of the executable-plan in a list and and trhe step named "batch_step1" holding this list and the batching tool of the MCP server specified as the tool for this step. Similarly, it will contain steps that clealy specify that compression tool needed to be called.
- After we have a final executable plan with implementation details, the workflow will enter the Execution Phase.
- In Execution Phase, a single, deterministic agent called the execution agent, will go through the final executable plan and instantiate sub-agents based on it. It will have functions that will return instances of the generic sub-agents with pre-defined system prompts for tool calling with filled place-holders and a set of tools identified by the execution-level agent.
- Once the sub-agents have been instantiated, the execution agent will go through the workflow, connecting sub-agents in a sequential workflow.
- If any step fails during execution, the executor shall have an error handling sub-agent to invoke as a tool that will inspect the error and try to rectify it, e.g. by modifying the arguments of a specific step etc. It will give it back to executor that will then, repeat the steps that failed in sequence (maybe without batching). All this logic shall be deterministic as the executor will just be calling tools and handling their errors via an LLM based sub-agent.
- The executor shall keep track of the state of each step and update it accordingly. It will also keep track of the plan's global state. This will help it identiofy any steps that the user may want to repeat in isolation or all steps after those e.g. regenerating visualizations or specific parts of some sheets e.g. change formatting, coloring etc.
- Either the plan will complete or exhaust retries after errors in a given step. In completed case, the executor shall return to the user-interface agent (root-agent) and the application shall return to the plan mode. The agent shall display the results to the user. It may download the final Spreadsheet and/or display it to the user in a tab.
- The user may give feedback to amend/modify or improve specific parts of some sheets. Since the plan is already in the session state, the high level planner would determine if the high level plan needs to change as a result of user's feedback. If it does need to change, it will modify that and subsequent steps (if required). However, if it does not need to change, it will just record feedback in the affected steps for the exectution-level planner.
- the execution level planner would update the affected steps and any subsequent steps dependent upon it and re-create the executable plan.
- The application shall enter the Execution mode again and the process shall repeat.

- We are going to build a Google Sheets MCP server for this application.
 - Here are our requirements for this Google Sheets MCP server:

 - The server is going to provide our AI agents, operations for creating/updating and manipulating Google Sheets based on a financial planning and analysis (FP&A) task given by the user.
 - We want our FP&A agents to be accurate and precise for FP&A tasks and scenarios.
 - We will use the Google Sheets API in Python.
 - Our agent should be able to insert/embed/apply the built-in formulas and functions provided by Google Sheets on any columns, rows or range or cells/rows/columns in any dimensions.
 - The issue is that while using LLMs, agents make too many mistakes in tool selection and tool arguments building and call structuring when given a large list of tools. Many a times, they are unable to pick the right tool for a step of the task solver plan. Therefore, our idea is to divide the operations that Google sheets API exposes into 3 to 4 MCP servers instead of creating one giant server exposing a large list of tool operations.
 - Same goes for formulas and functions. Google Sheets have over 200 (maybe even more) formulas/functions. Giving so many functions to an LLM would overwhelm it and again, it is expected to make too many mistakes. rendering the solution almost useless for FP&A tasks.
 - The formulas are mostly passed as simple strings in Google Sheets API operations with an = sign, therefore, our idea is to create .md files for available formulas. The functions available in the documentation have already been divided into classes/categories by Google themselves. The categories and the functions under them can be found at: https://support.google.com/docs/table/25273?hl=en. We can create multiple .md files, one for each class/category e.g. Array. Date, Financial etc. and put each available functions's description/documentation into its own category file. That way, we would also have the flexibility to not expose all the categories to our agents if they really don't need it.
 -  The idea is that one or more MCP servers would expose all their tools to our execution-level planning agent. Also, our main high level planing agent shall also have a .md file containing all the MCP servers and their well summarized dscriptions, similar to what we have for other servers in @docs/finn_tool_categories.md.
 - The execution-level planner would first select the right MCP server for a specific step of the plan, and then invoke a step level tool selector sub-agent that is only given that MCP server's tools.
 - The step level sub-agent will select the right tool for this specific plan step.
 - For formulas and values to be given to the sheets API, there will be a values generator sub-agent available to the exection-level planner. It will determine if a formula needs to be applied in this step. If yes, then it will select the class/category of the formula to be used based the operation being performed in the step. It will invoke another sub-agent that will be given only the formulas fo that category. It will select the actual formula from the given class/category and form the values string with the formula/function. If no formula needs to be applied, then values generator sub-agent will just generate the values string without the formula and return to the parent agent.
 - The execution-level planner would then take the values string from the values generating agent (may or may not be containing formulas for this step), and set the values in the step.

This is by and large the core of our application design. Of course we will add UI later and some user database and basic API on top of it later. Howeverm right now, we are more focused on getting the agentic design correct and solid. Of course this is a high level description of the design and not a formal one. We will ask you later to create a formal design spec once we have finalized it.

We would really appreciate if you can review it critically and identify any gaps, problems or blind spots we might have been overlooking. Think deeply and critically.
We will then start bottom up in our implementation. This means we will go step by step with MCP Server implementation for Google Sheets first to build the building blocks and tools on which our agents shall rely on. Then we will move on to high level planner, followed by execution-level planner with all its sub-agents for API tool and formula/function selectors, and then finally plan executioner with its tool calling sub-agents.
Use context7 for Google Sheets API and Google ADK documentation.
Again, think deeply.




Excellent critique and recommendations. Thank you very much!!
Let's answer and address these one by one and devise a solid strategy and design together:

Critical Gaps & Concerns

  1. State Management Complexity

  - Problem: Your plan tracks status at step level, but what about partial failures within a step?
  - Example: If step 3 writes to 10 sheets and fails on sheet 7, how do you track/recover?
  - Solution: Add sub-step tracking or checkpoint mechanisms within each step
  **Good Suggestion. We must add it in the design**.

  2. Formula Dependency Resolution

  - Problem: Google Sheets formulas can reference other cells/sheets. How will agents handle formula
  dependencies?
  - Example: Cell A1 has =B1+C1, but B1 has =Sheet2!A1. Order matters!
  - Solution: Implement dependency graph analysis before formula application
**Good Suggestion. Although advanced feature. Let's keep this in mind for future since we might follow a hybrid approach as you suggested and may mitigate it partially that way. We will implement it later though after our implementation is working reasonably**.

  3. Data Volume & Performance

  - Problem: Google Sheets API has strict rate limits (100 requests/100 seconds per user)
  - Risk: Complex FP&A tasks could hit limits quickly without proper batching
  - Solution: Implement aggressive batching at the API layer, not just MCP layer
**Good Suggestion. We must add it in the API layer.**.

  4. Agent Context Window Management

  - Problem: Large spreadsheets + plan history + formula docs could exceed context limits
  - Risk: Agents losing track of earlier steps or making inconsistent decisions
  - Solution: Implement context pruning and summarization between steps
**For plan history and formula docs it might be so but we are going to summarize formulas anyway.Also keep in mind that we are not sending large spread sheets to LLM in any case as far as we know. We will only be sending data schema and maybe a few samples. The idea is to plan with tool calls, not let the LLM do the analysis directly on data**.

  5. Error Recovery Granularity

  - Problem: Your design has "retry failed steps" but what about cascading failures?
  - Example: If step 5 fails and you retry it, but step 6-10 already executed based on step 5's partial
  output
  - Solution: Add rollback capabilities or versioning for each spreadsheet state
**Good Suggestion. We must add it in the design**.

  6. Formula Validation Challenge

  - Problem: LLMs can generate syntactically correct but semantically wrong formulas
  - Example: =SUM(A1:A10) vs =SUM(A:A) - both valid but different intents
  - Solution: Add formula validation agent that checks against expected output patterns
**Good Suggestion. We must add it in the design**.

  7. Multi-User Coordination

  - Problem: No mention of handling concurrent edits or multiple users
  - Risk: Data inconsistency if multiple agents/users modify same spreadsheet
  - Solution: Implement locking mechanisms or use Google Sheets revision history
**Good Suggestion. However, we are not going to provide this feature of multiple users updating the same plan/scenario in first version. It wil be an advanced feature down the roadmap**.

  8. Testing & Validation Gap

  - Problem: How do you validate that the generated spreadsheet actually solves the user's task?
  - Example: User asks for "customer payback analysis" - how do you verify correctness?
  - Solution: Add validation agent that checks outputs against expected patterns/ranges
**Excellent Suggestion. We must add it in the design. This is an absolute must**.

  🟡 Architectural Blind Spots

  1. Cross-Sheet References

  Google Sheets heavily uses cross-sheet references (Sheet2!A1:B10). Your design doesn't explicitly handle:
  - Creating these references programmatically
  - Validating reference integrity
  - Updating references when sheets are renamed/deleted

  2. Chart/Visualization Generation

  You mention "charts and visualizations" but don't detail:
  - Which agent generates them?
  - How do you map data to appropriate chart types?
  - How do you handle chart positioning and sizing?

**We thought it is part of Google Sheets API. But we found out that it is part of the Charets API and that API can be integrated with Google Sheets easily. So we should enhance our core low level API using Google Charts. look at https://developers.google.com/chart/interactive/docs/spreadsheets using Context7**.

  3. Named Ranges & Data Validation

  Professional spreadsheets use:
  - Named ranges for clarity
  - Data validation rules
  - Protected ranges
  These aren't mentioned in your MCP server design
**Excellent Suggestion. We must add it in MCP server design. This is an absolute must**.

  4. Incremental Updates

  Your workflow seems to regenerate entire spreadsheets. What about:
  - Updating existing spreadsheets based on new data?
  - Preserving user modifications while updating other parts?
**It was implied that the plan will also hold the generated sheets as internal artifacs. The user will of course not be expected to update the sheets while Agent is working on it. However, when updating based on new data as a result of further interaction within the same thread with the user, i.e. within the context of a preexisting plan, the artifacts shall be preserved and updated in their respective steps if some steps are repeated or new ones added to the existing plan**.

  💡 Recommendations

  1. Add Pre-Execution Validation
    - Before execution, validate that all required data exists
    - Check formula syntax validity
    - Estimate API calls to prevent rate limit issues
**Excellent Suggestion. We must add it**

  2. Implement Checkpointing
    - Save spreadsheet state after each successful step
    - Enable resume from last checkpoint on failure
**Excellent Suggestion. We must add it**

  3. Create Formula Composition Layer
    - Instead of raw formula strings, use structured formula objects
    - Validate formula dependencies before application

**Excellent Suggestion. We must add it**

  4. Add Observability
    - Log all API calls with request/response
    - Track token usage per step
    - Monitor execution time for optimization

**That is all provided by ADK already. Execution time monitoring and tool profiling could be added though to measure tool latency using Python code and log it via ADK mechanisms**

  5. Consider Hybrid Approach
    - Use Google Sheets for final output
    - Use local Polars DataFrames for intermediate calculations
    - Only push to Sheets when necessary (reduces API calls)
**It is a good suggestion. However, we ned to fix something in our Polar API. Actually we want you to suggest a good solution for this. Our Polars based MCP tools API as implemented in other MCP Servers takes a string for the dataframe path. However, it assumes that the rows and column ranges and dimensions etc. will already be determined by the caller and the caller will create a slice of the dataframe before calling the tool. This means that every time before calling any API, a dataframe selector/slicer tool needs to be called. Do you think it is a good approach and we have an appropriate slicer API that will always be called? Or you think that adding the inddexing/slicing information with dimensions and ranges as Google sheets API functions use is better? We want an hybrid approach as it is definitely much more edfficient and reduces API calls, avoids rate limits hits and is better overall but we need to modify the MCP server tools for all other Polar Dataframe based MCPs. Which may be worth it though. We need your comments on that.**

  6. Implement Feedback Loop
    - After execution, validate outputs
    - Check for formula errors in resulting spreadsheet
    - Provide automatic corrections for common issues
**This is already flagged as a MUST above. We definitely want to add it to our implementation.**

Based on above answers, revise the design and comment on our answer to recommendation 5 "Hybrid Approach" and it it is worth it, add it to the revised design too, otherwise, describe how you see it.
Again, think deeply. Thanks.


Hybrid Approach Analysis

  The Current Problem

  Your Polars MCP tools require pre-sliced DataFrames, meaning every operation needs:
  1. Call slicer tool → 2. Call actual operation tool

  This creates 2N tool calls for N operations, increasing complexity and failure points.

  My Recommendation: Adopt Google Sheets-style Range Parameters

  Transform your MCP tool signatures from:
  # Current approach
  def sum_tool(dataframe_path: str) -> FinnOutput:
      # Assumes df is already sliced

  To:
  # Recommended approach
  def sum_tool(
      dataframe_path: str,
      range_spec: Optional[RangeSpec] = None  # e.g., "A1:C10" or {"rows": [1,10], "cols": ["A","C"]}
  ) -> FinnOutput:
      # Tool handles slicing internally

  Why This Is Worth It

  1. Consistency: Both local Polars and Google Sheets operations use the same range syntax
  2. Efficiency: Reduces tool calls by 50%
  3. Atomicity: Each operation is self-contained (slice + compute in one call)
  4. Caching: Tools can cache DataFrames and only re-slice, not reload
  5. Plan Simplicity: Plans become more readable and maintainable

  Implementation Strategy

  Create a shared RangeResolver class:
  class RangeResolver:
      @staticmethod
      def resolve_range(df: pl.DataFrame, range_spec: Union[str, dict]) -> pl.DataFrame:
          """Converts Google Sheets-style ranges to Polars slices"""
          # "A1:C10" → df[0:10, 0:3]
          # {"rows": [1,10], "cols": ["revenue", "profit"]} → df[1:10, ["revenue", "profit"]]

  This one-time refactoring enables the hybrid approach and dramatically simplifies your workflow.

  ---
  Revised FP&A Application Design v2.0

  Core Architecture Enhancements

  1. Enhanced Plan Data Structure

  {
    "plan_id": "uuid",
    "version": 1,
    "checkpoints": [],  // NEW: State snapshots
    "validation_rules": [],  // NEW: Expected outcomes
    "plan": [
      {
        "step_id": 1,
        "description": "...",
        "sub_steps": [  // NEW: Granular tracking
          {
            "sub_id": "1.1",
            "operation": "write_range",
            "status": "completed",
            "checkpoint_id": "chk_001"
          }
        ],
        "execution_mode": "hybrid",  // NEW: local/sheets/hybrid
        "output_artifacts": {
          "local": "df_revenue_2024.parquet",
          "sheets": "Sheet1!A1:Z100"
        },
        "validation": {  // NEW: Validation rules
          "type": "range_check",
          "expected": {"min": 0, "max": 1000000}
        },
        "rollback_point": "chk_000"  // NEW: Recovery reference
      }
    ]
  }

  2. Hybrid Execution Layer

  class HybridExecutor:
      """Intelligently routes operations between local Polars and Google Sheets"""

      def execute_step(self, step):
          if step.involves_heavy_computation:
              # Execute locally with Polars
              result = self.execute_local(step)
              # Push only final results to Sheets
              self.push_to_sheets(result, step.output_range)
          else:
              # Direct Google Sheets operation
              self.execute_sheets(step)

  3. Formula Composition Layer

  class FormulaBuilder:
      """Structured formula generation with validation"""

      def build_formula(self, operation: str, ranges: List[str]) -> Formula:
          formula = Formula(
              text=f"={operation}({','.join(ranges)})",
              dependencies=self.extract_dependencies(ranges),
              validation_rules=self.get_validation_rules(operation)
          )
          return self.validate_syntax(formula)

  4. Checkpoint & Recovery System

  class CheckpointManager:
      """Manages state snapshots for recovery"""

      def create_checkpoint(self, step_id: str, state: dict):
          checkpoint = {
              "id": f"chk_{step_id}",
              "timestamp": datetime.now(),
              "spreadsheet_revision": self.get_sheets_revision(),
              "local_artifacts": self.snapshot_local_files(),
              "state": state
          }
          self.save_checkpoint(checkpoint)

      def rollback_to(self, checkpoint_id: str):
          # Restore both local and Sheets state
          checkpoint = self.load_checkpoint(checkpoint_id)
          self.restore_sheets(checkpoint["spreadsheet_revision"])
          self.restore_local(checkpoint["local_artifacts"])

  5. Validation Agent Architecture

  class ValidationAgent:
      """Validates outputs against expected patterns"""

      def validate_step_output(self, step, output):
          validations = [
              self.check_data_types(output),
              self.check_value_ranges(output, step.validation_rules),
              self.check_formula_results(output),
              self.check_business_rules(output, step.context)
          ]
          return all(validations)

      def validate_final_spreadsheet(self, spreadsheet_id, task_description):
          """End-to-end validation against user requirements"""
          # Uses LLM to verify if output matches task intent
          return self.llm_validate(spreadsheet_id, task_description)

  Updated MCP Server Structure

  /mcp_tooling/google_sheets/
  ├── api/
  │   ├── auth.py
  │   ├── spreadsheet_ops.py
  │   ├── value_ops.py
  │   ├── format_ops.py
  │   ├── formula_ops.py
  │   ├── batch_ops.py           # Aggressive batching implementation
  │   ├── chart_ops.py            # NEW: Google Charts API integration
  │   ├── named_range_ops.py     # NEW: Named ranges management
  │   ├── validation_ops.py      # NEW: Data validation rules
  │   └── range_resolver.py      # NEW: Unified range handling
  │
  ├── structure_server/
  ├── data_server/
  ├── formatting_server/
  ├── formula_server/
  ├── chart_server/               # NEW: Chart generation MCP
  │   └── sheets_chart_mcp.py
  └── validation_server/          # NEW: Validation rules MCP
      └── sheets_validation_mcp.py

  Workflow Enhancements

  Pre-Execution Phase (NEW)

  1. Data Availability Check: Verify all required data exists
  2. Formula Syntax Validation: Pre-validate all formulas
  3. API Call Estimation: Calculate expected API calls vs rate limits
  4. Resource Allocation: Determine local vs. Sheets execution split

  Execution Phase (Enhanced)

  1. Hybrid Routing: Each step executed locally or remotely based on complexity
  2. Sub-step Tracking: Granular progress monitoring
  3. Continuous Checkpointing: State saved after each sub-step
  4. Real-time Validation: Each output validated immediately

  Post-Execution Phase (NEW)

  1. Output Validation: Verify against expected patterns
  2. Formula Error Check: Scan for #ERROR, #REF, etc.
  3. Business Rule Validation: Check FP&A-specific constraints
  4. User Acceptance: Present summary with confidence score

  API Layer Enhancements

  class GoogleSheetsAPI:
      def __init__(self):
          self.batch_queue = BatchQueue(max_size=100)  # Aggressive batching
          self.rate_limiter = RateLimiter(
              max_requests=100,
              window_seconds=100,
              retry_strategy="exponential"
          )

      def execute_with_batching(self, operations: List[Operation]):
          """Automatically batches operations for efficiency"""
          batched = self.batch_queue.optimize(operations)
          for batch in batched:
              self.rate_limiter.wait_if_needed()
              self.execute_batch(batch)

  Critical Design Decisions

  1. Hybrid by Default: Every step evaluates whether to execute locally or in Sheets
  2. Range Unification: All tools accept Google Sheets-style ranges, resolved appropriately
  3. Validation-First: Every operation validated before and after execution
  4. Recovery-Oriented: Checkpoints enable rollback without full re-execution
  5. Formula Safety: Structured formula objects prevent syntax errors

  Implementation Priority

  1. Phase 1: Core Google Sheets API with batching + Range resolver
  2. Phase 2: Hybrid executor with local/remote routing
  3. Phase 3: Checkpoint system + Basic validation
  4. Phase 4: Formula composition layer
  5. Phase 5: Advanced validation agent
  6. Phase 6: Chart generation and named ranges

  This revised design addresses all critical gaps while maintaining the elegance of your original vision.
  The hybrid approach with unified range handling will significantly improve efficiency and reliability.

  The key insight is that treating local DataFrames and Google Sheets ranges uniformly simplifies the
  mental model for both agents and developers, making the system more robust and maintainable.

  Well there is a problem there. We can calculate in Polars but the user does need formulas in Google
  Sheets because Finance people are comfortable with Spreadsheets. They need formulas in cells. What we
  need to do it, compute everything locally using Polars and then for each step also push an update to the
  sheets after transforming the Polars formula applied to equivalent one in Google Sheets. We may do final
  calculations of sub-steps using Polars but eventually, we update the sheet with the formula also, not
  only the final result.

⏺ Excellent point! Finance users absolutely need to see and understand the formulas. This is actually a
  more sophisticated hybrid approach - let me revise:

  Dual-Layer Hybrid Architecture: Compute + Formula Transparency

  Core Principle

  Polars = Computation & Validation | Google Sheets = Formula Transparency & User Interaction

  We compute everything locally for accuracy/speed, but then push BOTH values AND equivalent formulas to
  Google Sheets for transparency and user editability.

  The Formula Translation Layer

  class FormulaTranslator:
      """Translates Polars operations to Google Sheets formulas"""

      def translate_operation(self, polars_op: dict, sheet_context: dict) -> str:
          """
          Converts a Polars operation to equivalent Google Sheets formula

          Example:
          Input: {"operation": "SUM", "range": df["revenue"][0:10]}
          Output: "=SUM(Revenue!A2:A11)"
          """

          translations = {
              "sum": self._translate_sum,
              "vlookup": self._translate_vlookup,
              "pivot": self._translate_pivot,
              "moving_average": self._translate_moving_avg,
              # ... all 200+ functions
          }

          return translations[polars_op["operation"]](
              polars_op["params"],
              sheet_context
          )

      def _translate_sum(self, params, context):
          # Convert Polars range to Sheets range
          sheets_range = self.polars_to_sheets_range(params["dataframe"], params["columns"])
          return f"=SUM({sheets_range})"

  Execution Flow with Formula Preservation

  class DualLayerExecutor:
      """Computes with Polars, pushes both values and formulas to Sheets"""

      def execute_step(self, step):
          # 1. Execute computation with Polars
          df = self.load_dataframe(step.input)
          result = self.execute_polars_operation(step.operation, df)

          # 2. Translate to Google Sheets formula
          formula = self.formula_translator.translate_operation(
              step.operation,
              {"sheet_name": step.sheet, "range": step.range}
          )

          # 3. Push BOTH to Google Sheets
          self.push_to_sheets_with_formula(
              computed_value=result,  # For immediate display
              formula=formula,         # For transparency/editing
              range=step.output_range,
              mode="formula_with_cached_value"  # Special mode
          )

          # 4. Validate that Sheets formula produces same result
          sheets_result = self.evaluate_sheets_formula(formula)
          assert abs(sheets_result - result) < 0.01  # Validation

  Smart Push Strategy

  class SmartSheetsPusher:
      """Intelligently pushes formulas vs values based on context"""

      def push_cell(self, cell_ref: str, polars_result: Any, formula: str, cell_type: str):
          if cell_type == "source_data":
              # Raw data: push values only
              self.push_value(cell_ref, polars_result)

          elif cell_type == "calculation":
              # Calculations: push formula (Sheets will auto-calculate)
              self.push_formula(cell_ref, formula)

          elif cell_type == "complex_calculation":
              # Complex calc: push both (formula + note with Polars result)
              self.push_formula(cell_ref, formula)
              self.add_note(cell_ref, f"Polars computed: {polars_result}")

          elif cell_type == "pivot_table":
              # For complex structures, create native Sheets pivot
              self.create_sheets_pivot(cell_ref, formula_config)

  Formula Mapping Structure

  # Store both Polars operation and Sheets formula for each step
  FORMULA_MAPPINGS = {
      "customer_lifetime_value": {
          "polars": lambda df: (df["ARPU"] / df["churn_rate"] * df["margin"]).mean(),
          "sheets": "=AVERAGE(ARRAYFORMULA(B:B / C:C * D:D))",
          "description": "LTV = (ARPU / Churn Rate) × Margin"
      },
      "cohort_retention": {
          "polars": lambda df: df.pivot_table(index="cohort", values="retained", aggfunc="mean"),
          "sheets": "=PIVOT(Data!A:D, 'cohort', 'retained', AVERAGE)",
          "validation": "check_percentages"
      }
  }

  Enhanced Plan Structure

  {
    "step_id": 1,
    "description": "Calculate monthly revenue growth",
    "computations": {
      "polars": {
        "operation": "period_over_period_growth",
        "params": {"column": "revenue", "period": "month"}
      },
      "sheets_formula": "=(B2-B1)/B1",  // Will be applied to each row
      "apply_to_range": "Growth!C2:C100"
    },
    "validation": {
      "mode": "dual_check",  // Verify Polars matches Sheets
      "tolerance": 0.001
    }
  }

  The Translation Challenge & Solution

  Some Polars operations don't have direct Google Sheets equivalents. We handle these with:

  class ComplexFormulaHandler:
      """Handles formulas that don't translate 1:1"""

      def handle_complex_operation(self, operation: str, params: dict):
          if operation == "exponential_smoothing":
              # No direct Sheets equivalent
              # Solution: Create helper columns with intermediate calculations
              return self.create_helper_formula_chain([
                  ("Helper1", "=A2*0.3 + A1*0.7"),  # Smoothing step 1
                  ("Helper2", "=Helper1*IF(...)"),   # Smoothing step 2
                  ("Result", "=AVERAGE(Helper2:Helper2)")
              ])

          elif operation == "complex_pivot":
              # Use Google Sheets QUERY function
              return f"=QUERY(Data!A:Z, 'SELECT A, SUM(B) GROUP BY A PIVOT C')"

  Benefits of This Approach

  1. Transparency: Finance users see familiar formulas
  2. Editability: Users can modify formulas and see updates
  3. Validation: We verify Polars and Sheets produce same results
  4. Education: Users learn how complex calculations work
  5. Auditability: Formulas are visible for compliance/review
  6. Trust: Users trust what they can see and understand

  Implementation Considerations

  class FormulaComplexityAnalyzer:
      """Determines how to handle each formula type"""

      def analyze_formula(self, polars_op):
          complexity = self.calculate_complexity(polars_op)

          if complexity == "simple":
              # Direct translation: SUM, AVERAGE, etc.
              return "direct_formula"

          elif complexity == "moderate":
              # Array formulas or QUERY
              return "array_formula"

          elif complexity == "complex":
              # Need helper columns or multiple steps
              return "helper_chain"

          else:  # "not_translatable"
              # Keep Polars result, add explanation
              return "value_with_note"

  Formula Documentation Structure

  /mcp_tooling/google_sheets/formula_mappings/
  ├── simple_formulas.json       # 1:1 mappings
  ├── array_formulas.json        # ARRAYFORMULA variants
  ├── pivot_formulas.json        # PIVOT/QUERY patterns
  ├── financial_formulas.json    # NPV, IRR, etc.
  └── complex_chains.json        # Multi-step formulas

  This approach gives finance users what they need - visible, editable formulas - while maintaining
  computational accuracy through Polars. The key is the translation layer that converts between the two
  worlds seamlessly.



We were in the process of implementing phase 2 of the plan in @docs/implementation_plan_v1.md section 2.1. We had created @mcp_tooling/google_sheets/standalone_math_functions.py and an enhanced mcp server in @mcp_tooling/google_sheets/math_aggregation_mcp_enhanced.py based on standalone_math_functions_enhanced.py. We also created additional Polars functions in sheets_compatible_functions.py in the same folder. The idea was to create Polars based Google Sheets compatible functions to simplify the translation layer. You have summarized context of what has been done before. Analyze the Polar compatible functions in standalone_math_functions.py and sheets_compatible_functions.py and create a unify them in sheets_compatible_functions.py. Then modify the math_aggregation_mcp_server.py to use the functions from this consolidated file for the tools it is exposing. Similarly, we want to create a few separate MCP Servers from this  sheets_compatible_functions.py. The idea is to separate these tools into a few MCP servers in same or similar categories as Google Sheets classifies them so that not only the translation is simplified but also tool selection for ADK agents later on is easier. Let's first plan this task as a sub-plan of the main plan and then implement it.



We were in the process of executing the plan in @docs/implementation_plan.md. The idea was to be able to expose tools to update Google Sheets with formulas using a few MCP Servers. However, we realized that using Polars and other Python based libraries to implement as many functions available in Google Sheets especially for formulas would be highly efficient and reduce burden of calling Google Sheets API. Therefore, we implemented a dual execution layer with translation of Google Sheets to Polars so that we wil always give preference to an implemented Python/Polars etc. function and then translate to Google Sheets so that the calls on actual data could be reduced for Sheets API.
Now, we want to integrate all implemented MCP servers that expose Google Sheets tools to the agents with Google ADK.
