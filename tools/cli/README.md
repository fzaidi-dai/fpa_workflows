# CLI Test Suite

This directory contains comprehensive pytest tests for the `execute_command_line` function.

## Test Coverage

The test suite covers all execution paths of the `execute_command_line` function:

### Success Paths
- Basic command execution
- Python script execution
- Various argument types (string, list, mixed)
- Different output scenarios (stdout only, stderr only, both, none)

### Failure Paths
- Command failure with non-zero exit codes
- Invalid command execution
- Empty command validation
- None argument handling

### Timeout Paths
- Command timeout scenarios
- Zero timeout behavior
- No timeout (None) scenarios

### Edge Cases
- Special characters in output
- Long output handling
- Unicode support
- Multiline output
- Working directory behavior
- Argument validation edge cases

## Running Tests

```bash
# Run all tests
uv run pytest tools/cli/cli_test.py -v

# Run specific test class
uv run pytest tools/cli/cli_test.py::TestExecuteCommandLine -v

# Run specific test
uv run pytest tools/cli/cli_test.py::TestExecuteCommandLine::test_basic_success -v
```

## Test Helper Script

The `test_helper.py` script provides various test scenarios:
- Success/failure exit codes
- Configurable timeout delays
- Custom stdout/stderr output
- Mixed output scenarios

## Test Structure

Tests are organized in the `TestExecuteCommandLine` class with descriptive method names that clearly indicate what each test covers.
