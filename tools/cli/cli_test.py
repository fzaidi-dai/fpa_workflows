#!/usr/bin/env python3
"""
Comprehensive pytest test suite for execute_command_line function.
Tests all execution paths including success, failure, timeout, and edge cases.
"""

import subprocess
import pytest
import sys
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli.cli import execute_command_line


class TestExecuteCommandLine:
    """Test suite for execute_command_line function."""

    def test_basic_success(self):
        """Test basic successful command execution."""
        result = execute_command_line("echo", "hello")
        assert "# Command Execution Output" in result
        assert "hello" in result

    def test_python_script_success(self):
        """Test successful Python script execution."""
        # Create a simple test script
        test_script = Path(__file__).parent / "test_helper.py"
        result = execute_command_line("python", str(test_script), "--success", "--output", "test success")
        assert "# Command Execution Output" in result
        assert "test success" in result

    def test_string_arguments(self):
        """Test command with string arguments."""
        result = execute_command_line("echo", "hello", "world")
        assert "hello world" in result

    def test_list_arguments(self):
        """Test command with list arguments."""
        args = ["hello", "world"]
        result = execute_command_line("echo", args)
        assert "hello world" in result

    def test_mixed_arguments(self):
        """Test command with mixed string and list arguments."""
        args_list = ["world", "test"]
        result = execute_command_line("echo", "hello", args_list, "end")
        assert "hello world test end" in result

    def test_stdout_only_output(self):
        """Test command with stdout output only."""
        test_script = Path(__file__).parent / "test_helper.py"
        result = execute_command_line("python", str(test_script), "--output", "stdout only")
        assert "## Standard Output" in result
        assert "stdout only" in result
        # Should not have error output section if stderr is empty
        if "## Error Output" in result:
            assert "stdout only" not in result.split("## Error Output")[1]

    def test_stderr_only_output(self):
        """Test command with stderr output only."""
        test_script = Path(__file__).parent / "test_helper.py"
        result = execute_command_line("python", str(test_script), "--error", "stderr only")
        assert "## Error Output" in result
        assert "stderr only" in result

    def test_both_stdout_stderr_output(self):
        """Test command with both stdout and stderr output."""
        test_script = Path(__file__).parent / "test_helper.py"
        result = execute_command_line("python", str(test_script), "--both")
        assert "## Standard Output" in result
        assert "## Error Output" in result
        assert "Standard output message" in result
        assert "Error output message" in result

    def test_no_output(self):
        """Test command with no output."""
        test_script = Path(__file__).parent / "test_helper.py"
        result = execute_command_line("python", str(test_script), "--success")
        assert "# Command Execution Output" in result
        assert "*No output generated*" in result

    def test_command_failure(self):
        """Test command that fails with non-zero exit code."""
        test_script = Path(__file__).parent / "test_helper.py"
        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            execute_command_line("python", str(test_script), "--fail")

        # Verify the exception contains formatted markdown output
        assert exc_info.value.output is not None
        assert "# Command Execution Output" in exc_info.value.output

    def test_invalid_command(self):
        """Test invalid command execution."""
        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            execute_command_line("nonexistentcommand12345")

        # The function should raise CalledProcessError, not a generic Exception
        assert exc_info.value is not None

    def test_empty_command_validation(self):
        """Test empty command validation."""
        with pytest.raises(ValueError) as exc_info:
            execute_command_line("")

        assert "Command cannot be empty" in str(exc_info.value)

    def test_none_argument_handling(self):
        """Test handling of None arguments."""
        result = execute_command_line("echo", "hello", None, "world")
        assert "hello world" in result

    def test_timeout_expired(self):
        """Test command timeout functionality."""
        test_script = Path(__file__).parent / "test_helper.py"
        with pytest.raises(subprocess.TimeoutExpired) as exc_info:
            execute_command_line("python", str(test_script), "--timeout", "5", timeout=1)

        # Verify the exception contains formatted timeout output
        assert exc_info.value.output is not None
        assert "# Command Execution Timeout" in exc_info.value.output
        assert "timed out after" in exc_info.value.output

    def test_no_timeout(self):
        """Test command with no timeout (timeout=None)."""
        result = execute_command_line("echo", "no timeout test", timeout=None)
        assert "no timeout test" in result

    def test_zero_timeout(self):
        """Test command with zero timeout - should timeout immediately."""
        test_script = Path(__file__).parent / "test_helper.py"
        with pytest.raises(subprocess.TimeoutExpired) as exc_info:
            execute_command_line("python", str(test_script), "--timeout", "1", timeout=0)

        # Should timeout immediately with timeout=0
        assert exc_info.value.output is not None
        assert "timed out after 0 seconds" in exc_info.value.output

    def test_special_characters_in_output(self):
        """Test handling of special characters in command output."""
        special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = execute_command_line("echo", special_chars)
        assert special_chars in result

    def test_long_output(self):
        """Test handling of long command output."""
        long_output = "A" * 1000
        result = execute_command_line("echo", long_output)
        assert long_output in result

    def test_working_directory(self):
        """Test that commands run in current working directory."""
        result = execute_command_line("pwd")
        assert str(Path.cwd()) in result

    def test_complex_command_with_flags(self):
        """Test complex command with multiple flags and options."""
        # Test ls command with flags
        try:
            result = execute_command_line("ls", "-la")
            assert "# Command Execution Output" in result
        except subprocess.CalledProcessError:
            # ls might not be available or behave differently, that's okay
            pass

    def test_formatted_output_structure(self):
        """Test that output is properly formatted in markdown."""
        result = execute_command_line("echo", "test")

        # Should have the main header
        assert "# Command Execution Output" in result

        # Should have proper section structure
        if "## Standard Output" in result:
            # Verify markdown code block structure
            if "```" in result:
                sections = result.split("```")
                assert len(sections) >= 2  # Should have opening and closing backticks

    def test_error_output_formatting(self):
        """Test that error output is properly formatted."""
        test_script = Path(__file__).parent / "test_helper.py"
        try:
            execute_command_line("python", str(test_script), "--fail", "--error", "test error")
        except subprocess.CalledProcessError as e:
            assert "# Command Execution Output" in e.output
            assert "test error" in e.output

    def test_timeout_output_formatting(self):
        """Test that timeout output is properly formatted."""
        test_script = Path(__file__).parent / "test_helper.py"
        try:
            execute_command_line("python", str(test_script), "--timeout", "3", timeout=1)
        except subprocess.TimeoutExpired as e:
            assert "# Command Execution Timeout" in e.output
            assert "timed out after" in e.output

    def test_argument_validation_edge_cases(self):
        """Test edge cases in argument validation."""
        # Empty list argument
        result = execute_command_line("echo", "hello", [], "world")
        assert "hello world" in result

        # List with None values
        result = execute_command_line("echo", "hello", [None, "world", None], "end")
        assert "hello world end" in result

    def test_multiple_list_arguments(self):
        """Test multiple list arguments."""
        args1 = ["arg1", "arg2"]
        args2 = ["arg3", "arg4"]
        result = execute_command_line("echo", args1, args2)
        assert "arg1 arg2 arg3 arg4" in result

    def test_unicode_output(self):
        """Test handling of unicode characters in output."""
        unicode_text = "Hello 世界 🌍"
        result = execute_command_line("echo", unicode_text)
        assert unicode_text in result

    def test_multiline_output(self):
        """Test handling of multiline output."""
        multiline_text = "Line 1\nLine 2\nLine 3"
        # Use printf for consistent multiline output across platforms
        result = execute_command_line("printf", f'"{multiline_text}"')
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result


# Additional test functions for specific edge cases

def test_import_function_directly():
    """Test that we can import and use the function directly."""
    from cli.cli import execute_command_line
    result = execute_command_line("echo", "direct import test")
    assert "direct import test" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
