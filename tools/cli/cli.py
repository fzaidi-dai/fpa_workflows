"""
Secure command execution utilities for uv environments.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any, Union


def execute_command_line(
    command: str,
    *args: Union[str, list[str]],
    timeout: int | None = 30
) -> str:
    """
    Execute a command securely using uv run in a child process.

    This function is designed as a tool for AI agents to execute generated code
    safely in sandboxed Docker environments. It automatically handles uv run prefix,
    captures complete output including Python stack traces, and provides structured
    error information.

    For AI Agents: Use this function to execute any command that needs to run
    in the uv environment. The function will automatically format output in
    markdown for easy parsing and will raise exceptions with detailed error
    information when commands fail.

    Args:
        command (str): The command to execute (e.g., 'python', 'script.py')
        *args (Union[str, list[str]]): Arguments to pass to the command.
            Can be individual strings or lists of strings for better validation.
        timeout (int | None): Timeout in seconds (None for no timeout, default 30)

    Returns:
        str: Formatted markdown output with separate stdout and stderr sections.
            Format:
            # Command Execution Output
            ## Standard Output
            [stdout content]
            ## Error Output
            [stderr content]

    Raises:
        subprocess.CalledProcessError: If the command exits with non-zero status.
            The .output attribute contains formatted markdown with error details.
        subprocess.TimeoutExpired: If the command times out.
            The .output attribute contains partial output before timeout.
        Exception: For other execution errors.

    Examples:
        # Execute a Python script
        output = execute_command_line("python", "test_script.py", "--verbose")

        # Execute with list arguments
        args = ["--input", "data.txt", "--output", "result.txt"]
        output = execute_command_line("python", "processor.py", args)

        # Execute a shell command
        output = execute_command_line("ls", "-la", "/workspace")

        # Handle errors
        try:
            output = execute_command_line("python", "buggy_script.py")
            return output  # Contains formatted markdown output
        except subprocess.CalledProcessError as e:
            # e.output contains formatted error information in markdown
            raise Exception(f"Script failed: {e.output}")
        except subprocess.TimeoutExpired as e:
            # e.output contains partial output before timeout
            raise Exception(f"Script timed out: {e.output}")
    """
    if not command:
        raise ValueError("Command cannot be empty")

    # Validate and flatten arguments
    validated_args = []
    for arg in args:
        if isinstance(arg, list):
            # Handle list arguments by flattening them
            validated_args.extend(str(item) for item in arg if item is not None)
        else:
            # Handle individual string arguments
            if arg is not None:
                validated_args.append(str(arg))

    # Build the full command with uv run prefix
    cmd_list = ["uv", "run", command] + validated_args

    try:
        # Execute command securely with timeout
        result = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path.cwd(),
            check=False  # We'll handle errors manually to capture full context
        )

        # Format output in markdown with separate sections
        stdout_output = result.stdout.strip() if result.stdout else ""
        stderr_output = result.stderr.strip() if result.stderr else ""

        # Create markdown formatted output
        output_sections = ["# Command Execution Output\n"]

        if stdout_output:
            output_sections.append("## Standard Output\n")
            output_sections.append("```\n")
            output_sections.append(stdout_output)
            output_sections.append("\n```")

        if stderr_output:
            output_sections.append("\n\n## Error Output\n")
            output_sections.append("```\n")
            output_sections.append(stderr_output)
            output_sections.append("\n```")

        # If no output, provide a clear message
        if not stdout_output and not stderr_output:
            output_sections.append("*No output generated*")

        formatted_output = "".join(output_sections)

        # Raise exception with formatted output if command failed
        if result.returncode != 0:
            error_msg = stderr_output if stderr_output else (
                stdout_output if stdout_output else f"Command failed with return code {result.returncode}"
            )
            raise subprocess.CalledProcessError(
                returncode=result.returncode,
                cmd=' '.join(cmd_list),
                output=formatted_output
            )

        return formatted_output

    except subprocess.TimeoutExpired as e:
        # Handle timeout with captured output
        timeout_stdout = e.output or ""
        timeout_stderr = e.stderr or ""

        # Format timeout output in markdown
        timeout_sections = ["# Command Execution Timeout\n"]
        timeout_sections.append(f"*Command timed out after {timeout} seconds*\n")

        if timeout_stdout:
            timeout_sections.append("\n## Standard Output (before timeout)\n```\n")
            timeout_sections.append(timeout_stdout)
            timeout_sections.append("\n```")

        if timeout_stderr:
            timeout_sections.append("\n\n## Error Output\n```\n")
            timeout_sections.append(timeout_stderr)
            timeout_sections.append("\n```")

        formatted_timeout_output = "".join(timeout_sections)

        raise subprocess.TimeoutExpired(
            cmd=' '.join(cmd_list),
            timeout=timeout,
            output=formatted_timeout_output
        )

    except subprocess.CalledProcessError:
        # Re-raise CalledProcessError (already handled above)
        raise

    except Exception as e:
        # Handle other execution errors
        raise Exception(f"Failed to execute command: {str(e)}")


