#!/usr/bin/env python3
"""
Helper script for testing execute_command_line function.
Contains various test scenarios for different execution paths.
"""

import sys
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description="Test helper script for CLI testing")
    parser.add_argument("--success", action="store_true", help="Exit with success (0)")
    parser.add_argument("--fail", action="store_true", help="Exit with failure (1)")
    parser.add_argument("--timeout", type=int, default=0, help="Sleep for specified seconds before exit")
    parser.add_argument("--output", type=str, default="", help="Output message to stdout")
    parser.add_argument("--error", type=str, default="", help="Error message to stderr")
    parser.add_argument("--both", action="store_true", help="Output to both stdout and stderr")

    args = parser.parse_args()

    # Handle timeout
    if args.timeout > 0:
        time.sleep(args.timeout)

    # Handle output
    if args.both:
        print("Standard output message")
        print("Error output message", file=sys.stderr)
    else:
        if args.output:
            print(args.output)
        if args.error:
            print(args.error, file=sys.stderr)

    # Handle exit codes
    if args.fail:
        sys.exit(1)
    elif args.success:
        sys.exit(0)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
