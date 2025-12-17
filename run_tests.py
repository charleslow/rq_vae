#!/usr/bin/env python3
"""Main test runner for RQ-VAE project.

This script provides a convenient way to run tests with various options.
It handles device selection (CPU/CUDA) and provides useful test execution modes.

Usage:
    # Run all tests
    python run_tests.py

    # Run on CPU only (useful even if CUDA is available)
    python run_tests.py --cpu

    # Run specific test file
    python run_tests.py tests/test_quantizer.py

    # Run specific test
    python run_tests.py tests/test_quantizer.py::TestResidualQuantizer::test_forward_shape

    # Run with coverage
    python run_tests.py --coverage

    # Run in verbose mode
    python run_tests.py -v

    # Run only fast tests (skip slow tests)
    python run_tests.py -m "not slow"
"""

import sys
import argparse
import subprocess
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Run RQ-VAE tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "path",
        nargs="?",
        default="tests/",
        help="Path to test file or directory (default: tests/)",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force tests to run on CPU even if CUDA is available",
    )

    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Only run tests that require CUDA (will fail if CUDA not available)",
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    parser.add_argument(
        "-s", "--no-capture",
        action="store_true",
        help="Don't capture stdout (useful for debugging)",
    )

    parser.add_argument(
        "-m",
        "--markers",
        help="Only run tests matching given mark expression (e.g., 'not slow')",
    )

    parser.add_argument(
        "-k",
        "--keyword",
        help="Only run tests matching given keyword expression",
    )

    parser.add_argument(
        "--failfast",
        action="store_true",
        help="Stop on first test failure",
    )

    args = parser.parse_args()

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()

    if args.cpu and cuda_available:
        print("🖥️  Forcing tests to run on CPU (CUDA available but disabled)")
        # Set environment variable to hide CUDA devices
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.cuda and not cuda_available:
        print("❌ Error: --cuda flag specified but CUDA is not available")
        return 1
    elif cuda_available:
        print(f"🚀 Running tests with CUDA support (GPU: {torch.cuda.get_device_name(0)})")
    else:
        print("🖥️  Running tests on CPU (CUDA not available)")

    # Build pytest command
    cmd = ["pytest"]

    # Add path
    cmd.append(args.path)

    # Add options
    if args.verbose:
        cmd.append("-vv")

    if args.no_capture:
        cmd.append("-s")

    if args.markers:
        cmd.extend(["-m", args.markers])

    if args.keyword:
        cmd.extend(["-k", args.keyword])

    if args.failfast:
        cmd.append("-x")

    if args.coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=html",
            "--cov-report=term-missing",
        ])

    # Print command
    print(f"\n📦 Running: {' '.join(cmd)}\n")

    # Run tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
