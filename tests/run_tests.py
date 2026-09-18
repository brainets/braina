# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pytest",
#   "mcp<2",
#   "frites>=0.4.5",
#   "hoi",
#   "numpy",
#   "xarray",
#   "pandas",
#   "netcdf4",
#   "h5netcdf",
#   "scipy",
#   "mne",
#   "matplotlib",
#   "PyMuPDF",
# ]
# ///
"""Run the braina test suite with the MCP server's dependencies.

Usage:  uv run tests/run_tests.py [pytest args...]
"""
import os
import sys

import pytest

here = os.path.dirname(os.path.abspath(__file__))
sys.exit(pytest.main([here, "-q", "-p", "no:cacheprovider", *sys.argv[1:]]))
