"""
tests/test_shopmind.py
ShopMind test suite — Block 11.

Covers:
  - Unit tests: feature engineering, demand forecast helpers, pricing model
  - Integration tests: FastAPI endpoints end-to-end
  - Agent tests: tool routing and response quality
  - Data quality tests: DB schema and feature matrix validation

Usage:
    pytest tests/test_shopmind.py -v                        # All tests
    pytest tests/test_shopmind.py -v -k "unit"              # Unit tests only
    pytest tests/test_shopmind.py -v -k "integration"       # Integration tests only
    pytest tests/test_shopmind.py -v -k "agent"             # Agent tests only
    pytest tests/test_shopmind.py --cov=models --cov=agent  # With coverage

Requirements:
    pip install pytest pytest-cov httpx
"""

import os
import sys
import json
import logging
import warnings
from datetime import date, datetime
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Stan.*")

# ── Path setup ────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

log = logging.getLogger("tests")

