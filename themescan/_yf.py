"""Import yfinance with its deprecation-warning noise switched off.

yfinance runs warnings.filterwarnings('default', DeprecationWarning, module='^yfinance') when imported, which re-enables warnings that the
CLI had silenced (newer pandas versions flag many calls inside yfinance). Importing it through this module restores our 'ignore' filter afterwards.
"""
import warnings

import yfinance as yf  # noqa: F401  (re-exported)

warnings.filterwarnings("ignore")
