# src/dcttools/__init__.py
# flake8: noqa
"""dcttools - A colletion of dictionairy utilites."""
from importlib.metadata import version

from .core import (
    depth,
    flaggregate,
    kfltr,
    kfrep,
    kswap,
    maggregate,
    naggregate,
    to_dataframe,
)

__version__ = version(__name__)
