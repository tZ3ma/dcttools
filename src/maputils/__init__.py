# src/maputils/__init__.py
# flake8: noqa
"""maputils - A colletion of dictionairy utilites."""
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
