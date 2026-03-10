"""
TDA-Enhanced Person Re-identification Package
Base package initialization
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from . import models
from . import datasets
from . import losses
from . import metrics

__all__ = [
    'models',
    'datasets',
    'losses',
    'metrics',
]
