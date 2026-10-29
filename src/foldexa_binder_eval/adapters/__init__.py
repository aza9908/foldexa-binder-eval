"""Adapters for upstream tools.

Each adapter turns one tool's output layout into the inputs
:func:`foldexa_binder_eval.metrics.evaluate_candidate` expects, so the rest of
the package never hard-codes a directory convention.
"""

from __future__ import annotations

from .boltzgen import BoltzGenRun, discover_boltzgen_run, read_boltzgen_metrics
from .ipsae_cli import IpsaeResult, parse_ipsae_output, run_ipsae
from .rf3 import RF3Prediction, discover_rf3_outputs

__all__ = [
    "BoltzGenRun",
    "IpsaeResult",
    "RF3Prediction",
    "discover_boltzgen_run",
    "discover_rf3_outputs",
    "parse_ipsae_output",
    "read_boltzgen_metrics",
    "run_ipsae",
]
