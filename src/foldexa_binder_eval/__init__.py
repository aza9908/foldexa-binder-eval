"""foldexa-binder-eval: twelve-metric acceptance testing for designed binders.

Pulls the binder-design stack into one decision:

* **RoseTTAFold3** (RosettaCommons ``foundry``) or **AlphaFold2/3** -- the
  predicted complex and its PAE/pLDDT.
* **BoltzGen** / **Boltz** -- generated binders and the independent refolds
  used for the self-consistency check.
* **ipSAE** (Dunbrack) -- interface-restricted PAE scoring, reimplemented
  natively here and cross-checked against the upstream script.
* **BindCraft** -- the filter set the default thresholds are calibrated
  against.

Typical use::

    from foldexa_binder_eval import evaluate_candidate, load_profile

    profile = load_profile("default")
    result = evaluate_candidate(
        structure="designs/design_0.cif",
        confidence=["designs/confidence_design_0.json"],
        refold="refolds/design_0.cif",
        thresholds=profile.thresholds,
        binder_chain="B",
    )
    print(result.verdict, [m.spec.key for m in result.failures])
"""

from __future__ import annotations

__version__ = "0.1.0"

from .confidence import ConfidenceData, load_confidence
from .interface import backbone_rmsd, compute_interface_geometry
from .ipsae import compute_interface_scores
from .metrics import (
    METRICS,
    METRICS_BY_KEY,
    Evaluation,
    MetricSpec,
    MetricValue,
    evaluate_candidate,
)
from .structure import Complex, load_structure
from .thresholds import ThresholdProfile, builtin_profiles, load_profile

__all__ = [
    "__version__",
    "Complex",
    "ConfidenceData",
    "Evaluation",
    "METRICS",
    "METRICS_BY_KEY",
    "MetricSpec",
    "MetricValue",
    "ThresholdProfile",
    "backbone_rmsd",
    "builtin_profiles",
    "compute_interface_geometry",
    "compute_interface_scores",
    "evaluate_candidate",
    "load_confidence",
    "load_profile",
    "load_structure",
]
