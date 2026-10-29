"""The twelve metrics, their provenance, and the evaluation pipeline.

A candidate binder is accepted only when every metric with a configured
threshold passes. The twelve sit in three tiers:

``confidence``
    Does the predictor believe the complex? Catches designs that never had a
    plausible pose to begin with.

``interface``
    Is the interface physically real? Catches poses that score well on
    confidence but bury nothing and make no contacts.

``developability``
    Will it survive expression, purification and storage? Catches binders
    that are real but sticky, aggregation-prone, or conformationally
    unstable.

Thresholds ship in ``config/``; the defaults trace to BindCraft's
``settings_filters/default_filters.json``, BoltzGen's ``Filter`` defaults, and
the ipSAE preprint. See ``docs/thresholds.md`` for the per-metric argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .confidence import ConfidenceData, load_confidence
from .interface import backbone_rmsd, compute_interface_geometry
from .ipsae import compute_interface_scores
from .structure import Complex

__all__ = [
    "MetricSpec",
    "MetricValue",
    "Evaluation",
    "METRICS",
    "METRICS_BY_KEY",
    "evaluate_candidate",
]


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    tier: str
    direction: str  # "higher" = bigger is better, "lower" = smaller is better
    units: str
    source: str
    description: str

    def passes(self, value: float, threshold: float) -> bool:
        if self.direction == "higher":
            return bool(value >= threshold)
        return bool(value <= threshold)


METRICS: tuple[MetricSpec, ...] = (
    # ---------------- tier 1: prediction confidence ----------------
    MetricSpec(
        key="binder_plddt",
        label="Binder pLDDT",
        tier="confidence",
        direction="higher",
        units="0-100",
        source="BindCraft Average_Binder_pLDDT (>= 0.8 on its 0-1 scale)",
        description=(
            "Mean pLDDT over binder residues in the complex. A binder the "
            "predictor models poorly is not a candidate regardless of how the "
            "interface scores."
        ),
    ),
    MetricSpec(
        key="complex_iptm",
        label="Complex ipTM",
        tier="confidence",
        direction="higher",
        units="0-1",
        source="BindCraft Average_i_pTM >= 0.5",
        description=(
            "Interface pTM as reported by the predictor. Global, so it is "
            "diluted on large targets; kept as a cheap first gate and "
            "cross-checked by ipSAE."
        ),
    ),
    MetricSpec(
        key="ipsae",
        label="ipSAE",
        tier="confidence",
        direction="higher",
        units="0-1",
        source="Dunbrack, doi:10.1101/2025.02.10.637595",
        description=(
            "Interface-restricted pTM: the pTM kernel averaged only over "
            "inter-chain residue pairs below the PAE cutoff, with d0 rescaled "
            "to the number of residues that clear it. The headline "
            "discriminator between a real binder and a confident-looking "
            "artefact."
        ),
    ),
    MetricSpec(
        key="pae_interaction",
        label="Interface PAE",
        tier="confidence",
        direction="lower",
        units="A",
        source="BindCraft Average_i_pAE <= 0.35 (0.35 x 31 A ~ 10.9 A)",
        description=(
            "Mean predicted aligned error across the binder:target block of "
            "the PAE matrix. Directly reads how confident the model is about "
            "the relative placement of the two chains."
        ),
    ),
    MetricSpec(
        key="pdockq",
        label="pDockQ",
        tier="confidence",
        direction="higher",
        units="0-1",
        source="Bryant et al., Nat Commun 13:1265 (2022)",
        description=(
            "Logistic fit of mean interface pLDDT against log10 contact count. "
            "0.23 is the published threshold separating acceptable from "
            "incorrect models."
        ),
    ),
    # ---------------- tier 2: interface quality ----------------
    MetricSpec(
        key="delta_sasa",
        label="Buried surface area",
        tier="interface",
        direction="higher",
        units="A^2",
        source="BindCraft Average_dSASA; typical de novo binder interfaces 800-2000 A^2",
        description=(
            "Total solvent-accessible surface buried on complex formation, "
            "both sides. Small interfaces rarely give measurable affinity."
        ),
    ),
    MetricSpec(
        key="n_interface_residues",
        label="Interface residues",
        tier="interface",
        direction="higher",
        units="count",
        source="BindCraft Average_n_InterfaceResidues >= 7",
        description=(
            "Binder residues with a heavy atom within 4 A of the target. "
            "Guards against single-loop grazing contacts."
        ),
    ),
    MetricSpec(
        key="n_interface_hbonds",
        label="Interface H-bonds",
        tier="interface",
        direction="higher",
        units="count",
        source="BindCraft Average_n_InterfaceHbonds >= 3",
        description=(
            "Cross-interface hydrogen bonds. Polar contacts carry the "
            "specificity that distinguishes a binder from a hydrophobic patch."
        ),
    ),
    MetricSpec(
        key="n_unsat_hbonds",
        label="Buried unsatisfied H-bonds",
        tier="interface",
        direction="lower",
        units="count",
        source="BindCraft Average_n_InterfaceUnsatHbonds <= 4",
        description=(
            "Polar atoms buried by binding that find no partner. Each one is "
            "a desolvation penalty paid for nothing, and the single best "
            "predictor of a design that scores well but does not bind."
        ),
    ),
    MetricSpec(
        key="shape_complementarity",
        label="Shape complementarity",
        tier="interface",
        direction="higher",
        units="0-1",
        source="BindCraft Average_ShapeComplementarity >= 0.6; Lawrence & Colman 1993",
        description=(
            "How well the two surfaces mate. Natural protein-protein "
            "interfaces cluster around 0.70-0.75; below 0.55 the interface is "
            "loosely packed."
        ),
    ),
    # ---------------- tier 3: developability ----------------
    MetricSpec(
        key="surface_hydrophobicity",
        label="Surface hydrophobicity",
        tier="developability",
        direction="lower",
        units="fraction",
        source="BindCraft Average_Surface_Hydrophobicity <= 0.35",
        description=(
            "Fraction of solvent-exposed binder residues that are apolar or "
            "aromatic. High values predict aggregation and non-specific "
            "binding, which is where otherwise good designs die in the lab."
        ),
    ),
    MetricSpec(
        key="binder_rmsd",
        label="Refold RMSD",
        tier="developability",
        direction="lower",
        units="A",
        source="BoltzGen refolding_rmsd_threshold = 2.5; BindCraft Average_Binder_RMSD <= 3.5",
        description=(
            "Backbone RMSD between the designed binder and an independent "
            "refold of its sequence. The self-consistency check: a sequence "
            "that does not fold back to its own design is not a real design."
        ),
    ),
)

METRICS_BY_KEY: dict[str, MetricSpec] = {m.key: m for m in METRICS}

TIERS: tuple[str, ...] = ("confidence", "interface", "developability")


@dataclass
class MetricValue:
    spec: MetricSpec
    value: float | None
    threshold: float | None = None
    passed: bool | None = None
    note: str = ""

    @property
    def status(self) -> str:
        if self.value is None or (
            isinstance(self.value, float) and not np.isfinite(self.value)
        ):
            return "missing"
        if self.threshold is None:
            return "unfiltered"
        return "pass" if self.passed else "fail"


@dataclass
class Evaluation:
    """Result of scoring one candidate against the threshold set."""

    design_id: str
    values: dict[str, MetricValue]
    extras: dict[str, Any] = field(default_factory=dict)
    profile: str = "default"
    missing_policy: str = "fail"

    @property
    def failures(self) -> list[MetricValue]:
        return [v for v in self.values.values() if v.status == "fail"]

    @property
    def missing(self) -> list[MetricValue]:
        return [v for v in self.values.values() if v.status == "missing"]

    @property
    def n_pass(self) -> int:
        return sum(1 for v in self.values.values() if v.status == "pass")

    @property
    def n_evaluated(self) -> int:
        return sum(
            1 for v in self.values.values() if v.status in {"pass", "fail"}
        )

    @property
    def verdict(self) -> str:
        """``PASS``, ``FAIL`` or ``INCOMPLETE``."""
        if self.failures:
            return "FAIL"
        blocking_missing = [
            v for v in self.missing if v.threshold is not None
        ]
        if blocking_missing:
            return "FAIL" if self.missing_policy == "fail" else "INCOMPLETE"
        return "PASS"

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {"design_id": self.design_id, "verdict": self.verdict}
        for key, mv in self.values.items():
            row[key] = mv.value
            row[f"{key}__pass"] = mv.passed
        row["n_pass"] = self.n_pass
        row["n_evaluated"] = self.n_evaluated
        row["failed_metrics"] = ";".join(v.spec.key for v in self.failures)
        return row


# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------
def evaluate_candidate(
    structure: str | Path | Complex,
    confidence: ConfidenceData | list[str | Path] | None = None,
    refold: str | Path | Complex | None = None,
    thresholds: dict[str, float | None] | None = None,
    binder_chain: str | None = None,
    target_chains: str | tuple[str, ...] | None = None,
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 10.0,
    rmsd_align_on: str = "target",
    design_id: str | None = None,
    compute_sc: bool = True,
    missing_policy: str = "fail",
    profile: str = "default",
) -> Evaluation:
    """Score one candidate and apply thresholds.

    ``structure`` is the designed binder:target complex. ``confidence`` is the
    predictor's PAE/pLDDT output for that same model. ``refold`` is an
    independent prediction of the binder sequence used for the self-consistency
    RMSD; omit it and that metric reports as missing.
    """
    cx = (
        structure
        if isinstance(structure, Complex)
        else Complex.from_file(structure, binder_chain, target_chains)
    )
    if design_id is None:
        design_id = cx.source.stem if cx.source else "design"

    conf: ConfidenceData | None
    if confidence is None:
        conf = None
    elif isinstance(confidence, ConfidenceData):
        conf = confidence
    else:
        conf = load_confidence(*confidence)
    if conf is not None:
        # Maps a token-indexed PAE matrix onto this complex's residues, which
        # matters whenever the prediction contained ligands or ions.
        conf = conf.align_to(cx)

    thresholds = dict(thresholds or {})
    extras: dict[str, Any] = {}
    raw: dict[str, float | None] = {key: None for key in METRICS_BY_KEY}

    # ---------------- confidence tier ----------------
    def _scaled(values: np.ndarray) -> np.ndarray | None:
        finite = values[np.isfinite(values)]
        if not finite.size:
            return None
        return values * 100.0 if finite.max() <= 1.0 + 1e-9 else values

    plddt_per_res = None
    plddt_representative = None
    if conf is not None and conf.plddt is not None:
        plddt_per_res = np.asarray(conf.plddt, dtype=float)
        plddt_representative = plddt_per_res
    else:
        plddt_per_res = _scaled(cx.per_residue_bfactor())
        # pDockQ is fit on CB pLDDT, so it gets the representative atom
        # rather than the residue mean.
        plddt_representative = _scaled(cx.representative_bfactor())
        if plddt_per_res is not None:
            extras["plddt_source"] = "structure B-factor column"

    if plddt_per_res is not None and plddt_per_res.shape[0] == cx.n_residues:
        binder_vals = plddt_per_res[cx.binder_residue_mask]
        if np.isfinite(binder_vals).any():
            raw["binder_plddt"] = float(np.nanmean(binder_vals))
        extras["complex_plddt"] = float(np.nanmean(plddt_per_res))

    if conf is not None and conf.iptm is not None:
        raw["complex_iptm"] = float(conf.iptm)
    if conf is not None and conf.ptm is not None:
        extras["ptm"] = float(conf.ptm)
    if conf is not None and conf.has_clash is not None:
        extras["has_clash"] = bool(conf.has_clash)

    if conf is not None and conf.pae is not None:
        distances = cx.residue_distance_matrix()
        plddt_for_scores = (
            plddt_representative
            if plddt_representative is not None
            and plddt_representative.shape[0] == cx.n_residues
            else np.full(cx.n_residues, np.nan)
        )
        scores = compute_interface_scores(
            pae=conf.pae,
            plddt=plddt_for_scores,
            distances=distances,
            binder_mask=cx.binder_residue_mask,
            target_mask=cx.target_residue_mask,
            pae_cutoff=pae_cutoff,
            dist_cutoff=dist_cutoff,
        )
        raw["ipsae"] = scores.ipsae
        raw["pae_interaction"] = scores.pae_interaction
        raw["pdockq"] = scores.pdockq
        extras.update(scores.as_dict())
        if raw["complex_iptm"] is None:
            raw["complex_iptm"] = scores.iptm_d0chn
            extras["iptm_source"] = "recomputed from PAE (d0chn)"

    # ---------------- interface tier ----------------
    geom = compute_interface_geometry(cx, compute_sc=compute_sc)
    raw["delta_sasa"] = geom.delta_sasa
    raw["n_interface_residues"] = float(geom.n_interface_residues_binder)
    raw["n_interface_hbonds"] = float(geom.n_interface_hbonds)
    raw["n_unsat_hbonds"] = float(geom.n_unsat_hbonds)
    raw["shape_complementarity"] = geom.shape_complementarity
    raw["surface_hydrophobicity"] = geom.surface_hydrophobicity
    extras.update(
        {
            "delta_sasa_binder": geom.delta_sasa_binder,
            "delta_sasa_target": geom.delta_sasa_target,
            "interface_fraction": geom.interface_fraction,
            "interface_hydrophobicity": geom.interface_hydrophobicity,
            "n_interface_residues_total": geom.n_interface_residues,
            "interface_residues": ",".join(geom.interface_residue_labels),
            "backend": geom.backend,
        }
    )

    # ---------------- developability tier ----------------
    if refold is not None:
        refold_cx = (
            refold
            if isinstance(refold, Complex)
            else Complex.from_file(refold, binder_chain, target_chains)
        )
        raw["binder_rmsd"] = backbone_rmsd(cx, refold_cx, align_on=rmsd_align_on)
        extras["rmsd_align_on"] = rmsd_align_on

    # ---------------- apply thresholds ----------------
    values: dict[str, MetricValue] = {}
    for spec in METRICS:
        value = raw[spec.key]
        if value is not None and isinstance(value, float) and not np.isfinite(value):
            value = None
        threshold = thresholds.get(spec.key)
        if value is None or threshold is None:
            values[spec.key] = MetricValue(spec, value, threshold, None)
        else:
            values[spec.key] = MetricValue(
                spec, value, threshold, spec.passes(value, threshold)
            )

    if raw["binder_rmsd"] is None and refold is None:
        values["binder_rmsd"].note = "no refold supplied"
    if conf is None:
        for key in ("complex_iptm", "ipsae", "pae_interaction", "pdockq"):
            values[key].note = "no confidence file supplied"

    return Evaluation(
        design_id=design_id,
        values=values,
        extras=extras,
        profile=profile,
        missing_policy=missing_policy,
    )
