"""Read BoltzGen output.

BoltzGen's ``Analyze`` step writes ``aggregate_metrics_*.csv`` and its
``Filter`` step consumes it. Those metrics overlap with ours but are not the
same set, so this adapter does two things:

* pairs each design CIF with its refold and its confidence file, so the
  twelve metrics can be computed from the structures directly;
* carries BoltzGen's own columns through into the report as extras, which is
  useful when comparing our gate against theirs.

Column names follow ``boltzgen.task.filter.filter.Filter``: ``design_ptm``,
``design_to_target_iptm``, ``min_design_to_target_pae``, ``filter_rmsd``,
``delta_sasa_refolded`` / ``delta_sasa_original``, ``plip_hbonds``,
``plip_saltbridge``, plus per-residue composition fractions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

__all__ = ["BoltzGenRun", "read_boltzgen_metrics", "BOLTZGEN_TO_FOLDEXA"]


# BoltzGen column -> our metric key, where the quantity is comparable.
# Deliberately partial: BoltzGen's PLIP H-bond count and our geometric count
# are not the same measurement, so they are reported side by side rather than
# mapped onto each other.
BOLTZGEN_TO_FOLDEXA: dict[str, str] = {
    "design_to_target_iptm": "complex_iptm",
    "filter_rmsd": "binder_rmsd",
    "delta_sasa_refolded": "delta_sasa",
    "delta_sasa_original": "delta_sasa",
}

_STRUCTURE_SUFFIXES = (".cif", ".pdb", ".mmcif", ".bcif")


@dataclass
class BoltzGenRun:
    """A BoltzGen design directory, resolved into scorable candidates."""

    root: Path
    designs: list[Path] = field(default_factory=list)
    refolds: dict[str, Path] = field(default_factory=dict)
    confidences: dict[str, list[Path]] = field(default_factory=dict)
    metrics: pd.DataFrame | None = None

    def pairs(self) -> list[tuple[Path, Path | None, list[Path]]]:
        """``(design, refold_or_None, confidence_files)`` per candidate."""
        out = []
        for design in self.designs:
            key = design.stem
            out.append((design, self.refolds.get(key), self.confidences.get(key, [])))
        return out

    def boltzgen_row(self, design_id: str) -> dict[str, float] | None:
        """BoltzGen's own metrics for one design, if the CSV was found."""
        if self.metrics is None or "design_id" not in self.metrics.columns:
            return None
        hit = self.metrics[self.metrics["design_id"].astype(str) == design_id]
        if hit.empty:
            return None
        return hit.iloc[0].to_dict()


def read_boltzgen_metrics(path: str | Path) -> pd.DataFrame:
    """Read an ``aggregate_metrics_*.csv`` or ``all_designs_metrics.csv``.

    A ``design_id`` column is derived from whichever identifier column the
    file carries, so downstream joins do not have to guess.
    """
    path = Path(path)
    if path.is_dir():
        found = sorted(path.glob("aggregate_metrics_*.csv")) or sorted(
            path.glob("*designs_metrics*.csv")
        )
        if not found:
            raise FileNotFoundError(
                f"No aggregate_metrics_*.csv or *designs_metrics*.csv in {path}"
            )
        path = found[0]

    frame = pd.read_csv(path)
    if "design_id" not in frame.columns:
        for candidate in ("name", "design", "id", "sample_name", "path"):
            if candidate in frame.columns:
                frame["design_id"] = (
                    frame[candidate].astype(str).map(lambda s: Path(s).stem)
                )
                break
    return frame


def discover_boltzgen_run(
    design_dir: str | Path,
    refold_dir: str | Path | None = None,
    metrics_csv: str | Path | None = None,
) -> BoltzGenRun:
    """Resolve a BoltzGen output directory into scorable candidates."""
    root = Path(design_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    designs = sorted(
        p
        for p in root.rglob("*")
        if p.suffix.lower() in _STRUCTURE_SUFFIXES and "refold" not in p.name.lower()
    )

    refolds: dict[str, Path] = {}
    refold_root = Path(refold_dir) if refold_dir else root
    for path in refold_root.rglob("*"):
        if path.suffix.lower() not in _STRUCTURE_SUFFIXES:
            continue
        name = path.stem
        if refold_dir is None and "refold" not in name.lower():
            continue
        key = name.lower().replace("refold_", "").replace("_refold", "")
        refolds[key] = path
        refolds.setdefault(name, path)

    confidences: dict[str, list[Path]] = {}
    for design in designs:
        stem = design.stem
        hits = [
            p
            for p in design.parent.iterdir()
            if p.suffix.lower() in {".json", ".npz"} and stem in p.stem
        ]
        if hits:
            confidences[stem] = sorted(hits)

    metrics = None
    try:
        metrics = read_boltzgen_metrics(metrics_csv or root)
    except FileNotFoundError:
        metrics = None

    return BoltzGenRun(
        root=root, designs=designs, refolds=refolds, confidences=confidences, metrics=metrics
    )
