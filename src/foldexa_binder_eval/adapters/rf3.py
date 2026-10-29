"""Read RoseTTAFold3 output from the RosettaCommons ``foundry`` package.

RF3 writes AF3-style confidence output: a ``summary_confidences`` block with
``ptm`` / ``iptm`` / ``has_clash`` / ``ranking_score``, and a ``confidences``
block with the ``pae`` and ``plddt`` arrays (see
``models/rf3/src/rf3/inference_engines/rf3.py`` upstream). That means
:func:`foldexa_binder_eval.confidence.load_confidence` parses it unchanged;
this adapter only has to pair each predicted structure with its JSON and
expose the ranking order.

Note the upstream caveat: the RF3 inference API, including confidence output
naming, is explicitly marked as still stabilising. If a future release renames
files, :func:`discover_rf3_outputs` is the single place to adjust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["RF3Prediction", "discover_rf3_outputs"]

_STRUCTURE_SUFFIXES = (".cif", ".pdb", ".mmcif", ".bcif")
_CONFIDENCE_HINTS = ("confidence", "summary", "scores", "full_data")


@dataclass
class RF3Prediction:
    """One RF3 model: its structure plus whatever confidence files exist."""

    structure: Path
    confidence: list[Path] = field(default_factory=list)
    example_id: str = ""
    sample_idx: int | None = None
    ranking_score: float | None = None

    @property
    def design_id(self) -> str:
        return self.example_id or self.structure.stem


def _looks_like_confidence(path: Path, stem: str) -> bool:
    if path.suffix.lower() not in {".json", ".npz"}:
        return False
    name = path.stem.lower()
    if stem.lower() in name:
        return True
    return any(hint in name for hint in _CONFIDENCE_HINTS)


def discover_rf3_outputs(out_dir: str | Path) -> list[RF3Prediction]:
    """Pair every predicted structure under ``out_dir`` with its confidence files.

    Sorted by ``ranking_score`` where the summary JSON provides it, best
    first, so taking the top model is a slice rather than a search.
    """
    root = Path(out_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    predictions: list[RF3Prediction] = []
    for structure in sorted(root.rglob("*")):
        if structure.suffix.lower() not in _STRUCTURE_SUFFIXES:
            continue
        stem = structure.stem
        confidence = sorted(
            p for p in structure.parent.iterdir() if _looks_like_confidence(p, stem)
        )
        # Prefer files that actually name this model over generic ones.
        specific = [p for p in confidence if stem.lower() in p.stem.lower()]
        chosen = specific or confidence

        prediction = RF3Prediction(
            structure=structure, confidence=chosen, example_id=stem
        )
        prediction.ranking_score = _read_ranking_score(chosen)
        predictions.append(prediction)

    predictions.sort(
        key=lambda p: (p.ranking_score is None, -(p.ranking_score or 0.0), p.design_id)
    )
    return predictions


def _read_ranking_score(paths: list[Path]) -> float | None:
    import json

    for path in paths:
        if path.suffix.lower() != ".json":
            continue
        try:
            with path.open() as fh:
                payload = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(payload, list) and payload:
            payload = payload[0]
        if isinstance(payload, dict) and payload.get("ranking_score") is not None:
            try:
                return float(payload["ranking_score"])
            except (TypeError, ValueError):
                continue
    return None
