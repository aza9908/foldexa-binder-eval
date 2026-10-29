"""Readers for predictor confidence output.

One loader covers the formats Foldexa actually sees:

* **AlphaFold2 / ColabFold** ``*_scores_*.json`` -- ``plddt``, ``pae`` (or
  ``predicted_aligned_error``), ``ptm``, ``iptm``.
* **AlphaFold3** ``*_full_data_*.json`` + ``*_summary_confidences_*.json`` --
  ``atom_plddts``, ``pae``, ``token_chain_ids``; summary carries ``iptm`` /
  ``ptm`` / ``chain_pair_iptm``.
* **RoseTTAFold3** (RosettaCommons ``foundry``) -- emits AF3-style confidence
  output, so the AF3 branch handles it. See ``models/rf3`` in that repo:
  ``compile_af3_style_confidence_outputs`` fills ``summary_confidences`` with
  ``ptm`` / ``iptm`` / ``has_clash`` / ``ranking_score``.
* **Boltz-1 / Boltz-2 / BoltzGen refolds** -- ``pae_*.npz`` and
  ``plddt_*.npz`` arrays plus ``confidence_*.json``.

Everything is normalised on the way out: pLDDT on a 0-100 scale, PAE in
angstroms, matrices indexed to match residue order in the structure file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["ConfidenceData", "load_confidence"]


@dataclass
class ConfidenceData:
    """Predictor confidence, normalised.

    Attributes
    ----------
    pae:
        Square predicted-aligned-error matrix in angstroms, one row/column per
        residue (token), or ``None`` if the file carried no PAE.
    plddt:
        Per-residue pLDDT on a 0-100 scale, or ``None``.
    ptm, iptm:
        Global scores as reported by the predictor, ``None`` if absent.
    chain_ids:
        Per-token chain ids when the format provides them (AF3/RF3). Used to
        sanity-check the matrix against the structure.
    source:
        Files the data came from, for the provenance block in the report.
    """

    pae: np.ndarray | None = None
    plddt: np.ndarray | None = None
    ptm: float | None = None
    iptm: float | None = None
    chain_ids: np.ndarray | None = None
    res_ids: np.ndarray | None = None
    has_clash: bool | None = None
    source: list[str] = field(default_factory=list)
    format: str = "unknown"

    @property
    def n_tokens(self) -> int | None:
        if self.pae is not None:
            return int(self.pae.shape[0])
        if self.plddt is not None:
            return int(self.plddt.shape[0])
        return None

    def check_against(self, n_residues: int) -> None:
        """Raise if the matrix cannot correspond to the loaded structure."""
        n = self.n_tokens
        if n is None or n == n_residues:
            return
        raise ValueError(
            f"Confidence arrays cover {n} tokens but the structure has "
            f"{n_residues} residues. This usually means the structure and the "
            "confidence file come from different predictions, or that hetero "
            "atoms / extra chains were dropped from one but not the other. "
            f"Sources: {', '.join(self.source) or 'n/a'}"
        )

    # ------------------------------------------------------------------
    def align_to(self, cx) -> ConfidenceData:
        """Reindex the arrays onto a :class:`~.structure.Complex`'s residues.

        AlphaFold3, RoseTTAFold3 and Boltz tokenise ligands, ions and
        modified residues as one token *per atom*, so a PAE matrix is
        routinely larger than the polymer residue count and the two are not
        in the same order. When the file carries ``token_chain_ids`` and
        ``token_res_ids`` the mapping is exact; otherwise a 1:1 correspondence
        is assumed and verified by length.
        """
        n_res = cx.n_residues
        if self.pae is None and self.plddt is None:
            return self

        if self.chain_ids is None or self.res_ids is None:
            self.check_against(n_res)
            return self

        lookup: dict[tuple[str, int], int] = {}
        for i, (chain, res) in enumerate(
            zip(self.chain_ids, self.res_ids, strict=True)
        ):
            lookup.setdefault((str(chain), int(res)), i)

        index = np.empty(n_res, dtype=int)
        missing: list[str] = []
        for residue in cx.residues:
            token = lookup.get((residue.chain_id, residue.res_id))
            if token is None:
                missing.append(residue.label)
                index[residue.index] = -1
            else:
                index[residue.index] = token

        if missing:
            preview = ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else "")
            raise ValueError(
                f"{len(missing)} residue(s) of the structure have no matching "
                f"token in the confidence file ({preview}). The structure and "
                "the confidence output are probably from different runs. "
                f"Sources: {', '.join(self.source) or 'n/a'}"
            )

        aligned = ConfidenceData(
            ptm=self.ptm,
            iptm=self.iptm,
            has_clash=self.has_clash,
            source=list(self.source),
            format=self.format,
        )
        if self.pae is not None:
            aligned.pae = self.pae[np.ix_(index, index)]
        if self.plddt is not None and self.plddt.shape[0] == len(self.chain_ids):
            aligned.plddt = self.plddt[index]
        elif self.plddt is not None and self.plddt.shape[0] == n_res:
            aligned.plddt = self.plddt
        return aligned


# ----------------------------------------------------------------------
# normalisation helpers
# ----------------------------------------------------------------------
def _normalise_plddt(values: np.ndarray) -> np.ndarray:
    """Return pLDDT on a 0-100 scale.

    AF2 PDB B-factors and ColabFold JSON use 0-100; AF3, RF3 and Boltz mmCIF
    use 0-1. Deciding on the observed maximum is the convention every
    downstream tool uses, including BindCraft.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size and finite.max() <= 1.0 + 1e-9:
        return arr * 100.0
    return arr


def _atoms_to_residues(
    atom_values: np.ndarray, token_counts: np.ndarray | None
) -> np.ndarray:
    """Collapse per-atom values to per-residue means."""
    if token_counts is None:
        return np.asarray(atom_values, dtype=float)
    out = np.zeros(len(token_counts))
    cursor = 0
    for i, count in enumerate(token_counts):
        count = int(count)
        out[i] = float(np.mean(atom_values[cursor : cursor + count]))
        cursor += count
    return out


def _as_square(matrix) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"PAE matrix is not square: shape {arr.shape}")
    return arr


# ----------------------------------------------------------------------
# per-format readers
# ----------------------------------------------------------------------
def _read_json(path: Path) -> ConfidenceData:
    with path.open() as fh:
        payload = json.load(fh)

    # ColabFold writes a one-element list wrapping the dict.
    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"{path} contains an empty JSON list")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported JSON confidence layout in {path}")

    data = ConfidenceData(source=[str(path)])

    # --- PAE -----------------------------------------------------------
    for key in ("pae", "predicted_aligned_error", "pae_matrix"):
        if key in payload and payload[key] is not None:
            data.pae = _as_square(payload[key])
            break
    else:
        # AF2 "distance" style: {"residue1": [...], "residue2": [...], "distance": [...]}
        if {"residue1", "residue2", "distance"} <= payload.keys():
            r1 = np.asarray(payload["residue1"], dtype=int) - 1
            r2 = np.asarray(payload["residue2"], dtype=int) - 1
            dist = np.asarray(payload["distance"], dtype=float)
            n = int(max(r1.max(), r2.max())) + 1
            mat = np.zeros((n, n))
            mat[r1, r2] = dist
            data.pae = mat

    # --- token identity (AF3 / RF3 / Boltz) -----------------------------
    if payload.get("token_chain_ids") is not None:
        data.chain_ids = np.asarray(
            [str(c) for c in payload["token_chain_ids"]], dtype=object
        )
    if payload.get("token_res_ids") is not None:
        data.res_ids = np.asarray(payload["token_res_ids"], dtype=int)

    # --- pLDDT ---------------------------------------------------------
    if "plddt" in payload and payload["plddt"] is not None:
        data.plddt = _normalise_plddt(np.asarray(payload["plddt"], dtype=float))
    elif payload.get("atom_plddts") is not None:
        # AF3 reports pLDDT per atom. Collapsing to tokens needs per-atom
        # residue ids, which the full-data file does not always carry; when
        # it does not, leave this empty and let the structure's B-factor
        # column supply pLDDT, which holds the same values.
        counts = _af3_atom_counts(payload)
        if counts is not None:
            data.plddt = _normalise_plddt(
                _atoms_to_residues(
                    np.asarray(payload["atom_plddts"], dtype=float), counts
                )
            )

    # --- global scores --------------------------------------------------
    for key in ("ptm", "pTM"):
        if key in payload and payload[key] is not None:
            data.ptm = float(np.asarray(payload[key]).reshape(-1)[0])
            break
    for key in ("iptm", "ipTM", "i_ptm"):
        if key in payload and payload[key] is not None:
            data.iptm = float(np.asarray(payload[key]).reshape(-1)[0])
            break

    if "has_clash" in payload and payload["has_clash"] is not None:
        value = payload["has_clash"]
        data.has_clash = bool(np.asarray(value).reshape(-1)[0])

    data.format = "af3-style" if data.chain_ids is not None else "af2-style"
    return data


def _af3_atom_counts(payload: dict) -> np.ndarray | None:
    """Number of atoms per token, derived from AF3 full-data arrays.

    Requires genuine *per-atom* residue ids. ``token_res_ids`` is not a
    substitute -- it is indexed by token, so zipping it against the atom
    arrays silently truncates and produces nonsense.
    """
    chain_ids = payload.get("atom_chain_ids")
    res_ids = payload.get("atom_res_ids")
    if chain_ids is None or res_ids is None:
        return None
    if len(chain_ids) != len(res_ids):
        return None
    if payload.get("atom_plddts") is not None and len(payload["atom_plddts"]) != len(
        chain_ids
    ):
        return None
    keys = list(zip(chain_ids, res_ids, strict=True))
    counts: list[int] = []
    previous = None
    for key in keys:
        if key != previous:
            counts.append(0)
            previous = key
        counts[-1] += 1
    return np.asarray(counts, dtype=int)


def _read_npz(path: Path) -> ConfidenceData:
    archive = np.load(path, allow_pickle=False)
    data = ConfidenceData(source=[str(path)], format="boltz-npz")
    keys = set(archive.files)

    for key in ("pae", "predicted_aligned_error"):
        if key in keys:
            data.pae = _as_square(archive[key])
            break
    for key in ("plddt", "pLDDT"):
        if key in keys:
            data.plddt = _normalise_plddt(np.asarray(archive[key], dtype=float))
            break

    if data.pae is None and data.plddt is None:
        raise ValueError(
            f"{path} has no 'pae' or 'plddt' array (found: {sorted(keys)})"
        )
    return data


# ----------------------------------------------------------------------
# public entry point
# ----------------------------------------------------------------------
def load_confidence(*paths: str | Path) -> ConfidenceData:
    """Load and merge one or more confidence files.

    Later files fill gaps left by earlier ones rather than overwriting them,
    so the usual Boltz/AF3 pattern works:

    >>> load_confidence("pae_model_0.npz", "confidence_model_0.json")

    and so does a single self-contained ColabFold JSON.
    """
    if not paths:
        raise ValueError("load_confidence() needs at least one path")

    merged = ConfidenceData()
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Confidence file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            part = _read_json(path)
        elif suffix in {".npz", ".npy"}:
            part = _read_npz(path)
        else:
            raise ValueError(
                f"Unsupported confidence format {suffix!r} for {path}. "
                "Expected .json, .npz or .npy."
            )

        for attr in (
            "pae",
            "plddt",
            "ptm",
            "iptm",
            "chain_ids",
            "res_ids",
            "has_clash",
        ):
            if getattr(merged, attr) is None:
                setattr(merged, attr, getattr(part, attr))
        merged.source.extend(part.source)
        if merged.format == "unknown":
            merged.format = part.format

    return merged
