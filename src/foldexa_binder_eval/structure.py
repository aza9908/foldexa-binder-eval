"""Structure loading and geometric primitives.

Thin wrapper over :mod:`biotite.structure` so the rest of the package never has
to care whether the model arrived as a PDB file or an mmCIF file, and so the
chain-splitting convention (binder chain vs target chain) lives in exactly one
place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import biotite.structure as struc
import numpy as np
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import BinaryCIFFile, CIFFile
from biotite.structure.io.pdbx import get_structure as cif_get_structure

__all__ = [
    "Complex",
    "load_structure",
    "HYDROPHOBIC_RESIDUES",
    "POLAR_DONORS",
    "POLAR_ACCEPTORS",
]


# Apolar + aromatic, matching the set BindCraft uses for interface
# hydrophobicity (functions/pyrosetta_utils.py).
HYDROPHOBIC_RESIDUES = frozenset("ACFILMPVWY")

# Heavy atoms that can donate a hydrogen bond, by (residue, atom) where the
# residue matters, plus a generic element rule for the rest.
POLAR_DONORS = frozenset({"N", "O"})
POLAR_ACCEPTORS = frozenset({"O", "N", "S"})

# Backbone atom names used to define "backbone RMSD".
BACKBONE_ATOMS = ("N", "CA", "C", "O")

_EXTRA_FIELDS = ["b_factor", "occupancy", "charge", "element"]


def _read_any(path: Path) -> struc.AtomArray:
    """Read the first model of a PDB / mmCIF / BinaryCIF file."""
    suffix = path.suffix.lower()
    if suffix in {".pdb", ".ent"}:
        pdb = PDBFile.read(str(path))
        return pdb.get_structure(model=1, extra_fields=["b_factor", "occupancy"])
    if suffix in {".cif", ".mmcif"}:
        cif = CIFFile.read(str(path))
        return cif_get_structure(cif, model=1, extra_fields=["b_factor", "occupancy"])
    if suffix in {".bcif", ".binarycif"}:
        bcif = BinaryCIFFile.read(str(path))
        return cif_get_structure(bcif, model=1, extra_fields=["b_factor", "occupancy"])
    raise ValueError(
        f"Unsupported structure format {suffix!r} for {path}. "
        "Expected one of .pdb, .cif, .mmcif, .bcif."
    )


@dataclass(frozen=True)
class Residue:
    """One residue, reduced to what the metrics actually need."""

    chain_id: str
    res_id: int
    ins_code: str
    res_name: str
    index: int  # position in Complex.residues

    @property
    def label(self) -> str:
        return f"{self.chain_id}{self.res_id}{self.ins_code}".strip()


class Complex:
    """A loaded binder:target complex.

    The binder is one chain, the target is one or more chains. Everything
    downstream addresses residues by their index in :attr:`residues`, which is
    the file order, so a PAE matrix emitted by the same predictor lines up
    without reordering.
    """

    def __init__(
        self,
        atoms: struc.AtomArray,
        binder_chain: str,
        target_chains: tuple[str, ...],
        source: Path | None = None,
    ) -> None:
        self.atoms = atoms
        self.binder_chain = binder_chain
        self.target_chains = target_chains
        self.source = source

        self._residues: list[Residue] | None = None
        self._res_index_of_atom: np.ndarray | None = None

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_file(
        cls,
        path: str | Path,
        binder_chain: str | None = None,
        target_chains: str | tuple[str, ...] | None = None,
        drop_hydrogens: bool = True,
        keep: str = "polymer",
    ) -> Complex:
        """Load a complex.

        ``keep`` controls what survives filtering:

        ``"polymer"`` (default)
            Amino acids and nucleotides, *including modified residues* such as
            MSE, TPO, SEP and PTR. Waters, ions and free ligands are dropped.
            Filtering on the ``hetero`` flag alone would discard modified
            residues, which breaks the chain and silently shifts every
            residue index against the predictor's PAE matrix.

        ``"all"``
            Everything except hydrogens. Use when the target is a small
            molecule or the ligand is part of the interface being scored.
        """
        path = Path(path)
        atoms = _read_any(path)

        if drop_hydrogens:
            atoms = atoms[atoms.element != "H"]

        if keep == "polymer":
            mask = struc.filter_amino_acids(atoms) | struc.filter_nucleotides(atoms)
            if mask.any():
                atoms = atoms[mask]
            else:
                # Nothing recognised as polymer: fall back rather than
                # returning an empty structure.
                atoms = atoms[~atoms.hetero]
        elif keep != "all":
            raise ValueError(f"keep must be 'polymer' or 'all', got {keep!r}")

        if atoms.array_length() == 0:
            raise ValueError(f"No atoms left after filtering in {path}")

        # altloc: keep the highest-occupancy copy per (chain, res, atom).
        atoms = _strip_altloc(atoms)

        present = _ordered_unique(atoms.chain_id.tolist())
        if binder_chain is None:
            binder_chain = present[0]
        if binder_chain not in present:
            raise ValueError(
                f"Binder chain {binder_chain!r} not found in {path}. "
                f"Chains present: {', '.join(present)}"
            )

        if target_chains is None:
            resolved = tuple(c for c in present if c != binder_chain)
        elif isinstance(target_chains, str):
            resolved = tuple(c.strip() for c in target_chains.split(",") if c.strip())
        else:
            resolved = tuple(target_chains)

        missing = [c for c in resolved if c not in present]
        if missing:
            raise ValueError(
                f"Target chain(s) {', '.join(missing)} not found in {path}. "
                f"Chains present: {', '.join(present)}"
            )
        if not resolved:
            raise ValueError(
                f"{path} contains only chain {binder_chain!r}; a binder:target "
                "complex needs at least one target chain."
            )
        if binder_chain in resolved:
            raise ValueError(
                f"Binder chain {binder_chain!r} also listed as a target chain."
            )

        # Every chain stays loaded even when only two are being scored. A PAE
        # matrix from AlphaFold2 or Boltz carries no token identifiers, so its
        # only alignment to the structure is residue order over the whole
        # model; dropping a chain here would silently shift that mapping.
        # Geometry is restricted separately, via pair_only().
        return cls(atoms, binder_chain, resolved, source=path)

    def pair_only(self) -> Complex:
        """A copy containing just the binder and target chains.

        Used for the structure-derived metrics, where a bystander chain would
        otherwise absorb part of the buried surface and distort the result.
        """
        chains = (self.binder_chain,) + self.target_chains
        if set(np.unique(self.atoms.chain_id)) <= set(chains):
            return self
        keep = np.isin(self.atoms.chain_id, np.array(chains))
        return Complex(
            self.atoms[keep], self.binder_chain, self.target_chains, source=self.source
        )

    # ------------------------------------------------------------------
    # residue bookkeeping
    # ------------------------------------------------------------------
    def _build_residues(self) -> None:
        atoms = self.atoms
        ins = getattr(atoms, "ins_code", None)
        if ins is None:
            ins = np.full(atoms.array_length(), "", dtype="U1")

        starts = struc.get_residue_starts(atoms)
        residues: list[Residue] = []
        res_index = np.empty(atoms.array_length(), dtype=int)

        bounds = list(starts) + [atoms.array_length()]
        for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
            residues.append(
                Residue(
                    chain_id=str(atoms.chain_id[lo]),
                    res_id=int(atoms.res_id[lo]),
                    ins_code=str(ins[lo]).strip(),
                    res_name=str(atoms.res_name[lo]),
                    index=i,
                )
            )
            res_index[lo:hi] = i

        self._residues = residues
        self._res_index_of_atom = res_index

    @property
    def residues(self) -> list[Residue]:
        if self._residues is None:
            self._build_residues()
        assert self._residues is not None
        return self._residues

    @property
    def res_index_of_atom(self) -> np.ndarray:
        if self._res_index_of_atom is None:
            self._build_residues()
        assert self._res_index_of_atom is not None
        return self._res_index_of_atom

    @property
    def n_residues(self) -> int:
        return len(self.residues)

    @property
    def residue_chain_ids(self) -> np.ndarray:
        return np.array([r.chain_id for r in self.residues])

    def residue_mask(self, chains: str | tuple[str, ...]) -> np.ndarray:
        """Boolean mask over :attr:`residues` for the given chain(s)."""
        if isinstance(chains, str):
            chains = (chains,)
        return np.isin(self.residue_chain_ids, np.array(chains))

    @property
    def binder_residue_mask(self) -> np.ndarray:
        return self.residue_mask(self.binder_chain)

    @property
    def target_residue_mask(self) -> np.ndarray:
        return self.residue_mask(self.target_chains)

    def atom_mask(self, chains: str | tuple[str, ...]) -> np.ndarray:
        if isinstance(chains, str):
            chains = (chains,)
        return np.isin(self.atoms.chain_id, np.array(chains))

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    def representative_coords(self) -> np.ndarray:
        """One coordinate per residue: CB where present, else CA, else centroid.

        This is the convention ipSAE and pDockQ use for contact counting.
        """
        coords = np.full((self.n_residues, 3), np.nan)
        atoms = self.atoms
        res_idx = self.res_index_of_atom
        names = atoms.atom_name

        for preferred in ("CB", "CA"):
            sel = names == preferred
            if not sel.any():
                continue
            idx = res_idx[sel]
            # np.isnan check keeps CB priority over CA.
            target = np.isnan(coords[idx, 0])
            coords[idx[target]] = atoms.coord[sel][target]

        # Residues with neither CB nor CA (nucleic acids, odd ligands): centroid.
        missing = np.isnan(coords[:, 0])
        if missing.any():
            for i in np.flatnonzero(missing):
                coords[i] = atoms.coord[res_idx == i].mean(axis=0)
        return coords

    def residue_distance_matrix(self) -> np.ndarray:
        """Pairwise CB-CB (CA fallback) distances, shape (n_res, n_res)."""
        coords = self.representative_coords()
        diff = coords[:, None, :] - coords[None, :, :]
        return np.sqrt((diff**2).sum(axis=-1))

    def min_heavy_atom_distances(self) -> np.ndarray:
        """Minimum heavy-atom distance between every pair of residues.

        Used for interface-residue definition, which is an atom-contact
        question rather than a CB-CB question.
        """
        n = self.n_residues
        res_idx = self.res_index_of_atom
        coord = self.atoms.coord
        out = np.full((n, n), np.inf)

        # Cell list keeps this tractable for large targets.
        cell = struc.CellList(self.atoms, cell_size=8.0)
        for i in range(n):
            sel = res_idx == i
            near = cell.get_atoms(coord[sel], radius=8.0)
            partners = np.unique(near[near >= 0])
            if partners.size == 0:
                continue
            d = np.linalg.norm(
                coord[sel][:, None, :] - coord[partners][None, :, :], axis=-1
            )
            per_atom_min = d.min(axis=0)
            # Both arrays are indexed by `partners`, so lengths always agree.
            for j, dist in zip(res_idx[partners], per_atom_min, strict=True):
                if dist < out[i, j]:
                    out[i, j] = dist
        np.fill_diagonal(out, 0.0)
        return np.minimum(out, out.T)

    # ------------------------------------------------------------------
    # confidence carried in the file
    # ------------------------------------------------------------------
    def representative_bfactor(self) -> np.ndarray:
        """B-factor at the representative atom (CB, else CA) of each residue.

        pDockQ was fit on CB pLDDT specifically, not on a per-residue mean, so
        the two are not interchangeable there.
        """
        atoms = self.atoms
        b = getattr(atoms, "b_factor", None)
        if b is None:
            return np.full(self.n_residues, np.nan)

        out = np.full(self.n_residues, np.nan)
        res_idx = self.res_index_of_atom
        for preferred in ("CA", "CB"):  # CB applied second so it wins
            sel = atoms.atom_name == preferred
            if sel.any():
                out[res_idx[sel]] = b[sel]
        # Residues with neither: fall back to the per-residue mean.
        missing = np.isnan(out)
        if missing.any():
            mean = self.per_residue_bfactor()
            out[missing] = mean[missing]
        return out

    def per_residue_bfactor(self) -> np.ndarray:
        """Mean B-factor per residue.

        For predictor output this column holds pLDDT, either 0-1 (AF3/Boltz
        mmCIF) or 0-100 (AF2 PDB). Scale normalisation happens in
        :mod:`foldexa_binder_eval.confidence`, not here.
        """
        b = getattr(self.atoms, "b_factor", None)
        if b is None:
            return np.full(self.n_residues, np.nan)
        res_idx = self.res_index_of_atom
        out = np.zeros(self.n_residues)
        counts = np.zeros(self.n_residues)
        np.add.at(out, res_idx, b)
        np.add.at(counts, res_idx, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(counts > 0, out / np.maximum(counts, 1), np.nan)

    def sequence(self, chains: str | tuple[str, ...] | None = None) -> str:
        """One-letter sequence for the given chain(s), file order."""
        if chains is None:
            chains = self.binder_chain
        mask = self.residue_mask(chains)
        return "".join(_three_to_one(r.res_name) for r in self.residues if mask[r.index])

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Complex(binder={self.binder_chain!r}, "
            f"target={','.join(self.target_chains)!r}, "
            f"residues={self.n_residues}, atoms={self.atoms.array_length()})"
        )


def load_structure(
    path: str | Path,
    binder_chain: str | None = None,
    target_chains: str | tuple[str, ...] | None = None,
) -> Complex:
    """Convenience wrapper around :meth:`Complex.from_file`."""
    return Complex.from_file(path, binder_chain, target_chains)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _ordered_unique(values: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for v in values:
        seen.setdefault(v, None)
    return list(seen)


def _strip_altloc(atoms: struc.AtomArray) -> struc.AtomArray:
    altloc = getattr(atoms, "altloc_id", None)
    if altloc is None:
        return atoms
    keep = np.isin(altloc, np.array(["", ".", "?", "A"]))
    return atoms[keep] if keep.any() else atoms


_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}


def _three_to_one(res_name: str) -> str:
    return _THREE_TO_ONE.get(res_name.upper(), "X")
