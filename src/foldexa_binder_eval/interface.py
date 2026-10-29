"""Structure-derived interface and developability metrics.

Everything here runs on open-source dependencies only (numpy / scipy /
biotite). PyRosetta is *optional*: when it is importable, ``shape
complementarity``, ``interface dG`` and ``buried unsatisfied H-bonds`` are
taken from Rosetta's ``InterfaceAnalyzerMover`` so numbers line up with
BindCraft; when it is not, the native implementations below are used and the
report records which backend produced each value.

The native implementations follow the published definitions:

* **ΔSASA** -- Shrake & Rupley solvent-accessible surface, probe 1.4 Å.
  ``ΔSASA = SASA(binder) + SASA(target) - SASA(complex)``, i.e. the total area
  buried on both sides.
* **Interface residues** -- any residue with a heavy atom within 4.0 Å of a
  heavy atom of the partner group. Same cutoff BindCraft uses.
* **H-bonds** -- heavy-atom donor/acceptor geometry, because predictor output
  carries no hydrogens. Counted across the interface only.
* **Buried unsatisfied H-bonds** -- polar heavy atoms that lose essentially
  all solvent accessibility on binding and gain no partner, the ``ddG``-style
  definition BindCraft's ``BuriedUnsatHbonds`` filter uses.
* **Shape complementarity** -- Lawrence & Colman, *J Mol Biol* **234**:946-950
  (1993): dot surfaces with outward normals, nearest-neighbour matching,
  ``S = n_A · n_B' · exp(-w d²)`` with ``w = 0.5``, then the mean of the two
  medians.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import biotite.structure as struc
import numpy as np
from scipy.spatial import cKDTree

from .structure import Complex

__all__ = ["InterfaceGeometry", "compute_interface_geometry", "backbone_rmsd"]


# Heavy-atom contact cutoff defining an interface residue (Å).
INTERFACE_CONTACT_CUTOFF = 4.0

# Heavy-atom donor-acceptor distance window for an H-bond (Å).
HBOND_MIN_DISTANCE = 2.4
HBOND_MAX_DISTANCE = 3.5

# An atom counts as buried below this absolute SASA (Å²).
BURIAL_SASA_CUTOFF = 0.1

# Shape-complementarity parameters (Lawrence & Colman 1993).
#
# The surfaces are built at the van der Waals radius, i.e. with a zero-radius
# probe, rather than as solvent-accessible surfaces. This matters: two
# solvent-accessible surfaces in van der Waals contact are separated by twice
# the probe radius (~3.4 A), which drives exp(-w d^2) to zero and collapses
# the score. Lawrence & Colman define SC on the molecular (solvent-excluded)
# surface, where the two sheets are in contact; the van der Waals envelope is
# the cheap approximation to it and keeps d in the 0-2 A range the kernel was
# designed for.
SC_WEIGHT = 0.5
SC_PROBE_RADIUS = 0.0
SC_DOTS_PER_ATOM = 150

# A dot on one side counts as an interface dot when a water probe could no
# longer reach it because of the partner, i.e. it lies within
# (r_partner + water) of a partner atom. This is the same burial criterion
# that defines dSASA, so the two interface definitions stay consistent.
SC_WATER_RADIUS = 1.4

# Radius over which dot normals are averaged before comparison. A raw van der
# Waals surface is bumpy at the atomic scale, so per-dot radial normals swing
# by tens of degrees between neighbouring atoms and depress the n_A . n_B
# term for interfaces that are in fact well mated. Averaging over 2.5 A
# recovers the local surface orientation the molecular surface would have.
SC_NORMAL_SMOOTHING = 2.5

# Residue is "surface" above this relative SASA.
SURFACE_RELATIVE_SASA = 0.25

# Apolar + aromatic side chains, for the surface-hydrophobicity metric.
# BindCraft takes Rosetta's ``is_apolar`` plus F/W/Y, which sweeps in glycine
# and proline. Both are excluded here: the metric is a proxy for aggregation
# and non-specific binding, and neither Gly nor Pro contributes a hydrophobic
# surface patch. Excluding them changes the scale, so the thresholds in
# ``profiles/`` are calibrated against this definition rather than copied
# from BindCraft. See docs/thresholds.md.
SURFACE_HYDROPHOBIC = frozenset("ACILMFVWY")

# Apolar + aromatic set BindCraft uses for *interface* hydrophobicity.
INTERFACE_HYDROPHOBIC = frozenset("ACFILMPVWY")

# Maximum SASA per residue in a Gly-X-Gly tripeptide (Å²), Tien et al. 2013
# theoretical values -- used to turn absolute SASA into relative SASA.
MAX_SASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
    "MSE": 224.0,
}

# van der Waals radii (Å) for the dot-surface construction.
VDW_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "SE": 1.90}
DEFAULT_VDW = 1.70

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}

# Side-chain atoms that can donate and/or accept, by residue.
_DONORS = {
    ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"), ("ASN", "ND2"),
    ("GLN", "NE2"), ("HIS", "ND1"), ("HIS", "NE2"), ("LYS", "NZ"),
    ("SER", "OG"), ("THR", "OG1"), ("TRP", "NE1"), ("TYR", "OH"),
    ("CYS", "SG"),
}
_ACCEPTORS = {
    ("ASN", "OD1"), ("ASP", "OD1"), ("ASP", "OD2"), ("GLN", "OE1"),
    ("GLU", "OE1"), ("GLU", "OE2"), ("HIS", "ND1"), ("HIS", "NE2"),
    ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"), ("MET", "SD"),
    ("CYS", "SG"),
}


@dataclass
class InterfaceGeometry:
    """Structure-derived numbers for one binder:target complex."""

    delta_sasa: float
    delta_sasa_binder: float
    delta_sasa_target: float
    binder_sasa_free: float
    interface_fraction: float
    n_interface_residues: int
    n_interface_residues_binder: int
    n_interface_hbonds: int
    n_unsat_hbonds: int
    shape_complementarity: float
    surface_hydrophobicity: float
    interface_hydrophobicity: float
    interface_residue_labels: list[str] = field(default_factory=list)
    backend: dict[str, str] = field(default_factory=dict)


# ----------------------------------------------------------------------
# SASA
# ----------------------------------------------------------------------
def _atom_sasa(atoms: struc.AtomArray, probe_radius: float = 1.4) -> np.ndarray:
    """Per-atom SASA, NaNs replaced with zero.

    ``biotite.structure.sasa`` returns NaN for atoms it does not recognise;
    treating those as zero keeps sums finite and matches what FreeSASA does
    with unknown atoms.
    """
    values = struc.sasa(atoms, probe_radius=probe_radius, vdw_radii="Single")
    return np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)


def _residue_sums(values: np.ndarray, res_index: np.ndarray, n_res: int) -> np.ndarray:
    out = np.zeros(n_res)
    np.add.at(out, res_index, values)
    return out


# ----------------------------------------------------------------------
# hydrogen bonds (heavy-atom geometry)
# ----------------------------------------------------------------------
def _polar_role(res_name: str, atom_name: str) -> tuple[bool, bool]:
    """Return ``(is_donor, is_acceptor)`` for a heavy atom."""
    res_name = res_name.upper()
    if atom_name == "N":  # backbone amide
        return True, False
    if atom_name == "O" or atom_name == "OXT":  # backbone carbonyl
        return False, True
    key = (res_name, atom_name)
    return key in _DONORS, key in _ACCEPTORS


def _near_partner_mask(
    coord: np.ndarray, side: np.ndarray, other: np.ndarray, cutoff: float
) -> np.ndarray:
    """Atoms of ``side`` within ``cutoff`` of any atom of ``other``."""
    if not side.any() or not other.any():
        return np.zeros_like(side)
    tree = cKDTree(coord[other])
    dist, _ = tree.query(coord[side])
    out = np.zeros_like(side)
    out[np.flatnonzero(side)[dist < cutoff]] = True
    return out


def _count_interface_hbonds(
    cx: Complex, binder_atom_mask: np.ndarray, target_atom_mask: np.ndarray
) -> int:
    atoms = cx.atoms
    coord = atoms.coord

    # Only atoms that could possibly reach across need classifying.
    near_b = _near_partner_mask(coord, binder_atom_mask, target_atom_mask, HBOND_MAX_DISTANCE)
    near_t = _near_partner_mask(coord, target_atom_mask, binder_atom_mask, HBOND_MAX_DISTANCE)

    donors_b, acceptors_b = _polar_atom_indices(atoms, near_b)
    donors_t, acceptors_t = _polar_atom_indices(atoms, near_t)

    # Count distinct atom pairs, not donor->acceptor assignments. Hydroxyls
    # and histidine nitrogens (SER OG, THR OG1, TYR OH, HIS ND1/NE2, CYS SG)
    # are both donor and acceptor, so one SER-OG...THR-OG1 bond is found once
    # in each sweep direction and would otherwise be counted twice.
    found: set[tuple[int, int]] = set()
    for donors, acceptors in ((donors_b, acceptors_t), (donors_t, acceptors_b)):
        if donors.size == 0 or acceptors.size == 0:
            continue
        pairs = cKDTree(coord[donors]).sparse_distance_matrix(
            cKDTree(coord[acceptors]), HBOND_MAX_DISTANCE, output_type="ndarray"
        )
        if not pairs.size:
            continue
        keep = pairs["v"] >= HBOND_MIN_DISTANCE
        for di, ai in zip(pairs["i"][keep], pairs["j"][keep], strict=True):
            lo, hi = int(donors[di]), int(acceptors[ai])
            found.add((lo, hi) if lo < hi else (hi, lo))
    return len(found)


def _polar_atom_indices(
    atoms: struc.AtomArray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    donors: list[int] = []
    acceptors: list[int] = []
    for idx in np.flatnonzero(mask):
        element = str(atoms.element[idx]).upper()
        if element not in {"N", "O", "S"}:
            continue
        is_d, is_a = _polar_role(str(atoms.res_name[idx]), str(atoms.atom_name[idx]))
        if is_d:
            donors.append(idx)
        if is_a:
            acceptors.append(idx)
    return np.asarray(donors, dtype=int), np.asarray(acceptors, dtype=int)


def _count_buried_unsatisfied(
    cx: Complex,
    binder_atom_mask: np.ndarray,
    target_atom_mask: np.ndarray,
    sasa_complex: np.ndarray,
    sasa_binder_free: np.ndarray,
    sasa_target_free: np.ndarray,
) -> int:
    """ddG-style buried unsatisfied polar atoms.

    An atom counts when it is polar, essentially solvent-inaccessible in the
    complex, was accessible before binding, and has no H-bond partner within
    ``HBOND_MAX_DISTANCE`` anywhere in the complex.
    """
    atoms = cx.atoms
    coord = atoms.coord

    free = np.zeros(atoms.array_length())
    free[binder_atom_mask] = sasa_binder_free
    free[target_atom_mask] = sasa_target_free

    polar = np.array(
        [str(e).upper() in {"N", "O"} for e in atoms.element], dtype=bool
    )
    newly_buried = polar & (sasa_complex <= BURIAL_SASA_CUTOFF) & (free > BURIAL_SASA_CUTOFF)
    if not newly_buried.any():
        return 0

    # Any polar partner in the whole complex satisfies the atom.
    partner_mask = polar.copy()
    partner_idx = np.flatnonzero(partner_mask)
    tree = cKDTree(coord[partner_idx])
    res_idx = cx.res_index_of_atom

    unsatisfied = 0
    for idx in np.flatnonzero(newly_buried):
        is_d, is_a = _polar_role(str(atoms.res_name[idx]), str(atoms.atom_name[idx]))
        if not (is_d or is_a):
            continue
        hits = tree.query_ball_point(coord[idx], r=HBOND_MAX_DISTANCE)
        satisfied = False
        for hi in hits:
            j = partner_idx[hi]
            if j == idx or res_idx[j] == res_idx[idx]:
                continue
            d = float(np.linalg.norm(coord[idx] - coord[j]))
            if d < HBOND_MIN_DISTANCE:
                continue
            other_d, other_a = _polar_role(
                str(atoms.res_name[j]), str(atoms.atom_name[j])
            )
            if (is_d and other_a) or (is_a and other_d):
                satisfied = True
                break
        if not satisfied:
            unsatisfied += 1
    return unsatisfied


# ----------------------------------------------------------------------
# shape complementarity
# ----------------------------------------------------------------------
def _fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1
    )


def _dot_surface(
    coord: np.ndarray,
    radii: np.ndarray,
    generate_mask: np.ndarray | None = None,
    n_dots: int = SC_DOTS_PER_ATOM,
) -> tuple[np.ndarray, np.ndarray]:
    """Solvent-accessible dot surface with outward normals.

    Dots occluded by a neighbouring atom of the *same* molecule are dropped,
    which leaves the molecular envelope the SC definition operates on.

    ``generate_mask`` restricts which atoms *emit* dots; occlusion is still
    tested against every atom passed in. Generating only around interface
    atoms is what keeps this tractable on a full-size target, and it does not
    change the result because distant dots are discarded by the interface
    band anyway.
    """
    unit = _fibonacci_sphere(n_dots)
    expanded = radii + SC_PROBE_RADIUS

    if generate_mask is None:
        generate_mask = np.ones(len(coord), dtype=bool)
    emit = np.flatnonzero(generate_mask)
    if emit.size == 0:
        return np.empty((0, 3)), np.empty((0, 3))

    points = (
        coord[emit][:, None, :] + unit[None, :, :] * expanded[emit][:, None, None]
    ).reshape(-1, 3)
    normals = np.tile(unit, (emit.size, 1))
    owner = np.repeat(emit, n_dots)

    atom_tree = cKDTree(coord)
    dot_tree = cKDTree(points)
    max_r = float(expanded.max())

    # Vectorised occlusion: every (dot, atom) pair within max_r at once.
    pairs = dot_tree.sparse_distance_matrix(
        atom_tree, max_r, output_type="ndarray"
    )
    if pairs.size:
        dot_i = pairs["i"]
        atom_j = pairs["j"]
        occluded = (pairs["v"] < expanded[atom_j] - 1e-6) & (owner[dot_i] != atom_j)
        keep = np.ones(len(points), dtype=bool)
        keep[np.unique(dot_i[occluded])] = False
    else:
        keep = np.ones(len(points), dtype=bool)

    return points[keep], normals[keep]


def _smooth_normals(
    dots: np.ndarray, normals: np.ndarray, radius: float = SC_NORMAL_SMOOTHING
) -> np.ndarray:
    """Average each dot's normal with its neighbours on the same surface."""
    if len(dots) == 0 or radius <= 0:
        return normals
    tree = cKDTree(dots)
    pairs = tree.sparse_distance_matrix(tree, radius, output_type="ndarray")
    if not pairs.size:
        return normals
    accumulated = np.zeros_like(normals)
    np.add.at(accumulated, pairs["i"], normals[pairs["j"]])
    norm = np.linalg.norm(accumulated, axis=1, keepdims=True)
    return np.where(norm > 1e-9, accumulated / np.maximum(norm, 1e-9), normals)


def _buried_dots(
    dots: np.ndarray, partner_coord: np.ndarray, partner_radii: np.ndarray
) -> np.ndarray:
    """Mask of dots a water probe can no longer reach because of the partner."""
    if len(dots) == 0 or len(partner_coord) == 0:
        return np.zeros(len(dots), dtype=bool)
    reach = partner_radii + SC_WATER_RADIUS
    pairs = cKDTree(dots).sparse_distance_matrix(
        cKDTree(partner_coord), float(reach.max()), output_type="ndarray"
    )
    mask = np.zeros(len(dots), dtype=bool)
    if pairs.size:
        hit = pairs["v"] < reach[pairs["j"]]
        mask[np.unique(pairs["i"][hit])] = True
    return mask


def _shape_complementarity(
    coord_a: np.ndarray,
    radii_a: np.ndarray,
    coord_b: np.ndarray,
    radii_b: np.ndarray,
) -> float:
    tree_a = cKDTree(coord_a)
    tree_b = cKDTree(coord_b)

    # Only atoms close to the partner can carry interface dots.
    margin = float(max(radii_a.max(), radii_b.max())) * 2 + 2 * SC_WATER_RADIUS + 2.0
    near_a = np.asarray(tree_b.query(coord_a)[0]) < margin
    near_b = np.asarray(tree_a.query(coord_b)[0]) < margin
    if not near_a.any() or not near_b.any():
        return float("nan")

    dots_a, norm_a = _dot_surface(coord_a, radii_a, generate_mask=near_a)
    dots_b, norm_b = _dot_surface(coord_b, radii_b, generate_mask=near_b)
    if len(dots_a) == 0 or len(dots_b) == 0:
        return float("nan")

    norm_a = _smooth_normals(dots_a, norm_a)
    norm_b = _smooth_normals(dots_b, norm_b)

    band_a = _buried_dots(dots_a, coord_b, radii_b)
    band_b = _buried_dots(dots_b, coord_a, radii_a)
    if band_a.sum() < 10 or band_b.sum() < 10:
        return float("nan")

    iface_a, iface_na = dots_a[band_a], norm_a[band_a]
    iface_b, iface_nb = dots_b[band_b], norm_b[band_b]

    def side(points, normals, other_points, other_normals) -> float:
        dist, idx = cKDTree(other_points).query(points)
        dot = np.einsum("ij,ij->i", normals, -other_normals[idx])
        return float(np.median(dot * np.exp(-SC_WEIGHT * dist**2)))

    sa = side(iface_a, iface_na, iface_b, iface_nb)
    sb = side(iface_b, iface_nb, iface_a, iface_na)
    return float((sa + sb) / 2.0)


# ----------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------
def compute_interface_geometry(
    cx: Complex, compute_sc: bool = True
) -> InterfaceGeometry:
    """Compute every structure-derived metric for ``cx``.

    Only the binder and target chains take part: a bystander chain in the
    same model would otherwise absorb buried surface and contacts that belong
    to neither side of the interface being scored.
    """
    cx = cx.pair_only()
    atoms = cx.atoms
    res_idx = cx.res_index_of_atom
    n_res = cx.n_residues

    binder_atoms_mask = cx.atom_mask(cx.binder_chain)
    target_atoms_mask = cx.atom_mask(cx.target_chains)

    binder_only = atoms[binder_atoms_mask]
    target_only = atoms[target_atoms_mask]

    sasa_complex = _atom_sasa(atoms)
    sasa_binder_free = _atom_sasa(binder_only)
    sasa_target_free = _atom_sasa(target_only)

    binder_free_total = float(sasa_binder_free.sum())
    target_free_total = float(sasa_target_free.sum())
    complex_total = float(sasa_complex.sum())

    delta_sasa = binder_free_total + target_free_total - complex_total
    delta_sasa_binder = binder_free_total - float(sasa_complex[binder_atoms_mask].sum())
    delta_sasa_target = target_free_total - float(sasa_complex[target_atoms_mask].sum())
    interface_fraction = (
        100.0 * delta_sasa_binder / binder_free_total if binder_free_total > 0 else 0.0
    )

    # --- interface residues --------------------------------------------
    tree_t = cKDTree(atoms.coord[target_atoms_mask])
    binder_atom_idx = np.flatnonzero(binder_atoms_mask)
    hits = tree_t.query_ball_point(
        atoms.coord[binder_atoms_mask], r=INTERFACE_CONTACT_CUTOFF
    )
    binder_iface_res: set[int] = set()
    target_atom_idx = np.flatnonzero(target_atoms_mask)
    target_iface_res: set[int] = set()
    for atom_i, partners in zip(binder_atom_idx, hits, strict=True):
        if partners:
            binder_iface_res.add(int(res_idx[atom_i]))
            for p in partners:
                target_iface_res.add(int(res_idx[target_atom_idx[p]]))

    residues = cx.residues
    labels = sorted(
        (residues[i].label for i in binder_iface_res),
        key=lambda s: (s[0], int("".join(ch for ch in s[1:] if ch.isdigit()) or 0)),
    )

    # --- interface hydrophobicity ---------------------------------------
    if binder_iface_res:
        codes = [
            _THREE_TO_ONE.get(residues[i].res_name.upper(), "X")
            for i in binder_iface_res
        ]
        n_phobic = sum(1 for c in codes if c in INTERFACE_HYDROPHOBIC)
        interface_hydrophobicity = 100.0 * n_phobic / len(codes)
    else:
        interface_hydrophobicity = 0.0

    # --- surface hydrophobicity of the free binder ----------------------
    binder_res_mask = cx.binder_residue_mask
    binder_res_ids = np.flatnonzero(binder_res_mask)
    binder_res_index_local = res_idx[binder_atoms_mask]
    per_res_free = np.zeros(n_res)
    np.add.at(per_res_free, binder_res_index_local, sasa_binder_free)

    exposed_apolar = 0
    exposed_total = 0
    for i in binder_res_ids:
        name = residues[i].res_name.upper()
        max_sasa = MAX_SASA.get(name)
        if max_sasa is None:
            continue
        rel = per_res_free[i] / max_sasa
        if rel >= SURFACE_RELATIVE_SASA:
            exposed_total += 1
            if _THREE_TO_ONE.get(name, "X") in SURFACE_HYDROPHOBIC:
                exposed_apolar += 1
    surface_hydrophobicity = (
        exposed_apolar / exposed_total if exposed_total else float("nan")
    )

    # --- hydrogen bonds --------------------------------------------------
    n_hbonds = _count_interface_hbonds(cx, binder_atoms_mask, target_atoms_mask)
    n_unsat = _count_buried_unsatisfied(
        cx,
        binder_atoms_mask,
        target_atoms_mask,
        sasa_complex,
        sasa_binder_free,
        sasa_target_free,
    )

    # --- shape complementarity -------------------------------------------
    if compute_sc:
        radii = np.array(
            [VDW_RADII.get(str(e).upper(), DEFAULT_VDW) for e in atoms.element]
        )
        sc = _shape_complementarity(
            atoms.coord[binder_atoms_mask],
            radii[binder_atoms_mask],
            atoms.coord[target_atoms_mask],
            radii[target_atoms_mask],
        )
    else:
        sc = float("nan")

    return InterfaceGeometry(
        delta_sasa=delta_sasa,
        delta_sasa_binder=delta_sasa_binder,
        delta_sasa_target=delta_sasa_target,
        binder_sasa_free=binder_free_total,
        interface_fraction=interface_fraction,
        n_interface_residues=len(binder_iface_res) + len(target_iface_res),
        n_interface_residues_binder=len(binder_iface_res),
        n_interface_hbonds=n_hbonds,
        n_unsat_hbonds=n_unsat,
        shape_complementarity=sc,
        surface_hydrophobicity=surface_hydrophobicity,
        interface_hydrophobicity=interface_hydrophobicity,
        interface_residue_labels=labels,
        backend={
            "sasa": "biotite Shrake-Rupley",
            "hbonds": "native heavy-atom geometry",
            "unsat_hbonds": "native ddG-style burial",
            "shape_complementarity": "native Lawrence-Colman" if compute_sc else "skipped",
        },
    )


# ----------------------------------------------------------------------
# RMSD
# ----------------------------------------------------------------------
def _backbone_subset(cx: Complex, chains) -> struc.AtomArray:
    atoms = cx.atoms
    mask = cx.atom_mask(chains) & np.isin(atoms.atom_name, ["N", "CA", "C", "O"])
    return atoms[mask]


def _atom_keys(atoms: struc.AtomArray) -> list[tuple[str, int, str, str]]:
    """``(chain, res_id, ins_code, atom_name)`` per atom.

    The chain id and insertion code are part of the key, not decoration.
    A multi-chain target restarts residue numbering per chain -- every Fab
    has an H and an L chain both beginning at 1 -- so a key of
    ``(res_id, atom_name)`` collides across chains and silently matches the
    wrong atoms. Antibody numbering (100, 100A, 100B) collides the same way
    without the insertion code.
    """
    ins = getattr(atoms, "ins_code", None)
    if ins is None:
        ins = np.full(atoms.array_length(), "", dtype="U1")
    return [
        (str(c), int(r), str(i).strip(), str(n))
        for c, r, i, n in zip(
            atoms.chain_id, atoms.res_id, ins, atoms.atom_name, strict=True
        )
    ]


def _match_by_residue(
    a: struc.AtomArray, b: struc.AtomArray
) -> tuple[struc.AtomArray, struc.AtomArray]:
    """Keep only the atoms present in both arrays, in ``b`` order."""
    key_a: dict[tuple[str, int, str, str], int] = {}
    for i, key in enumerate(_atom_keys(a)):
        key_a.setdefault(key, i)

    idx_a: list[int] = []
    idx_b: list[int] = []
    for j, key in enumerate(_atom_keys(b)):
        i = key_a.get(key)
        if i is not None:
            idx_a.append(i)
            idx_b.append(j)
    return a[idx_a], b[idx_b]


def backbone_rmsd(
    reference: Complex, mobile: Complex, align_on: str = "target"
) -> float:
    """Backbone RMSD of the binder between two models, in angstroms.

    ``align_on="target"`` superposes on the target backbone first, so the
    result answers "does the binder land in the same pose" -- BindCraft's
    ``Binder_RMSD``. ``align_on="binder"`` superposes the binder on itself,
    answering "does the binder fold the same way" -- BoltzGen's
    ``filter_rmsd`` self-consistency check.
    """
    if align_on not in {"target", "binder"}:
        raise ValueError("align_on must be 'target' or 'binder'")

    ref_binder = _backbone_subset(reference, reference.binder_chain)
    mob_binder = _backbone_subset(mobile, mobile.binder_chain)

    if align_on == "binder":
        a, b = _match_by_residue(ref_binder, mob_binder)
        if a.array_length() < 3:
            return float("nan")
        transform = struc.superimpose(a, b)
        return float(struc.rmsd(a, transform.apply(b)))

    ref_target = _backbone_subset(reference, reference.target_chains)
    mob_target = _backbone_subset(mobile, mobile.target_chains)
    ta, tb = _match_by_residue(ref_target, mob_target)
    if ta.array_length() < 3:
        return float("nan")

    # Fit on the target, then carry the binder through the same transform so
    # the RMSD reports the binder's pose relative to the target.
    transform = struc.superimpose(ta, tb)
    moved_binder = transform.apply(mob_binder)
    a, b = _match_by_residue(ref_binder, moved_binder)
    if a.array_length() < 3:
        return float("nan")
    return float(struc.rmsd(a, b))
