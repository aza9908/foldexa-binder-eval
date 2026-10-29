"""Shared fixtures.

Two kinds of test data:

* **Synthetic** -- built in-process from coordinates this file generates, so
  the core maths is tested without shipping any third-party structures.
* **Upstream** -- the ``Example/`` directory of the pinned ipSAE submodule,
  used for the exactness regression against the reference implementation.
  Those tests skip when the submodule has not been cloned.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
IPSAE_EXAMPLE = REPO_ROOT / "external" / "ipsae" / "Example"

requires_ipsae_example = pytest.mark.skipif(
    not IPSAE_EXAMPLE.is_dir(),
    reason="ipSAE submodule not cloned; run scripts/setup_upstream.sh",
)


def _write_pdb(path: Path, residues: list[tuple[str, int, str, np.ndarray, float]]) -> Path:
    """Write a minimal PDB with N, CA, C, O, CB per residue.

    ``residues`` is a list of ``(chain, res_id, res_name, ca_xyz, b_factor)``.
    The other backbone atoms are placed at small fixed offsets from CA, which
    is geometrically crude but entirely sufficient: every metric under test
    cares about inter-chain distances, not bond geometry.
    """
    offsets = {
        "N": np.array([-1.20, 0.30, 0.00]),
        "CA": np.array([0.00, 0.00, 0.00]),
        "C": np.array([1.20, 0.40, 0.00]),
        "O": np.array([1.60, 0.40, 1.10]),
        "CB": np.array([-0.30, -1.20, 0.60]),
    }
    lines: list[str] = []
    serial = 1
    for chain, res_id, res_name, ca, bfac in residues:
        for atom_name, delta in offsets.items():
            if res_name == "GLY" and atom_name == "CB":
                continue
            x, y, z = ca + delta
            element = atom_name[0]
            lines.append(
                f"ATOM  {serial:5d} {atom_name:<4s}{res_name:>4s} {chain}"
                f"{res_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"{1.00:6.2f}{bfac:6.2f}          {element:>2s}"
            )
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


BINDER_SEPARATION = 5.0
"""Chain separation in the two-chain fixture, in angstroms.

Chosen so the closest heavy atoms across the interface sit at about 3.4 A:
inside the 4.0 A contact cutoff that defines an interface residue, and
outside van der Waals clash range.
"""


@pytest.fixture
def two_chain_pdb(tmp_path: Path) -> Path:
    """A 20-residue binder packed against a 30-residue target.

    Both chains are straight rods laid side by side, so there is a genuine,
    countable interface: contacts and buried surface are non-zero but modest.
    The geometry is deliberately crude; what matters is that the inter-chain
    distances land in the ranges the metrics key off.
    """
    residues: list[tuple[str, int, str, np.ndarray, float]] = []

    # Target chain A: rod along x at y = 0.
    for i in range(30):
        ca = np.array([i * 3.3, 0.0, 0.0])
        name = ["ALA", "SER", "LEU", "GLU", "LYS", "THR"][i % 6]
        residues.append(("A", i + 1, name, ca, 90.0))

    # Binder chain B: parallel rod, overlapping the middle of A.
    for i in range(20):
        ca = np.array([(i + 5) * 3.3, BINDER_SEPARATION, 0.0])
        name = ["ILE", "ASN", "ARG", "VAL", "ASP", "GLN"][i % 6]
        residues.append(("B", i + 1, name, ca, 88.0))

    return _write_pdb(tmp_path / "complex.pdb", residues)


@pytest.fixture
def hbond_pdb(tmp_path: Path) -> Path:
    """Two residues whose backbone carbonyl and amide sit 2.9 A apart.

    A parallel-rod fixture cannot produce hydrogen bonds without also
    producing clashes, so the H-bond counter gets a purpose-built case: one
    donor-acceptor pair at textbook distance and nothing else in range.
    """
    lines = [
        # Chain A: a single residue, carbonyl oxygen at the origin.
        "ATOM      1 N    ALA A   1       0.000   3.000   0.000  1.00 90.00           N",
        "ATOM      2 CA   ALA A   1       0.000   2.000   0.000  1.00 90.00           C",
        "ATOM      3 C    ALA A   1       0.000   1.000   0.000  1.00 90.00           C",
        "ATOM      4 O    ALA A   1       0.000   0.000   0.000  1.00 90.00           O",
        "ATOM      5 CB   ALA A   1       1.500   2.000   0.000  1.00 90.00           C",
        # Chain B: amide nitrogen 2.9 A from that oxygen, along -y.
        "ATOM      6 N    ALA B   1       0.000  -2.900   0.000  1.00 88.00           N",
        "ATOM      7 CA   ALA B   1       0.000  -3.900   0.000  1.00 88.00           C",
        "ATOM      8 C    ALA B   1       0.000  -4.900   0.000  1.00 88.00           C",
        "ATOM      9 O    ALA B   1       0.000  -5.900   0.000  1.00 88.00           O",
        "ATOM     10 CB   ALA B   1       1.500  -3.900   0.000  1.00 88.00           C",
        "END",
    ]
    path = tmp_path / "hbond.pdb"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def refold_pdb(tmp_path: Path, two_chain_pdb: Path) -> Path:
    """The same complex with the binder displaced by a known amount.

    Every binder CA moves 1.0 A along z and the target is untouched, so the
    target-aligned backbone RMSD of the binder is exactly 1.0 A.
    """
    text = two_chain_pdb.read_text().splitlines()
    out: list[str] = []
    for line in text:
        if line.startswith("ATOM") and line[21] == "B":
            z = float(line[46:54]) + 1.0
            line = f"{line[:46]}{z:8.3f}{line[54:]}"
        out.append(line)
    path = tmp_path / "refold.pdb"
    path.write_text("\n".join(out) + "\n")
    return path


# Synthetic PAE system: chain A is residues 0..SPLIT-1, chain B the rest.
PAE_N = 300
PAE_SPLIT = 150


@pytest.fixture
def synthetic_pae() -> np.ndarray:
    """A 300x300 PAE matrix with a confidently predicted interface.

    The system is deliberately large. d0 is floored at 1.0 A, so when only a
    handful of residue pairs clear the PAE cutoff the pTM kernel is tiny no
    matter how low the error is -- that is ipSAE working as designed, not a
    defect, but it means a toy 50-residue complex cannot produce a high
    score. A 100x100 confident block reproduces the regime real binders sit
    in.
    """
    pae = np.full((PAE_N, PAE_N), 25.0)
    pae[:PAE_SPLIT, :PAE_SPLIT] = 2.0
    pae[PAE_SPLIT:, PAE_SPLIT:] = 2.0
    np.fill_diagonal(pae, 0.4)

    lo_a, hi_a = 25, 125
    lo_b, hi_b = PAE_SPLIT + 10, PAE_SPLIT + 110
    pae[lo_b:hi_b, lo_a:hi_a] = 2.0
    pae[lo_a:hi_a, lo_b:hi_b] = 2.0
    return pae


@pytest.fixture
def synthetic_plddt() -> np.ndarray:
    values = np.full(PAE_N, 92.0)
    values[PAE_SPLIT:] = 89.0
    return values


def d0_reference(length: float) -> float:
    """Yang & Skolnick d0, written out independently of the package."""
    length = max(float(length), 26.0)
    return max(1.0, 1.24 * (length - 15.0) ** (1.0 / 3.0) - 1.8)


def ptm_reference(pae: float, d0: float) -> float:
    return 1.0 / (1.0 + (pae / d0) ** 2)


__all__ = [
    "BINDER_SEPARATION",
    "IPSAE_EXAMPLE",
    "PAE_N",
    "PAE_SPLIT",
    "REPO_ROOT",
    "d0_reference",
    "ptm_reference",
    "requires_ipsae_example",
]
