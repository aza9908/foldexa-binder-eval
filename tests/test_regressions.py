"""Regressions for bugs that produced plausible but wrong numbers.

Each test here corresponds to a defect that did not crash and did not fail
any existing test — it returned a number that looked reasonable and was
wrong. They are grouped in one file because that shared property is what
makes them worth pinning down explicitly.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from foldexa_binder_eval import (
    Complex,
    backbone_rmsd,
    compute_interface_geometry,
    evaluate_candidate,
    load_confidence,
)
from foldexa_binder_eval.cli import _auto_confidence
from foldexa_binder_eval.metrics import METRICS_BY_KEY


# ----------------------------------------------------------------------
# Residue matching across a multi-chain target
# ----------------------------------------------------------------------
def _multi_chain_target(tmp_path, name: str, binder_dz: float = 0.0):
    """Binder B against a two-chain target H+L that both restart at residue 1.

    This is the layout of every Fab complex, and it is where a residue key
    without the chain id collides: H1 and L1 are different residues with the
    same number.
    """
    lines: list[str] = []
    serial = 1

    def emit(chain, res_id, res_name, xyz, bfac):
        nonlocal serial
        for atom_name, delta in (
            ("N", (-1.2, 0.3, 0.0)),
            ("CA", (0.0, 0.0, 0.0)),
            ("C", (1.2, 0.4, 0.0)),
            ("O", (1.6, 0.4, 1.1)),
        ):
            x, y, z = (xyz[i] + delta[i] for i in range(3))
            lines.append(
                f"ATOM  {serial:5d} {atom_name:<4s}{res_name:>4s} {chain}"
                f"{res_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"{1.00:6.2f}{bfac:6.2f}          {atom_name[0]:>2s}"
            )
            serial += 1

    # Target chain H, residues 1..10, and chain L, also residues 1..10 but
    # displaced well away so the two are unmistakably different residues.
    for i in range(10):
        emit("H", i + 1, "ALA", (i * 3.3, 0.0, 0.0), 90.0)
    for i in range(10):
        emit("L", i + 1, "GLY", (i * 3.3, 40.0, 0.0), 90.0)
    # Binder chain B against chain H.
    for i in range(10):
        emit("B", i + 1, "LEU", (i * 3.3, 5.0, binder_dz), 88.0)

    path = tmp_path / name
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def test_rmsd_is_correct_with_a_multi_chain_target(tmp_path):
    """The binder is displaced exactly 1.0 A; the target is identical.

    With a residue key that ignores the chain id, H and L residues collide
    and the measured RMSD was 8.0 A instead of 1.0.
    """
    reference = Complex.from_file(
        _multi_chain_target(tmp_path, "ref.pdb", binder_dz=0.0),
        binder_chain="B",
        target_chains="H,L",
    )
    mobile = Complex.from_file(
        _multi_chain_target(tmp_path, "mob.pdb", binder_dz=1.0),
        binder_chain="B",
        target_chains="H,L",
    )
    assert backbone_rmsd(reference, mobile, align_on="target") == pytest.approx(
        1.0, abs=1e-3
    )


def test_rmsd_of_identical_multi_chain_models_is_zero(tmp_path):
    cx = Complex.from_file(
        _multi_chain_target(tmp_path, "same.pdb"),
        binder_chain="B",
        target_chains="H,L",
    )
    # Tolerance is float32 coordinate precision, not slack in the fit.
    assert backbone_rmsd(cx, cx, align_on="target") == pytest.approx(0.0, abs=1e-4)


# ----------------------------------------------------------------------
# Hydrogen bonds counted once per atom pair
# ----------------------------------------------------------------------
def test_dual_role_hydroxyl_pair_counts_once(tmp_path):
    """SER OG and THR OG1 are both donor and acceptor.

    One bond between them was found in each sweep direction and counted
    twice.
    """
    lines = [
        "ATOM      1 N    SER A   1       0.000   3.000   0.000  1.00 90.00           N",
        "ATOM      2 CA   SER A   1       0.000   2.000   0.000  1.00 90.00           C",
        "ATOM      3 C    SER A   1       1.200   2.400   0.000  1.00 90.00           C",
        "ATOM      4 CB   SER A   1      -1.000   1.400   0.000  1.00 90.00           C",
        "ATOM      5 OG   SER A   1       0.000   0.000   0.000  1.00 90.00           O",
        "ATOM      6 N    THR B   1       0.000  -5.900   0.000  1.00 88.00           N",
        "ATOM      7 CA   THR B   1       0.000  -4.900   0.000  1.00 88.00           C",
        "ATOM      8 C    THR B   1       1.200  -5.300   0.000  1.00 88.00           C",
        "ATOM      9 CB   THR B   1      -1.000  -3.800   0.000  1.00 88.00           C",
        "ATOM     10 OG1  THR B   1       0.000  -2.900   0.000  1.00 88.00           O",
    ]
    path = tmp_path / "hydroxyls.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")

    cx = Complex.from_file(path, binder_chain="B", target_chains="A")
    geometry = compute_interface_geometry(cx, compute_sc=False)
    assert geometry.n_interface_hbonds == 1


# ----------------------------------------------------------------------
# Token alignment
# ----------------------------------------------------------------------
def test_insertion_codes_map_to_distinct_pae_rows(tmp_path):
    """Residues 100, 100A, 100B share a res_id but are three residues.

    Keyed on (chain, res_id) alone they all mapped onto one PAE row, leaving
    two rows unused and silently duplicating a third.
    """
    lines = []
    serial = 1
    for chain, res_id, ins in (
        ("A", 100, " "),
        ("A", 100, "A"),
        ("A", 100, "B"),
        ("B", 1, " "),
        ("B", 2, " "),
    ):
        for atom_name in ("N", "CA", "C", "O"):
            lines.append(
                f"ATOM  {serial:5d} {atom_name:<4s} ALA {chain}"
                f"{res_id:4d}{ins}   {serial * 1.0:8.3f}{0.0:8.3f}{0.0:8.3f}"
                f"{1.00:6.2f}{90.00:6.2f}          {atom_name[0]:>2s}"
            )
            serial += 1
    path = tmp_path / "insertion.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")

    cx = Complex.from_file(path, binder_chain="B", target_chains="A")
    assert cx.n_residues == 5

    n = 5
    pae = np.arange(n * n, dtype=float).reshape(n, n)
    payload = {
        "pae": pae.tolist(),
        "token_chain_ids": ["A", "A", "A", "B", "B"],
        "token_res_ids": [100, 100, 100, 1, 2],
    }
    conf_path = tmp_path / "full_data.json"
    conf_path.write_text(json.dumps(payload))

    aligned = load_confidence(conf_path).align_to(cx)
    # A correct mapping is the identity here, so the matrix is unchanged.
    assert np.array_equal(aligned.pae, pae)


def test_modified_residue_uses_its_ca_token(tmp_path):
    """A per-atom-tokenised residue is attributed to its CA, not its first atom."""
    lines = [
        # Chain A: one canonical residue, then a TPO tokenised per atom.
        "ATOM      1 N    ALA A   1       0.000   0.000   0.000  1.00 90.00           N",
        "ATOM      2 CA   ALA A   1       1.000   0.000   0.000  1.00 90.00           C",
        "HETATM    3 N    TPO A   2       2.000   0.000   0.000  1.00 90.00           N",
        "HETATM    4 CA   TPO A   2       3.000   0.000   0.000  1.00 90.00           C",
        "HETATM    5 CB   TPO A   2       4.000   0.000   0.000  1.00 90.00           C",
        "ATOM      6 N    GLY B   1       0.000   5.000   0.000  1.00 88.00           N",
        "ATOM      7 CA   GLY B   1       1.000   5.000   0.000  1.00 88.00           C",
    ]
    path = tmp_path / "modified.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")

    cx = Complex.from_file(path, binder_chain="B", target_chains="A")
    assert cx.n_residues == 3  # ALA, TPO, GLY

    # Tokens: ALA(1), then three for TPO(2) -- N, CA, CB -- then GLY(1).
    n = 5
    pae = np.zeros((n, n))
    # Mark the CA token of TPO (index 2) distinctly from its N (1) and CB (3).
    pae[2, :] = 7.0
    pae[:, 2] = 7.0
    payload = {
        "pae": pae.tolist(),
        "token_chain_ids": ["A", "A", "A", "A", "B"],
        "token_res_ids": [1, 2, 2, 2, 1],
    }
    conf_path = tmp_path / "full_data.json"
    conf_path.write_text(json.dumps(payload))

    aligned = load_confidence(conf_path).align_to(cx)
    assert aligned.pae.shape == (3, 3)
    # Residue index 1 is the TPO; it must have picked up the CA token's row.
    assert aligned.pae[1, 0] == pytest.approx(7.0)


# ----------------------------------------------------------------------
# Sparse AF2 PAE
# ----------------------------------------------------------------------
def test_one_sided_pae_triangle_is_mirrored(tmp_path):
    """The residue1/residue2/distance layout, supplied as one triangle."""
    n = 4
    r1, r2, dist = [], [], []
    for i in range(n):
        for j in range(i, n):
            r1.append(i + 1)
            r2.append(j + 1)
            dist.append(float(abs(i - j) + 1))
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"residue1": r1, "residue2": r2, "distance": dist}))

    pae = load_confidence(path).pae
    assert pae.shape == (n, n)
    assert np.allclose(pae, pae.T)
    # No cell was left at the fabricated zero-error value.
    assert (pae > 0).all()


def test_genuinely_sparse_pae_is_rejected(tmp_path):
    """Missing entries must error rather than read as perfect confidence."""
    path = tmp_path / "scores.json"
    path.write_text(
        json.dumps({"residue1": [1, 2], "residue2": [1, 2], "distance": [1.0, 1.0]})
    )
    with pytest.raises(ValueError, match="sparse PAE"):
        load_confidence(path)


# ----------------------------------------------------------------------
# pLDDT fallback
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "bfactor,reason",
    [(1.00, "constant 1.00 would rescale to a perfect 100"), (0.00, "all zero")],
)
def test_constant_bfactor_is_not_treated_as_plddt(tmp_path, bfactor, reason):
    lines = []
    serial = 1
    for chain in ("A", "B"):
        for res_id in range(1, 6):
            for atom_name in ("N", "CA", "C", "O"):
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:<4s} ALA {chain}"
                    f"{res_id:4d}    {serial * 1.0:8.3f}{0.0:8.3f}{0.0:8.3f}"
                    f"{1.00:6.2f}{bfactor:6.2f}          {atom_name[0]:>2s}"
                )
                serial += 1
    path = tmp_path / "no_plddt.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")

    result = evaluate_candidate(
        structure=path,
        binder_chain="B",
        target_chains="A",
        thresholds={**dict.fromkeys(METRICS_BY_KEY, None), "binder_plddt": 80.0},
        compute_sc=False,
    )
    assert result.values["binder_plddt"].status == "missing", reason
    assert result.verdict != "PASS"


def test_crystallographic_bfactors_are_not_treated_as_plddt(tmp_path):
    """Real B-factors run well above the 0-100 pLDDT range."""
    lines = []
    serial = 1
    for chain in ("A", "B"):
        for res_id in range(1, 6):
            for atom_name in ("N", "CA", "C", "O"):
                bfac = 20.0 + serial * 4.0  # climbs past 100
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:<4s} ALA {chain}"
                    f"{res_id:4d}    {serial * 1.0:8.3f}{0.0:8.3f}{0.0:8.3f}"
                    f"{1.00:6.2f}{bfac:6.2f}          {atom_name[0]:>2s}"
                )
                serial += 1
    path = tmp_path / "crystal.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")

    result = evaluate_candidate(
        structure=path,
        binder_chain="B",
        target_chains="A",
        thresholds=dict.fromkeys(METRICS_BY_KEY, None),
        compute_sc=False,
    )
    assert result.values["binder_plddt"].status == "missing"


# ----------------------------------------------------------------------
# Sibling models must not share a confidence file
# ----------------------------------------------------------------------
def test_sibling_models_do_not_share_confidence(tmp_path, two_chain_pdb):
    """An AF3 directory holds several models and several full_data files.

    Stripping the trailing sample index made every model match every
    confidence file, and the first one won — so model_1 was scored with
    model_0's PAE matrix.
    """
    for i in (0, 1):
        shutil.copy(two_chain_pdb, tmp_path / f"fold_job_model_{i}.pdb")
        (tmp_path / f"fold_job_full_data_{i}.json").write_text("{}")

    for i in (0, 1):
        found = _auto_confidence(tmp_path / f"fold_job_model_{i}.pdb", ())
        assert found == [tmp_path / f"fold_job_full_data_{i}.json"]


def test_sibling_models_score_from_their_own_confidence(tmp_path, two_chain_pdb):
    """End to end: two models, two PAE matrices, two different verdicts."""
    cx = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    n = cx.n_residues

    for i, pae_value in ((0, 3.0), (1, 29.0)):
        shutil.copy(two_chain_pdb, tmp_path / f"fold_job_model_{i}.pdb")
        pae = np.full((n, n), pae_value)
        np.fill_diagonal(pae, 0.4)
        (tmp_path / f"fold_job_full_data_{i}.json").write_text(
            json.dumps({"pae": pae.tolist(), "plddt": [92.0] * n, "iptm": 0.85})
        )

    values = []
    for i in (0, 1):
        structure = tmp_path / f"fold_job_model_{i}.pdb"
        result = evaluate_candidate(
            structure=structure,
            confidence=_auto_confidence(structure, ()),
            binder_chain="B",
            target_chains="A",
            thresholds=dict.fromkeys(METRICS_BY_KEY, None),
            compute_sc=False,
        )
        values.append(result.values["pae_interaction"].value)

    # The confident model and the hopeless one must not report the same PAE.
    assert values[0] == pytest.approx(3.0, abs=0.1)
    assert values[1] == pytest.approx(29.0, abs=0.1)
