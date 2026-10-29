"""Structure-derived interface metrics and RMSD."""

from __future__ import annotations

import numpy as np
import pytest

from foldexa_binder_eval import Complex, backbone_rmsd, compute_interface_geometry


@pytest.fixture
def geometry(two_chain_pdb):
    cx = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    return compute_interface_geometry(cx)


def test_buried_area_is_positive(geometry):
    assert geometry.delta_sasa > 0
    assert geometry.delta_sasa_binder > 0
    assert geometry.delta_sasa_target > 0


def test_buried_area_sides_sum_to_total(geometry):
    total = geometry.delta_sasa_binder + geometry.delta_sasa_target
    assert total == pytest.approx(geometry.delta_sasa, rel=1e-6)


def test_interface_residues_found(geometry):
    assert geometry.n_interface_residues_binder > 0
    assert geometry.n_interface_residues >= geometry.n_interface_residues_binder
    assert len(geometry.interface_residue_labels) == geometry.n_interface_residues_binder


def test_interface_labels_name_the_binder_chain(geometry):
    assert all(label.startswith("B") for label in geometry.interface_residue_labels)


def test_counts_are_non_negative(geometry):
    assert geometry.n_interface_hbonds >= 0
    assert geometry.n_unsat_hbonds >= 0


def test_hbond_detected_at_textbook_distance(hbond_pdb):
    """One carbonyl O and one amide N, 2.9 A apart across the interface."""
    cx = Complex.from_file(hbond_pdb, binder_chain="B", target_chains="A")
    geometry = compute_interface_geometry(cx, compute_sc=False)
    assert geometry.n_interface_hbonds == 1


def test_hbond_not_counted_beyond_cutoff(tmp_path, hbond_pdb):
    """Push the same pair to 4.5 A: outside the donor-acceptor window."""
    out = []
    for line in hbond_pdb.read_text().splitlines():
        if line.startswith("ATOM") and line[21] == "B":
            y = float(line[38:46]) - 1.6
            line = f"{line[:38]}{y:8.3f}{line[46:]}"
        out.append(line)
    path = tmp_path / "far.pdb"
    path.write_text("\n".join(out) + "\n")

    cx = Complex.from_file(path, binder_chain="B", target_chains="A")
    assert compute_interface_geometry(cx, compute_sc=False).n_interface_hbonds == 0


def test_hydrophobicity_fractions_in_range(geometry):
    assert 0.0 <= geometry.surface_hydrophobicity <= 1.0
    assert 0.0 <= geometry.interface_hydrophobicity <= 100.0


def test_shape_complementarity_in_range(geometry):
    sc = geometry.shape_complementarity
    assert np.isnan(sc) or -1.0 <= sc <= 1.0


def test_backend_is_recorded(geometry):
    assert "sasa" in geometry.backend
    assert "shape_complementarity" in geometry.backend


def test_skipping_sc_is_honoured(two_chain_pdb):
    cx = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    geometry = compute_interface_geometry(cx, compute_sc=False)
    assert np.isnan(geometry.shape_complementarity)
    assert geometry.backend["shape_complementarity"] == "skipped"


def test_separated_chains_bury_nothing(tmp_path, two_chain_pdb):
    """Push the binder 100 A away: no interface should be detected."""
    out = []
    for line in two_chain_pdb.read_text().splitlines():
        if line.startswith("ATOM") and line[21] == "B":
            y = float(line[38:46]) + 100.0
            line = f"{line[:38]}{y:8.3f}{line[46:]}"
        out.append(line)
    path = tmp_path / "separated.pdb"
    path.write_text("\n".join(out) + "\n")

    cx = Complex.from_file(path, binder_chain="B", target_chains="A")
    geometry = compute_interface_geometry(cx)
    assert geometry.delta_sasa == pytest.approx(0.0, abs=1.0)
    assert geometry.n_interface_residues_binder == 0
    assert geometry.n_interface_hbonds == 0


# ----------------------------------------------------------------------
# RMSD
# ----------------------------------------------------------------------
def test_rmsd_against_self_is_zero(two_chain_pdb):
    cx = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    assert backbone_rmsd(cx, cx, align_on="target") == pytest.approx(0.0, abs=1e-5)
    assert backbone_rmsd(cx, cx, align_on="binder") == pytest.approx(0.0, abs=1e-5)


def test_target_aligned_rmsd_detects_known_shift(two_chain_pdb, refold_pdb):
    """The refold fixture moves every binder atom 1.0 A along z."""
    reference = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    mobile = Complex.from_file(refold_pdb, binder_chain="B", target_chains="A")
    assert backbone_rmsd(reference, mobile, align_on="target") == pytest.approx(1.0, abs=1e-3)


def test_binder_aligned_rmsd_ignores_rigid_shift(two_chain_pdb, refold_pdb):
    """A rigid translation of the binder is removed by binder-on-binder fit."""
    reference = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    mobile = Complex.from_file(refold_pdb, binder_chain="B", target_chains="A")
    assert backbone_rmsd(reference, mobile, align_on="binder") == pytest.approx(0.0, abs=1e-3)


def test_invalid_alignment_mode_rejected(two_chain_pdb):
    cx = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    with pytest.raises(ValueError, match="align_on"):
        backbone_rmsd(cx, cx, align_on="sideways")
