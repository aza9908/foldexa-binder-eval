"""Structure loading, chain resolution and residue bookkeeping."""

from __future__ import annotations

import numpy as np
import pytest

from foldexa_binder_eval import Complex


def test_loads_two_chains(two_chain_pdb):
    cx = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    assert cx.binder_chain == "B"
    assert cx.target_chains == ("A",)
    assert cx.n_residues == 50
    assert cx.binder_residue_mask.sum() == 20
    assert cx.target_residue_mask.sum() == 30


def test_default_binder_is_first_chain(two_chain_pdb):
    cx = Complex.from_file(two_chain_pdb)
    assert cx.binder_chain == "A"
    assert cx.target_chains == ("B",)


def test_unknown_binder_chain_is_rejected(two_chain_pdb):
    with pytest.raises(ValueError, match="not found"):
        Complex.from_file(two_chain_pdb, binder_chain="Z")


def test_unknown_target_chain_is_rejected(two_chain_pdb):
    with pytest.raises(ValueError, match="not found"):
        Complex.from_file(two_chain_pdb, binder_chain="A", target_chains="Q")


def test_binder_cannot_also_be_target(two_chain_pdb):
    with pytest.raises(ValueError, match="also listed"):
        Complex.from_file(two_chain_pdb, binder_chain="A", target_chains="A")


def test_masks_do_not_overlap(two_chain_pdb):
    cx = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    assert not (cx.binder_residue_mask & cx.target_residue_mask).any()


def test_residue_order_is_file_order(two_chain_pdb):
    cx = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    chains = cx.residue_chain_ids
    # Chain A was written first, so its residues come first.
    assert chains[0] == "A"
    assert chains[-1] == "B"


def test_representative_coords_are_finite(two_chain_pdb):
    cx = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    coords = cx.representative_coords()
    assert coords.shape == (50, 3)
    assert np.isfinite(coords).all()


def test_distance_matrix_is_symmetric_with_zero_diagonal(two_chain_pdb):
    cx = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    d = cx.residue_distance_matrix()
    assert d.shape == (50, 50)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)


def test_bfactor_is_read(two_chain_pdb):
    cx = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    b = cx.per_residue_bfactor()
    assert np.isfinite(b).all()
    assert b[cx.target_residue_mask].mean() == pytest.approx(90.0, abs=0.1)
    assert b[cx.binder_residue_mask].mean() == pytest.approx(88.0, abs=0.1)


def test_representative_bfactor_prefers_cb(two_chain_pdb):
    cx = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    rep = cx.representative_bfactor()
    assert rep.shape == (50,)
    assert np.isfinite(rep).all()


def test_sequence_is_one_letter(two_chain_pdb):
    cx = Complex.from_file(two_chain_pdb, binder_chain="B", target_chains="A")
    seq = cx.sequence("B")
    assert len(seq) == 20
    assert set(seq) <= set("INRVDQ")


def test_pair_only_drops_bystander_chains(tmp_path, two_chain_pdb):
    """A third chain stays loaded for PAE alignment but out of the geometry."""
    lines = two_chain_pdb.read_text().splitlines()
    extra = []
    serial = 10000
    for i in range(5):
        x, y, z = 100.0 + i * 3.3, 50.0, 0.0
        extra.append(
            f"ATOM  {serial:5d} CA   ALA C{i + 1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{70.00:6.2f}           C"
        )
        serial += 1
    path = tmp_path / "three_chain.pdb"
    path.write_text("\n".join([ln for ln in lines if ln != "END"] + extra + ["END"]) + "\n")

    cx = Complex.from_file(path, binder_chain="B", target_chains="A")
    assert cx.n_residues == 55  # all three chains retained
    assert cx.pair_only().n_residues == 50  # chain C excluded from geometry


def test_unsupported_format_is_rejected(tmp_path):
    bad = tmp_path / "model.xyz"
    bad.write_text("nothing here\n")
    with pytest.raises(ValueError, match="Unsupported structure format"):
        Complex.from_file(bad)


def test_invalid_keep_policy_rejected(two_chain_pdb):
    with pytest.raises(ValueError, match="keep must be"):
        Complex.from_file(two_chain_pdb, keep="some-other-policy")
