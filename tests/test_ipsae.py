"""ipSAE, pDockQ, pDockQ2 and LIS.

The important test here is :func:`test_matches_upstream_reference`, which
checks the native implementation against the score files Dunbrack's
``ipsae.py`` produced for its own example models. Those files are committed
upstream, so this is a true regression against the reference implementation
rather than against our own earlier output.
"""

from __future__ import annotations

import numpy as np
import pytest

from foldexa_binder_eval import Complex, compute_interface_scores, load_confidence
from foldexa_binder_eval.adapters import parse_ipsae_output
from foldexa_binder_eval.ipsae import calc_d0, ptm_func

from .conftest import (
    IPSAE_EXAMPLE,
    PAE_N,
    PAE_SPLIT,
    d0_reference,
    ptm_reference,
    requires_ipsae_example,
)


# ----------------------------------------------------------------------
# formulas
# ----------------------------------------------------------------------
@pytest.mark.parametrize("length", [0, 1, 25, 26, 27, 40, 100, 319, 1571])
def test_d0_matches_reference_formula(length):
    assert calc_d0(length) == pytest.approx(d0_reference(length), abs=1e-12)


def test_d0_is_floored_at_one():
    assert float(calc_d0(0)) >= 1.0
    assert float(calc_d0(26)) == pytest.approx(d0_reference(26))


def test_d0_nucleic_acid_floor_is_two():
    assert float(calc_d0(1, pair_type="nucleic_acid")) >= 2.0


@pytest.mark.parametrize("pae,d0", [(0.0, 5.0), (5.0, 5.0), (12.0, 6.5), (30.0, 2.0)])
def test_ptm_kernel(pae, d0):
    assert float(ptm_func(pae, d0)) == pytest.approx(ptm_reference(pae, d0))


def test_ptm_kernel_is_one_at_zero_error():
    assert float(ptm_func(0.0, 7.0)) == pytest.approx(1.0)


def test_ptm_kernel_decreases_with_error():
    values = [float(ptm_func(p, 6.0)) for p in (0, 2, 5, 10, 20)]
    assert values == sorted(values, reverse=True)


# ----------------------------------------------------------------------
# behaviour on synthetic input
# ----------------------------------------------------------------------
def _masks(n=PAE_N, split=PAE_SPLIT):
    target = np.zeros(n, dtype=bool)
    binder = np.zeros(n, dtype=bool)
    target[:split] = True
    binder[split:] = True
    return binder, target


def _distances(n=PAE_N, split=PAE_SPLIT):
    """Chain A along x; chain B parallel 6 A away, offset by 10 residues."""
    coords = np.zeros((n, 3))
    coords[:split, 0] = np.arange(split) * 3.3
    coords[split:, 0] = (np.arange(n - split) + 10) * 3.3
    coords[split:, 1] = 6.0
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def test_confident_interface_scores_high(synthetic_pae, synthetic_plddt):
    binder, target = _masks()
    scores = compute_interface_scores(
        synthetic_pae, synthetic_plddt, _distances(), binder, target
    )
    assert scores.ipsae > 0.5
    assert scores.n_contacts > 0
    assert 0.0 <= scores.pdockq <= 1.0


def test_unconfident_interface_scores_low(synthetic_plddt):
    binder, target = _masks()
    pae = np.full((PAE_N, PAE_N), 28.0)
    pae[:PAE_SPLIT, :PAE_SPLIT] = 3.0
    pae[PAE_SPLIT:, PAE_SPLIT:] = 3.0
    scores = compute_interface_scores(pae, synthetic_plddt, _distances(), binder, target)
    assert scores.ipsae == pytest.approx(0.0)
    assert scores.lis == pytest.approx(0.0)


def test_ipsae_is_symmetric_under_role_swap(synthetic_pae, synthetic_plddt):
    binder, target = _masks()
    distances = _distances()
    a = compute_interface_scores(synthetic_pae, synthetic_plddt, distances, binder, target)
    b = compute_interface_scores(synthetic_pae, synthetic_plddt, distances, target, binder)
    assert a.ipsae == pytest.approx(b.ipsae)
    assert a.pdockq == pytest.approx(b.pdockq)
    assert a.n_contacts == b.n_contacts


def test_overlapping_masks_rejected(synthetic_pae, synthetic_plddt):
    overlap = np.ones(PAE_N, dtype=bool)
    with pytest.raises(ValueError, match="overlap"):
        compute_interface_scores(
            synthetic_pae, synthetic_plddt, _distances(), overlap, overlap
        )


def test_shape_mismatch_rejected(synthetic_plddt):
    binder, target = _masks()
    with pytest.raises(ValueError, match="covers"):
        compute_interface_scores(
            np.zeros((10, 10)), synthetic_plddt, _distances(), binder, target
        )


def test_no_contacts_gives_zero_pdockq(synthetic_pae, synthetic_plddt):
    binder, target = _masks()
    far = _distances() + 500.0
    np.fill_diagonal(far, 0.0)
    scores = compute_interface_scores(synthetic_pae, synthetic_plddt, far, binder, target)
    assert scores.pdockq == pytest.approx(0.0)
    assert scores.n_contacts == 0


# ----------------------------------------------------------------------
# exactness against the upstream reference implementation
# ----------------------------------------------------------------------
# (structure, confidence, binder, target, pae/dist cutoff, reference file)
UPSTREAM_CASES = [
    pytest.param(
        "fold_aurka_0_tpx2_0_model_0.cif",
        "fold_aurka_0_tpx2_0_full_data_0.json",
        "B",
        "A",
        10,
        "fold_aurka_0_tpx2_0_model_0_10_10.txt",
        id="af3-aurka-tpx2",
    ),
    pytest.param(
        "RAF1_KSR1_MEK1_9f755_unrelaxed_alphafold2_multimer_v3_model_1_seed_000.pdb",
        "RAF1_KSR1_MEK1_9f755_scores_alphafold2_multimer_v3_model_1_seed_000.json",
        "B",
        "A",
        15,
        "RAF1_KSR1_MEK1_9f755_unrelaxed_alphafold2_multimer_v3_model_1_seed_000_15_15.txt",
        id="af2-raf1-ksr1",
    ),
]


def _reference_row(path, chain1: str, chain2: str) -> dict:
    """The 'max' row for a chain pair, in either column order."""
    for row in parse_ipsae_output(path):
        if row.type != "max":
            continue
        if {row.chain1, row.chain2} == {chain1, chain2}:
            return row.values
    raise AssertionError(f"no 'max' row for {chain1}/{chain2} in {path}")


@requires_ipsae_example
@pytest.mark.parametrize(
    "structure,confidence,binder,target,cutoff,reference", UPSTREAM_CASES
)
def test_matches_upstream_reference(
    structure, confidence, binder, target, cutoff, reference
):
    structure_path = IPSAE_EXAMPLE / structure
    confidence_path = IPSAE_EXAMPLE / confidence
    if not confidence_path.exists():
        gz = confidence_path.with_suffix(confidence_path.suffix + ".gz")
        if not gz.exists():
            pytest.skip(f"{confidence_path.name} not present")
        import gzip
        import shutil

        with gzip.open(gz, "rb") as src, confidence_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    cx = Complex.from_file(structure_path, binder_chain=binder, target_chains=target)
    conf = load_confidence(confidence_path).align_to(cx)

    scores = compute_interface_scores(
        pae=conf.pae,
        plddt=cx.representative_bfactor(),
        distances=cx.residue_distance_matrix(),
        binder_mask=cx.binder_residue_mask,
        target_mask=cx.target_residue_mask,
        pae_cutoff=cutoff,
        dist_cutoff=cutoff,
    )
    expected = _reference_row(IPSAE_EXAMPLE / reference, binder, target)

    # Tolerance 1e-4. Every structural quantity the score is built from --
    # n0chn, n0dom, n0res, d0res, and the set of residue pairs clearing the
    # cutoff -- agrees with the reference exactly; what is left is a
    # floating-point difference of a few times 1e-5 in the final mean. The
    # AF2 case agrees to the full six printed decimals.
    assert scores.ipsae == pytest.approx(expected["ipSAE"], abs=1e-4)
    assert scores.ipsae_d0chn == pytest.approx(expected["ipSAE_d0chn"], abs=1e-4)
    assert scores.ipsae_d0dom == pytest.approx(expected["ipSAE_d0dom"], abs=1e-4)
    assert scores.iptm_d0chn == pytest.approx(expected["ipTM_d0chn"], abs=1e-4)
    assert scores.pdockq == pytest.approx(expected["pDockQ"], abs=1e-3)
    assert scores.lis == pytest.approx(expected["LIS"], abs=1e-3)
